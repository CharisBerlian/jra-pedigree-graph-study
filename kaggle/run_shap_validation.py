"""Validation-only grouped SHAP explanations for the locked model manifest.

The production entry point is Kaggle-only. It reads validation rows and the
four manifest-selected seed-0 refits for S6/S7, but never reads test rows,
fits a model, selects a model, or computes validation performance.
"""
import argparse
from collections import defaultdict
from importlib import metadata as importlib_metadata
import hashlib
import json
import math
from pathlib import Path
import sys

import experiment_features as features
import run_experiments as runner
import run_test_inference as inference


SURFACES = ("turf", "dirt")
SETTINGS = ("S6", "S7")
SEED = 20260913
MIN_ENTRIES = 2000
MODEL_NAMES = {"xgboost": "model.json", "lightgbm": "model.txt", "catboost": "model.cbm"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def validation_sample(rows, seed=SEED, min_entries=MIN_ENTRIES):
    """Shuffle complete validation races until the entry target is reached."""
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("NumPy is required for deterministic SHAP sampling") from error
    try:
        min_entries = int(min_entries)
    except (TypeError, ValueError) as error:
        raise ValueError("min_entries must be a positive integer") from error
    if min_entries < 1:
        raise ValueError("min_entries must be positive")
    groups = defaultdict(list)
    for row in rows:
        if row.get("split") != "validation":
            raise ValueError("SHAP input must contain validation rows only")
        groups[str(row["race_id"])].append(dict(row))
    if not groups:
        raise ValueError("validation sample is empty")
    ordered = {}
    for race_id, entries in groups.items():
        sizes = {int(row["field_size"]) for row in entries
                 if row.get("field_size") not in (None, "")}
        if len(sizes) != 1 or len(entries) != next(iter(sizes)):
            raise ValueError(f"validation race is incomplete: {race_id}")
        if len({row.get("surface") for row in entries}) != 1:
            raise ValueError(f"validation race has mixed surfaces: {race_id}")
        ordered[race_id] = sorted(entries, key=lambda row: (
            int(row["horse_number"]), str(row["horse_id_netkeiba"])))
    race_ids = sorted(ordered)
    permutation = np.random.default_rng(int(seed)).permutation(race_ids)
    selected_ids = []
    selected_rows = []
    count = 0
    for race_id in permutation:
        race_id = str(race_id)
        selected_ids.append(race_id)
        selected_rows.extend(ordered[race_id])
        count += len(ordered[race_id])
        if count >= min_entries:
            break
    if count < min_entries:
        raise ValueError(f"validation rows contain only {count} entries; need {min_entries}")
    return selected_rows, selected_ids


def attribution_groups(setting, columns):
    """Return prespecified grouped feature indices for one locked setting."""
    columns = tuple(columns)
    groups = {}
    pedigree = tuple(column for column in features.PEDIGREE_CATS if column in columns)
    embeddings = tuple(column for column in features.EMBEDDING_COLUMNS if column in columns)
    graph_meta = tuple(column for column in features.GRAPH_META if column in columns)
    cluster = tuple(column for column in features.GRAPH_CLUSTER if column in columns)
    if setting in ("S3", "M3") and pedigree:
        groups["pedigree"] = pedigree
    if setting in ("S4", "S5", "S6", "S7"):
        if graph_meta:
            groups["graph_availability"] = graph_meta
        if embeddings:
            groups["graph_embeddings"] = embeddings
        if cluster:
            groups["graph_cluster"] = cluster
    return {name: tuple(columns.index(column) for column in selected)
            for name, selected in groups.items()}


def normalize_shap_values(values, row_count, feature_count):
    """Normalize SHAP's binary-output list/array variants to rows by features."""
    import numpy as np

    if hasattr(values, "values"):
        values = values.values
    if isinstance(values, list):
        if len(values) == 1:
            values = values[0]
        elif len(values) == 2:
            values = values[1]
        else:
            raise ValueError("SHAP returned an unexpected output list")
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:
        if values.shape[:2] == (row_count, feature_count) and values.shape[2] == 2:
            values = values[:, :, 1]
        elif values.shape[0] == 2 and values.shape[1:] == (row_count, feature_count):
            values = values[1]
        else:
            raise ValueError("SHAP returned an unexpected three-dimensional shape")
    if values.shape != (row_count, feature_count):
        raise ValueError(f"SHAP shape mismatch: {values.shape} != {(row_count, feature_count)}")
    if not np.isfinite(values).all():
        raise ValueError("SHAP values must be finite")
    return values


def summarize_attributions(values, columns, groups):
    """Summarize individual features and mean abs(sum(phi)) for groups."""
    import numpy as np

    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(columns):
        raise ValueError("attribution matrix does not match feature columns")
    feature_mean_abs = {column: float(np.mean(np.abs(values[:, index])))
                        for index, column in enumerate(columns)}
    group_mean_abs_sum = {}
    for name, indices in groups.items():
        if not indices:
            raise ValueError(f"attribution group is empty: {name}")
        group_mean_abs_sum[name] = float(np.mean(np.abs(values[:, indices].sum(axis=1))))
    return {"feature_mean_abs": feature_mean_abs,
            "group_mean_abs_sum": group_mean_abs_sum}


def _load_model(family, model_path):
    if family == "xgboost":
        import xgboost as xgb
        model = xgb.Booster()
    elif family == "lightgbm":
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(model_path))
        return model
    elif family == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
    else:
        raise ValueError(f"unsupported model family: {family}")
    model.load_model(str(model_path))
    return model


def explain_model(model, frame, columns, groups, shap_module=None):
    """Explain a native tree model with the locked raw-margin TreeExplainer."""
    if shap_module is None:
        import shap as shap_module
    explainer = shap_module.TreeExplainer(
        model, feature_perturbation="tree_path_dependent", model_output="raw")
    raw = explainer.shap_values(frame)
    values = normalize_shap_values(raw, len(frame), len(columns))
    return summarize_attributions(values, columns, groups), explainer


def _selected_choices(manifest):
    choices = manifest.get("shap_family_choices")
    expected = {(surface, setting) for surface in SURFACES for setting in SETTINGS}
    if not isinstance(choices, list) or len(choices) != len(expected):
        raise ValueError("manifest must contain exactly four SHAP family choices")
    result = {}
    for choice in choices:
        key = (choice.get("surface"), choice.get("setting"))
        family = choice.get("selected_family")
        if key in result or key not in expected or family not in inference.FAMILIES:
            raise ValueError("manifest SHAP choices are incomplete or duplicated")
        result[key] = family
    if set(result) != expected:
        raise ValueError("manifest SHAP choices do not cover S6/S7 on both surfaces")
    return result


def run_shap(manifest_path, spec_path, model_root, input_path, output_dir,
             source_root=None, require_kaggle=True, shap_module=None):
    """Write grouped validation SHAP output for the four selected seed-0 models."""
    if require_kaggle and not inference._on_kaggle():
        raise RuntimeError("SHAP explanation is restricted to Kaggle")
    manifest_path, spec_path, input_path = map(Path, (manifest_path, spec_path, input_path))
    model_root, output_dir = Path(model_root), Path(output_dir)
    source_root = Path(source_root) if source_root is not None else Path(__file__).resolve().parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    choices = _selected_choices(manifest)
    model_keys = {(family, surface, setting) for (surface, setting), family in choices.items()}
    entries = inference._validate_manifest(manifest, spec, spec_path, source_root,
                                           model_keys=model_keys)
    rows, columns = runner.read_rows(input_path)
    metadata = runner._metadata(input_path)
    if metadata.get("seed") != 0:
        raise ValueError("SHAP input must be the seed-0 train/validation export")
    validation_by_surface = {
        surface: [row for row in rows
                  if row.get("surface") == surface and row.get("split") == "validation"]
        for surface in SURFACES}
    samples = {}
    for surface, surface_rows in validation_by_surface.items():
        samples[surface] = validation_sample(surface_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite SHAP output: {output_dir}")
    if shap_module is None:
        import shap as shap_module
    shap_version = getattr(shap_module, "__version__", None)
    if shap_version is None:
        shap_version = importlib_metadata.version("shap")
    package_versions = {"shap": str(shap_version)}
    for distribution in ("numpy", "pandas", "xgboost", "lightgbm", "catboost"):
        package_versions[distribution] = importlib_metadata.version(distribution)
    results = []
    for (surface, setting), family in sorted(choices.items()):
        key = (family, surface, setting)
        entry = entries[key]
        selected_round, refits = inference._validate_model_entry(entry, model_root, spec, source_root)
        seed0 = [value for value in refits if value[0] == 0]
        if len(seed0) != 1:
            raise ValueError(f"manifest refit seed0 is incomplete: {key}")
        _, refit_dir, config, preprocessing, _ = seed0[0]
        sample_rows, sample_races = samples[surface]
        frame = inference._prepare_frame(sample_rows, preprocessing, family)
        selected_columns = tuple(preprocessing["features"])
        expected_columns = features.feature_columns(setting, columns)
        if selected_columns != expected_columns:
            raise ValueError(f"manifest preprocessing differs from frozen allowlist: {key}")
        groups = attribution_groups(setting, selected_columns)
        model_path = refit_dir / MODEL_NAMES[family]
        model = _load_model(family, model_path)
        summary, _ = explain_model(model, frame, selected_columns, groups, shap_module)
        results.append({
            "family": family, "surface": surface, "setting": setting, "seed": 0,
            "selected_round": int(selected_round), "model_sha256": sha256_file(model_path),
            "config_sha256": sha256_file(refit_dir / "config.json"),
            "preprocessing_sha256": sha256_file(refit_dir / "preprocessing.json"),
            "validation_race_ids": sample_races,
            "validation_entry_count": len(sample_rows),
            "validation_race_count": len(sample_races),
            "sample_seed": SEED, "sample_min_entries": MIN_ENTRIES,
            "sample_ids_sha256": canonical_hash(sample_races), "features": list(selected_columns),
            **summary,
        })
    report = {
        "purpose": "validation-only grouped SHAP explanations",
        "script_sha256": sha256_file(Path(__file__)),
        "runner_sha256": sha256_file(source_root / "run_experiments.py"),
        "inference_sha256": sha256_file(source_root / "run_test_inference.py"),
        "feature_helper_sha256": sha256_file(source_root / "experiment_features.py"),
        "manifest_sha256": sha256_file(manifest_path), "spec_sha256": sha256_file(spec_path),
        "input_sha256": sha256_file(input_path),
        "input_metadata_sha256": sha256_file(input_path.with_suffix(".json")),
        "input_seed": metadata["seed"], "input_splits": metadata["splits"],
        "python": sys.version.split()[0], "packages": package_versions,
        "model_output": "raw_margin",
        "feature_perturbation": "tree_path_dependent",
        "group_attribution": "mean_abs_sum_of_group_phi", "results": results,
        "test_rows_read": False, "test_performance_computed": False,
    }
    output = output_dir / "shap_validation_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                      encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args(argv)
    report = run_shap(args.manifest, args.spec, args.model_root, args.input,
                      args.output_dir, source_root=args.source_root)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
