"""Inference-only test runner for a locked selection manifest.

This module reads test features and frozen refit artifacts on Kaggle. It has
no fit, tune, or validation-selection operation; test labels are never used.
"""
import csv
import gzip
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path

import experiment_features as features


FAMILIES = ("xgboost", "lightgbm", "catboost")
SURFACES = ("turf", "dirt")
SEEDS = (0, 1, 2)
SETTINGS = ("B1", "S1", "S2", "S3", "S4", "S5", "S6", "M2", "M3", "S7")
KEYS = ("race_id", "horse_id_netkeiba", "horse_number")
MODEL_NAMES = {"xgboost": "model.json", "lightgbm": "model.txt", "catboost": "model.cbm"}
THRESHOLD_RULE = "validation_grid_0.01_to_0.99_step_0.01_max_f1"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()


def _runtime_spec_sha256(spec):
    """Match the exact sorted JSON bytes written into Kaggle runtime work."""
    payload = json.dumps(spec, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing inference artifact: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid inference JSON: {path}") from error


def _on_kaggle():
    return Path("/kaggle").is_dir()


def _integer(value, name):
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be integer-valued") from error
    if not math.isfinite(numeric) or numeric != parsed:
        raise ValueError(f"{name} must be integer-valued")
    return parsed


def _test_sidecar(path):
    path = Path(path)
    sidecars = [path.with_suffix(".json"), Path(str(path) + ".json")]
    sidecars = [candidate for candidate in sidecars if candidate.is_file()]
    if len(sidecars) != 1:
        raise ValueError(f"expected one test-input sidecar for {path}: {sidecars}")
    return sidecars[0]


def read_test_rows(path, expected_seed=None):
    """Read only a test-split feature file and verify complete race keys.

    The parser deliberately does not access a label column. A sidecar beside
    the file must bind its bytes and declare the sole split as ``test``.
    """
    path = Path(path)
    sidecar = _test_sidecar(path)
    metadata = _json(sidecar)
    actual_sha256 = _sha256(path)
    if metadata.get("data_sha256") != actual_sha256:
        if path.suffix != ".csv" or metadata.get("uncompressed_csv_sha256") != actual_sha256:
            raise ValueError("test input checksum does not match sidecar")
    if metadata.get("splits") != ["test"]:
        raise ValueError("test input sidecar must declare exactly the test split")
    if expected_seed is not None and metadata.get("seed") not in (None, expected_seed):
        raise ValueError("test input sidecar seed does not match its resolved seed")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("test feature input is empty")
    missing = [key for key in (*KEYS, "split", "surface", "field_size")
               if key not in (rows[0] or {})]
    if missing:
        raise ValueError(f"test feature input lacks required columns: {missing}")
    seen = set()
    groups = {}
    for row in rows:
        if row.get("split") != "test":
            raise ValueError("test inference input contains a non-test split")
        key = (str(row["race_id"]), str(row["horse_id_netkeiba"]),
               _integer(row["horse_number"], "horse_number"))
        if key in seen:
            raise ValueError(f"duplicate test row key: {key}")
        seen.add(key)
        row["race_id"], row["horse_id_netkeiba"], row["horse_number"] = key
        field_size = _integer(row.get("field_size"), "field_size")
        if field_size < 3:
            raise ValueError("test field_size must be at least three")
        row["field_size"] = field_size
        groups.setdefault(key[0], []).append(row)
    for race_id, entries in groups.items():
        sizes = {row["field_size"] for row in entries}
        if len(sizes) != 1 or len(entries) != next(iter(sizes)) or len({row["surface"] for row in entries}) != 1:
            raise ValueError(f"test race is incomplete: {race_id}")
    return rows, metadata


def _seed_token(path):
    match = re.search(r"(?:^|[_-])seed[_-]?([012])(?:\.|$)", Path(path).name.lower())
    return int(match.group(1)) if match else None


def resolve_test_inputs(test_input):
    """Resolve exactly one ``test_seed0/1/2`` feature file per seed.

    ``test_input`` may be a directory containing the three canonical files or
    an explicit mapping whose keys are the integer seed IDs. The resolver only
    inspects paths; row contents are opened later, after manifest validation.
    """
    if isinstance(test_input, Mapping):
        resolved = {}
        for raw_seed, raw_path in test_input.items():
            try:
                seed = _integer(raw_seed, "test seed")
            except ValueError as error:
                raise ValueError("explicit test input mapping keys must be seeds 0, 1, and 2") from error
            if seed in resolved or seed not in SEEDS:
                raise ValueError("explicit test input mapping keys must be exactly 0, 1, and 2")
            path = Path(raw_path)
            if not path.is_file():
                raise ValueError(f"test input file is missing for seed {seed}: {path}")
            token = _seed_token(path)
            if token is not None and token != seed:
                raise ValueError(f"test input filename seed disagrees with mapping: {path}")
            resolved[seed] = path
        if set(resolved) != set(SEEDS):
            raise ValueError("explicit test input mapping must contain exactly seeds 0, 1, and 2")
        return resolved
    path = Path(test_input)
    if not path.is_dir():
        raise ValueError("test input must be a directory or an explicit seed mapping")
    resolved = {}
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        match = re.fullmatch(r"test[_-]?seed[_-]?([012])\.csv(?:\.gz)?", candidate.name.lower())
        if not match:
            continue
        seed = int(match.group(1))
        if seed in resolved:
            raise ValueError(f"duplicate test input file for seed {seed}")
        resolved[seed] = candidate
    if set(resolved) != set(SEEDS):
        raise ValueError("test input directory must contain exactly test_seed0/1/2.csv[.gz]")
    return resolved


def _artifact_hashes(folder):
    folder = Path(folder)
    hashes = {}
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.name == "complete.json":
            continue
        name = path.relative_to(folder).as_posix()
        key = name.removesuffix(".gz") if path.name.endswith(".csv.gz") else name
        if key in hashes:
            raise ValueError(f"duplicate canonical artifact path: {key}")
        if path.name.endswith(".csv.gz"):
            digest = hashlib.sha256()
            with gzip.open(path, "rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            hashes[key] = digest.hexdigest()
        else:
            hashes[key] = _sha256(path)
    return hashes


def _canonical_artifact_manifest(artifacts):
    canonical = {}
    for name, digest in (artifacts or {}).items():
        key = name.removesuffix(".gz") if name.endswith(".csv.gz") else name
        if key in canonical:
            raise ValueError(f"duplicate canonical artifact path: {key}")
        canonical[key] = digest
    return canonical


def _complete(folder, expected_artifacts):
    folder = Path(folder)
    marker = _json(folder / "complete.json")
    expected = _canonical_artifact_manifest(expected_artifacts)
    marker_artifacts = _canonical_artifact_manifest(marker.get("artifacts"))
    if marker.get("status") != "complete" or marker_artifacts != expected:
        raise ValueError(f"inference checkpoint is incomplete or changed: {folder}")
    if marker_artifacts != _artifact_hashes(folder):
        raise ValueError(f"inference artifact hash mismatch: {folder}")
    return marker


def _validate_manifest(manifest, spec, spec_path, source_root, model_keys=None):
    if manifest.get("purpose") != "JRA test-release model selection manifest" or \
            manifest.get("status") != "locked_for_test_release":
        raise ValueError("selection manifest is not locked for test release")
    if manifest.get("test_performance_opened") is not False or \
            manifest.get("contains_test_predictions") is not False or \
            manifest.get("no_test_performance") is not True:
        raise ValueError("selection manifest test gate is not closed")
    if spec.get("status") != "locked" or spec.get("test_performance_opened") is not False:
        raise ValueError("scientific specification is not locked with test performance closed")
    runtime_spec_hash = _runtime_spec_sha256(spec)
    if (manifest.get("spec_id") != spec.get("spec_id") or
            manifest.get("spec_sha256") != runtime_spec_hash or
            manifest.get("spec_runtime_sha256") != runtime_spec_hash or
            manifest.get("spec_source_sha256") != _sha256(spec_path)):
        raise ValueError("manifest/spec identity or checksum mismatch")
    if tuple(spec.get("families", ())) != FAMILIES or tuple(spec.get("surfaces", ())) != SURFACES or \
            tuple(spec.get("settings", ())) != SETTINGS:
        raise ValueError("locked specification matrix mismatch")
    if spec.get("matched_seeds") != list(SEEDS):
        raise ValueError("locked specification seed matrix mismatch")
    if manifest.get("audited_counts") != {"tuning_runs": 60, "candidate_runs": 240, "refit_runs": 180}:
        raise ValueError("selection manifest does not record complete audited counts")
    expected_sources = {
        "audit_experiment_outputs.py": manifest.get("audit_script_sha256"),
        "evaluate_locked.py": manifest.get("evaluator_script_sha256"),
        "run_experiments.py": manifest.get("provenance", {}).get("script_sha256"),
        "experiment_features.py": manifest.get("provenance", {}).get("helper_sha256"),
    }
    for name, expected in expected_sources.items():
        if not isinstance(expected, str) or _sha256(Path(source_root) / name) != expected:
            raise ValueError(f"locked source checksum mismatch: {name}")
    models = manifest.get("models")
    expected_keys = {(family, surface, setting) for family in FAMILIES
                     for surface in SURFACES for setting in SETTINGS}
    if model_keys is None:
        requested = expected_keys
    else:
        requested = set(model_keys)
        if not requested or not requested.issubset(expected_keys):
            raise ValueError("requested inference model keys are outside the locked matrix")
    if not isinstance(models, list) or len(models) != len(expected_keys):
        raise ValueError("selection manifest must contain all 60 model entries")
    by_key = {}
    for entry in models:
        key = (entry.get("family"), entry.get("surface"), entry.get("setting"))
        if key in by_key or key not in expected_keys:
            raise ValueError("selection manifest model matrix is incomplete or duplicated")
        by_key[key] = entry
    if set(by_key) != expected_keys:
        raise ValueError("selection manifest model matrix is incomplete")
    return {key: by_key[key] for key in requested}


def _validate_model_entry(entry, model_root, spec, source_root):
    family, surface, setting = entry["family"], entry["surface"], entry["setting"]
    tune_info = entry.get("tuning")
    refit_info = entry.get("refits")
    if not isinstance(tune_info, dict) or not isinstance(refit_info, list) or len(refit_info) != 3:
        raise ValueError(f"manifest model entry is incomplete: {family}/{surface}/{setting}")
    base = Path(model_root) / "scientific" / family / surface / setting
    expected_tune_rel = Path("scientific") / family / surface / setting / "tune" / "seed0"
    tune_rel = Path(tune_info.get("relative_dir", expected_tune_rel.as_posix()))
    if tune_rel.as_posix() != expected_tune_rel.as_posix():
        raise ValueError(f"tuning relative path mismatch: {family}/{surface}/{setting}")
    tune = Path(model_root) / tune_rel
    tune_marker = _complete(tune, tune_info.get("artifact_hashes", {}))
    tune_config_path = tune / "config.json"
    tune_config = _json(tune_config_path)
    if _sha256(tune_config_path) != tune_info.get("config_sha256") or \
            tune_marker.get("config_hash") != tune_info.get("config_hash") or \
            tune_marker.get("config_hash") != _canonical(tune_config) or \
            tune_config.get("candidate_grid_sha256") != tune_info.get("candidate_grid_sha256") or \
            tune_config.get("candidate_grid_sha256") != _canonical(spec["tuning"]["candidates"][family]):
        raise ValueError(f"tuning provenance mismatch: {tune}")
    selected_path = tune / "selected.json"
    selected = _json(selected_path)
    if _sha256(selected_path) != tune_info.get("selected_sha256") or selected != tune_info.get("selected"):
        raise ValueError(f"tuning selection mismatch: {tune}")
    if selected.get("status") != "complete" or selected.get("candidate_id") not in ("c0", "c1", "c2", "c3"):
        raise ValueError(f"tuning selection is incomplete: {tune}")
    candidate_manifest = tune_info.get("candidates", {}).get(selected["candidate_id"])
    if not isinstance(candidate_manifest, dict) or selected.get("params") != candidate_manifest.get("params"):
        raise ValueError(f"tuning selected parameters are not bound to candidate grid: {tune}")
    selected_round = _integer(selected.get("best_round"), "selected round")
    if selected_round < 1 or selected_round > int(spec["tuning"]["max_rounds"]):
        raise ValueError(f"selected round is outside locked tuning cap: {tune}")
    expected_provenance = entry["tuning"].get("config_provenance", {})
    if not expected_provenance:
        raise ValueError(f"tuning provenance is missing: {tune}")
    outputs = []
    for refit in refit_info:
        seed = _integer(refit.get("seed"), "refit seed")
        if seed not in SEEDS:
            raise ValueError(f"invalid refit seed: {base}")
        expected_refit_rel = Path("scientific") / family / surface / setting / "refit" / f"seed{seed}"
        refit_rel = Path(refit.get("relative_dir", expected_refit_rel.as_posix()))
        if refit_rel.as_posix() != expected_refit_rel.as_posix():
            raise ValueError(f"refit relative path mismatch: {family}/{surface}/{setting}/seed{seed}")
        refit_dir = Path(model_root) / refit_rel
        marker = _complete(refit_dir, refit.get("artifact_hashes", {}))
        config_path = refit_dir / "config.json"
        config = _json(config_path)
        metrics_path = refit_dir / "metrics.json"
        metrics = _json(metrics_path)
        if _sha256(config_path) != refit.get("config_sha256") or \
                _sha256(metrics_path) != refit.get("metrics_sha256") or \
                _sha256(refit_dir / "complete.json") != refit.get("complete_sha256") or \
                marker.get("config_hash") != _canonical(config):
            raise ValueError(f"refit manifest hashes mismatch: {refit_dir}")
        for key, value in {"family": family, "surface": surface, "setting": setting,
                           "stage": "refit", "seed": seed, "spec_id": spec["spec_id"],
                           "selected_candidate": selected["candidate_id"],
                           "selected_round": selected_round, "params": selected["params"]}.items():
            if config.get(key) != value:
                raise ValueError(f"refit config mismatch for {key}: {refit_dir}")
        if config.get("features") != tune_config.get("features") or \
                config.get("categorical") != tune_config.get("categorical"):
            raise ValueError(f"refit feature schema mismatch: {refit_dir}")
        if config.get("spec_sha256") != expected_provenance.get("spec_sha256") or \
                config.get("script_sha256") != expected_provenance.get("script_sha256") or \
                config.get("helper_sha256") != expected_provenance.get("helper_sha256") or \
                config.get("environment_sha256") != expected_provenance.get("environment_sha256"):
            raise ValueError(f"refit source provenance mismatch: {refit_dir}")
        if refit.get("config_provenance") != {key: config.get(key) for key in (
                "spec_sha256", "script_sha256", "helper_sha256", "environment_sha256",
                "input_sha256", "input_metadata_sha256")}:
            raise ValueError(f"refit config provenance differs from manifest: {refit_dir}")
        threshold = metrics.get("validation_threshold")
        selected_threshold = threshold.get("selected") if isinstance(threshold, dict) else None
        if not isinstance(threshold, dict) or threshold.get("rule") != THRESHOLD_RULE or \
                threshold.get("grid") != [0.01, 0.99, 0.01] or not isinstance(selected_threshold, dict):
            raise ValueError(f"refit validation threshold is missing or changed: {refit_dir}")
        threshold_value = float(selected_threshold.get("threshold"))
        if not math.isfinite(threshold_value) or threshold_value < .01 or threshold_value > .99:
            raise ValueError(f"refit threshold is outside the validation grid: {refit_dir}")
        preprocessing = _json(refit_dir / "preprocessing.json")
        artifact_hashes = refit.get("artifact_hashes", {})
        if artifact_hashes.get("preprocessing.json") != _sha256(refit_dir / "preprocessing.json") or \
                artifact_hashes.get(MODEL_NAMES[family]) != _sha256(refit_dir / MODEL_NAMES[family]):
            raise ValueError(f"refit model/preprocessing hash mismatch: {refit_dir}")
        expected_features = list(features.feature_columns(setting))
        expected_categorical = list(features.categorical_columns(expected_features))
        if preprocessing.get("spec_id") != spec["spec_id"] or preprocessing.get("setting") != setting or \
                preprocessing.get("features") != expected_features or \
                preprocessing.get("categorical_columns") != expected_categorical or \
                preprocessing.get("features") != config.get("features") or \
                preprocessing.get("categorical_columns") != config.get("categorical"):
            raise ValueError(f"refit preprocessing schema mismatch: {refit_dir}")
        category_maps = preprocessing.get("category_maps")
        if not isinstance(category_maps, dict) or set(category_maps) != set(config.get("categorical", ())):
            raise ValueError(f"refit category maps are incomplete: {refit_dir}")
        outputs.append((seed, refit_dir, config, preprocessing, selected_threshold))
    if {item[0] for item in outputs} != set(SEEDS):
        raise ValueError(f"refit seed matrix is incomplete: {base}")
    return selected_round, outputs


def _number(value):
    if value in (None, "", "NULL", "None"):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"non-numeric predictor value: {value!r}") from error


def _prepare_frame(rows, preprocessing, family):
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("pandas is required for native test inference") from error
    columns = tuple(preprocessing["features"])
    maps = preprocessing["category_maps"]
    transformed = list(features.transform_rows(rows, columns, maps))
    frame = pd.DataFrame([[ _number(row.get(column)) for column in columns]
                          for row in transformed], columns=columns)
    for column, mapping in maps.items():
        categories = sorted(set(mapping.values()))
        if family == "catboost":
            frame[column] = frame[column].astype(str)
        else:
            frame[column] = pd.Categorical(frame[column], categories=categories)
    return frame


def _load_predict(family, model_path, frame, selected_round):
    if family == "xgboost":
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(str(model_path))
        if hasattr(model, "num_boosted_rounds") and model.num_boosted_rounds() != selected_round:
            raise ValueError("XGBoost model round count differs from locked selected round")
        scores = model.predict(xgb.DMatrix(frame, enable_categorical=True))
    elif family == "lightgbm":
        os.environ["MPLBACKEND"] = "Agg"
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(model_path))
        scores = model.predict(frame, num_iteration=selected_round)
    elif family == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        if hasattr(model, "tree_count_") and model.tree_count_ != selected_round:
            raise ValueError("CatBoost model round count differs from locked selected round")
        scores = model.predict_proba(frame)[:, 1]
    else:
        raise ValueError(f"unsupported model family: {family}")
    values = [float(value) for value in scores]
    if len(values) != len(frame) or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("native model returned invalid test probabilities")
    return values


def _write_predictions(path, rows, scores):
    ranked = {}
    for row, score in zip(rows, scores):
        ranked.setdefault(row["race_id"], []).append((row, score))
    selected = set()
    for entries in ranked.values():
        selected.update((row["race_id"], row["horse_id_netkeiba"], row["horse_number"])
                        for row, _ in sorted(
                            entries, key=lambda item: (-item[1], item[0]["horse_number"],
                                                       item[0]["horse_id_netkeiba"]))[:3])
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["race_id", "horse_id_netkeiba", "horse_number", "surface", "prediction", "selected_top3"])
        for row, score in zip(rows, scores):
            key = (row["race_id"], row["horse_id_netkeiba"], row["horse_number"])
            writer.writerow([row["race_id"], row["horse_id_netkeiba"], row["horse_number"],
                             row["surface"], "%.17g" % score, int(key in selected)])


def run_inference(manifest_path, spec_path, model_root, test_input, output_dir,
                  source_root=None, model_keys=None, require_kaggle=True):
    """Run locked native inference and write one test prediction file per refit.

    ``model_keys`` is intended only for local synthetic checks; production
    calls leave it unset and therefore require all 60 manifest model entries.
    """
    if require_kaggle and not _on_kaggle():
        raise RuntimeError("test inference is restricted to Kaggle")
    source_root = Path(source_root) if source_root is not None else Path(__file__).resolve().parent
    manifest_path, spec_path = Path(manifest_path), Path(spec_path)
    manifest = _json(manifest_path)
    spec = _json(spec_path)
    entries = _validate_manifest(manifest, spec, spec_path, source_root, model_keys=model_keys)
    test_inputs = resolve_test_inputs(test_input)
    rows_by_seed = {seed: read_test_rows(path, expected_seed=seed)[0]
                    for seed, path in sorted(test_inputs.items())}
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite inference output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for key, entry in sorted(entries.items()):
        selected_round, refits = _validate_model_entry(entry, model_root, spec, source_root)
        for seed, refit_dir, config, preprocessing, threshold in sorted(refits):
            rows = rows_by_seed[seed]
            surface_rows = [row for row in rows if row["surface"] == key[1]]
            if not surface_rows:
                raise ValueError(f"test input seed {seed} has no rows for surface {key[1]}")
            frame = _prepare_frame(surface_rows, preprocessing, key[0])
            scores = _load_predict(key[0], refit_dir / MODEL_NAMES[key[0]], frame, selected_round)
            destination = output_dir / key[0] / key[1] / key[2] / f"seed{seed}" / "test_predictions.csv.gz"
            _write_predictions(destination, surface_rows, scores)
            written.append({"family": key[0], "surface": key[1], "setting": key[2],
                            "seed": seed, "selected_round": selected_round,
                            "threshold": float(threshold["threshold"]),
                            "path": str(destination), "rows": len(surface_rows)})
    report = {"status": "complete", "purpose": "JRA locked test inference; no metrics",
              "manifest_sha256": _sha256(manifest_path), "spec_sha256": _sha256(spec_path),
              "spec_runtime_sha256": _runtime_spec_sha256(spec),
              "spec_source_sha256": _sha256(spec_path),
              "test_input_sha256": {str(seed): _sha256(path) for seed, path in sorted(test_inputs.items())},
              "test_input_metadata_sha256": {str(seed): _sha256(_test_sidecar(path))
                                             for seed, path in sorted(test_inputs.items())},
              "test_rows": {str(seed): len(rows_by_seed[seed]) for seed in SEEDS},
              "model_outputs": len(written),
              "test_labels_read": False, "test_metrics_computed": False, "outputs": written}
    (output_dir / "inference_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--test-input", type=Path, required=True,
                        help="directory containing test_seed0/1/2.csv[.gz]")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_inference(args.manifest, args.spec, args.model_root,
                                   args.test_input, args.output), indent=2, sort_keys=True))
