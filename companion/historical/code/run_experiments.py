"""Run one bounded train/validation experiment on Kaggle.

The command owns one surface/setting/family at a time.  Tuning is seed 0 only;
refits consume the frozen seed-0 winner and round count.  Test rows are
rejected before any model library is imported.
"""
import argparse
import csv
import gzip
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
from pathlib import Path
import sys
import time

import experiment_features as features

FAMILIES = ("xgboost", "lightgbm", "catboost")
SURFACES = ("turf", "dirt")
SEEDS = (0, 1, 2)
# Checkpoints written before canonical CSV hashing used this runner revision.
LEGACY_ARTIFACT_SCRIPT_SHA256 = "be4d1c546a683cc45c85f755a86ab29d0ba652b13625f58c8c4919f7a2b6a5ab"


def sha256_file(path):
    digest = hashlib.sha256()
    path = Path(path)
    opener = gzip.open if path.name.endswith(".csv.gz") else open
    with opener(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_uncompressed_sha256(path):
    """Hash CSV payload bytes, independent of gzip container metadata."""
    digest = hashlib.sha256()
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_sha256(path):
    return csv_uncompressed_sha256(path) if Path(path).suffix == ".gz" else sha256_file(path)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_rows(path):
    """Read and structurally validate only the train/validation export."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("feature input is empty")
    if any(row.get("split") not in ("train", "validation") for row in rows):
        raise ValueError("test or unknown split found; runner is validation-only")
    keys = {(row["race_id"], row["horse_id_netkeiba"], row["horse_number"]) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("duplicate feature keys")
    by_race = {}
    for row in rows:
        by_race.setdefault(row["race_id"], []).append(row)
    for race_id, entries in by_race.items():
        sizes = {int(row["field_size"]) for row in entries if row.get("field_size") not in (None, "")}
        if len(sizes) != 1 or len(entries) != next(iter(sizes)):
            raise ValueError(f"incomplete or inconsistent field_size for race {race_id}")
    return rows, tuple(rows[0])


def feature_allowlist(setting, columns):
    """Use the required companion helper for the frozen feature policy."""
    return tuple(features.feature_columns(setting, columns))


def prepare_values(rows, columns, maps, family):
    """Create native categorical columns from helper encoded values."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required on Kaggle for native categories") from exc
    values = []
    for row in features.transform_rows(rows, columns, maps):
        values.append([_number(row.get(col)) for col in columns])
    frame = pd.DataFrame(values, columns=columns)
    for col, mapping in maps.items():
        categories = sorted(set(mapping.values()))
        if family == "catboost":
            frame[col] = frame[col].astype(str)
        else:
            frame[col] = pd.Categorical(frame[col], categories=categories)
    return frame


def _number(value):
    if value in (None, "", "NULL", "None"):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non-numeric predictor value: {value!r}") from exc


def labels(rows):
    return [int(row["label"]) for row in rows]


def top_three(rows, scores):
    ranked = sorted(zip(rows, scores), key=lambda pair: (-pair[1], int(pair[0]["horse_number"]), pair[0]["horse_id_netkeiba"]))
    return ranked[:3]


def mean_hit_at_3(rows, scores):
    groups = {}
    for row, score in zip(rows, scores):
        groups.setdefault(row["race_id"], []).append((row, score))
    values = []
    for entries in groups.values():
        if len(entries) >= 3:
            selected = top_three([row for row, _ in entries], [score for _, score in entries])
            values.append(sum(int(row["label"]) for row, _ in selected) / 3.0)
    return (sum(values) / len(values), len(values)) if values else (float("nan"), 0)


def metric_values(rows, scores):
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    y = labels(rows)
    hit3, race_count = mean_hit_at_3(rows, scores)
    return {"average_precision": float(average_precision_score(y, scores)),
            "roc_auc": (float(roc_auc_score(y, scores)) if len(set(y)) == 2 else None),
            "brier": float(brier_score_loss(y, scores)),
            "mean_hit_at_3": hit3, "complete_validation_races": race_count}


def _binary_threshold_metrics(y, scores, threshold):
    tp = int(sum(label == 1 and score >= threshold for label, score in zip(y, scores)))
    fp = int(sum(label == 0 and score >= threshold for label, score in zip(y, scores)))
    fn = int(sum(label == 1 and score < threshold for label, score in zip(y, scores)))
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1,
            "predicted_positive": int(tp + fp)}


def validation_threshold(rows, scores, spec):
    """Fit the preregistered threshold on validation rows for final refit evidence."""
    rule = spec.get("threshold", {}).get("fitted_settings")
    expected = "validation_grid_0.01_to_0.99_step_0.01_max_f1"
    if rule != expected:
        raise ValueError(f"unsupported validation threshold rule: {rule!r}")
    y = labels(rows)
    candidates = [_binary_threshold_metrics(y, scores, index / 100.0) for index in range(1, 100)]
    winner = max(candidates, key=lambda value: (value["f1"], value["precision"],
                                                  -abs(value["threshold"] - 0.5),
                                                  value["threshold"]))
    return {"rule": rule, "grid": [0.01, 0.99, 0.01], "selected": winner}


def _ap_curve_best(curve):
    if not curve:
        raise ValueError("empty validation AP curve")
    best = max(float(value) for value in curve)
    return min(index + 1 for index, value in enumerate(curve) if abs(float(value) - best) <= 1e-15), best


def _import_lightgbm():
    """Import LightGBM safely from notebook processes with an inline backend."""
    import os
    os.environ["MPLBACKEND"] = "Agg"
    matplotlib = sys.modules.get("matplotlib")
    if matplotlib is not None:
        try:
            matplotlib.use("Agg", force=True)
        except (ImportError, ValueError):
            pass
    import lightgbm as lgb
    return lgb


def _fit_xgboost(x_train, y_train, x_val, y_val, params, seed, policy):
    import xgboost as xgb
    from sklearn.metrics import average_precision_score
    def ap_metric(prediction, dataset):
        return "average_precision", float(average_precision_score(dataset.get_label(), prediction))
    dtrain = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(x_val, label=y_val, enable_categorical=True)
    config = {"objective": "binary:logistic", "tree_method": "hist", "enable_categorical": True,
              "max_cat_to_onehot": 32, "max_cat_threshold": 64, "nthread": policy["threads"],
              "seed": seed, **params}
    evals_result = {}
    model = xgb.train(config, dtrain, num_boost_round=policy["max_rounds"],
                      evals=[(dval, "validation")], custom_metric=ap_metric,
                      evals_result=evals_result,
                      callbacks=[xgb.callback.EarlyStopping(rounds=policy["patience"], maximize=True,
                                                             save_best=True, data_name="validation",
                                                             metric_name="average_precision")], verbose_eval=False)
    curve = [float(value) for value in evals_result.get("validation", {}).get("average_precision", ())]
    return model, curve, model.predict(dval)


def _fit_lightgbm(x_train, y_train, x_val, y_val, params, seed, categorical, policy):
    lgb = _import_lightgbm()
    from sklearn.metrics import average_precision_score
    def ap_metric(y_true, prediction):
        return "average_precision", float(average_precision_score(y_true, prediction)), True
    evals_result = {}
    model = lgb.LGBMClassifier(objective="binary", n_estimators=policy["max_rounds"],
                               random_state=seed, n_jobs=policy["threads"], verbosity=-1,
                               metric="None", cat_smooth=20, cat_l2=10,
                               min_data_per_group=20, max_cat_threshold=64,
                               **params)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric=ap_metric,
              categorical_feature=list(categorical), callbacks=[
                  lgb.record_evaluation(evals_result),
                  lgb.early_stopping(policy["patience"], first_metric_only=True, verbose=False)])
    curve = [float(value) for value in evals_result.get("valid_0", {}).get("average_precision", ())]
    return model, curve, model.predict_proba(x_val)[:, 1]


def _fit_catboost(x_train, y_train, x_val, y_val, params, seed, categorical, policy):
    from catboost import CatBoostClassifier
    from sklearn.metrics import average_precision_score
    class AveragePrecisionMetric:
        def is_max_optimal(self):
            return True
        def get_final_error(self, error, weight):
            return error
        def evaluate(self, approxes, target, weight):
            prediction = approxes[0]
            prediction = [1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value)))) for value in prediction]
            return float(average_precision_score(target, prediction)), 1.0
    model = CatBoostClassifier(iterations=policy["max_rounds"], random_seed=seed, thread_count=policy["threads"],
                               loss_function="Logloss", eval_metric=AveragePrecisionMetric(),
                               cat_features=list(categorical), one_hot_max_size=32,
                               max_ctr_complexity=1, bootstrap_type="Bayesian",
                               bagging_temperature=1, verbose=False, allow_writing_files=False,
                               **params)
    model.fit(x_train, y_train, eval_set=(x_val, y_val), early_stopping_rounds=policy["patience"],
              use_best_model=True, verbose=False)
    evaluation = model.get_evals_result().get("validation", {})
    metric_name = "AveragePrecisionMetric"
    curve = [float(value) for value in evaluation.get(metric_name, ())]
    if not curve:
        raise ValueError("CatBoost did not record the custom AveragePrecisionMetric curve")
    best_round, _ = _ap_curve_best(curve)
    if model.get_best_iteration() < 0 or model.get_best_iteration() + 1 != best_round:
        raise ValueError("CatBoost best iteration disagrees with custom AP curve")
    return model, curve, model.predict_proba(x_val)[:, 1]


def fit_model(family, x_train, y_train, x_val, y_val, params, seed, categorical, policy):
    if family == "xgboost":
        return _fit_xgboost(x_train, y_train, x_val, y_val, params, seed, policy)
    if family == "lightgbm":
        return _fit_lightgbm(x_train, y_train, x_val, y_val, params, seed, categorical, policy)
    return _fit_catboost(x_train, y_train, x_val, y_val, params, seed, categorical, policy)


def choose_candidate(records, tolerance):
    valid = [record for record in records if record.get("status") == "complete"]
    if len(valid) != 4:
        raise ValueError("all four tuning candidates must complete")
    maximum_ap = max(record["average_precision"] for record in valid)
    eligible = [record for record in valid
                if maximum_ap - record["average_precision"] <= tolerance]
    return min(eligible, key=lambda record: (
        -record["mean_hit_at_3"], record["best_round"], record["candidate_id"]))


def _environment(spec):
    runtime = spec.get("runtime")
    if not isinstance(runtime, dict):
        raise RuntimeError("spec runtime pins are required")
    package_keys = (
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("scikit_learn", "scikit-learn"),
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("catboost", "catboost"),
    )
    expected_python = runtime.get("python")
    actual_python = sys.version.split()[0]
    if expected_python is None:
        raise RuntimeError("spec runtime pin is missing: python")
    if actual_python != str(expected_python):
        raise RuntimeError(f"python pin mismatch: {actual_python} != {expected_python}")
    versions = {}
    for spec_key, distribution in package_keys:
        wanted = runtime.get(spec_key)
        if wanted is None:
            raise RuntimeError(f"spec runtime pin is missing: {spec_key}")
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError as error:
            raise RuntimeError(f"required package is not installed: {distribution}") from error
        if versions[distribution] != str(wanted):
            raise RuntimeError(
                f"package pin mismatch for {distribution}: {versions[distribution]} != {wanted}")
    frozen = "python==%s\n%s" % (actual_python, "\n".join(
        f"{distribution}=={versions[distribution]}" for _, distribution in package_keys))
    frozen += "\n"
    return frozen, hashlib.sha256(frozen.encode()).hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_predictions(path, rows, scores):
    selected = {(row["race_id"], row["horse_id_netkeiba"], row["horse_number"])
               for entries in _group(rows, scores).values()
               for row, _ in top_three([r for r, _ in entries], [s for _, s in entries])}
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["race_id", "horse_id_netkeiba", "horse_number", "surface", "label", "prediction", "selected_top3"])
        for row, score in zip(rows, scores):
            key = (row["race_id"], row["horse_id_netkeiba"], row["horse_number"])
            writer.writerow([row["race_id"], row["horse_id_netkeiba"], row["horse_number"], row["surface"], row["label"], "%.17g" % score, int(key in selected)])


def _group(rows, scores):
    groups = {}
    for row, score in zip(rows, scores):
        groups.setdefault(row["race_id"], []).append((row, score))
    return groups


def _save_model(model, family, path):
    if family == "xgboost":
        model.save_model(str(path.with_suffix(".json")))
        return path.with_suffix(".json")
    if family == "lightgbm":
        model.booster_.save_model(str(path.with_suffix(".txt")))
        return path.with_suffix(".txt")
    model.save_model(str(path.with_suffix(".cbm")))
    return path.with_suffix(".cbm")


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
        hashes[key] = sha256_file(path)
    return hashes


def _legacy_artifact_hashes(folder):
    """Hash artifact bytes as used by checkpoints written before CSV canonicalization."""
    folder = Path(folder)
    return {path.relative_to(folder).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(folder.rglob("*"))
            if path.is_file() and path.name != "complete.json"}


def _artifact_hash_diff(expected, actual):
    """Return a compact, deterministic diff for a failed checkpoint audit."""
    keys = sorted(set(expected) | set(actual))
    return {key: {"expected": expected.get(key), "actual": actual.get(key)}
            for key in keys if expected.get(key) != actual.get(key)}


def _canonical_artifact_manifest(artifacts):
    canonical = {}
    for name, digest in artifacts.items():
        key = name.removesuffix(".gz") if name.endswith(".csv.gz") else name
        if key in canonical:
            raise ValueError(f"duplicate canonical artifact path: {key}")
        canonical[key] = digest
    return canonical


def _finish(folder, config_hash):
    artifacts = _artifact_hashes(folder)
    marker = {"status": "complete", "config_hash": config_hash, "artifacts": artifacts}
    _write_json(Path(folder) / "complete.json", marker)
    return marker


def _check_existing(folder, config_hash):
    marker_path = Path(folder) / "complete.json"
    if not marker_path.exists():
        return None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "complete" or marker.get("config_hash") != config_hash:
        raise FileExistsError(f"completed checkpoint differs: {folder}")
    if marker.get("artifacts") != _artifact_hashes(folder):
        raise ValueError(f"completed checkpoint artifact hash mismatch: {folder}")
    return {"status": "skipped", "run_dir": str(folder), "config_hash": config_hash}


def _metadata(input_path):
    sidecar = Path(input_path).with_suffix(".json")
    if not sidecar.exists():
        raise FileNotFoundError(f"input provenance sidecar is required: {sidecar}")
    value = json.loads(sidecar.read_text(encoding="utf-8"))
    if value.get("splits") != ["train", "validation"]:
        raise ValueError("input metadata splits must be exactly ['train', 'validation']")
    if value.get("seed") not in SEEDS:
        raise ValueError("input metadata must declare seed 0, 1, or 2")
    expected = input_sha256(input_path)
    declared = value.get("data_sha256")
    if declared != expected:
        # Older refit notebooks wrote the gzip-container hash in data_sha256.
        # Keep accepting that format only with a matching, well-formed payload hash.
        raw_ok = (Path(input_path).suffix == ".gz" and
                  declared == raw_sha256_file(input_path) and
                  value.get("uncompressed_sha256") == expected and
                  isinstance(value.get("purpose"), str) and
                  value["purpose"].startswith("JRA train-validation prediction inputs"))
        if not raw_ok:
            raise ValueError("input data checksum does not match its provenance sidecar")
    return value


def _base_provenance(input_path, spec_path, spec):
    helper_path = Path(__file__).with_name("experiment_features.py")
    metadata_path = Path(input_path).with_suffix(".json")
    frozen, environment_hash = _environment(spec)
    values = {"input_sha256": input_sha256(input_path),
              "input_metadata_sha256": sha256_file(metadata_path),
              "script_sha256": sha256_file(__file__),
              "spec_sha256": sha256_file(spec_path), "environment_sha256": environment_hash,
              "helper": str(helper_path.resolve()) if helper_path.is_file() else None,
              "helper_sha256": sha256_file(helper_path) if helper_path.is_file() else None}
    return values, frozen


def _load_spec(path, mode):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    spec_id = value.get("spec_id", "")
    if mode == "scientific" and (spec_id != "experiment_spec_2026-09-13" or
                                  value.get("status") not in ("accepted", "locked", "approved")):
        raise ValueError("unexpected experiment specification")
    if mode == "smoke" and not spec_id.endswith("_smoke"):
        raise ValueError("smoke runs require a *_smoke specification")
    if not value.get("families") or not value.get("settings"):
        raise ValueError("experiment specification lacks family/setting matrix")
    tuning = value.get("tuning", {})
    if tuning.get("tuned_seed") != 0 or tuning.get("candidates_per_family_setting_surface") != 4:
        raise ValueError("experiment specification tuning policy mismatch")
    if mode == "smoke" and (tuning.get("max_rounds"), tuning.get("patience")) != (8, 3):
        raise ValueError("smoke specification must use max_rounds=8 and patience=3")
    if int(tuning.get("max_rounds", 0)) < 1 or int(tuning.get("patience", 0)) < 1:
        raise ValueError("invalid max_rounds or patience")
    return value


def _policy(spec):
    tuning = spec["tuning"]
    threads = tuning.get("threads", spec.get("models", {}).get("threads",
                           spec.get("graph", {}).get("workers")))
    if threads is None or int(threads) < 1:
        raise ValueError("experiment specification lacks positive model thread count")
    return {"max_rounds": int(tuning["max_rounds"]), "patience": int(tuning["patience"]),
            "threads": int(threads), "ap_tolerance": float(tuning["ap_tie_tolerance"])}


def _candidates(spec, family):
    tuning = spec.get("tuning", {})
    grid = tuning.get("candidates", tuning.get("candidate_grid", spec.get("candidates")))
    if not isinstance(grid, dict) or family not in grid:
        raise ValueError("experiment specification lacks candidate parameters for " + family)
    candidates = grid[family]
    if not isinstance(candidates, dict) or tuple(sorted(candidates)) != ("c0", "c1", "c2", "c3"):
        raise ValueError("specification must provide exactly c0..c3 for " + family)
    return {candidate_id: candidates[candidate_id] for candidate_id in ("c0", "c1", "c2", "c3")}


def _purpose(mode):
    return "JRA smoke experiment checkpoint" if mode == "smoke" else "JRA scientific experiment checkpoint"


def _verified_tune_selection(tune_dir, current_config):
    """Load seed-0 selection only after verifying its complete checkpoint and provenance."""
    tune_dir = Path(tune_dir)
    marker_path = tune_dir / "complete.json"
    config_path = tune_dir / "config.json"
    selected_path = tune_dir / "selected.json"
    for path in (marker_path, config_path, selected_path):
        if not path.exists():
            raise FileNotFoundError(f"verified seed-0 tuning artifact is required: {path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "complete":
        raise ValueError("seed-0 tuning checkpoint is not complete")
    actual_artifacts = _artifact_hashes(tune_dir)
    expected_artifacts = _canonical_artifact_manifest(marker.get("artifacts", {}))
    legacy_artifacts = marker.get("artifacts") == _legacy_artifact_hashes(tune_dir)
    if expected_artifacts != actual_artifacts:
        diff = _artifact_hash_diff(expected_artifacts, actual_artifacts)
        # Tuning outputs mounted from the prior runner carry compressed-byte
        # hashes. Accept that exact legacy marker once; new checkpoints use
        # canonical CSV content hashes above.
        if not legacy_artifacts:
            raise ValueError("seed-0 tuning checkpoint artifact hash mismatch: " +
                             json.dumps(diff, sort_keys=True))
    tune_config = json.loads(config_path.read_text(encoding="utf-8"))
    if marker.get("config_hash") != canonical_hash(tune_config):
        raise ValueError("seed-0 tuning config hash mismatch")
    comparable = ("purpose", "family", "surface", "setting", "stage", "seed", "run_mode",
                  "spec_id", "spec_sha256", "script_sha256", "helper_sha256",
                  "environment_sha256", "features", "categorical", "max_rounds",
                  "patience", "threads", "candidate_grid_sha256")
    for key in comparable:
        if key == "script_sha256" and legacy_artifacts and tune_config.get(key) == LEGACY_ARTIFACT_SCRIPT_SHA256:
            continue
        if tune_config.get(key) != current_config.get(key):
            raise ValueError(f"seed-0 tuning provenance mismatch for {key}")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if selected.get("status") != "complete" or selected.get("candidate_id") not in ("c0", "c1", "c2", "c3"):
        raise ValueError("seed-0 tuning selection is not complete")
    try:
        selected["best_round"] = int(selected["best_round"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("seed-0 tuning selection has no valid selected round") from error
    if selected["best_round"] < 1 or selected["best_round"] > int(tune_config["max_rounds"]):
        raise ValueError("seed-0 selected round is outside the tuning policy")
    selected["selection_sha256"] = sha256_file(selected_path)
    selected["tuning_complete_sha256"] = sha256_file(marker_path)
    selected["tuning_config_sha256"] = sha256_file(config_path)
    return selected


def _run_tune(args, spec, policy, rows, columns, provenance, frozen, run_dir):
    surface_rows = [row for row in rows if row["surface"] == args.surface]
    train = [row for row in surface_rows if row["split"] == "train"]
    validation = [row for row in surface_rows if row["split"] == "validation"]
    selected = feature_allowlist(args.setting, columns)
    maps = features.fit_category_maps(train, selected)
    categorical = tuple(features.categorical_columns(selected))
    x_train = prepare_values(train, selected, maps, args.family)
    x_val = prepare_values(validation, selected, maps, args.family)
    config_base = {"purpose": _purpose(args.mode), "family": args.family, "surface": args.surface, "setting": args.setting,
                   "stage": "tune", "seed": 0, "run_mode": args.mode, "spec_id": spec["spec_id"], "features": list(selected),
                   "categorical": list(categorical),
                   "max_rounds": policy["max_rounds"], "patience": policy["patience"],
                   "threads": policy["threads"],
                   "candidate_grid_sha256": canonical_hash(_candidates(spec, args.family)), **provenance}
    config_hash = canonical_hash(config_base)
    existing = _check_existing(run_dir, config_hash)
    if existing:
        return existing
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", config_base)
    _write_json(run_dir / "preprocessing.json", features.preprocessing_metadata(args.setting, selected, maps))
    (run_dir / "environment.txt").write_text(frozen, encoding="utf-8")
    records = []
    for candidate_id, candidate_params in _candidates(spec, args.family).items():
        candidate_dir = run_dir / "candidates" / candidate_id
        candidate_config = {"parent_config_hash": config_hash, "candidate_id": candidate_id,
                            "params": candidate_params}
        candidate_hash = canonical_hash(candidate_config)
        existing_candidate = _check_existing(candidate_dir, candidate_hash)
        if existing_candidate:
            record = json.loads((candidate_dir / "metrics.json").read_text(encoding="utf-8"))
            records.append(record)
            continue
        candidate_dir.mkdir(parents=True, exist_ok=True)
        _write_json(candidate_dir / "config.json", candidate_config)
        started = time.perf_counter()
        try:
            model, curve, scores = fit_model(args.family, x_train, labels(train), x_val,
                                             labels(validation), candidate_params, 0, categorical, policy)
            best_round, best_ap = _ap_curve_best(curve)
            metrics = metric_values(validation, scores)
            record = {"status": "complete", "candidate_id": candidate_id,
                      "average_precision": best_ap, "mean_hit_at_3": metrics["mean_hit_at_3"],
                      "best_round": best_round, "hit_cap": len(curve) >= policy["max_rounds"],
                      "seconds": time.perf_counter() - started, "curve": curve,
                      "params": candidate_params}
            _write_json(candidate_dir / "metrics.json", record)
            _write_json(candidate_dir / "ap_curve.json", {"metric": "sklearn.metrics.average_precision_score", "values": curve})
            _write_predictions(candidate_dir / "validation_predictions.csv.gz", validation, scores)
            _save_model(model, args.family, candidate_dir / "model")
            _finish(candidate_dir, candidate_hash)
        except Exception as error:
            record = {"status": "failed", "candidate_id": candidate_id,
                      "seconds": time.perf_counter() - started, "error": repr(error),
                      "params": candidate_params}
            _write_json(candidate_dir / "failure.json", record)
        records.append(record)
    _write_json(run_dir / "candidate_results.json", records)
    winner = choose_candidate(records, policy["ap_tolerance"])
    _write_json(run_dir / "selected.json", winner)
    _finish(run_dir, config_hash)
    return {"status": "complete", "run_dir": str(run_dir), "config_hash": config_hash, "selected": winner}


def _run_refit(args, spec, policy, rows, columns, provenance, frozen, run_dir):
    if args.seed not in SEEDS:
        raise ValueError("refit seed must be 0, 1, or 2")
    tune_dir = Path(args.output) / args.mode / args.family / args.surface / args.setting / "tune" / "seed0"
    surface_rows = [row for row in rows if row["surface"] == args.surface]
    train = [row for row in surface_rows if row["split"] == "train"]
    validation = [row for row in surface_rows if row["split"] == "validation"]
    selected_features = feature_allowlist(args.setting, columns)
    maps = features.fit_category_maps(train, selected_features)
    categorical = tuple(features.categorical_columns(selected_features))
    x_train = prepare_values(train, selected_features, maps, args.family)
    x_val = prepare_values(validation, selected_features, maps, args.family)
    tune_probe = {"purpose": _purpose(args.mode), "family": args.family, "surface": args.surface,
                  "setting": args.setting, "stage": "tune", "seed": 0, "run_mode": args.mode,
                  "spec_id": spec["spec_id"], "features": list(selected_features), "categorical": list(categorical),
                  "max_rounds": policy["max_rounds"], "patience": policy["patience"],
                  "threads": policy["threads"],
                  "candidate_grid_sha256": canonical_hash(_candidates(spec, args.family)), **provenance}
    selected = _verified_tune_selection(tune_dir, tune_probe)
    params = dict(_candidates(spec, args.family)[selected["candidate_id"]])
    config = {"purpose": _purpose(args.mode), "family": args.family, "surface": args.surface, "setting": args.setting,
              "stage": "refit", "seed": args.seed, "run_mode": args.mode, "spec_id": spec["spec_id"], "features": list(selected_features),
              "categorical": list(categorical), "selected_candidate": selected["candidate_id"],
              "selected_round": int(selected["best_round"]), "threads": policy["threads"],
              "params": params, "selection_sha256": selected["selection_sha256"],
              "tuning_complete_sha256": selected["tuning_complete_sha256"],
              "tuning_config_sha256": selected["tuning_config_sha256"], **provenance}
    config_hash = canonical_hash(config)
    existing = _check_existing(run_dir, config_hash)
    if existing:
        return existing
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "preprocessing.json", features.preprocessing_metadata(args.setting, selected_features, maps))
    (run_dir / "environment.txt").write_text(frozen, encoding="utf-8")
    # Final refits deliberately use the frozen number of rounds and no eval set.
    model, _, _ = fit_model_fixed(args.family, x_train, labels(train), params,
                                   args.seed, categorical, int(selected["best_round"]), policy["threads"])
    scores = predict_model(args.family, model, x_val)
    metrics = metric_values(validation, scores)
    metrics["validation_threshold"] = validation_threshold(validation, scores, spec)
    metrics.update(seed=args.seed, selected_candidate=selected["candidate_id"],
                   selected_round=int(selected["best_round"]), train_rows=len(train),
                   validation_rows=len(validation))
    _write_json(run_dir / "metrics.json", metrics)
    _write_predictions(run_dir / "validation_predictions.csv.gz", validation, scores)
    _save_model(model, args.family, run_dir / "model")
    _finish(run_dir, config_hash)
    return {"status": "complete", "run_dir": str(run_dir), "config_hash": config_hash, "metrics": metrics}


def fit_model_fixed(family, x_train, y_train, params, seed, categorical, rounds, threads):
    if family == "xgboost":
        import xgboost as xgb
        dtrain = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
        config = {"objective": "binary:logistic", "tree_method": "hist", "enable_categorical": True,
                  "max_cat_to_onehot": 32, "max_cat_threshold": 64, "nthread": threads,
                  "seed": seed, **params}
        return xgb.train(config, dtrain, num_boost_round=rounds, verbose_eval=False), [], []
    if family == "lightgbm":
        lgb = _import_lightgbm()
        model = lgb.LGBMClassifier(objective="binary", n_estimators=rounds, random_state=seed,
                                   n_jobs=threads, verbosity=-1, cat_smooth=20, cat_l2=10,
                                   min_data_per_group=20, max_cat_threshold=64, **params)
        model.fit(x_train, y_train, categorical_feature=list(categorical))
        return model, [], []
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(iterations=rounds, random_seed=seed, thread_count=threads,
                               loss_function="Logloss", cat_features=list(categorical),
                               one_hot_max_size=32, max_ctr_complexity=1,
                               bootstrap_type="Bayesian", bagging_temperature=1,
                               verbose=False, allow_writing_files=False, **params)
    model.fit(x_train, y_train, verbose=False)
    return model, [], []


def predict_model(family, model, frame):
    if family == "xgboost":
        import xgboost as xgb
        return model.predict(xgb.DMatrix(frame, enable_categorical=True))
    return model.predict_proba(frame)[:, 1]


def run(args):
    if not Path("/kaggle").is_dir():
        raise RuntimeError("model fitting is restricted to Kaggle")
    if args.family not in FAMILIES or args.surface not in SURFACES:
        raise ValueError("unsupported family, surface, or setting")
    if args.stage == "tune" and args.seed != 0:
        raise ValueError("tuning is seed 0 only")
    spec = Path(args.spec)
    spec_data = _load_spec(spec, args.mode)
    if args.family not in spec_data["families"] or args.setting not in spec_data["settings"]:
        raise ValueError("requested family/setting is not in the specification")
    if args.surface not in SURFACES:
        raise ValueError("unsupported surface")
    policy = _policy(spec_data)
    rows, columns = read_rows(args.input)
    metadata = _metadata(args.input)
    if metadata["seed"] != args.seed:
        raise ValueError("input seed does not match classifier seed")
    provenance, frozen = _base_provenance(args.input, spec, spec_data)
    run_dir = Path(args.output) / args.mode / args.family / args.surface / args.setting / args.stage / f"seed{args.seed}"
    if args.stage == "tune":
        return _run_tune(args, spec_data, policy, rows, columns, provenance, frozen, run_dir)
    return _run_refit(args, spec_data, policy, rows, columns, provenance, frozen, run_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--surface", required=True, choices=SURFACES)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--stage", required=True, choices=("tune", "refit"))
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--input", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("scientific", "smoke"), default="scientific")
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
