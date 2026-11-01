"""Read-only audit for the frozen tuning/refit output tree.

The audit checks the runner's JSON/checkpoint schemas and validation prediction
artifacts. It never loads a model or reads a test file.
"""
import hashlib
import gzip
import json
import math
from pathlib import Path

import evaluate_locked as evaluation


FAMILIES = ("xgboost", "lightgbm", "catboost")
SURFACES = ("turf", "dirt")
SEEDS = (0, 1, 2)
MODEL_NAMES = {"xgboost": "model.json", "lightgbm": "model.txt", "catboost": "model.cbm"}
AP_VALIDATION_TOLERANCE = 1e-6
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


def _json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing required JSON artifact: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error


def _finite_unit(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def _integer(value, name):
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be integer-valued") from error
    if not math.isfinite(numeric) or numeric != parsed:
        raise ValueError(f"{name} must be integer-valued")
    return parsed


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


def _complete(folder):
    folder = Path(folder)
    marker_path = folder / "complete.json"
    marker = _json(marker_path)
    if marker.get("status") != "complete":
        raise ValueError(f"checkpoint is not complete: {folder}")
    expected = _canonical_artifact_manifest(marker.get("artifacts"))
    if expected != _artifact_hashes(folder):
        raise ValueError(f"artifact hash mismatch: {folder}")
    marker = dict(marker)
    marker["artifacts"] = expected
    if not isinstance(marker.get("config_hash"), str) or not marker["config_hash"]:
        raise ValueError(f"checkpoint config hash is missing: {folder}")
    return marker


def _reject_test_paths(root):
    root = Path(root)
    for path in root.rglob("*"):
        if any("test" in part.lower() for part in path.relative_to(root).parts):
            raise ValueError(f"test artifact is outside this audit's scope: {path}")


def _candidate_grid(spec, family):
    try:
        grid = spec["tuning"]["candidates"][family]
    except (KeyError, TypeError) as error:
        raise ValueError(f"missing candidate grid for {family}") from error
    if not isinstance(grid, dict) or tuple(sorted(grid)) != ("c0", "c1", "c2", "c3"):
        raise ValueError(f"candidate grid must contain exactly c0..c3 for {family}")
    return {candidate_id: grid[candidate_id] for candidate_id in ("c0", "c1", "c2", "c3")}


def _policy(spec):
    tuning = spec.get("tuning", {})
    required = ("max_rounds", "patience", "threads", "ap_tie_tolerance")
    if any(key not in tuning for key in required):
        raise ValueError("specification lacks complete tuning policy")
    policy = {key: _integer(tuning[key], key) for key in ("max_rounds", "patience", "threads")}
    if any(value < 1 for value in policy.values()):
        raise ValueError("tuning policy values must be positive")
    policy["ap_tie_tolerance"] = float(tuning["ap_tie_tolerance"])
    if not math.isfinite(policy["ap_tie_tolerance"]) or policy["ap_tie_tolerance"] < 0:
        raise ValueError("AP tie tolerance must be finite and nonnegative")
    return policy


def _check_config(config, spec, mode, family, surface, setting, stage, seed, policy,
                  expected_provenance=None, tune_config=None):
    expected = {
        "purpose": "JRA smoke experiment checkpoint" if mode == "smoke" else "JRA scientific experiment checkpoint",
        "family": family, "surface": surface, "setting": setting,
        "stage": stage, "seed": seed, "run_mode": mode, "spec_id": spec.get("spec_id"),
        "threads": policy["threads"],
    }
    if stage == "tune":
        expected.update(max_rounds=policy["max_rounds"], patience=policy["patience"])
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"config {key} mismatch for {family}/{surface}/{setting}/{stage}/seed{seed}")
    for key in ("spec_sha256", "script_sha256", "helper_sha256", "environment_sha256",
                "input_sha256", "input_metadata_sha256"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"config provenance field is missing: {key}")
    if expected_provenance:
        for key, value in expected_provenance.items():
            if config.get(key) != value:
                raise ValueError(f"config provenance mismatch for {key}")
    if tune_config is not None:
        for key in ("spec_id", "spec_sha256", "script_sha256", "helper_sha256",
                    "environment_sha256", "features", "categorical", "threads"):
            if config.get(key) != tune_config.get(key):
                raise ValueError(f"refit/tune provenance mismatch for {key}")


def _validate_curve(candidate_dir, candidate_config, expected_params, policy, family):
    marker = _complete(candidate_dir)
    expected_names = {"ap_curve.json", "config.json", "metrics.json", MODEL_NAMES[family],
                      "validation_predictions.csv"}
    if set(marker["artifacts"]) != expected_names:
        raise ValueError(f"unexpected candidate artifacts: {candidate_dir}")
    if marker["config_hash"] != _canonical(candidate_config):
        raise ValueError(f"candidate config hash mismatch: {candidate_dir}")
    if candidate_config.get("candidate_id") not in ("c0", "c1", "c2", "c3"):
        raise ValueError(f"invalid candidate ID: {candidate_dir}")
    if candidate_config.get("params") != expected_params:
        raise ValueError(f"candidate parameters differ from locked grid: {candidate_dir}")
    metrics = _json(candidate_dir / "metrics.json")
    if metrics.get("status") != "complete" or metrics.get("candidate_id") != candidate_config["candidate_id"]:
        raise ValueError(f"candidate metrics are incomplete: {candidate_dir}")
    if metrics.get("params") != expected_params:
        raise ValueError(f"candidate metrics parameters differ from locked grid: {candidate_dir}")
    curve_file = _json(candidate_dir / "ap_curve.json")
    if curve_file.get("metric") != "sklearn.metrics.average_precision_score":
        raise ValueError(f"candidate AP curve metric is not sklearn AP: {candidate_dir}")
    curve = curve_file.get("values")
    if not isinstance(curve, list) or not curve:
        raise ValueError(f"candidate AP curve is empty: {candidate_dir}")
    if metrics.get("curve") != curve:
        raise ValueError(f"candidate metrics/AP curve mismatch: {candidate_dir}")
    for index, value in enumerate(curve, 1):
        _finite_unit(value, f"candidate AP curve round {index}")
    best_round = _integer(metrics.get("best_round"), "best_round")
    if best_round < 1 or best_round > min(len(curve), policy["max_rounds"]):
        raise ValueError(f"candidate best round is outside its AP curve: {candidate_dir}")
    best_ap = max(float(value) for value in curve)
    earliest = next(index for index, value in enumerate(curve, 1)
                    if abs(float(value) - best_ap) <= 1e-15)
    if best_round != earliest or abs(float(metrics.get("average_precision")) - best_ap) > 1e-12:
        raise ValueError(f"candidate AP/best round disagrees with AP curve: {candidate_dir}")
    if bool(metrics.get("hit_cap")) != (len(curve) >= policy["max_rounds"]):
        raise ValueError(f"candidate cap flag disagrees with AP curve: {candidate_dir}")
    _finite_unit(metrics.get("mean_hit_at_3"), "candidate mean Hit@3")
    prediction_rows = evaluation.read_prediction_csv(candidate_dir / "validation_predictions.csv.gz")
    evaluation.audit_selection_flags(prediction_rows)
    if abs(_average_precision(prediction_rows) - best_ap) > AP_VALIDATION_TOLERANCE:
        raise ValueError(f"candidate AP disagrees with validation predictions: {candidate_dir}")
    return metrics


def _winner(records, tolerance):
    if len(records) != 4 or {record.get("candidate_id") for record in records} != {"c0", "c1", "c2", "c3"}:
        raise ValueError("all four candidate records are required")
    maximum = max(float(record["average_precision"]) for record in records)
    eligible = [record for record in records if maximum - float(record["average_precision"]) <= tolerance]
    return min(eligible, key=lambda record: (-float(record["mean_hit_at_3"]),
                                               _integer(record["best_round"], "best_round"),
                                               record["candidate_id"])), maximum


def _threshold_metrics(rows, threshold):
    threshold = float(threshold)
    labels = [row["label"] for row in rows]
    scores = [row["score"] for row in rows]
    tp = sum(label == 1 and score >= threshold for label, score in zip(labels, scores))
    fp = sum(label == 0 and score >= threshold for label, score in zip(labels, scores))
    fn = sum(label == 1 and score < threshold for label, score in zip(labels, scores))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": threshold, "precision": precision, "recall": recall,
            "f1": f1, "predicted_positive": tp + fp}


def _average_precision(rows):
    """Compute sklearn-style non-interpolated AP without importing sklearn."""
    positives = sum(row["label"] for row in rows)
    if not positives:
        return 0.0
    ordered = sorted(rows, key=lambda row: -row["score"])
    true_positive = seen = previous_recall = value = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end]["score"] == ordered[index]["score"]:
            end += 1
        true_positive += sum(row["label"] for row in ordered[index:end])
        seen += end - index
        recall = true_positive / positives
        value += (recall - previous_recall) * (true_positive / seen)
        previous_recall = recall
        index = end
    return value


def _audit_threshold(rows, threshold_report, path):
    if threshold_report.get("rule") != THRESHOLD_RULE or threshold_report.get("grid") != [0.01, 0.99, 0.01]:
        raise ValueError(f"validation threshold policy mismatch: {path}")
    selected = threshold_report.get("selected")
    if not isinstance(selected, dict):
        raise ValueError(f"validation threshold selection is missing: {path}")
    candidates = [_threshold_metrics(rows, index / 100.0) for index in range(1, 100)]
    expected = max(candidates, key=lambda value: (value["f1"], value["precision"],
                                                   -abs(value["threshold"] - 0.5), value["threshold"]))
    selected_threshold = float(selected.get("threshold"))
    if selected_threshold != expected["threshold"]:
        raise ValueError(f"validation threshold selection disagrees with predictions: {path}")
    for key in ("precision", "recall", "f1"):
        if abs(float(selected.get(key)) - expected[key]) > 1e-12:
            raise ValueError(f"validation threshold {key} disagrees with predictions: {path}")
    if _integer(selected.get("predicted_positive"), "predicted_positive") != expected["predicted_positive"]:
        raise ValueError(f"validation threshold count disagrees with predictions: {path}")


def _audit_refit(refit_dir, spec, mode, family, surface, setting, seed, policy, tune_config,
                 selected, expected_provenance):
    marker = _complete(refit_dir)
    expected_names = {"config.json", "environment.txt", "metrics.json", MODEL_NAMES[family],
                      "preprocessing.json", "validation_predictions.csv"}
    if set(marker["artifacts"]) != expected_names:
        raise ValueError(f"unexpected refit artifacts: {refit_dir}")
    config = _json(refit_dir / "config.json")
    if marker["config_hash"] != _canonical(config):
        raise ValueError(f"refit config hash mismatch: {refit_dir}")
    _check_config(config, spec, mode, family, surface, setting, "refit", seed, policy,
                  expected_provenance, tune_config=tune_config)
    if config.get("selected_candidate") != selected["candidate_id"] or \
            _integer(config.get("selected_round"), "selected_round") != selected["best_round"]:
        raise ValueError(f"refit selection differs from seed-0 tuning winner: {refit_dir}")
    if config.get("params") != selected.get("params"):
        raise ValueError(f"refit parameters differ from seed-0 tuning winner: {refit_dir}")
    metrics = _json(refit_dir / "metrics.json")
    if _integer(metrics.get("seed"), "metrics seed") != seed:
        raise ValueError(f"refit metrics seed mismatch: {refit_dir}")
    if metrics.get("selected_candidate") != selected["candidate_id"] or \
            _integer(metrics.get("selected_round"), "metrics selected_round") != selected["best_round"]:
        raise ValueError(f"refit metrics selection mismatch: {refit_dir}")
    for key in ("average_precision", "brier", "mean_hit_at_3"):
        _finite_unit(metrics.get(key), f"refit {key}")
    if metrics.get("roc_auc") is not None:
        _finite_unit(metrics["roc_auc"], "refit roc_auc")
    prediction_rows = evaluation.read_prediction_csv(refit_dir / "validation_predictions.csv.gz")
    audit = evaluation.audit_selection_flags(prediction_rows)
    if abs(_average_precision(prediction_rows) - float(metrics["average_precision"])) > AP_VALIDATION_TOLERANCE:
        raise ValueError(f"refit AP disagrees with validation predictions: {refit_dir}")
    if _integer(metrics.get("complete_validation_races"), "complete_validation_races") != audit["complete_races"]:
        raise ValueError(f"refit race count disagrees with prediction artifact: {refit_dir}")
    _audit_threshold(prediction_rows, metrics.get("validation_threshold", {}), refit_dir / "metrics.json")
    return {"seed": seed, "selected_candidate": selected["candidate_id"],
            "selected_round": selected["best_round"], "validation_races": audit["complete_races"]}


def audit_output_tree(root, spec, mode="scientific", expected_provenance=None,
                      families=None, surfaces=None, settings=None, stages="all"):
    """Audit every expected family/surface/setting tune and refit checkpoint.

    ``root`` is the output parent containing ``<mode>/<family>/<surface>``.
    Only validation prediction files under expected run paths are opened.
    """
    root = Path(root)
    if mode not in ("scientific", "smoke"):
        raise ValueError("mode must be scientific or smoke")
    if stages not in ("tune", "all"):
        raise ValueError("stages must be tune or all")
    _reject_test_paths(root)
    if not root.is_dir():
        raise ValueError(f"output root is missing: {root}")
    families = tuple(spec.get("families", ())) if families is None else tuple(families)
    surfaces = tuple(spec.get("surfaces", ())) if surfaces is None else tuple(surfaces)
    settings = tuple(spec.get("settings", ())) if settings is None else tuple(settings)
    if not families or not set(families).issubset(set(FAMILIES)) or not surfaces or \
            not set(surfaces).issubset(set(SURFACES)) or not settings or \
            not set(settings).issubset(set(spec.get("settings", ()) )):
        raise ValueError("specification matrix does not match the locked family/surface protocol")
    if tuple(spec.get("matched_seeds", SEEDS)) != SEEDS:
        raise ValueError("specification must declare matched seeds 0, 1, and 2")
    if stages == "all" and (set(families) != set(FAMILIES) or set(surfaces) != set(SURFACES) or
                             set(settings) != set(spec.get("settings", ()) )):
        raise ValueError("all-stage audit requires the full locked family/surface/setting matrix")
    policy = _policy(spec)
    mode_root = root / mode
    if not mode_root.is_dir():
        raise ValueError(f"mode output is missing: {mode_root}")
    available_families = {path.name for path in mode_root.iterdir() if path.is_dir()}
    if ((stages == "all" and available_families != set(families)) or
            (stages == "tune" and not set(families).issubset(available_families))):
        raise ValueError("output is missing or has extra model families")
    summary = {"status": "PASS", "mode": mode, "families": len(families),
               "surfaces": len(surfaces), "settings": len(settings),
               "tuning_runs": 0, "candidate_runs": 0, "refit_runs": 0, "refits": []}
    for family in families:
        grid = _candidate_grid(spec, family)
        for surface in surfaces:
            surface_root = mode_root / family / surface
            available_settings = {path.name for path in surface_root.iterdir() if path.is_dir()}
            if ((stages == "all" and available_settings != set(settings)) or
                    (stages == "tune" and not set(settings).issubset(available_settings))):
                raise ValueError(f"output setting matrix mismatch: {surface_root}")
            for setting in settings:
                base = surface_root / setting
                tune_dir = base / "tune" / "seed0"
                marker = _complete(tune_dir)
                tune_config = _json(tune_dir / "config.json")
                _check_config(tune_config, spec, mode, family, surface, setting, "tune", 0, policy,
                              expected_provenance)
                if marker["config_hash"] != _canonical(tune_config):
                    raise ValueError(f"tuning config hash mismatch: {tune_dir}")
                if tune_config.get("candidate_grid_sha256") != _canonical(grid):
                    raise ValueError(f"candidate grid hash mismatch: {tune_dir}")
                candidate_root = tune_dir / "candidates"
                if {path.name for path in candidate_root.iterdir() if path.is_dir()} != set(grid):
                    raise ValueError(f"candidate matrix mismatch: {candidate_root}")
                records = []
                for candidate_id, params in grid.items():
                    candidate_dir = candidate_root / candidate_id
                    candidate_config = _json(candidate_dir / "config.json")
                    if candidate_config.get("parent_config_hash") != marker["config_hash"]:
                        raise ValueError(f"candidate parent config hash mismatch: {candidate_dir}")
                    records.append(_validate_curve(candidate_dir, candidate_config, params, policy, family))
                    summary["candidate_runs"] += 1
                results = _json(tune_dir / "candidate_results.json")
                if results != records:
                    raise ValueError(f"candidate_results.json disagrees with candidate metrics: {tune_dir}")
                selected = _json(tune_dir / "selected.json")
                winner, maximum_ap = _winner(records, policy["ap_tie_tolerance"])
                if selected != winner:
                    raise ValueError(f"selected candidate disagrees with global AP/tie rule: {tune_dir}")
                summary["tuning_runs"] += 1
                if stages == "tune":
                    continue
                refit_root = base / "refit"
                if {path.name for path in refit_root.iterdir() if path.is_dir()} != {f"seed{seed}" for seed in SEEDS}:
                    raise ValueError(f"refit seed matrix mismatch: {refit_root}")
                for seed in SEEDS:
                    result = _audit_refit(refit_root / f"seed{seed}", spec, mode, family, surface,
                                          setting, seed, policy, tune_config, selected,
                                          expected_provenance)
                    summary["refit_runs"] += 1
                    summary["refits"].append({"family": family, "surface": surface,
                                              "setting": setting, **result})
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--mode", choices=("scientific", "smoke"), default="scientific")
    parser.add_argument("--family", dest="families", action="append")
    parser.add_argument("--surface", dest="surfaces", action="append")
    parser.add_argument("--setting", dest="settings", action="append")
    parser.add_argument("--stages", choices=("tune", "all"), default="all")
    args = parser.parse_args()
    report = audit_output_tree(args.root, _json(args.spec), args.mode,
                               families=args.families, surfaces=args.surfaces,
                               settings=args.settings, stages=args.stages)
    print(json.dumps(report, indent=2, sort_keys=True))
