"""Build the immutable, validation-only manifest used before test release."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import audit_experiment_outputs as audit


HERE = Path(__file__).resolve().parent
FAMILIES = ("xgboost", "lightgbm", "catboost")
SURFACES = ("turf", "dirt")
SEEDS = (0, 1, 2)
SHAP_SETTINGS = ("S6", "S7")
SHAP_FAMILY_ORDER = ("catboost", "lightgbm", "xgboost")
MODEL_NAMES = audit.MODEL_NAMES
THRESHOLD_RULE = audit.THRESHOLD_RULE


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _config_provenance(config):
    keys = ("spec_sha256", "script_sha256", "helper_sha256", "environment_sha256",
            "input_sha256", "input_metadata_sha256")
    result = {key: config.get(key) for key in keys}
    if any(not isinstance(value, str) or not value for value in result.values()):
        raise ValueError("artifact config provenance is incomplete")
    return result


def _validate_spec(spec):
    if spec.get("status") != "locked" or spec.get("test_performance_opened") is not False:
        raise ValueError("only a locked specification with closed test performance is eligible")
    if tuple(spec.get("families", ())) != FAMILIES or tuple(spec.get("surfaces", ())) != SURFACES:
        raise ValueError("specification family/surface matrix is not the locked protocol")
    settings = tuple(spec.get("settings", ()))
    tuning = spec.get("tuning", {})
    if tuning.get("fit_count") != 240 or tuning.get("candidates_per_family_setting_surface") != 4:
        raise ValueError("specification does not declare all 240 tuning fits")
    if tuning.get("ap_tie_tolerance") != 1e-6 or tuning.get("tie_break") != [
            "higher_mean_hit_at_3", "lower_best_round", "lexicographic_candidate_id"]:
        raise ValueError("specification AP tie rule is not locked")
    if spec.get("final_fit_count") != 180 or spec.get("matched_seeds") != list(SEEDS):
        raise ValueError("specification does not declare all 180 matched-seed refits")
    if len(FAMILIES) * len(SURFACES) * len(settings) * 4 != 240:
        raise ValueError("settings do not account for all 240 candidate fits")
    if len(FAMILIES) * len(SURFACES) * len(settings) * len(SEEDS) != 180:
        raise ValueError("settings do not account for all 180 refits")
    if spec.get("refit", {}).get("fit_scope") != "train_only" or spec["refit"].get("train_plus_validation") is not False:
        raise ValueError("refit policy is not train-only")
    if spec.get("threshold", {}).get("fitted_settings") != THRESHOLD_RULE:
        raise ValueError("fixed validation F1 threshold rule is not locked")
    if spec.get("shap", {}).get("selection") != "highest_mean_validation_AP_across_seeds_per_surface_and_S6_or_S7":
        raise ValueError("SHAP family selection rule is not locked")
    return settings


def _runtime_spec_sha256(spec):
    """Hash the exact sorted JSON written by the Kaggle notebook at runtime."""
    payload = json.dumps(spec, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _expected_provenance(spec_path, spec=None):
    if spec is None:
        spec = _json(spec_path)
    return {
        "spec_sha256": _runtime_spec_sha256(spec),
        "script_sha256": _sha256(HERE / "run_experiments.py"),
        "helper_sha256": _sha256(HERE / "experiment_features.py"),
    }


def _compact_candidate(root, family, candidate_dir, parent_provenance):
    marker = audit._complete(candidate_dir)
    config_path = candidate_dir / "config.json"
    metrics_path = candidate_dir / "metrics.json"
    config = audit._json(config_path)
    metrics = audit._json(metrics_path)
    return {
        "relative_dir": candidate_dir.relative_to(root).as_posix(),
        "candidate_id": config["candidate_id"],
        "params": config["params"],
        "average_precision": metrics["average_precision"],
        "mean_hit_at_3": metrics["mean_hit_at_3"],
        "best_round": metrics["best_round"],
        "config_hash": marker["config_hash"],
        "parent_config_hash": config["parent_config_hash"],
        "config_provenance": parent_provenance,
        "config_sha256": _sha256(config_path),
        "metrics_sha256": _sha256(metrics_path),
        "complete_sha256": _sha256(Path(candidate_dir) / "complete.json"),
        "artifact_hashes": marker["artifacts"],
        "model_artifact": MODEL_NAMES[family],
    }


def _collect_entry(root, family, surface, setting):
    base = Path(root) / "scientific" / family / surface / setting
    tune = base / "tune" / "seed0"
    tune_marker = audit._complete(tune)
    tune_config_path = tune / "config.json"
    selected_path = tune / "selected.json"
    tune_config = audit._json(tune_config_path)
    selected = audit._json(selected_path)
    candidates = {
        candidate_id: _compact_candidate(root, family, tune / "candidates" / candidate_id,
                                         _config_provenance(tune_config))
        for candidate_id in ("c0", "c1", "c2", "c3")
    }
    refits = []
    for seed in SEEDS:
        refit = base / "refit" / f"seed{seed}"
        marker = audit._complete(refit)
        config_path = refit / "config.json"
        metrics_path = refit / "metrics.json"
        config = audit._json(config_path)
        metrics = audit._json(metrics_path)
        threshold = metrics.get("validation_threshold")
        if threshold is None or threshold.get("rule") != THRESHOLD_RULE:
            raise ValueError(f"fixed validation threshold is missing: {refit}")
        selected_threshold = threshold.get("selected")
        if not isinstance(selected_threshold, dict) or selected_threshold.get("threshold") is None:
            raise ValueError(f"selected validation F1 threshold is missing: {refit}")
        refits.append({
            "relative_dir": refit.relative_to(root).as_posix(),
            "seed": seed,
            "selected_candidate": config["selected_candidate"],
            "selected_round": config["selected_round"],
            "params": config["params"],
            "average_precision": metrics["average_precision"],
            "mean_hit_at_3": metrics["mean_hit_at_3"],
            "validation_threshold": threshold,
            "config_provenance": _config_provenance(config),
            "config_sha256": _sha256(config_path),
            "metrics_sha256": _sha256(metrics_path),
            "complete_sha256": _sha256(refit / "complete.json"),
            "artifact_hashes": marker["artifacts"],
            "model_artifact": MODEL_NAMES[family],
        })
    return {
        "family": family, "surface": surface, "setting": setting,
        "tuning": {
            "relative_dir": tune.relative_to(root).as_posix(),
            "config_hash": tune_marker["config_hash"],
            "config_sha256": _sha256(tune_config_path),
            "complete_sha256": _sha256(tune / "complete.json"),
            "selected_sha256": _sha256(selected_path),
            "selected": selected,
            "config_provenance": _config_provenance(tune_config),
            "candidate_grid_sha256": tune_config["candidate_grid_sha256"],
            "artifact_hashes": tune_marker["artifacts"],
            "candidates": candidates,
        },
        "refits": refits,
    }


def _shap_choices(entries, spec):
    tolerance = float(spec["tuning"]["ap_tie_tolerance"])
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("SHAP AP tie tolerance is invalid")
    choices = []
    for surface in SURFACES:
        for setting in SHAP_SETTINGS:
            candidates = []
            for family in FAMILIES:
                entry = entries[(family, surface, setting)]
                scores = [float(refit["average_precision"]) for refit in entry["refits"]]
                hit3 = [float(refit["mean_hit_at_3"]) for refit in entry["refits"]]
                rounds = [int(refit["selected_round"]) for refit in entry["refits"]]
                if any(not math.isfinite(value) for value in scores + hit3):
                    raise ValueError(f"non-finite SHAP selection metric for {family}/{surface}/{setting}")
                if len(set(rounds)) != 1:
                    raise ValueError(f"SHAP refit rounds are not frozen for {family}/{surface}/{setting}")
                candidates.append({
                    "family": family, "seed_average_precision": scores,
                    "mean_validation_average_precision": sum(scores) / len(scores),
                    "seed_mean_hit_at_3": hit3,
                    "mean_validation_hit_at_3": sum(hit3) / len(hit3),
                    "seed_selected_round": rounds,
                    "selected_round": rounds[0],
                })
            maximum = max(item["mean_validation_average_precision"] for item in candidates)
            eligible = [item for item in candidates
                        if maximum - item["mean_validation_average_precision"] <= tolerance]
            winner = min(eligible, key=lambda item: (
                -item["mean_validation_hit_at_3"],
                item["selected_round"],
                SHAP_FAMILY_ORDER.index(item["family"])))
            choices.append({
                "surface": surface, "setting": setting,
                "rule": "highest_mean_validation_AP_across_seeds",
                "ap_tie_tolerance": tolerance,
                "tie_break": ["higher_mean_validation_hit_at_3", "lower_selected_round",
                               "fixed_family_order_catboost_lightgbm_xgboost"],
                "selected_family": winner["family"],
                "candidates": candidates,
            })
    return choices


def build_manifest(root, spec_path, output):
    root = Path(root)
    spec_path = Path(spec_path)
    output = Path(output)
    spec = _json(spec_path)
    settings = _validate_spec(spec)
    provenance = _expected_provenance(spec_path, spec)
    source_spec_sha256 = _sha256(spec_path)
    summary = audit.audit_output_tree(root, spec, mode="scientific",
                                      expected_provenance=provenance)
    if summary.get("tuning_runs") != 60 or summary.get("candidate_runs") != 240 or summary.get("refit_runs") != 180:
        raise ValueError("audited artifact counts are incomplete; no release manifest was written")
    entries = {}
    for family in FAMILIES:
        for surface in SURFACES:
            for setting in settings:
                entry = _collect_entry(root, family, surface, setting)
                entries[(family, surface, setting)] = entry
    global_provenance = {}
    for entry in entries.values():
        config = audit._json(Path(root) / "scientific" / entry["family"] / entry["surface"] /
                             entry["setting"] / "tune" / "seed0" / "config.json")
        for key in ("spec_sha256", "script_sha256", "helper_sha256", "environment_sha256"):
            value = config.get(key)
            if key in global_provenance and global_provenance[key] != value:
                raise ValueError(f"provenance is inconsistent across artifacts: {key}")
            global_provenance[key] = value
    manifest = {
        "purpose": "JRA test-release model selection manifest",
        "status": "locked_for_test_release",
        "spec_id": spec["spec_id"],
        "spec_sha256": provenance["spec_sha256"],
        "spec_runtime_sha256": provenance["spec_sha256"],
        "spec_source_sha256": source_spec_sha256,
        "test_performance_opened": False,
        "test_evaluation_authorized": False,
        "contains_test_predictions": False,
        "no_test_performance": True,
        "audited_counts": {"tuning_runs": 60, "candidate_runs": 240, "refit_runs": 180},
        "audit_summary": summary,
        "audit_script_sha256": _sha256(HERE / "audit_experiment_outputs.py"),
        "evaluator_script_sha256": _sha256(HERE / "evaluate_locked.py"),
        "provenance": global_provenance,
        "models": [entries[key] for key in sorted(entries)],
        "shap_family_choices": _shap_choices(entries, spec),
        "selection_rules": {
            "candidate_selection": "seed0 validation average precision with locked tie rule",
            "refit_rounds": "frozen seed0 selected round for seeds 0,1,2",
            "threshold": THRESHOLD_RULE,
            "shap": "highest mean validation AP across matched seeds; AP tolerance 1e-6, then higher mean Hit@3, lower selected round, then family order catboost/lightgbm/xgboost",
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing selection manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
                      encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_manifest(args.root, args.spec, args.output),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
