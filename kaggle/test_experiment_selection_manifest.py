"""Synthetic no-fit checks for the test-release selection manifest."""
import hashlib
import json
import gzip
from pathlib import Path
import tempfile
from unittest.mock import patch

import audit_experiment_outputs as audit
import build_experiment_selection_manifest as builder


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value, encoding="utf-8")


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finish(folder, config_hash):
    artifacts = audit._artifact_hashes(folder)
    _write(folder / "complete.json", json.dumps({
        "status": "complete", "config_hash": config_hash, "artifacts": artifacts,
    }))


def _provenance():
    return {key: f"{key}-hash" for key in (
        "spec_sha256", "script_sha256", "helper_sha256", "environment_sha256",
        "input_sha256", "input_metadata_sha256")}


def _fixture(root, spec):
    settings = spec["settings"]
    families = spec["families"]
    surfaces = spec["surfaces"]
    params = {family: spec["tuning"]["candidates"][family]["c0"] for family in families}
    shap_ap = {
        ("turf", "S6"): {"xgboost": 0.90, "lightgbm": 0.80, "catboost": 0.70},
        ("turf", "S7"): {"xgboost": 0.70, "lightgbm": 0.90, "catboost": 0.80},
        ("dirt", "S6"): {"xgboost": 0.70, "lightgbm": 0.80, "catboost": 0.90},
        ("dirt", "S7"): {"xgboost": 0.80, "lightgbm": 0.70, "catboost": 0.90},
    }
    default_ap = {"xgboost": 0.60, "lightgbm": 0.70, "catboost": 0.80}
    for family in families:
        for surface in surfaces:
            for setting in settings:
                base = root / "scientific" / family / surface / setting
                prov = _provenance()
                prov["spec_sha256"] = builder._runtime_spec_sha256(spec)
                tune = base / "tune" / "seed0"
                tune_config = {**prov, "candidate_grid_sha256": "grid-hash"}
                _write(tune / "config.json", json.dumps(tune_config))
                _write(tune / "preprocessing.json", "{}")
                _write(tune / "environment.txt", "python==3.12.13\n")
                records = []
                for candidate_id in ("c0", "c1", "c2", "c3"):
                    candidate = tune / "candidates" / candidate_id
                    candidate_params = params[family]
                    ap = (shap_ap.get((surface, setting), default_ap)[family]
                          if candidate_id == "c0" else 0.1)
                    config = {"candidate_id": candidate_id, "params": candidate_params,
                              "parent_config_hash": "tune-config-hash"}
                    metrics = {"status": "complete", "candidate_id": candidate_id,
                               "average_precision": ap, "mean_hit_at_3": 0.5,
                               "best_round": 3, "params": candidate_params}
                    _write(candidate / "config.json", json.dumps(config))
                    _write(candidate / "metrics.json", json.dumps(metrics))
                    _write(candidate / "ap_curve.json", json.dumps({
                        "metric": "sklearn.metrics.average_precision_score", "values": [ap]}))
                    _write(candidate / audit.MODEL_NAMES[family], "model")
                    with gzip.open(candidate / "validation_predictions.csv.gz", "wb") as stream:
                        stream.write(b"synthetic")
                    _finish(candidate, "candidate-config-hash")
                    records.append(metrics)
                selected = {**records[0]}
                _write(tune / "candidate_results.json", json.dumps(records))
                _write(tune / "selected.json", json.dumps(selected))
                _finish(tune, "tune-config-hash")
                for seed in (0, 1, 2):
                    refit = base / "refit" / f"seed{seed}"
                    refit_config = {**prov, "selected_candidate": "c0",
                                    "selected_round": 3, "params": params[family]}
                    score = (shap_ap.get((surface, setting), default_ap)[family]
                             + seed / 1000.0)
                    metrics = {"seed": seed, "selected_candidate": "c0",
                               "selected_round": 3, "average_precision": score,
                               "mean_hit_at_3": 0.5,
                               "validation_threshold": {
                                   "rule": builder.THRESHOLD_RULE,
                                   "selected": {"threshold": 0.5, "f1": 0.5},
                               }}
                    _write(refit / "config.json", json.dumps(refit_config))
                    _write(refit / "environment.txt", "python==3.12.13\n")
                    _write(refit / "preprocessing.json", "{}")
                    _write(refit / "metrics.json", json.dumps(metrics))
                    _write(refit / audit.MODEL_NAMES[family], "model")
                    with gzip.open(refit / "validation_predictions.csv.gz", "wb") as stream:
                        stream.write(b"synthetic")
                    _finish(refit, "refit-config-hash")


def test_manifest_refuses_audit_failure_without_writing():
    spec_path = Path(__file__).resolve().parents[1] / "plans" / "experiment_spec_2026-09-13.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp) / "artifacts"
        output = Path(temp) / "selection.json"
        with patch.object(builder.audit, "audit_output_tree",
                          side_effect=ValueError("missing required scientific batch")):
            try:
                builder.build_manifest(root, spec_path, output)
            except ValueError as error:
                assert "missing required scientific batch" in str(error)
            else:
                raise AssertionError("incomplete artifact tree must be rejected")
        assert not output.exists()


def test_manifest_locks_all_artifacts_and_shap_choices():
    spec_path = Path(__file__).resolve().parents[1] / "plans" / "experiment_spec_2026-09-13.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp) / "artifacts"
        output = Path(temp) / "selection.json"
        _fixture(root, spec)
        summary = {"tuning_runs": 60, "candidate_runs": 240, "refit_runs": 180}
        with patch.object(builder.audit, "audit_output_tree", return_value=summary):
            manifest = builder.build_manifest(root, spec_path, output)
        assert manifest["test_performance_opened"] is False
        assert manifest["test_evaluation_authorized"] is False
        assert manifest["contains_test_predictions"] is False
        assert len(manifest["models"]) == 60
        assert all(len(entry["tuning"]["candidates"]) == 4 for entry in manifest["models"])
        assert all(len(entry["refits"]) == 3 for entry in manifest["models"])
        assert len(manifest["shap_family_choices"]) == 4
        assert {(item["surface"], item["setting"])
                for item in manifest["shap_family_choices"]} == {
                    ("turf", "S6"), ("turf", "S7"), ("dirt", "S6"), ("dirt", "S7")}
        assert manifest["spec_runtime_sha256"] == builder._runtime_spec_sha256(spec)
        assert manifest["spec_source_sha256"] == builder._sha256(spec_path)
        assert json.loads(output.read_text(encoding="utf-8"))["status"] == "locked_for_test_release"


def test_shap_ap_ties_use_locked_hit_round_family_order():
    spec_path = Path(__file__).resolve().parents[1] / "plans" / "experiment_spec_2026-09-13.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    entries = {}
    for surface in builder.SURFACES:
        for setting in builder.SHAP_SETTINGS:
            for family in builder.FAMILIES:
                ap, hit3, selected_round = 0.60, 0.5, 3
                if (surface, setting) == ("turf", "S6"):
                    ap, hit3, selected_round = 0.75, {
                        "catboost": 0.8, "lightgbm": 0.8, "xgboost": 0.7}[family], {
                            "catboost": 7, "lightgbm": 6, "xgboost": 5}[family]
                elif (surface, setting) == ("turf", "S7"):
                    ap, hit3, selected_round = 0.75, 0.8, 6
                elif family == "xgboost":
                    ap = 0.9
                entries[(family, surface, setting)] = {
                    "refits": [{"average_precision": ap, "mean_hit_at_3": hit3,
                                 "selected_round": selected_round} for _ in builder.SEEDS]
                }
    choices = builder._shap_choices(entries, spec)
    selected = {(item["surface"], item["setting"]): item["selected_family"]
                for item in choices}
    assert selected[("turf", "S6")] == "lightgbm"  # lower round after equal AP/Hit@3
    assert selected[("turf", "S7")] == "catboost"  # fixed catboost/lightgbm/xgboost order


if __name__ == "__main__":
    test_manifest_refuses_audit_failure_without_writing()
    test_manifest_locks_all_artifacts_and_shap_choices()
    test_shap_ap_ties_use_locked_hit_round_family_order()
    print("PASS: selection manifest rejects incomplete artifacts, binds runtime spec, and applies SHAP tie rules")
