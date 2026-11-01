"""Synthetic no-fit checks for the tuning/refit output-tree audit."""
import gzip
import json
import tempfile
from pathlib import Path

import audit_experiment_outputs as audit


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _marker(folder, config_hash):
    artifacts = audit._artifact_hashes(folder)
    _write_json(folder / "complete.json", {"status": "complete", "config_hash": config_hash,
                                            "artifacts": artifacts})


def _prediction(path):
    rows = []
    for race in ("r1", "r2"):
        for horse in range(1, 5):
            rows.append((race, f"{race}-h{horse}", horse, int(horse <= 3),
                         (.9, .8, .7, .1)[horse - 1]))
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        stream.write("race_id,horse_id_netkeiba,horse_number,surface,label,prediction,selected_top3\n")
        for race, horse_id, horse, label, score in rows:
            stream.write(f"{race},{horse_id},{horse},turf,{label},{score},{int(horse <= 3)}\n")


def _spec():
    families = list(audit.FAMILIES)
    grid = {family: {f"c{i}": {"learning_rate": .01 * (i + 1)} for i in range(4)}
            for family in families}
    return {"spec_id": "experiment_spec_2026-09-13_smoke", "families": families,
            "surfaces": list(audit.SURFACES), "settings": ["S1"],
            "matched_seeds": list(audit.SEEDS),
            "tuning": {"max_rounds": 8, "patience": 3, "threads": 1,
                       "ap_tie_tolerance": 1e-6, "candidates": grid}}


def _make_tree(root, spec):
    grid = spec["tuning"]["candidates"]
    provenance = {"spec_sha256": "spec", "script_sha256": "script",
                  "helper_sha256": "helper", "environment_sha256": "environment",
                  "input_sha256": "input", "input_metadata_sha256": "input-meta"}
    for family in audit.FAMILIES:
        for surface in audit.SURFACES:
            base = root / "smoke" / family / surface / "S1"
            tune = base / "tune" / "seed0"
            candidate_records = []
            tune_config = {"purpose": "JRA smoke experiment checkpoint", "family": family,
                           "surface": surface, "setting": "S1", "stage": "tune", "seed": 0,
                           "run_mode": "smoke", "spec_id": spec["spec_id"], "features": ["x"],
                           "categorical": [], "max_rounds": 8, "patience": 3, "threads": 1,
                           "candidate_grid_sha256": audit._canonical(grid[family]), **provenance}
            tune.mkdir(parents=True)
            config_hash = audit._canonical(tune_config)
            _write_json(tune / "config.json", tune_config)
            _write_json(tune / "preprocessing.json", {"columns": ["x"]})
            for i in range(4):
                candidate_id = f"c{i}"
                candidate = tune / "candidates" / candidate_id
                candidate.mkdir(parents=True)
                candidate_config = {"parent_config_hash": config_hash, "candidate_id": candidate_id,
                                    "params": grid[family][candidate_id]}
                _write_json(candidate / "config.json", candidate_config)
                curve = [1.0, 1.0, 1.0]
                metrics = {"status": "complete", "candidate_id": candidate_id,
                           "average_precision": max(curve), "mean_hit_at_3": .5 + i * .01,
                           "best_round": 1, "hit_cap": False, "seconds": 1.0,
                           "curve": curve, "params": grid[family][candidate_id]}
                _write_json(candidate / "metrics.json", metrics)
                _write_json(candidate / "ap_curve.json",
                            {"metric": "sklearn.metrics.average_precision_score", "values": curve})
                (candidate / audit.MODEL_NAMES[family]).write_text("model", encoding="utf-8")
                _prediction(candidate / "validation_predictions.csv.gz")
                _marker(candidate, audit._canonical(candidate_config))
                candidate_records.append(metrics)
            _write_json(tune / "candidate_results.json", candidate_records)
            # c3 has the largest AP and Hit@3, so it is the deterministic winner.
            _write_json(tune / "selected.json", candidate_records[3])
            _marker(tune, config_hash)
            tune_marker_hash = audit._sha256(tune / "complete.json")
            tune_config_hash = audit._sha256(tune / "config.json")
            selected_hash = audit._sha256(tune / "selected.json")
            for seed in audit.SEEDS:
                refit = base / "refit" / f"seed{seed}"
                refit.mkdir(parents=True)
                refit_config = {"purpose": "JRA smoke experiment checkpoint", "family": family,
                                "surface": surface, "setting": "S1", "stage": "refit", "seed": seed,
                                "run_mode": "smoke", "spec_id": spec["spec_id"], "features": ["x"],
                                "categorical": [], "max_rounds": 8, "patience": 3, "threads": 1,
                                "selected_candidate": "c3", "selected_round": 1,
                                "params": grid[family]["c3"], "selection_sha256": selected_hash,
                                "tuning_complete_sha256": tune_marker_hash,
                                "tuning_config_sha256": tune_config_hash, **provenance}
                _write_json(refit / "config.json", refit_config)
                _write_json(refit / "preprocessing.json", {"columns": ["x"]})
                (refit / "environment.txt").write_text("python", encoding="utf-8")
                _write_json(refit / "metrics.json", {
                    "average_precision": 1.0, "brier": .2, "roc_auc": .9,
                    "complete_validation_races": 2, "mean_hit_at_3": 1.0, "seed": seed,
                    "selected_candidate": "c3", "selected_round": 1,
                    "train_rows": 8, "validation_rows": 8,
                    "validation_threshold": {"rule": audit.THRESHOLD_RULE,
                        "grid": [0.01, 0.99, 0.01],
                        "selected": {"threshold": .5, "precision": 1.0, "recall": 1.0,
                                     "f1": 1.0, "predicted_positive": 6}}})
                (refit / audit.MODEL_NAMES[family]).write_text("model", encoding="utf-8")
                _prediction(refit / "validation_predictions.csv.gz")
                _marker(refit, audit._canonical(refit_config))


def test_audit_complete_tree_and_reject_tampering():
    spec = _spec()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _make_tree(root, spec)
        report = audit.audit_output_tree(root, spec, "smoke")
        assert report["status"] == "PASS"
        assert report["tuning_runs"] == 6
        assert report["candidate_runs"] == 24
        assert report["refit_runs"] == 18
        partial = audit.audit_output_tree(root, spec, "smoke", families=("xgboost",),
                                          surfaces=("turf",), settings=("S1",), stages="tune")
        assert partial["candidate_runs"] == 4
        assert partial["refit_runs"] == 0
        try:
            audit.audit_output_tree(root, spec, "smoke", families=("xgboost",),
                                    surfaces=("turf",), settings=("S1",), stages="all")
        except ValueError as error:
            assert "full locked" in str(error)
        else:
            raise AssertionError("all-stage audit must require the full matrix")
        selected = root / "smoke" / "xgboost" / "turf" / "S1" / "tune" / "seed0" / "selected.json"
        value = json.loads(selected.read_text(encoding="utf-8"))
        value["candidate_id"] = "c0"
        _write_json(selected, value)
        try:
            audit.audit_output_tree(root, spec, "smoke")
        except ValueError as error:
            assert "artifact hash mismatch" in str(error)
        else:
            raise AssertionError("audit must reject changed selected artifact")


def test_audit_rejects_test_paths_and_threshold_drift():
    spec = _spec()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _make_tree(root, spec)
        (root / "smoke" / "xgboost" / "turf" / "S1" / "refit" / "seed0" / "test_metrics.json").write_text("{}")
        try:
            audit.audit_output_tree(root, spec, "smoke")
        except ValueError as error:
            assert "test artifact" in str(error)
        else:
            raise AssertionError("test artifacts must stay outside the audit scope")


if __name__ == "__main__":
    test_audit_complete_tree_and_reject_tampering()
    test_audit_rejects_test_paths_and_threshold_drift()
    print("PASS: synthetic tuning/refit tree audit, hashes, selection, thresholds, and test guard")
