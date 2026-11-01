"""Synthetic no-fit checks for locked test inference plumbing."""
import csv
import gzip
import hashlib
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import run_test_inference as inference


def _test_rows():
    return [{"race_id": "r1", "horse_id_netkeiba": f"h{i}", "horse_number": i,
             "split": "test", "surface": "turf", "field_size": 4}
            for i in range(1, 5)]


def _write_input(path):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(_test_rows()[0]))
        writer.writeheader()
        writer.writerows(_test_rows())
    path.with_suffix(".json").write_text(json.dumps({
        "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "splits": ["test"]
    }), encoding="utf-8")


def test_resolve_test_inputs_directory_and_explicit_mapping():
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        paths = {}
        for seed in inference.SEEDS:
            path = folder / f"test_seed{seed}.csv.gz"
            path.write_bytes(f"synthetic-seed-{seed}".encode("ascii"))
            paths[seed] = path
        resolved = inference.resolve_test_inputs(folder)
        assert resolved == paths
        assert inference.resolve_test_inputs({str(seed): path for seed, path in paths.items()}) == paths

        (folder / "test_seed0.csv").write_bytes(b"duplicate")
        try:
            inference.resolve_test_inputs(folder)
        except ValueError as error:
            assert "duplicate" in str(error)
        else:
            raise AssertionError("duplicate seed files must be rejected")


def test_resolve_test_inputs_rejects_missing_and_mismatched_seeds():
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        seed0 = folder / "features0.csv"
        seed1 = folder / "features1.csv"
        seed2 = folder / "test_seed2.csv"
        for path in (seed0, seed1, seed2):
            path.write_bytes(b"synthetic")
        try:
            inference.resolve_test_inputs({0: seed0, 1: seed1})
        except ValueError as error:
            assert "exactly" in str(error)
        else:
            raise AssertionError("missing explicit seed must be rejected")
        try:
            inference.resolve_test_inputs({0: seed2, 1: seed0, 2: seed1})
        except ValueError as error:
            assert "disagrees" in str(error)
        else:
            raise AssertionError("filename/mapping seed disagreement must be rejected")


def test_test_input_and_key_preserving_output():
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        source = folder / "test.csv"
        _write_input(source)
        rows, metadata = inference.read_test_rows(source)
        assert metadata["splits"] == ["test"]
        destination = folder / "predictions.csv.gz"
        inference._write_predictions(destination, rows, [.9, .8, .7, .1])
        with gzip.open(destination, "rt", encoding="utf-8", newline="") as stream:
            output = list(csv.DictReader(stream))
        assert [(r["race_id"], r["horse_id_netkeiba"], int(r["horse_number"])) for r in output] == [
            ("r1", "h1", 1), ("r1", "h2", 2), ("r1", "h3", 3), ("r1", "h4", 4)]
        assert [int(r["selected_top3"]) for r in output] == [1, 1, 1, 0]
        source.write_text(source.read_text(encoding="utf-8").replace(",test,", ",validation,"), encoding="utf-8")
        try:
            inference.read_test_rows(source)
        except ValueError as error:
            assert "checksum" in str(error)
        else:
            raise AssertionError("changed test input must fail its sidecar checksum")


def test_csv_materialization_sidecar_alias():
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        source = folder / "test_seed0.csv"
        source.write_bytes(b"csv-materialized")
        sidecar = Path(str(source) + ".json")
        sidecar.write_text(json.dumps({"data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                                       "splits": ["test"]}), encoding="utf-8")
        assert inference._test_sidecar(source) == sidecar


def test_manifest_gate_and_kaggle_gate_fail_closed():
    spec = {"status": "locked", "test_performance_opened": False}
    manifest = {"purpose": "JRA test-release model selection manifest",
                "status": "draft", "test_performance_opened": False}
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        spec_path = folder / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        try:
            inference._validate_manifest(manifest, spec, spec_path, Path(__file__).parent)
        except ValueError as error:
            assert "locked" in str(error)
        else:
            raise AssertionError("unlocked manifest must be rejected")
        try:
            with patch.object(inference, "_on_kaggle", return_value=False):
                inference.run_inference(folder / "manifest.json", spec_path, folder, folder / "test.csv", folder / "out")
        except RuntimeError as error:
            assert "Kaggle" in str(error)
        else:
            raise AssertionError("test inference must be Kaggle-only")


def test_minimal_valid_manifest_hash_gate():
    spec = {"spec_id": "experiment_spec_2026-09-13", "status": "locked",
            "test_performance_opened": False, "families": list(inference.FAMILIES),
            "surfaces": list(inference.SURFACES), "settings": list(inference.SETTINGS),
            "matched_seeds": list(inference.SEEDS)}
    source_root = Path(__file__).parent
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        spec_path = folder / "spec.json"
        spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        provenance = {"script_sha256": inference._sha256(source_root / "run_experiments.py"),
                      "helper_sha256": inference._sha256(source_root / "experiment_features.py")}
        entries = [{"family": family, "surface": surface, "setting": setting}
                   for family in inference.FAMILIES for surface in inference.SURFACES
                   for setting in inference.SETTINGS]
        manifest = {"purpose": "JRA test-release model selection manifest",
                    "status": "locked_for_test_release", "spec_id": spec["spec_id"],
                    "spec_sha256": inference._runtime_spec_sha256(spec),
                    "spec_runtime_sha256": inference._runtime_spec_sha256(spec),
                    "spec_source_sha256": inference._sha256(spec_path),
                    "test_performance_opened": False, "contains_test_predictions": False,
                    "no_test_performance": True,
                    "audited_counts": {"tuning_runs": 60, "candidate_runs": 240, "refit_runs": 180},
                    "audit_script_sha256": inference._sha256(source_root / "audit_experiment_outputs.py"),
                    "evaluator_script_sha256": inference._sha256(source_root / "evaluate_locked.py"),
                    "provenance": provenance, "models": entries}
        selected = inference._validate_manifest(manifest, spec, spec_path, source_root,
                                                model_keys={("xgboost", "turf", "S1")})
        assert set(selected) == {("xgboost", "turf", "S1")}


def test_native_loader_uses_reload_and_prediction_only():
    calls = []
    class Booster:
        def __init__(self):
            self.rounds = 2
        def load_model(self, path):
            calls.append(("load", path))
        def num_boosted_rounds(self):
            return self.rounds
        def predict(self, matrix):
            calls.append(("predict", matrix))
            return [.2, .8]
    xgb = types.ModuleType("xgboost")
    xgb.Booster = Booster
    xgb.DMatrix = lambda frame, enable_categorical: ("dmatrix", frame, enable_categorical)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.json"
        path.write_text("native model", encoding="utf-8")
        with patch.dict(sys.modules, {"xgboost": xgb}):
            scores = inference._load_predict("xgboost", path, ["frame-1", "frame-2"], 2)
    assert scores == [.2, .8]
    assert calls[0][0] == "load" and calls[1][0] == "predict"


if __name__ == "__main__":
    test_resolve_test_inputs_directory_and_explicit_mapping()
    test_resolve_test_inputs_rejects_missing_and_mismatched_seeds()
    test_test_input_and_key_preserving_output()
    test_manifest_gate_and_kaggle_gate_fail_closed()
    test_native_loader_uses_reload_and_prediction_only()
    print("PASS: synthetic test input, manifest/Kaggle gates, key preservation, and native reload inference")
