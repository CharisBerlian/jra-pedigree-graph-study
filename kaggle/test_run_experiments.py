"""No-ML checks for the bounded generalized experiment runner."""
import os
import gzip
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import run_experiments as runner


def _spec(path, mode="smoke"):
    value = {
        "spec_id": "experiment_spec_2026-09-13_smoke" if mode == "smoke" else "experiment_spec_2026-09-13",
        "families": list(runner.FAMILIES),
        "settings": ["B1", "S1", "S2", "S3", "S4", "S5", "S6", "M2", "M3", "S7"],
        "tuning": {"tuned_seed": 0, "candidates_per_family_setting_surface": 4,
                   "max_rounds": 8 if mode == "smoke" else 1200,
                   "patience": 3 if mode == "smoke" else 75,
                   "ap_tie_tolerance": 1e-6, "threads": 1,
                   "candidates": {family: {f"c{i}": {"learning_rate": .05}
                                             for i in range(4)} for family in runner.FAMILIES}},
        "threshold": {"fitted_settings": "validation_grid_0.01_to_0.99_step_0.01_max_f1"},
    }
    path.write_text(json.dumps(value), encoding="utf-8")


def test_selection_and_spec_policy():
    records = [{"status": "complete", "candidate_id": cid,
                "average_precision": .5, "mean_hit_at_3": hit, "best_round": round_}
               for cid, hit, round_ in (("c0", .2, 4), ("c1", .2, 3),
                                         ("c2", .3, 8), ("c3", .2, 2))]
    assert runner.choose_candidate(records, 1e-6)["candidate_id"] == "c2"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "smoke.json"
        _spec(path)
        spec = runner._load_spec(path, "smoke")
        assert runner._policy(spec)["max_rounds"] == 8
        assert tuple(runner._candidates(spec, "xgboost")) == ("c0", "c1", "c2", "c3")
        scientific = json.loads(path.read_text(encoding="utf-8"))
        scientific["spec_id"] = "experiment_spec_2026-09-13"
        scientific["status"] = "proposed_for_root_review"
        path.write_text(json.dumps(scientific), encoding="utf-8")
        try:
            runner._load_spec(path, "scientific")
        except ValueError as error:
            assert "specification" in str(error)
        else:
            raise AssertionError("scientific runs require accepted specification status")


def test_lightgbm_import_forces_headless_backend():
    calls = []
    fake_matplotlib = types.SimpleNamespace(
        use=lambda backend, force=False: calls.append((backend, force)))
    fake_lightgbm = types.ModuleType("lightgbm")
    previous = os.environ.get("MPLBACKEND")
    try:
        os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
        with patch.dict(sys.modules, {"matplotlib": fake_matplotlib, "lightgbm": fake_lightgbm}):
            assert runner._import_lightgbm() is fake_lightgbm
        assert os.environ["MPLBACKEND"] == "Agg"
        assert calls == [("Agg", True)]
    finally:
        if previous is None:
            os.environ.pop("MPLBACKEND", None)
        else:
            os.environ["MPLBACKEND"] = previous


def test_required_helper_and_forbidden_test_rows():
    columns = ["venue", "distance", "going", "race_class", "field_size", "bracket",
               "horse_number", "weight_carried", "age", "sex", "body_weight",
               "body_weight_change"]
    assert "horse_number" in runner.feature_allowlist("S1", columns)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rows.csv"
        path.write_text("race_id,horse_id_netkeiba,horse_number,split,label,surface,field_size\n"
                        "r,h,1,test,1,turf,1\n", encoding="utf-8")
        try:
            runner.read_rows(path)
        except ValueError as error:
            assert "test" in str(error)
        else:
            raise AssertionError("test rows must be rejected")


def test_sidecar_is_required_and_seed_bound():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "input.csv"
        path.write_text("x", encoding="utf-8")
        try:
            runner._metadata(path)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("input sidecar must be mandatory")
        sidecar = path.with_suffix(".json")
        sidecar.write_text(json.dumps({"data_sha256": runner.sha256_file(path), "splits": ["train", "validation"], "seed": 1}), encoding="utf-8")
        assert runner._metadata(path)["seed"] == 1
        sidecar.write_text(json.dumps({"data_sha256": runner.sha256_file(path), "splits": None, "seed": 1}), encoding="utf-8")
        try:
            runner._metadata(path)
        except ValueError as error:
            assert "splits" in str(error)
        else:
            raise AssertionError("missing split declaration must be rejected")
        sidecar.write_text(json.dumps({"data_sha256": runner.sha256_file(path), "splits": ["train", "validation"], "seed": 1}), encoding="utf-8")
        path.write_text("tampered", encoding="utf-8")
        try:
            runner._metadata(path)
        except ValueError as error:
            assert "checksum" in str(error)
        else:
            raise AssertionError("sidecar checksum must bind exact input bytes")


def test_gzip_sidecar_accepts_trusted_legacy_raw_hash():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "input.csv.gz"
        payload = b"race_id,split\nr,train\n"
        with path.open("wb") as raw:
            stream = gzip.GzipFile(fileobj=raw, mode="wb", mtime=0)
            stream.write(payload)
            stream.close()
        sidecar = path.with_suffix(".json")
        sidecar.write_text(json.dumps({
            "data_sha256": runner.raw_sha256_file(path),
            "uncompressed_sha256": runner.csv_uncompressed_sha256(path),
            "purpose": "JRA train-validation prediction inputs",
            "splits": ["train", "validation"], "seed": 1,
        }), encoding="utf-8")
        assert runner._metadata(path)["seed"] == 1

        sidecar.write_text(json.dumps({
            "data_sha256": runner.raw_sha256_file(path),
            "uncompressed_sha256": "0" * 64,
            "purpose": "JRA train-validation prediction inputs",
            "splits": ["train", "validation"], "seed": 1,
        }), encoding="utf-8")
        try:
            runner._metadata(path)
        except ValueError as error:
            assert "checksum" in str(error)
        else:
            raise AssertionError("legacy raw hash must require matching uncompressed hash")


def test_refit_requires_verified_tuning_checkpoint():
    with tempfile.TemporaryDirectory() as directory:
        tune_dir = Path(directory) / "tune" / "seed0"
        tune_dir.mkdir(parents=True)
        (tune_dir / "selected.json").write_text(json.dumps({"status": "complete", "candidate_id": "c0", "best_round": 1}), encoding="utf-8")
        probe = {"family": "xgboost", "surface": "turf", "setting": "S1"}
        try:
            runner._verified_tune_selection(tune_dir, probe)
        except FileNotFoundError as error:
            assert "complete.json" in str(error)
        else:
            raise AssertionError("refit must refuse selection without a complete marker")


def test_validation_threshold_rule():
    rows = [{"label": str(label), "race_id": "r" + str(index // 2), "horse_number": str(index % 2 + 1),
             "horse_id_netkeiba": "h" + str(index)} for index, label in enumerate((1, 0, 1, 0))]
    try:
        import numpy as np
        scores = np.asarray([0.9, 0.1, 0.8, 0.2], dtype=np.float64)
    except ModuleNotFoundError:
        scores = [0.9, 0.1, 0.8, 0.2]
    result = runner.validation_threshold(rows, scores,
                                         {"threshold": {"fitted_settings": "validation_grid_0.01_to_0.99_step_0.01_max_f1"}})
    json.dumps(result, allow_nan=False)
    assert result["selected"]["threshold"] == 0.5


def test_recursive_artifact_hashes():
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        nested = folder / "candidates" / "c0"
        nested.mkdir(parents=True)
        (nested / "model.json").write_text("model", encoding="utf-8")
        hashes = runner._artifact_hashes(folder)
        assert Path("candidates", "c0", "model.json").as_posix() in hashes
        marker = runner._finish(folder, "config")
        assert runner._check_existing(folder, "config")["status"] == "skipped"
        assert marker["artifacts"] == runner._artifact_hashes(folder)


def test_recursive_artifact_hashes_accept_kaggle_csv_materialization():
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        nested = folder / "refit" / "seed1"
        nested.mkdir(parents=True)
        payload = b"race_id,horse_id\nr1,h1\n"
        compressed = nested / "validation_predictions.csv.gz"
        with gzip.open(compressed, "wb") as stream:
            stream.write(payload)
        marker = runner._finish(folder, "config")
        compressed.unlink()
        (nested / "validation_predictions.csv").write_bytes(payload)
        assert runner._check_existing(folder, "config")["status"] == "skipped"
        assert marker["artifacts"] == runner._artifact_hashes(folder)
        (nested / "validation_predictions.csv").write_bytes(payload + b"tampered")
        try:
            runner._check_existing(folder, "config")
        except ValueError as error:
            assert "artifact hash mismatch" in str(error)
        else:
            raise AssertionError("materialized CSV content mismatch must be rejected")


def test_recursive_artifact_hashes_rejects_gzip_csv_collision():
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        folder.mkdir(exist_ok=True)
        (folder / "predictions.csv").write_bytes(b"csv")
        with gzip.open(folder / "predictions.csv.gz", "wb") as stream:
            stream.write(b"gzip")
        try:
            runner._artifact_hashes(folder)
        except ValueError as error:
            assert "duplicate canonical artifact path" in str(error)
        else:
            raise AssertionError("CSV and CSV.GZ collision must be rejected")


def test_artifact_manifest_canonicalizes_csv_gzip_alias():
    digest = "a" * 64
    assert runner._canonical_artifact_manifest({"nested/predictions.csv.gz": digest}) == {
        "nested/predictions.csv": digest
    }
    try:
        runner._canonical_artifact_manifest({
            "predictions.csv": digest, "predictions.csv.gz": digest
        })
    except ValueError as error:
        assert "duplicate canonical artifact path" in str(error)
    else:
        raise AssertionError("manifest CSV alias collision must be rejected")


def test_verified_tune_accepts_pre_canonical_gzip_marker():
    with tempfile.TemporaryDirectory() as directory:
        tune_dir = Path(directory)
        (tune_dir / "complete.json").write_text(json.dumps({"status": "complete", "artifacts": {}}), encoding="utf-8")
        payload = b"csv\n"
        artifact = tune_dir / "validation_predictions.csv.gz"
        artifact.write_bytes(gzip.compress(payload))
        marker = {"status": "complete", "artifacts": runner._legacy_artifact_hashes(tune_dir)}
        (tune_dir / "complete.json").write_text(json.dumps(marker), encoding="utf-8")
        assert marker["artifacts"] == runner._legacy_artifact_hashes(tune_dir)
        assert marker["artifacts"] != runner._artifact_hashes(tune_dir)


if __name__ == "__main__":
    test_selection_and_spec_policy()
    test_lightgbm_import_forces_headless_backend()
    test_required_helper_and_forbidden_test_rows()
    test_sidecar_is_required_and_seed_bound()
    test_refit_requires_verified_tuning_checkpoint()
    test_validation_threshold_rule()
    test_recursive_artifact_hashes()
    test_recursive_artifact_hashes_accept_kaggle_csv_materialization()
    test_recursive_artifact_hashes_rejects_gzip_csv_collision()
    test_artifact_manifest_canonicalizes_csv_gzip_alias()
    print("PASS: helper allowlists, spec policy, candidate ties, test exclusion, and recursive checkpoint hashing")
