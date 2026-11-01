"""Small no-ML checks for the seed-aware prediction exporter."""
import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path

import prepare_prediction_seeds as exporter
from prepare_prediction_seeds import (
    FROZEN_INPUT_MANIFEST_SHA256,
    FROZEN_SCRIPT_SHA256,
    coverage_by_split_surface,
    load_graph_seed,
    prepare_exports,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_coverage_counts_entries_and_horses():
    rows = [
        {"split": "train", "surface": "turf", "horse_id_netkeiba": "a", "embedding_status": "direct"},
        {"split": "train", "surface": "turf", "horse_id_netkeiba": "a", "embedding_status": "direct"},
        {"split": "train", "surface": "turf", "horse_id_netkeiba": "b", "embedding_status": "fallback"},
        {"split": "validation", "surface": "dirt", "horse_id_netkeiba": "c", "embedding_status": "missing"},
    ]
    coverage = coverage_by_split_surface(rows)
    assert coverage["train"]["turf"] == {
        "entries": {"total": 3, "direct": 2, "fallback": 1, "missing": 0},
        "unique_horses": {"total": 2, "direct": 1, "fallback": 1, "missing": 0},
    }
    assert coverage["validation"]["dirt"]["unique_horses"]["missing"] == 1


def _graph(folder, seed):
    import numpy as np

    folder.mkdir()
    vectors = np.asarray([[1.0] * 64, [2.0] * 64], dtype="float32")
    np.savez_compressed(folder / f"nodevectors_seed{seed}.npz",
                        node_ids=np.asarray(["h", "anc"]), vectors=vectors)
    centers = np.zeros((64, 64), dtype="float32")
    centers[1, :] = 10 + seed
    np.save(folder / f"cluster_centers_seed{seed}.npy", centers)
    (folder / "training_graph.edg").write_text("anc\th\n", encoding="utf-8")
    artifacts = {
        path.name: _sha(path)
        for path in (folder / f"nodevectors_seed{seed}.npz",
                     folder / f"cluster_centers_seed{seed}.npy",
                     folder / "training_graph.edg")
    }
    report = {
        "status": "complete",
        "settings": {"seed": seed, "dimensions": 64},
        "selected_k": 64,
        "graph_sha256": artifacts["training_graph.edg"],
        "input_manifest_sha256": FROZEN_INPUT_MANIFEST_SHA256,
        "script_sha256": FROZEN_SCRIPT_SHA256,
        "target_outcomes_read": False,
        "validation_test_roots_input": False,
        "artifact_sha256": artifacts,
    }
    (folder / "production_report.json").write_text(json.dumps(report), encoding="utf-8")


def _toy_inputs(folder):
    features = folder / "features.db"
    con = sqlite3.connect(features)
    con.executescript("""
        CREATE TABLE entries (
          race_id TEXT, horse_id_netkeiba TEXT, horse_number INTEGER,
          split TEXT, label INTEGER, surface TEXT, venue TEXT, distance INTEGER,
          going TEXT, race_class TEXT, field_size INTEGER, bracket INTEGER,
          weight_carried REAL, age INTEGER, sex TEXT, body_weight INTEGER,
          body_weight_change INTEGER, hist_count INTEGER, has_hist1 INTEGER,
          has_hist2 INTEGER, has_hist3 INTEGER);
        CREATE TABLE history_long (
          race_id TEXT, horse_id_netkeiba TEXT, horse_number INTEGER,
          hist_rank INTEGER, prior_race_id TEXT, prior_race_date TEXT,
          prior_finish_pos INTEGER, prior_result_status TEXT, prior_norm_finish REAL,
          prior_field_size INTEGER, prior_distance INTEGER, prior_surface TEXT,
          prior_going TEXT, prior_race_class TEXT, prior_weight_carried REAL,
          days_since INTEGER);
        CREATE TABLE meta (key TEXT, value TEXT);
    """)
    con.execute("INSERT INTO meta VALUES (?,?)",
                ("uncertain_history_races_excluded", "201009040210,201407010204"))
    base = ("r1", "h", 1, "train", 1, "turf", "V", 1200, "G", "C", 1, 1,
            55, 3, "M", 450, 0, 0, 0, 0, 0)
    con.execute("INSERT INTO entries VALUES (" + ",".join("?" * len(base)) + ")", base)
    val = list(base)
    val[0], val[1], val[3], val[4], val[5] = "r2", "new", "validation", 0, "dirt"
    con.execute("INSERT INTO entries VALUES (" + ",".join("?" * len(val)) + ")", val)
    con.commit(); con.close()

    source = folder / "source.db"
    con = sqlite3.connect(source)
    con.executescript("CREATE TABLE races_r (race_id TEXT, race_date TEXT);"
                      "CREATE TABLE runners_r (race_id TEXT, horse_id_netkeiba TEXT,"
                      "horse_number INTEGER, odds REAL, popularity INTEGER);")
    con.executemany("INSERT INTO races_r VALUES (?,?)", [("r1", "20200101"), ("r2", "20220101")])
    con.executemany("INSERT INTO runners_r VALUES (?,?,?,?,?)",
                    [("r1", "h", 1, 3.2, 2), ("r2", "new", 1, 5.0, 4)])
    con.commit(); con.close()

    pedigree = folder / "pedigree.db"
    con = sqlite3.connect(pedigree)
    con.execute("CREATE TABLE pedigree_edges (child_jbis_id TEXT, parent_jbis_id TEXT, relation TEXT)")
    con.executemany("INSERT INTO pedigree_edges VALUES (?,?,?)",
                    [("new", "anc", "sire"), ("new", "anc", "dam")])
    con.commit(); con.close()
    return features, source, pedigree


def test_three_seed_export_uses_each_graph_and_never_test():
    try:
        import numpy  # noqa: F401
    except ImportError:
        return
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        features, source, pedigree = _toy_inputs(folder)
        source_before = _sha(source)
        features_before = _sha(features)
        graph_dirs = {}
        for seed in (0, 1, 2):
            graph_dirs[seed] = folder / f"seed-{seed}"
            _graph(graph_dirs[seed], seed)
        # The toy edge list is not the frozen production graph; keep the
        # loader's provenance rule exercised without inventing its checksum.
        exporter.FROZEN_GRAPH_SHA256 = _sha(graph_dirs[0] / "training_graph.edg")
        for seed in (0, 1, 2):
            vectors, centers, meta = load_graph_seed(graph_dirs[seed], seed)
            assert meta["seed"] == seed and len(vectors) == 2 and len(centers) == 64
        try:
            load_graph_seed(graph_dirs[0], 1)
        except ValueError as error:
            assert "seed mismatch" in str(error)
        else:
            raise AssertionError("a seed must use its own graph directory/report")
        outputs = prepare_exports(features, source, [pedigree], graph_dirs, folder / "out")
        assert set(outputs) == {0, 1, 2}
        assert _sha(source) == source_before and _sha(features) == features_before
        for seed, output in outputs.items():
            assert Path(output).is_file()
            metadata = json.loads(Path(output).with_suffix(".json").read_text(encoding="utf-8"))
            assert metadata["seed"] == seed
            assert metadata["splits"] == ("train", "validation") or metadata["splits"] == ["train", "validation"]
            assert metadata["coverage"]["validation"]["dirt"]["entries"]["fallback"] == 1
            assert metadata["coverage"]["validation"]["dirt"]["unique_horses"]["fallback"] == 1
            assert "test" not in metadata["splits"]


def test_provenance_hashes_are_keyed():
    try:
        import numpy  # noqa: F401
    except ImportError:
        return
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        _graph(folder / "seed-0", 0)
        report_path = folder / "seed-0" / "production_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["input_manifest_sha256"], report["script_sha256"] = (
            report["script_sha256"], report["input_manifest_sha256"])
        report_path.write_text(json.dumps(report), encoding="utf-8")
        try:
            load_graph_seed(folder / "seed-0", 0)
        except ValueError as error:
            assert "provenance" in str(error)
        else:
            raise AssertionError("swapped provenance fields must be rejected")


if __name__ == "__main__":
    test_coverage_counts_entries_and_horses()
    test_three_seed_export_uses_each_graph_and_never_test()
    test_provenance_hashes_are_keyed()
    print("PASS: seed-specific artifacts, frozen fallback, split/surface coverage, provenance, and no-test export")
