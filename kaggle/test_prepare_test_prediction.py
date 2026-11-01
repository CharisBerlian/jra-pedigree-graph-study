"""Synthetic no-fit checks for test feature export."""
import csv
import hashlib
import json
import sqlite3
from pathlib import Path
import tempfile
from unittest.mock import patch

import prepare_test_prediction as exporter
import run_test_inference as inference


def _dbs(folder):
    features = folder / "features.db"
    connection = sqlite3.connect(features)
    connection.executescript("""
        CREATE TABLE entries (
          race_id TEXT, horse_id_netkeiba TEXT, horse_number INTEGER,
          split TEXT, label INTEGER, surface TEXT, venue TEXT, distance INTEGER,
          going TEXT, race_class TEXT, field_size INTEGER, bracket INTEGER,
          weight_carried REAL, age INTEGER, sex TEXT, body_weight REAL,
          body_weight_change REAL, hist_count INTEGER, has_hist1 INTEGER,
          has_hist2 INTEGER, has_hist3 INTEGER);
        CREATE TABLE history_long (
          race_id TEXT, horse_id_netkeiba TEXT, horse_number INTEGER,
          hist_rank INTEGER, prior_race_id TEXT, prior_race_date TEXT,
          prior_finish_pos INTEGER, prior_result_status TEXT,
          prior_norm_finish REAL, prior_field_size INTEGER, prior_distance INTEGER,
          prior_surface TEXT, prior_going TEXT, prior_race_class TEXT,
          prior_weight_carried REAL, days_since INTEGER);
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
    """)
    values = []
    for number, horse, label in ((1, "h1", 1), (2, "h2", 0), (3, "h3", 0)):
        values.append(("r1", horse, number, "test", label, "turf", "V", 1600,
                       "good", "C", 3, 1, 55.0, 4, "M", 470.0, 0.0, 1, 1, 0, 0))
    connection.executemany("INSERT INTO entries VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", values)
    connection.execute("INSERT INTO history_long VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       ("r1", "h1", 1, 1, "old", "20230101", 2, "finished", .5,
                        4, 1400, "turf", "good", "C", 54.0, 300))
    connection.executemany("INSERT INTO meta VALUES (?,?)", [
        ("uncertain_history_races_excluded", "201009040210,201407010204")])
    connection.commit()
    connection.close()

    source = folder / "source.db"
    connection = sqlite3.connect(source)
    connection.executescript("""
        CREATE TABLE runners_r (
          race_id TEXT, horse_id_netkeiba TEXT, horse_number INTEGER,
          odds REAL, popularity INTEGER, finish_pos INTEGER);
        CREATE TABLE races_r (race_id TEXT, race_date TEXT);
    """)
    connection.execute("INSERT INTO races_r VALUES ('r1','20240101')")
    connection.executemany("INSERT INTO runners_r VALUES (?,?,?,?,?,?)", [
        ("r1", "h1", 1, 2.0, 1, 1), ("r1", "h2", 2, 4.0, 2, 2),
        ("r1", "h3", 3, 8.0, 3, 3)])
    connection.commit()
    connection.close()

    pedigree = folder / "pedigree.db"
    connection = sqlite3.connect(pedigree)
    connection.execute("CREATE TABLE pedigree_edges (child_jbis_id TEXT, parent_jbis_id TEXT, relation TEXT)")
    connection.execute("INSERT INTO pedigree_edges VALUES ('h2','ancestor','sire')")
    connection.commit()
    connection.close()
    return features, source, pedigree


def _graph_dirs(folder):
    result = {}
    for seed in exporter.SEEDS:
        path = folder / f"graph{seed}"
        path.mkdir()
        for name in (f"nodevectors_seed{seed}.npz", f"cluster_centers_seed{seed}.npy",
                     "training_graph.edg", "production_report.json"):
            (path / name).write_bytes(f"seed-{seed}-{name}".encode())
        result[seed] = path
    return result


def test_test_export_is_label_free_and_seed_bound():
    features, source, pedigree = (None, None, None)
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        features, source, pedigree = _dbs(folder)
        graph_dirs = _graph_dirs(folder)

        def fake_graph(path, seed):
            vectors = {"h1": tuple([1.0] * 64), "ancestor": tuple([2.0] * 64)}
            metadata = {"seed": seed, "graph_sha256": "g", "input_manifest_sha256": "i",
                        "script_sha256": "s", "artifact_sha256": {}}
            return vectors, [tuple([0.0] * 64)], metadata

        with patch.object(exporter, "load_graph_seed", side_effect=fake_graph):
            outputs = exporter.prepare_exports(features, source, [pedigree], graph_dirs,
                                               folder / "out")
        assert set(outputs) == {0, 1, 2}
        for seed, raw_path in outputs.items():
            path = Path(raw_path)
            rows, metadata = inference.read_test_rows(path)
            assert metadata["seed"] == seed and metadata["splits"] == ["test"]
            assert "label" not in metadata["columns"]
            assert "finish_pos" not in metadata["columns"]
            assert [row["embedding_status"] for row in rows] == ["direct", "fallback", "missing"]
            assert metadata["test_labels_read"] is False
            assert metadata["data_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
            assert metadata["input_sha256"][str(features)]
            assert metadata["graph"]["seed"] == seed


def test_test_rows_refuse_non_test_split_and_do_not_depend_on_labels():
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        features, source, pedigree = _dbs(folder)
        graph_dirs = _graph_dirs(folder)
        graph = {"h1": tuple([1.0] * 64)}
        loader = lambda path, seed: (graph, [tuple([0.0] * 64)], {
            "seed": seed, "graph_sha256": "g", "input_manifest_sha256": "i", "script_sha256": "s"})
        with patch.object(exporter, "load_graph_seed", side_effect=loader):
            columns_a, rows_a, _ = exporter.build_test_rows(features, source, [pedigree], graph_dirs[0], 0)
            connection = sqlite3.connect(features)
            connection.execute("UPDATE entries SET label = 99")
            connection.commit()
            connection.close()
            columns_b, rows_b, _ = exporter.build_test_rows(features, source, [pedigree], graph_dirs[0], 0)
            assert columns_a == columns_b
            assert rows_a == rows_b
            connection = sqlite3.connect(features)
            connection.execute("UPDATE entries SET split = 'validation'")
            connection.commit()
            connection.close()
            try:
                exporter.build_test_rows(features, source, [pedigree], graph_dirs[0], 0)
            except ValueError as error:
                assert "no test rows" in str(error)
            else:
                raise AssertionError("non-test rows must not enter test export")


if __name__ == "__main__":
    test_test_export_is_label_free_and_seed_bound()
    test_test_rows_refuse_non_test_split_and_do_not_depend_on_labels()
    print("PASS: label-free test export, strict test split, graph fallback, seed provenance, and hashes")
