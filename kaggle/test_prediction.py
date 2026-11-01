"""No-ML checks for the bounded prediction preparation and metrics."""
import csv
import gzip
import json
import sqlite3
import tempfile
from pathlib import Path

from prepare_prediction import build_rows, load_graph, resolve_embedding, write_export
from predict_validation import (feature_columns, fit_category_maps, mean_hit_at_3,
                                read_rows, run_one, top_three)


def row(race, horse, number, label, split="validation", surface="turf", **extra):
    d = {"race_id": race, "horse_id_netkeiba": horse, "horse_number": str(number),
         "split": split, "label": str(label), "surface": surface}
    d.update(extra)
    return d


def test_embedding_fallbacks():
    vectors = {"h": (1.0, 2.0), "a": (3.0, 5.0), "b": (5.0, 7.0)}
    parents = {("u", "sire"): "a", ("u", "dam"): "a",
               ("v", "sire"): "b", ("v", "dam"): "b",
               ("x", "sire"): "u", ("x", "dam"): "v"}
    direct = resolve_embedding("h", vectors, parents, [(0.0, 0.0), (2.0, 2.0)])
    assert direct[:5] == ((1.0, 2.0), "direct", 0, 0, 0)
    fallback = resolve_embedding("x", vectors, parents)
    assert fallback[:5] == ((4.0, 6.0), "fallback", 2, 2, 0)
    missing = resolve_embedding("none", vectors, parents)
    assert missing[:5] == ((0.0, 0.0), "missing", None, 0, 1)
    assert resolve_embedding("none", vectors, parents)[5] == "MISSING"


def test_allowlists_and_train_categories():
    columns = set(("venue", "distance", "going", "race_class", "field_size", "bracket",
                   "horse_number", "weight_carried", "age", "sex", "body_weight",
                   "body_weight_change", "hist_count", "has_hist1", "has_hist2", "has_hist3",
                   "sire", "dam", "odds", "popularity", "embedding_0", "embedding_status",
                   "fallback_depth", "ancestor_count", "embedding_missing", "graph_cluster"))
    columns.update(f"{name}_{rank}" for name in ("prior_finish_pos", "prior_norm_finish",
                                                   "prior_field_size", "prior_distance",
                                                   "prior_surface", "prior_going",
                                                   "prior_race_class", "prior_weight_carried",
                                                   "days_since") for rank in (1, 2, 3))
    columns.update(f"embedding_{i}" for i in range(64))
    assert "odds" not in feature_columns("S2", columns)
    assert "popularity" not in feature_columns("S6", columns)
    assert {"odds", "popularity"} <= set(feature_columns("M2", columns))
    missing_dimension = set(columns) - {"embedding_63"}
    try:
        feature_columns("S5", missing_dimension)
    except ValueError as exc:
        assert "exactly embedding_0..embedding_63" in str(exc)
    else:
        raise AssertionError("missing graph dimension must be rejected")
    extra_dimension = set(columns) | {"embedding_64"}
    try:
        feature_columns("S5", extra_dimension)
    except ValueError as exc:
        assert "exactly embedding_0..embedding_63" in str(exc)
    else:
        raise AssertionError("unexpected graph dimension must be rejected")
    train = [row("r1", "a", 1, 1, sire="known"), row("r1", "b", 2, 0, sire="other")]
    maps = fit_category_maps(train, ("sire",))
    assert maps["sire"] == {"known": 1, "other": 2}
    assert maps["sire"].get("unseen", 0) == 0


def test_metrics_and_ties():
    rows = [row("r", "z", 2, 1), row("r", "a", 1, 0), row("r", "b", 3, 1), row("r", "c", 4, 0)]
    selected = top_three(rows, [0.5, 0.5, 0.5, 0.5])
    assert [r["horse_id_netkeiba"] for r, _ in selected] == ["a", "z", "b"]
    hit, count = mean_hit_at_3(rows, [0.5, 0.5, 0.5, 0.5])
    assert (hit, count) == (2 / 3, 1)


def test_initial_batch_seed_guard():
    try:
        run_one([], (), "missing.csv.gz", tempfile.gettempdir(), "turf", "S2", seed=1)
    except ValueError as exc:
        assert "seed 0" in str(exc)
    else:
        raise AssertionError("nonzero seed must be rejected for initial batch")


def test_toy_feature_join_and_train_validation_only():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        fdb, sdb, pdb = p / "features.db", p / "source.db", p / "pedigree.db"
        c = sqlite3.connect(fdb)
        c.execute("CREATE TABLE entries (race_id TEXT, horse_id_netkeiba TEXT, horse_number INTEGER, split TEXT, label INTEGER, surface TEXT, venue TEXT, distance INTEGER, going TEXT, race_class TEXT, field_size INTEGER, bracket INTEGER, weight_carried REAL, age INTEGER, sex TEXT, body_weight INTEGER, body_weight_change INTEGER, hist_count INTEGER, has_hist1 INTEGER, has_hist2 INTEGER, has_hist3 INTEGER)")
        c.execute("CREATE TABLE history_long (race_id TEXT, horse_id_netkeiba TEXT, horse_number INTEGER, hist_rank INTEGER, prior_race_id TEXT, prior_race_date TEXT, prior_finish_pos INTEGER, prior_result_status TEXT, prior_norm_finish REAL, prior_field_size INTEGER, prior_distance INTEGER, prior_surface TEXT, prior_going TEXT, prior_race_class TEXT, prior_weight_carried REAL, days_since INTEGER)")
        c.execute("CREATE TABLE meta (key TEXT, value TEXT)")
        c.execute("INSERT INTO meta VALUES (?,?)", ("uncertain_history_races_excluded", "201009040210,201407010204"))
        base = ("r1", "h", 1, "train", 1, "turf", "V", 1200, "G", "C", 1, 1, 55, 3, "M", 450, 0, 1, 1, 0, 0)
        c.execute("INSERT INTO entries VALUES (" + ",".join("?" * len(base)) + ")", base)
        val = list(base); val[0], val[1], val[3], val[4] = "r2", "new", "validation", 0
        c.execute("INSERT INTO entries VALUES (" + ",".join("?" * len(val)) + ")", val)
        c.execute("INSERT INTO history_long VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ("r1", "h", 1, 1, "old", "20190101", 2, "finished", .5, 4, 1000, "turf", "G", "C", 55, 100))
        c.commit(); c.close()
        c = sqlite3.connect(sdb)
        c.execute("CREATE TABLE races_r (race_id TEXT, race_date TEXT)")
        c.executemany("INSERT INTO races_r VALUES (?,?)", [("r1", "20200101"), ("r2", "20220101")])
        c.execute("CREATE TABLE runners_r (race_id TEXT, horse_id_netkeiba TEXT, horse_number INTEGER, odds REAL, popularity INTEGER)")
        c.executemany("INSERT INTO runners_r VALUES (?,?,?,?,?)", [("r1", "h", 1, 3.2, 2), ("r2", "new", 1, 5.0, 4)])
        c.commit(); c.close()
        c = sqlite3.connect(pdb)
        c.execute("CREATE TABLE pedigree_edges (child_jbis_id TEXT,parent_jbis_id TEXT,relation TEXT)")
        c.executemany("INSERT INTO pedigree_edges VALUES (?,?,?)", [("new", "anc", "sire"), ("new", "anc", "dam")])
        c.commit(); c.close()
        cols, rows, _ = build_rows(fdb, sdb, [pdb])
        assert {r["split"] for r in rows} == {"train", "validation"}
        assert len(cols) == len(set(cols))
        assert len(rows) == 2 and rows[0]["prior_finish_pos_1"] == 2
        assert "finish_pos" not in cols and "result_status" not in cols
        out = p / "export.csv.gz"; write_export(out, cols, rows, {}, [fdb, sdb])
        assert len(read_rows(out)[0]) == 2


def test_canonical_graph_npz_when_numpy_available():
    try:
        import numpy as np
    except ImportError:
        return  # local checks intentionally have no heavy ML dependencies
    with tempfile.TemporaryDirectory() as td:
        folder = Path(td)
        np.savez_compressed(folder / "nodevectors_seed0.npz",
                            node_ids=np.asarray(["a", "b"]),
                            vectors=np.asarray([[1.] * 64, [3.] * 64], dtype="float32"))
        np.save(folder / "cluster_centers_seed0.npy", np.asarray([[0.] * 64, [3.] * 64]))
        (folder / "training_graph.edg").write_text("a\\tb\\n", encoding="utf-8")
        import hashlib
        report = {"status": "complete", "settings": {"seed": 0, "dimensions": 64},
                  "graph_sha256": "0" * 64, "input_manifest_sha256": "1" * 64,
                  "script_sha256": "2" * 64,
                  "artifact_sha256": {
                      "nodevectors_seed0.npz": hashlib.sha256((folder / "nodevectors_seed0.npz").read_bytes()).hexdigest(),
                      "cluster_centers_seed0.npy": hashlib.sha256((folder / "cluster_centers_seed0.npy").read_bytes()).hexdigest(),
                      "training_graph.edg": hashlib.sha256((folder / "training_graph.edg").read_bytes()).hexdigest()}}
        report["graph_sha256"] = report["artifact_sha256"]["training_graph.edg"]
        (folder / "production_report.json").write_text(json.dumps(report), encoding="utf-8")
        vectors, centers, meta = load_graph(folder)
        assert vectors["a"] == tuple([1.0] * 64) and len(centers) == 2 and meta["dimensions"] == 64
        np.savez_compressed(folder / "nodevectors_seed0.npz",
                            node_ids=np.asarray(["a", "a"]), vectors=np.asarray([[1.] * 64, [2.] * 64]))
        try:
            load_graph(folder)
        except ValueError as exc:
            assert "unique" in str(exc)
        else:
            raise AssertionError("duplicate graph IDs must be rejected before dict construction")


if __name__ == "__main__":
    test_embedding_fallbacks(); test_allowlists_and_train_categories(); test_metrics_and_ties(); test_initial_batch_seed_guard(); test_toy_feature_join_and_train_validation_only(); test_canonical_graph_npz_when_numpy_available()
    print("PASS: fallback, unique ancestor averaging, allowlists, categories, ties, metrics, prior joins, validation-only export, and canonical graph artifact checks")
