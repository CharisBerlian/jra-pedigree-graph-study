"""Prepare a train+validation-only feature export for the Kaggle batch.

The script reads the rebuilt ``features.db`` and source databases read-only. It
never selects or writes test rows. Graph vectors and centroids are expected to
be produced by the completed production graph job.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import sqlite3
from pathlib import Path

from graph_pilot import load_parents, read_db


SPLITS = ("train", "validation")
SURFACES = ("turf", "dirt")
MARKET_SETTINGS = ("M2", "M3", "S7")
GRAPH_SETTINGS = ("S4", "S5", "S6", "S7")
TARGET_COLUMNS = {"label", "finish_pos", "result_status", "time", "margin",
                  "last_3f", "corner_passing", "top_three"}
CONTEXT = ("venue", "distance", "going", "race_class", "field_size",
           "bracket", "horse_number", "weight_carried", "age", "sex",
           "body_weight", "body_weight_change")
HISTORY = ("hist_count", "has_hist1", "has_hist2", "has_hist3")
HISTORY_FIELDS = ("prior_finish_pos", "prior_norm_finish", "prior_field_size",
                  "prior_distance", "prior_surface", "prior_going",
                  "prior_race_class", "prior_weight_carried", "days_since")
PEDIGREE_CATS = ("sire", "dam")
GRAPH_META = ("embedding_status", "fallback_depth", "ancestor_count",
              "embedding_missing", "graph_cluster")
KEYS = ("race_id", "horse_id_netkeiba", "horse_number", "split", "label", "surface")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_graph(folder):
    """Load the completed seed-0 production artifacts.

    Canonical files are ``nodevectors_seed0.npz`` and
    ``cluster_centers_seed0.npy``. The report is required so an old pilot
    artifact cannot silently become a model input.
    """
    folder = Path(folder)
    vector_path = folder / "nodevectors_seed0.npz"
    if not vector_path.exists():
        raise FileNotFoundError("missing canonical nodevectors_seed0.npz")
    vectors = None
    if vector_path.suffix == ".npz":
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("NumPy is required to read production embeddings.npz") from exc
        with np.load(vector_path, allow_pickle=False) as data:
            if "node_ids" not in data or "vectors" not in data:
                raise ValueError("embeddings NPZ needs node_ids and vectors")
            ids = [str(x) for x in data["node_ids"].tolist()]
            if len(ids) != len(set(ids)) or any(not x for x in ids):
                raise ValueError("embeddings must have unique nonempty node IDs")
            matrix = data["vectors"].tolist()
            if len(ids) != len(matrix):
                raise ValueError("embedding node/vector row count mismatch")
        vectors = {n: tuple(float(x) for x in v) for n, v in zip(ids, matrix)}
    if not vectors or len(set(vectors)) != len(vectors):
        raise ValueError("embeddings must have unique nonempty node IDs")
    dims = {len(v) for v in vectors.values()}
    if len(dims) != 1 or next(iter(dims)) != 64:
        raise ValueError("embedding dimensions are inconsistent")
    if any(not math.isfinite(x) for v in vectors.values() for x in v):
        raise ValueError("embeddings contain nonfinite values")

    center_path = folder / "cluster_centers_seed0.npy"
    if not center_path.exists():
        raise FileNotFoundError("missing canonical cluster_centers_seed0.npy")
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required to read cluster centers") from exc
    centers = np.load(center_path, allow_pickle=False).tolist()
    centers = [tuple(float(x) for x in c) for c in centers]
    if not centers or any(len(c) != next(iter(dims)) for c in centers):
        raise ValueError("cluster center dimensions are inconsistent")
    if any(not math.isfinite(x) for c in centers for x in c):
        raise ValueError("cluster centers contain nonfinite values")
    report_path = folder / "production_report.json"
    if not report_path.exists():
        raise FileNotFoundError("missing production_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    settings = report.get("settings", {})
    if report.get("status") != "complete" or settings.get("seed") != 0:
        raise ValueError("production graph report is not a completed seed-0 run")
    if settings.get("dimensions") != 64:
        raise ValueError("production graph report does not declare 64 dimensions")
    for key in ("graph_sha256", "input_manifest_sha256", "script_sha256"):
        value = report.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("production report lacks valid %s" % key)
    # Output hashes are independently checked against the saved artifacts.
    expected = report.get("artifact_sha256")
    if not isinstance(expected, dict) or vector_path.name not in expected or center_path.name not in expected:
        raise ValueError("production report lacks output artifact hashes")
    graph_path = folder / "training_graph.edg"
    if not graph_path.exists() or expected.get("training_graph.edg") != hashlib.sha256(graph_path.read_bytes()).hexdigest():
        raise ValueError("training graph checksum missing or mismatched")
    if expected.get("training_graph.edg") != report.get("graph_sha256"):
        raise ValueError("production graph hash does not match saved edge list")
    for path in (vector_path, center_path):
        digest = expected.get(path.name)
        if digest != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("graph artifact checksum mismatch: %s" % path.name)
    return vectors, centers, {
        "vectors_file": str(vector_path),
        "centers_file": str(center_path),
        "report_file": str(report_path),
        "dimensions": next(iter(dims), 0),
        "graph_sha256": report["graph_sha256"],
        "input_manifest_sha256": report["input_manifest_sha256"],
        "script_sha256": report["script_sha256"],
    }


def generation_ancestors(horse, parents, max_depth=3):
    """Return unique IDs by generation, preserving pedigree collapse safely."""
    frontier = {horse}
    out = []
    for _ in range(max_depth):
        following = set()
        for child in frontier:
            for relation in ("sire", "dam"):
                parent = parents.get((child, relation))
                if parent:
                    following.add(parent)
        out.append(following)
        frontier = following
    return out


def resolve_embedding(horse, vectors, parents, centers=None):
    """Apply frozen direct, nearest-generation unique-ancestor, zero fallback."""
    if horse in vectors:
        vec, status, depth, count = vectors[horse], "direct", 0, 0
    else:
        vec = None
        status, depth, count = "missing", None, 0
        for depth_i, ancestors in enumerate(generation_ancestors(horse, parents), 1):
            available = sorted(a for a in ancestors if a in vectors)
            if available:
                count = len(available)
                dim = len(vectors[available[0]])
                vec = tuple(sum(vectors[a][j] for a in available) / count
                            for j in range(dim))
                status, depth = "fallback", depth_i
                break
        if vec is None:
            dim = len(next(iter(vectors.values()))) if vectors else 0
            vec = tuple(0.0 for _ in range(dim))
    missing = status == "missing"
    cluster = "MISSING"
    if not missing and centers:
        cluster = min(range(len(centers)),
                      key=lambda i: sum((vec[j] - centers[i][j]) ** 2
                                        for j in range(len(vec))))
    return vec, status, depth, count, int(missing), cluster


def nearest_clusters(vectors, centers):
    """Assign direct vectors in one NumPy distance pass when available."""
    if not centers:
        return {}
    try:
        import numpy as np
    except ImportError:
        return {horse: min(range(len(centers)),
                           key=lambda i: sum((vec[j] - centers[i][j]) ** 2
                                             for j in range(len(vec))))
                for horse, vec in vectors.items()}
    ids = list(vectors)
    matrix = np.asarray([vectors[x] for x in ids], dtype="float32")
    center_matrix = np.asarray(centers, dtype="float32")
    center_norm = (center_matrix * center_matrix).sum(axis=1)
    result = {}
    # Keep the distance tensor bounded: N x K x 64 is unnecessarily large.
    for start in range(0, len(ids), 8192):
        batch = matrix[start:start + 8192]
        distances = ((batch * batch).sum(axis=1)[:, None] + center_norm[None, :]
                     - 2.0 * batch.dot(center_matrix.T))
        result.update((horse, int(cluster)) for horse, cluster in
                      zip(ids[start:start + 8192], distances.argmin(axis=1).tolist()))
    return result


def _history_pivot(con):
    rows = {}
    names = {rank: tuple(f"{field}_{rank}" for field in HISTORY_FIELDS)
             for rank in (1, 2, 3)}
    for r in con.execute("""
            SELECT h.* FROM history_long h
            JOIN entries e USING (race_id,horse_id_netkeiba,horse_number)
            WHERE e.split IN ('train','validation') AND LOWER(e.surface) IN ('turf','dirt')
            ORDER BY h.race_id,h.horse_id_netkeiba,h.horse_number,h.hist_rank
            """):
        key = (r["race_id"], r["horse_id_netkeiba"], r["horse_number"])
        d = rows.setdefault(key, {})
        rank = int(r["hist_rank"])
        for name, field in zip(names[rank], HISTORY_FIELDS):
            d[name] = r[field]
    return rows


def _market(con):
    out = {}
    for r in con.execute("""
            SELECT n.race_id,n.horse_id_netkeiba,n.horse_number,n.odds,n.popularity
            FROM runners_r n JOIN races_r r USING (race_id)
            WHERE r.race_date <= '20231231'
            """):
        key = (r["race_id"], r["horse_id_netkeiba"], r["horse_number"])
        if key in out:
            raise ValueError("duplicate market key: %s" % (key,))
        out[key] = (r["odds"], r["popularity"])
    return out


def build_rows(features_db, source_db, pedigree_paths=(), graph_dir=None):
    """Return only train/validation rows and provenance metadata."""
    if not pedigree_paths:
        raise ValueError("at least one Netkeiba pedigree database is required")
    fcon = read_db(features_db)
    scon = read_db(source_db)
    fcon.row_factory = sqlite3.Row
    scon.row_factory = sqlite3.Row
    try:
        pivots = _history_pivot(fcon)
        market = _market(scon)
        policy = {r[0]: r[1] for r in fcon.execute("SELECT key,value FROM meta")}
        expected_history_exclusions = {"201009040210", "201407010204"}
        actual_exclusions = set(filter(None, policy.get("uncertain_history_races_excluded", "").split(",")))
        if not expected_history_exclusions.issubset(actual_exclusions):
            raise ValueError("features.db lacks the rebuilt uncertain-history exclusion policy")
        parents = load_parents(pedigree_paths)
        vectors, centers, graph_meta = load_graph(graph_dir) if graph_dir else ({}, None, {})
        direct_clusters = nearest_clusters(vectors, centers)
        cols = list(dict.fromkeys(KEYS + CONTEXT + HISTORY))
        for rank in (1, 2, 3):
            cols += ["%s_%d" % (x, rank) for x in HISTORY_FIELDS]
        cols += list(PEDIGREE_CATS) + ["odds", "popularity"] + list(GRAPH_META)
        if graph_dir:
            cols += ["embedding_%d" % j for j in range(64)]
        cols = list(dict.fromkeys(cols))
        # ponytail: keep this fixed cohort in memory for whole-race validation;
        # share predictor-name objects and stream batches if the cohort grows.
        embedding_names = tuple(f"embedding_{j}" for j in range(64))
        rows = []
        resolved = {}
        base_cols = list(dict.fromkeys(KEYS + CONTEXT + HISTORY))
        query = "SELECT %s FROM entries WHERE split IN ('train','validation') AND LOWER(surface) IN ('turf','dirt') ORDER BY split,race_id,horse_number,horse_id_netkeiba" % ",".join(base_cols)
        for r in fcon.execute(query):
            key = (r["race_id"], r["horse_id_netkeiba"], r["horse_number"])
            d = {c: r[c] for c in base_cols}
            d["surface"] = str(r["surface"]).lower()
            d.update({c: None for c in cols if c not in d})
            d.update(pivots.get(key, {}))
            if d["sire"] is None:
                d["sire"] = parents.get((r["horse_id_netkeiba"], "sire"))
            if d["dam"] is None:
                d["dam"] = parents.get((r["horse_id_netkeiba"], "dam"))
            if key not in market:
                raise ValueError("missing exact odds/popularity join key: %s" % (key,))
            odds, popularity = market[key]
            d["odds"], d["popularity"] = odds, popularity
            if graph_dir:
                horse = r["horse_id_netkeiba"]
                if horse not in resolved:
                    resolved[horse] = resolve_embedding(
                        horse, vectors, parents, None if horse in direct_clusters else centers)
                    if horse in direct_clusters:
                        old = resolved[horse]
                        resolved[horse] = old[:5] + (direct_clusters[horse],)
                vec, status, depth, count, missing, cluster = resolved[horse]
                d.update(zip(embedding_names, vec))
                d.update(embedding_status=status, fallback_depth=depth,
                         ancestor_count=count, embedding_missing=missing,
                         graph_cluster=cluster)
            rows.append(d)
        if any(d["split"] not in SPLITS for d in rows):
            raise AssertionError("test rows entered export")
        if len({(d["race_id"], d["horse_id_netkeiba"], d["horse_number"]) for d in rows}) != len(rows):
            raise ValueError("many-to-many or duplicate feature keys")
        counts = {}
        for d in rows:
            counts.setdefault(d["race_id"], []).append(d)
        for race_id, entries in counts.items():
            field_sizes = {int(e["field_size"]) for e in entries if e["field_size"] is not None}
            if len(field_sizes) != 1 or len(entries) != next(iter(field_sizes)):
                raise ValueError("incomplete or inconsistent field_size for race %s" % race_id)
        return cols, rows, {"graph": graph_meta, "rows": len(rows), "splits": SPLITS,
                            "forbidden_predictor_columns": sorted(TARGET_COLUMNS)}
    finally:
        fcon.close()
        scon.close()


def write_export(output, cols, rows, metadata, inputs=()):
    output = Path(output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing export: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    meta = dict(metadata, purpose="JRA train-validation prediction inputs",
                data_sha256=sha256_file(output), columns=cols,
                input_sha256={str(p): sha256_file(p)
                                                       for p in inputs if Path(p).is_file()})
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-db", required=True)
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--pedigree", action="append", default=[])
    parser.add_argument("--graph-dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    cols, rows, meta = build_rows(args.features_db, args.source_db, args.pedigree, args.graph_dir)
    write_export(args.output, cols, rows, meta,
                 [args.features_db, args.source_db] + args.pedigree)
    print(json.dumps(dict(meta, output=str(args.output)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
