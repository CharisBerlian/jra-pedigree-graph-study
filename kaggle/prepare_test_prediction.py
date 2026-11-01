"""Prepare seed-specific, inference-only test feature exports.

The source databases are opened read-only.  This module selects only test
entry columns (never target columns), rebuilds chronological prior-history
features, joins target-race odds, and resolves graph features from one frozen
graph seed at a time.  No model is fitted and no test metric is computed.
"""
import argparse
import csv
import gzip
import hashlib
import json
import sqlite3
from pathlib import Path

from graph_pilot import load_parents, read_db
from prepare_prediction import (
    CONTEXT,
    GRAPH_META,
    HISTORY,
    HISTORY_FIELDS,
    PEDIGREE_CATS,
    resolve_embedding,
)
from prepare_prediction_seeds import SEEDS, load_graph_seed


TEST_SPLIT = "test"
IDENTITY = ("race_id", "horse_id_netkeiba", "horse_number", "split", "surface")
EMBEDDING_COLUMNS = tuple(f"embedding_{index}" for index in range(64))
TARGET_COLUMNS = frozenset((
    "label", "finish_pos", "result_status", "time", "margin", "last_3f",
    "corner_passing", "top_three",
))
UNCERTAIN_HISTORY_RACES = frozenset(("201009040210", "201407010204"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _history_pivot(connection):
    """Pivot prior outcomes for test targets without selecting target outcomes."""
    fields = tuple(HISTORY_FIELDS)
    rows = {}
    query = """
        SELECT h.race_id, h.horse_id_netkeiba, h.horse_number, h.hist_rank,
               h.prior_race_id, h.prior_race_date, h.prior_finish_pos,
               h.prior_norm_finish, h.prior_field_size, h.prior_distance,
               h.prior_surface, h.prior_going, h.prior_race_class,
               h.prior_weight_carried, h.days_since
        FROM history_long h
        JOIN entries e USING (race_id, horse_id_netkeiba, horse_number)
        WHERE e.split = 'test' AND LOWER(e.surface) IN ('turf', 'dirt')
        ORDER BY h.race_id, h.horse_id_netkeiba, h.horse_number, h.hist_rank
    """
    for row in connection.execute(query):
        rank = int(row["hist_rank"])
        if rank not in (1, 2, 3):
            raise ValueError("history rank outside the frozen three-row window")
        key = (row["race_id"], row["horse_id_netkeiba"], row["horse_number"])
        target = rows.setdefault(key, {})
        if any(name in target for name in (f"{field}_{rank}" for field in fields)):
            raise ValueError(f"duplicate test history rank: {key}/{rank}")
        for field in fields:
            target[f"{field}_{rank}"] = row[field]
    return rows


def _market_for_keys(connection, keys):
    """Read only odds/popularity for test entry keys from the source DB."""
    if not keys:
        return {}
    races = sorted({key[0] for key in keys})
    placeholders = ",".join("?" for _ in races)
    query = f"""
        SELECT n.race_id, n.horse_id_netkeiba, n.horse_number, n.odds, n.popularity
        FROM runners_r n
        WHERE n.race_id IN ({placeholders})
    """
    market = {}
    for row in connection.execute(query, races):
        key = (row["race_id"], row["horse_id_netkeiba"], row["horse_number"])
        if key in market:
            raise ValueError(f"duplicate market key: {key}")
        market[key] = (row["odds"], row["popularity"])
    return market


def _columns(with_graph=True):
    columns = list(IDENTITY) + list(CONTEXT) + list(HISTORY)
    for rank in (1, 2, 3):
        columns.extend(f"{field}_{rank}" for field in HISTORY_FIELDS)
    columns.extend(PEDIGREE_CATS)
    columns.extend(("odds", "popularity"))
    if with_graph:
        columns.extend(GRAPH_META)
        columns.extend(EMBEDDING_COLUMNS)
    return list(dict.fromkeys(columns))


def build_test_rows(features_db, source_db, pedigree_paths, graph_dir, seed):
    """Build one seed's test rows and provenance without reading test labels."""
    if seed not in SEEDS:
        raise ValueError(f"unsupported graph seed: {seed}")
    if not pedigree_paths:
        raise ValueError("at least one pedigree database is required")
    vectors, centers, graph_meta = load_graph_seed(graph_dir, seed)
    features_connection = read_db(features_db)
    source_connection = read_db(source_db)
    features_connection.row_factory = sqlite3.Row
    source_connection.row_factory = sqlite3.Row
    try:
        policy = {row[0]: row[1] for row in features_connection.execute(
            "SELECT key, value FROM meta")}
        actual_exclusions = set(filter(None, policy.get(
            "uncertain_history_races_excluded", "").split(",")))
        if not UNCERTAIN_HISTORY_RACES.issubset(actual_exclusions):
            raise ValueError("features.db lacks the rebuilt history-exclusion policy")
        parents = load_parents(pedigree_paths)
        history = _history_pivot(features_connection)
        base_columns = list(CONTEXT) + list(HISTORY)
        query = "SELECT race_id, horse_id_netkeiba, horse_number, split, surface, %s " \
                "FROM entries WHERE split = 'test' AND LOWER(surface) IN ('turf','dirt') " \
                "ORDER BY race_id, horse_number, horse_id_netkeiba" % ",".join(base_columns)
        raw_rows = list(features_connection.execute(query))
        if not raw_rows:
            raise ValueError("features database has no test rows")
        keys = [(row["race_id"], row["horse_id_netkeiba"], row["horse_number"])
                for row in raw_rows]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate test feature keys")
        market = _market_for_keys(source_connection, keys)
        columns = _columns()
        rows = []
        resolved = {}
        for raw in raw_rows:
            key = (raw["race_id"], raw["horse_id_netkeiba"], raw["horse_number"])
            row = {column: raw[column] for column in IDENTITY + CONTEXT + HISTORY}
            row["surface"] = str(row["surface"]).lower()
            row.update({column: None for column in columns if column not in row})
            row.update(history.get(key, {}))
            horse = str(row["horse_id_netkeiba"])
            row["sire"] = parents.get((horse, "sire"))
            row["dam"] = parents.get((horse, "dam"))
            if key not in market:
                raise ValueError(f"missing exact test odds/popularity key: {key}")
            row["odds"], row["popularity"] = market[key]
            if horse not in resolved:
                resolved[horse] = resolve_embedding(horse, vectors, parents, centers)
            vector, status, depth, count, missing, cluster = resolved[horse]
            row.update(zip(EMBEDDING_COLUMNS, vector))
            row.update(embedding_status=status, fallback_depth=depth,
                       ancestor_count=count, embedding_missing=missing,
                       graph_cluster=cluster)
            rows.append(row)
        _validate_rows(rows, columns)
        return columns, rows, graph_meta
    finally:
        features_connection.close()
        source_connection.close()


def _validate_rows(rows, columns):
    if not rows or set(columns) & TARGET_COLUMNS:
        raise ValueError("test export contains target or historical outcome columns")
    if any(row.get("split") != TEST_SPLIT for row in rows):
        raise ValueError("non-test row entered test export")
    groups = {}
    for row in rows:
        key = (row["race_id"], row["horse_id_netkeiba"], row["horse_number"])
        groups.setdefault(row["race_id"], []).append((key, row))
    if len({key for entries in groups.values() for key, _ in entries}) != len(rows):
        raise ValueError("duplicate test export key")
    for race_id, entries in groups.items():
        sizes = {int(row["field_size"]) for _, row in entries}
        if len(sizes) != 1 or len(entries) != next(iter(sizes)) or \
                len({row["surface"] for _, row in entries}) != 1:
            raise ValueError(f"incomplete or inconsistent test race: {race_id}")


def write_test_export(output, columns, rows, metadata, inputs=()):
    """Write one gzip CSV and its adjacent ``.csv.json`` sidecar."""
    output = Path(output)
    sidecar = output.with_suffix(".json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite test export: {output}")
    _validate_rows(rows, columns)
    input_paths = [Path(path) for path in inputs]
    missing_inputs = [str(path) for path in input_paths if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"missing provenance inputs: {missing_inputs}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    sidecar_metadata = dict(metadata,
                            purpose="JRA test prediction inputs",
                            split=TEST_SPLIT, splits=[TEST_SPLIT], columns=columns,
                            rows=len(rows), data_sha256=sha256_file(output),
                            input_sha256={str(path): sha256_file(path) for path in input_paths},
                            target_columns_excluded=sorted(TARGET_COLUMNS),
                            test_labels_read=False, test_metrics_computed=False)
    sidecar.write_text(json.dumps(sidecar_metadata, indent=2, sort_keys=True,
                                  allow_nan=False) + "\n", encoding="utf-8")
    return sidecar_metadata


def _parse_graph_dirs(values):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--graph-dir must be SEED=PATH")
        raw_seed, raw_path = value.split("=", 1)
        seed = int(raw_seed)
        if seed in result or seed not in SEEDS or not raw_path:
            raise ValueError("--graph-dir must provide unique seeds 0, 1 and 2")
        result[seed] = Path(raw_path)
    if set(result) != set(SEEDS):
        raise ValueError("exactly graph seeds 0, 1 and 2 are required")
    return result


def prepare_exports(features_db, source_db, pedigree_paths, graph_dirs, output_dir):
    """Preflight all graph seeds, then write test_seed0/1/2 exports."""
    graph_dirs = {int(seed): Path(path) for seed, path in graph_dirs.items()}
    if set(graph_dirs) != set(SEEDS):
        raise ValueError("exactly graph seeds 0, 1 and 2 are required")
    output_dir = Path(output_dir)
    outputs = {seed: output_dir / f"test_seed{seed}.csv.gz" for seed in SEEDS}
    if any(path.exists() or path.with_suffix(".json").exists() for path in outputs.values()):
        raise FileExistsError("refusing to overwrite an existing test export")
    graph_meta = {}
    graph_reference = None
    for seed in SEEDS:
        _, _, current = load_graph_seed(graph_dirs[seed], seed)
        graph_meta[seed] = current
        comparable = {key: current[key] for key in (
            "graph_sha256", "input_manifest_sha256", "script_sha256")}
        if graph_reference is None:
            graph_reference = comparable
        elif comparable != graph_reference:
            raise ValueError("graph seeds do not share frozen input/graph provenance")
    results = {}
    for seed in SEEDS:
        columns, rows, current_graph = build_test_rows(
            features_db, source_db, pedigree_paths, graph_dirs[seed], seed)
        inputs = [Path(features_db), Path(source_db), *(Path(path) for path in pedigree_paths),
                  Path(__file__), Path(load_parents.__code__.co_filename),
                  graph_dirs[seed] / f"nodevectors_seed{seed}.npz",
                  graph_dirs[seed] / f"cluster_centers_seed{seed}.npy",
                  graph_dirs[seed] / "training_graph.edg",
                  graph_dirs[seed] / "production_report.json"]
        metadata = {"seed": seed, "graph": current_graph,
                    "exporter": str(Path(__file__).resolve()),
                    "exporter_sha256": sha256_file(__file__),
                    "graph_helper": str(Path(load_parents.__code__.co_filename).resolve()),
                    "graph_helper_sha256": sha256_file(load_parents.__code__.co_filename)}
        write_test_export(outputs[seed], columns, rows, metadata, inputs)
        results[seed] = str(outputs[seed])
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-db", required=True)
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--pedigree", action="append", required=True)
    parser.add_argument("--graph-dir", action="append", required=True, metavar="SEED=PATH")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result = prepare_exports(args.features_db, args.source_db, args.pedigree,
                             _parse_graph_dirs(args.graph_dir), args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
