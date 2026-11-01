"""Audit compact graph tables against the frozen prediction base."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


KEY = "horse_id_netkeiba"
ROW_KEYS = ("race_id", "horse_id_netkeiba", "horse_number", "split")
GRAPH_META = ("embedding_status", "fallback_depth", "ancestor_count",
              "embedding_missing", "graph_cluster")
EMBEDDINGS = tuple(f"embedding_{i}" for i in range(64))
GRAPH_COLUMNS = GRAPH_META + EMBEDDINGS
NON_GRAPH_EXCLUDE = set(GRAPH_COLUMNS)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_stats(path):
    digest = hashlib.sha256()
    size = 0
    with gzip.open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def update_digest(digest, row, columns):
    digest.update((json.dumps([row[column] for column in columns],
                              ensure_ascii=False, separators=(",", ":")) + "\n").encode())


def audit_base(path):
    ids = set()
    keys = set()
    rows = 0
    races = set()
    labels = {"train": 0, "validation": 0}
    split_surface = {}
    non_graph = None
    non_graph_columns = None
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError("base CSV has no header")
        if not set(GRAPH_COLUMNS).issubset(reader.fieldnames):
            raise ValueError("base CSV lacks frozen graph columns")
        non_graph_columns = [c for c in reader.fieldnames if c not in NON_GRAPH_EXCLUDE]
        non_graph = hashlib.sha256()
        for row in reader:
            row_key = tuple(row[column] for column in ROW_KEYS)
            if row_key in keys:
                raise ValueError(f"duplicate base key: {row_key}")
            keys.add(row_key)
            horse = row[KEY]
            if not horse:
                raise ValueError("empty base horse ID")
            ids.add(horse)
            races.add(row["race_id"])
            labels[row["split"]] = labels.get(row["split"], 0) + int(row["label"])
            cell = f"{row['split']}/{row['surface']}"
            split_surface[cell] = split_surface.get(cell, 0) + 1
            update_digest(non_graph, row, non_graph_columns)
            rows += 1
    return {
        "rows": rows, "races": len(races), "unique_horses": len(ids),
        "unique_keys": len(keys), "labels": labels,
        "split_surface_entries": dict(sorted(split_surface.items())),
        "columns": reader.fieldnames, "non_graph_columns": non_graph_columns,
        "non_graph_sha256": non_graph.hexdigest(), "horse_ids": ids,
        "csv_stats": csv_stats(path), "compressed_sha256": sha256_file(path),
    }


def audit_table(path, base):
    sidecar = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
    compressed = {"bytes": Path(path).stat().st_size, "sha256": sha256_file(path)}
    uncompressed = csv_stats(path)
    if sidecar.get("data_sha256") != compressed["sha256"]:
        raise ValueError(f"sidecar compressed hash mismatch: {path}")
    if sidecar.get("compressed_csv") != compressed or sidecar.get("uncompressed_csv") != uncompressed:
        raise ValueError(f"sidecar CSV stats mismatch: {path}")
    if sidecar.get("base_csv_sha256") != base["compressed_sha256"]:
        raise ValueError(f"base provenance mismatch: {path}")
    ids = set()
    statuses = {"direct": 0, "fallback": 0, "missing": 0}
    finite_bad = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = [KEY, *GRAPH_COLUMNS]
        if reader.fieldnames != expected:
            raise ValueError(f"graph table schema mismatch: {path}")
        for row in reader:
            horse = row[KEY]
            if not horse or horse in ids:
                raise ValueError(f"duplicate/empty graph key: {horse!r}")
            ids.add(horse)
            status = row["embedding_status"]
            if status not in statuses:
                raise ValueError(f"invalid graph status: {status}")
            statuses[status] += 1
            for column in EMBEDDINGS:
                if not math.isfinite(float(row[column])):
                    finite_bad += 1
    if ids != base["horse_ids"]:
        raise ValueError(f"base/table horse-ID cohort mismatch: {path}")
    if finite_bad:
        raise ValueError(f"nonfinite graph values: {path}")
    if sidecar.get("rows") != len(ids) or sidecar.get("coverage") != statuses:
        raise ValueError(f"table sidecar counts mismatch: {path}")
    graph = sidecar.get("graph", {})
    if (sidecar.get("seed") not in (1, 2) or graph.get("seed") != sidecar.get("seed") or
            graph.get("selected_k") != 64 or graph.get("dimensions") != 64):
        raise ValueError(f"seed/dimension metadata mismatch: {path}")
    return {"path": str(Path(path).resolve()), "seed": sidecar["seed"],
            "rows": len(ids), "coverage": statuses, "finite_embedding_values_bad": finite_bad,
            "compressed_csv": compressed, "uncompressed_csv": uncompressed,
            "sidecar_sha256": sha256_file(Path(path).with_suffix(".json")),
            "graph": graph}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", required=True, type=Path)
    parser.add_argument("--table", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    base = audit_base(args.base_csv)
    tables = [audit_table(path, base) for path in args.table]
    if {item["seed"] for item in tables} != {1, 2}:
        raise ValueError("audit requires seed-1 and seed-2 tables")
    report = {"status": "PASS", "purpose": "No-fit compact graph-table and frozen-base audit",
              "base": {key: value for key, value in base.items() if key != "horse_ids"},
              "tables": tables,
              "provenance": {"graph_sha256": "5cdd9acf5bd3ad3dc749300cc2fb985163233fc4d30faae0910b2b9193976acc",
                             "input_manifest_sha256": "a2b316aee5fa255df908450ac5f90960c1a05b329a1a93609d55579cb3d26447",
                             "script_sha256": "3a2c4e77bd70af6059a2854b601ac84a47de8ed752972bde656b97bbd162d59f"}}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
