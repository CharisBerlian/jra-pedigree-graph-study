"""Stream a frozen base prediction bundle plus one keyed graph table."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


KEY = "horse_id_netkeiba"
ROW_KEYS = ("race_id", "horse_id_netkeiba", "horse_number", "split")
GRAPH_COLUMNS = ("embedding_status", "fallback_depth", "ancestor_count",
                 "embedding_missing", "graph_cluster",
                 *(f"embedding_{index}" for index in range(64)))
FROZEN_PROVENANCE = {
    "graph_sha256": "5cdd9acf5bd3ad3dc749300cc2fb985163233fc4d30faae0910b2b9193976acc",
    "input_manifest_sha256": "a2b316aee5fa255df908450ac5f90960c1a05b329a1a93609d55579cb3d26447",
    "script_sha256": "3a2c4e77bd70af6059a2854b601ac84a47de8ed752972bde656b97bbd162d59f",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_stream_stats(path):
    digest = hashlib.sha256()
    size = 0
    with gzip.open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def _update_row_digest(digest, row, columns):
    encoded = json.dumps([row[column] for column in columns],
                         ensure_ascii=False, separators=(",", ":"))
    digest.update((encoded + "\n").encode("utf-8"))


def load_graph_table(path):
    values = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = [KEY, *GRAPH_COLUMNS]
        if reader.fieldnames != expected:
            raise ValueError("graph table columns are not the frozen keyed schema")
        for row in reader:
            horse = row[KEY]
            if not horse or horse in values:
                raise ValueError(f"duplicate/empty graph key: {horse!r}")
            values[horse] = {column: row[column] for column in GRAPH_COLUMNS}
    if not values:
        raise ValueError("graph table is empty")
    return values


def verify_graph_metadata(path, expected_seed=None, base_data_sha256=None):
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    graph = metadata.get("graph", {})
    if expected_seed is not None and metadata.get("seed") != expected_seed:
        raise ValueError("graph table metadata seed does not match assembly seed")
    if graph.get("seed") != metadata.get("seed") or graph.get("selected_k") != 64 or graph.get("dimensions") != 64:
        raise ValueError("graph table metadata lacks the expected 64-dimensional seed")
    if any(graph.get(key) != value for key, value in FROZEN_PROVENANCE.items()):
        raise ValueError("graph table metadata provenance differs from frozen graph run")
    if base_data_sha256 is not None and metadata.get("base_csv_sha256") != base_data_sha256:
        raise ValueError("graph table was built from a different frozen base bundle")
    return metadata


def assemble(base_csv, graph_csv, output_csv, graph_metadata=None,
             base_metadata=None, expected_seed=None, source_base_sha256=None,
             source_base_metadata=None):
    graph = load_graph_table(graph_csv)
    seen = set()
    base_keys = set()
    rows = 0
    base_sidecar = (Path(base_metadata) if base_metadata else
                    Path(base_csv).with_suffix(".json"))
    base_meta = json.loads(base_sidecar.read_text(encoding="utf-8"))
    if base_meta.get("data_sha256") != sha256_file(base_csv):
        raise ValueError("base bundle checksum does not match its sidecar")
    graph_meta = (verify_graph_metadata(graph_metadata, expected_seed=expected_seed,
                                        base_data_sha256=(source_base_sha256 or
                                                          base_meta["data_sha256"]))
                  if graph_metadata is not None else None)
    non_graph_digest = hashlib.sha256()
    assembled_digest = hashlib.sha256()
    output_csv = Path(output_csv)
    with gzip.open(base_csv, "rt", encoding="utf-8", newline="") as source, \
            gzip.open(output_csv, "wt", encoding="utf-8", newline="") as target:
        reader = csv.DictReader(source)
        if not reader.fieldnames or KEY not in reader.fieldnames:
            raise ValueError("base bundle lacks horse key")
        if not set(GRAPH_COLUMNS).issubset(reader.fieldnames):
            raise ValueError("base bundle lacks frozen graph columns")
        non_graph_columns = [column for column in reader.fieldnames
                             if column not in GRAPH_COLUMNS]
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            horse = row[KEY]
            if horse not in graph:
                raise ValueError(f"base horse has no graph row: {horse}")
            row_key = tuple(row[column] for column in ROW_KEYS)
            if row_key in base_keys:
                raise ValueError(f"duplicate base row key: {row_key}")
            base_keys.add(row_key)
            _update_row_digest(non_graph_digest, row, non_graph_columns)
            seen.add(horse)
            row.update(graph[horse])
            _update_row_digest(assembled_digest, row, non_graph_columns)
            writer.writerow(row)
            rows += 1
    missing = set(graph) - seen
    if missing:
        raise ValueError(f"graph table has {len(missing)} unused horse IDs")
    if non_graph_digest.digest() != assembled_digest.digest():
        raise AssertionError("non-graph row content changed during assembly")
    if base_meta.get("splits") != ["train", "validation"]:
        raise ValueError("base bundle sidecar must declare train/validation splits")
    base_csv_stats = {"compressed": {"bytes": Path(base_csv).stat().st_size,
                                     "sha256": sha256_file(base_csv)},
                      "uncompressed": csv_stream_stats(base_csv)}
    output_csv_stats = {"compressed": {"bytes": output_csv.stat().st_size,
                                       "sha256": sha256_file(output_csv)},
                        "uncompressed": csv_stream_stats(output_csv)}
    result = {
        "purpose": "JRA train-validation prediction inputs; graph-only remote assembly",
        "seed": expected_seed,
        "splits": ["train", "validation"],
        "rows": rows,
        "base_csv_sha256": base_csv_stats["compressed"]["sha256"],
        "base_csv_stats": base_csv_stats,
        "graph_table_sha256": sha256_file(graph_csv),
        "data_sha256": output_csv_stats["compressed"]["sha256"],
        "csv_stats": output_csv_stats,
        "columns": reader.fieldnames,
        "non_graph_columns": non_graph_columns,
        "base_non_graph_sha256": non_graph_digest.hexdigest(),
        "assembled_non_graph_sha256": assembled_digest.hexdigest(),
        "base_metadata": base_meta,
    }
    if source_base_sha256 is not None:
        result["source_base_sha256"] = source_base_sha256
    if source_base_metadata is not None:
        result["source_base_metadata"] = source_base_metadata
    if graph_metadata is not None:
        result["graph_metadata"] = graph_meta
    output_csv.with_suffix(".json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", required=True, type=Path)
    parser.add_argument("--graph-csv", required=True, type=Path)
    parser.add_argument("--graph-metadata", type=Path)
    parser.add_argument("--base-metadata", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--source-base-sha256")
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.graph_metadata is not None:
        verify_graph_metadata(args.graph_metadata, expected_seed=args.seed)
    print(json.dumps(assemble(args.base_csv, args.graph_csv, args.output_csv,
                              args.graph_metadata, args.base_metadata, args.seed,
                              args.source_base_sha256),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
