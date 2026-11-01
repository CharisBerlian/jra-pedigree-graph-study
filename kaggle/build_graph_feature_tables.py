"""Build compact seed-specific graph feature tables for remote assembly.

The frozen Kaggle prediction bundle already contains every non-graph column.
This helper reads that bundle only to collect its horse IDs, then writes one
keyed graph row per horse for seeds 1 and 2.  It never reads labels into the
output and never fits a model.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

from graph_pilot import load_parents
from prepare_prediction_seeds import load_graph_seed
from prepare_prediction import resolve_embedding


SEEDS = (1, 2)
GRAPH_META = ("embedding_status", "fallback_depth", "ancestor_count",
              "embedding_missing", "graph_cluster")
EMBEDDINGS = tuple(f"embedding_{index}" for index in range(64))
GRAPH_COLUMNS = GRAPH_META + EMBEDDINGS
KEY = "horse_id_netkeiba"


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


def base_horse_ids(base_csv):
    """Read only the base key column and reject duplicate header/key errors."""
    ids = set()
    with gzip.open(base_csv, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or KEY not in reader.fieldnames:
            raise ValueError(f"base bundle lacks {KEY}: {base_csv}")
        if any(column in reader.fieldnames for column in GRAPH_COLUMNS) is False:
            raise ValueError("base bundle has no graph columns to replace")
        for row in reader:
            horse = str(row[KEY])
            if not horse:
                raise ValueError("base bundle contains an empty horse ID")
            ids.add(horse)
    if not ids:
        raise ValueError("base bundle contains no horse IDs")
    return ids


def graph_row(horse, vectors, centers, parents, seed):
    vector, status, depth, count, missing, cluster = resolve_embedding(
        horse, vectors, parents, centers)
    if len(vector) != 64:
        raise ValueError(f"seed {seed} produced a non-64-dimensional vector")
    return {
        KEY: horse,
        "embedding_status": status,
        "fallback_depth": "" if depth is None else depth,
        "ancestor_count": count,
        "embedding_missing": missing,
        "graph_cluster": cluster,
        **{name: value for name, value in zip(EMBEDDINGS, vector)},
    }


def write_table(path, rows, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(KEY,) + GRAPH_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    compressed = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    uncompressed = csv_stream_stats(path)
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps(
        dict(metadata, data_sha256=compressed["sha256"],
             compressed_csv=compressed, uncompressed_csv=uncompressed,
             columns=(KEY,) + GRAPH_COLUMNS),
        indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_tables(base_csv, pedigree_paths, graph_dirs, output_dir):
    base_csv = Path(base_csv).resolve()
    output_dir = Path(output_dir).resolve()
    horses = base_horse_ids(base_csv)
    parents = load_parents(pedigree_paths)
    outputs = {}
    for seed in SEEDS:
        output = output_dir / f"graph_features_seed{seed}.csv.gz"
        if output.exists() or output.with_suffix(".json").exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        vectors, centers, graph = load_graph_seed(graph_dirs[seed], seed)
        rows = [graph_row(horse, vectors, centers, parents, seed)
                for horse in sorted(horses)]
        direct = sum(row["embedding_status"] == "direct" for row in rows)
        fallback = sum(row["embedding_status"] == "fallback" for row in rows)
        missing = sum(row["embedding_status"] == "missing" for row in rows)
        metadata = {
            "purpose": "JRA graph-only keyed feature table; remote assembly input",
            "seed": seed,
            "rows": len(rows),
            "base_csv": str(base_csv),
            "base_csv_sha256": sha256_file(base_csv),
            "graph": graph,
            "coverage": {"direct": direct, "fallback": fallback, "missing": missing},
            "pedigree_sources": [str(Path(path).resolve()) for path in pedigree_paths],
        }
        write_table(output, rows, metadata)
        outputs[seed] = str(output)
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", required=True, type=Path)
    parser.add_argument("--pedigree", action="append", required=True)
    parser.add_argument("--graph-dir", action="append", required=True,
                        metavar="SEED=PATH")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    graph_dirs = {}
    for value in args.graph_dir:
        raw_seed, separator, raw_path = value.partition("=")
        seed = int(raw_seed) if raw_seed.isdigit() else -1
        if not separator or seed not in SEEDS or seed in graph_dirs or not raw_path:
            raise ValueError("--graph-dir must provide unique seeds 1 and 2")
        graph_dirs[seed] = Path(raw_path)
    if set(graph_dirs) != set(SEEDS):
        raise ValueError("--graph-dir must provide both seeds 1 and 2")
    print(json.dumps(build_tables(args.base_csv, args.pedigree, graph_dirs,
                                  args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
