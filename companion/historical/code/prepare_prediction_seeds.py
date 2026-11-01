"""Prepare seed-specific train/validation prediction bundles.

The source databases are opened read-only.  Base rows are built once, then
reused in place for seeds 0, 1 and 2 so the three exports do not retain three
copies of the feature table in memory.  Test rows are never selected.
"""
import argparse
import json
from pathlib import Path

from graph_pilot import load_parents
from prepare_prediction import (
    GRAPH_META,
    build_rows,
    resolve_embedding,
    sha256_file,
    write_export,
)


SEEDS = (0, 1, 2)
DIMENSIONS = 64
EMBEDDING_COLUMNS = tuple(f"embedding_{i}" for i in range(DIMENSIONS))
FROZEN_GRAPH_SHA256 = "5cdd9acf5bd3ad3dc749300cc2fb985163233fc4d30faae0910b2b9193976acc"
FROZEN_INPUT_MANIFEST_SHA256 = "a2b316aee5fa255df908450ac5f90960c1a05b329a1a93609d55579cb3d26447"
FROZEN_SCRIPT_SHA256 = "3a2c4e77bd70af6059a2854b601ac84a47de8ed752972bde656b97bbd162d59f"


def _report(path):
    report_path = Path(path) / "production_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"missing production_report.json: {report_path}")
    return report_path, json.loads(report_path.read_text(encoding="utf-8"))


def load_graph_seed(folder, seed):
    """Load and checksum one completed production graph seed.

    Centroids are loaded from this seed's own artifact; no seed-0 fallback is
    allowed.  The graph report is retained in the returned provenance.
    """
    folder = Path(folder).resolve()
    report_path, report = _report(folder)
    settings = report.get("settings", {})
    if report.get("status") != "complete":
        raise ValueError(f"graph seed {seed} is not complete")
    if settings.get("seed") != seed:
        raise ValueError(f"graph report seed mismatch: expected {seed}")
    if settings.get("dimensions") != DIMENSIONS or report.get("selected_k") != DIMENSIONS:
        raise ValueError("production graph must declare 64 dimensions and selected_k=64")
    if report.get("target_outcomes_read") is not False or report.get("validation_test_roots_input") is not False:
        raise ValueError("production graph report does not prove outcome-free input use")
    for key in ("graph_sha256", "input_manifest_sha256", "script_sha256"):
        value = report.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"production report lacks valid {key}")
    expected_provenance = {
        "graph_sha256": FROZEN_GRAPH_SHA256,
        "input_manifest_sha256": FROZEN_INPUT_MANIFEST_SHA256,
        "script_sha256": FROZEN_SCRIPT_SHA256,
    }
    actual_provenance = {key: report.get(key) for key in expected_provenance}
    if actual_provenance != expected_provenance:
        raise ValueError("production graph provenance differs from the frozen run")

    vector_path = folder / f"nodevectors_seed{seed}.npz"
    center_path = folder / f"cluster_centers_seed{seed}.npy"
    graph_path = folder / "training_graph.edg"
    artifact_hashes = report.get("artifact_sha256", {})
    for path in (vector_path, center_path, graph_path):
        expected = artifact_hashes.get(path.name)
        if not path.is_file() or expected != sha256_file(path):
            raise ValueError(f"graph artifact checksum missing or mismatched: {path.name}")
    if report["graph_sha256"] != artifact_hashes[graph_path.name]:
        raise ValueError("production graph hash does not match saved edge list")

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required to read production graph artifacts") from exc
    with np.load(vector_path, allow_pickle=False) as archive:
        if set(archive.files) != {"node_ids", "vectors"}:
            raise ValueError("embeddings NPZ needs exactly node_ids and vectors")
        ids = [str(value) for value in archive["node_ids"].tolist()]
        matrix = archive["vectors"]
    if len(ids) == 0 or len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("embeddings must have unique nonempty node IDs")
    if matrix.shape != (len(ids), DIMENSIONS) or matrix.dtype != np.float32:
        raise ValueError("unexpected embedding shape or dtype")
    if not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix, axis=1) == 0):
        raise ValueError("embeddings contain nonfinite or zero rows")
    vectors = {node: tuple(float(value) for value in vector)
               for node, vector in zip(ids, matrix.tolist())}

    centers = np.load(center_path, allow_pickle=False)
    if centers.shape != (DIMENSIONS, DIMENSIONS) or centers.dtype.kind not in "fc":
        raise ValueError("cluster centers must have shape (64, 64)")
    if not np.isfinite(centers).all():
        raise ValueError("cluster centers contain nonfinite values")
    centers = [tuple(float(value) for value in center) for center in centers.tolist()]
    graph_meta = {
        "seed": seed,
        "directory": str(folder),
        "report_file": str(report_path),
        "report_sha256": sha256_file(report_path),
        "dimensions": DIMENSIONS,
        "selected_k": DIMENSIONS,
        "graph_sha256": report["graph_sha256"],
        "input_manifest_sha256": report["input_manifest_sha256"],
        "script_sha256": report["script_sha256"],
        "artifact_sha256": dict(sorted(artifact_hashes.items())),
    }
    return vectors, centers, graph_meta


def coverage_by_split_surface(rows):
    """Return entry and unique-horse status counts for every cohort cell."""
    result = {}
    for split in ("train", "validation"):
        result[split] = {}
        for surface in ("turf", "dirt"):
            selected = [row for row in rows
                        if row["split"] == split and row["surface"] == surface]
            statuses = ("direct", "fallback", "missing")
            horses = {status: {row["horse_id_netkeiba"] for row in selected
                               if row["embedding_status"] == status}
                      for status in statuses}
            result[split][surface] = {
                "entries": {"total": len(selected), **{
                    status: sum(row["embedding_status"] == status for row in selected)
                    for status in statuses}},
                "unique_horses": {"total": len({row["horse_id_netkeiba"] for row in selected}),
                                  **{status: len(horses[status]) for status in statuses}},
            }
    return result


def add_seed_features_with_parents(rows, vectors, centers, parents):
    """Inject one seed's graph features using the frozen ancestor rule."""
    resolved = {}
    for row in rows:
        horse = str(row["horse_id_netkeiba"])
        if horse not in resolved:
            resolved[horse] = resolve_embedding(horse, vectors, parents, centers)
        vector, status, depth, count, missing, cluster = resolved[horse]
        row.update(zip(EMBEDDING_COLUMNS, vector))
        row.update(embedding_status=status, fallback_depth=depth,
                   ancestor_count=count, embedding_missing=missing,
                   graph_cluster=cluster)
    return coverage_by_split_surface(rows)


def prepare_exports(features_db, source_db, pedigree_paths, graph_dirs, output_dir):
    """Build all three seed exports and return their paths and metadata."""
    graph_dirs = {int(seed): Path(path) for seed, path in graph_dirs.items()}
    if set(graph_dirs) != set(SEEDS):
        raise ValueError("exactly graph seeds 0, 1 and 2 are required")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        output = output_dir / f"train_validation_seed{seed}.csv.gz"
        if output.exists() or output.with_suffix(".json").exists():
            raise FileExistsError(f"Refusing to overwrite an existing export: {output}")

    # Validate every repeat before the large base table is loaded or any output
    # is written.  Artifacts are read again below one seed at a time to bound
    # memory rather than retaining three vector dictionaries.
    graph_reference = None
    for seed in SEEDS:
        _, _, graph_meta = load_graph_seed(graph_dirs[seed], seed)
        current = {key: graph_meta[key] for key in
                   ("graph_sha256", "input_manifest_sha256", "script_sha256")}
        if graph_reference is None:
            graph_reference = current
        elif current != graph_reference:
            raise ValueError("graph seeds do not share the immutable graph/input provenance")

    # Existing audited join and race completeness checks are reused once.
    cols, rows, base_meta = build_rows(features_db, source_db, pedigree_paths)
    if not set(GRAPH_META).issubset(cols):
        raise ValueError("audited base exporter did not provide all graph metadata columns")
    parents = load_parents(pedigree_paths)
    cols = list(dict.fromkeys(cols + list(EMBEDDING_COLUMNS)))
    outputs = {}
    graph_reference = None
    for seed in SEEDS:
        vectors, centers, graph_meta = load_graph_seed(graph_dirs[seed], seed)
        if graph_reference is None:
            graph_reference = {key: graph_meta[key] for key in
                               ("graph_sha256", "input_manifest_sha256", "script_sha256")}
        elif graph_reference != {key: graph_meta[key] for key in graph_reference}:
            raise ValueError("graph seeds do not share the immutable graph/input provenance")
        coverage = add_seed_features_with_parents(rows, vectors, centers, parents)
        output = output_dir / f"train_validation_seed{seed}.csv.gz"
        metadata = dict(base_meta, seed=seed, graph=graph_meta, coverage=coverage,
                        exporter=str(Path(__file__).resolve()),
                        exporter_sha256=sha256_file(__file__),
                        helper=str(Path(build_rows.__code__.co_filename).resolve()),
                        helper_sha256=sha256_file(build_rows.__code__.co_filename),
                        graph_helper=str(Path(load_parents.__code__.co_filename).resolve()),
                        graph_helper_sha256=sha256_file(load_parents.__code__.co_filename))
        inputs = [features_db, source_db, *pedigree_paths,
                  Path(build_rows.__code__.co_filename),
                  Path(load_parents.__code__.co_filename),
                  graph_dirs[seed] / f"nodevectors_seed{seed}.npz",
                  graph_dirs[seed] / f"cluster_centers_seed{seed}.npy",
                  graph_dirs[seed] / "training_graph.edg",
                  graph_dirs[seed] / "production_report.json"]
        write_export(output, cols, rows, metadata, inputs)
        outputs[seed] = str(output)
    return outputs


def _parse_graph_dirs(values):
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--graph-dir must be SEED=PATH")
        raw_seed, raw_path = value.split("=", 1)
        seed = int(raw_seed)
        if seed in parsed or seed not in SEEDS or not raw_path:
            raise ValueError("--graph-dir must provide unique seeds 0, 1 and 2")
        parsed[seed] = raw_path
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-db", required=True)
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--pedigree", action="append", default=[])
    parser.add_argument("--graph-dir", action="append", required=True,
                        metavar="SEED=PATH")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    outputs = prepare_exports(args.features_db, args.source_db, args.pedigree,
                              _parse_graph_dirs(args.graph_dir), args.output_dir)
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
