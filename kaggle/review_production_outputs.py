"""Read-only audit of saved production graph outputs; never fits a model."""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BASE / "kaggle" / "upload" / "jra-pedigree-pilot-v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_production_module():
    source = BASE / "kaggle" / "graph_production.py"
    spec = importlib.util.spec_from_file_location("review_graph_production", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load graph_production.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_input(output, explicit):
    candidates = ([Path(explicit)] if explicit else []) + [
        DEFAULT_INPUT, output.parent.parent, output.parent, Path("/kaggle/input")]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "manifest.json").is_file():
            return candidate.resolve()
        if candidate.is_dir() and candidate.name == "input":
            matches = list(candidate.rglob("manifest.json"))
            if len(matches) == 1:
                return matches[0].parent.resolve()
    raise FileNotFoundError("Immutable input bundle not found; pass --input-dir")


def find_script(output, explicit, expected_hash):
    candidates = ([Path(explicit)] if explicit else []) + [
        output.parent.parent / "graph_production.py",
        output.parent / "graph_production.py",
        BASE / "kaggle" / "graph_production.py",
        Path("/kaggle/working/graph_production.py")]
    for candidate in candidates:
        if candidate.is_file() and sha256_file(candidate) == expected_hash:
            return candidate
    raise ValueError("Saved production script hash does not match any available graph_production.py")


def choose_k(diagnostics):
    valid = [(int(k), value.get("silhouette_sample"))
             for k, value in diagnostics.items()
             if value.get("silhouette_sample") is not None and
             math.isfinite(float(value["silhouette_sample"]))]
    if not valid:
        raise ValueError("No finite silhouette score available for k selection")
    return min(valid, key=lambda pair: (-float(pair[1]), pair[0]))[0]


def config_hash(config):
    return hashlib.sha256(json.dumps(
        config, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_run_metadata(report, status, run_config):
    if report.get("status") != "complete" or status.get("status") != "complete":
        raise ValueError("Production report and run status are not complete")
    hashes = (report.get("config_hash"), status.get("config_hash"),
              run_config.get("config_hash"))
    if not (hashes[0] == hashes[1] == hashes[2]):
        raise ValueError("Report, status and run_config hashes differ")
    saved_config = {key: run_config[key] for key in
                    ("input_manifest_sha256", "graph_sha256", "script_sha256",
                     "settings", "candidate_clusters")}
    if config_hash(saved_config) != hashes[0]:
        raise ValueError("run_config config_hash does not match its saved settings")
    for key in ("input_manifest_sha256", "graph_sha256", "script_sha256", "settings"):
        if report.get(key) != run_config.get(key):
            raise ValueError(f"Report and run_config differ for {key}")
    if status.get("selected_k") != report.get("selected_k"):
        raise ValueError("Completed selected_k differs between report and run status")
    return saved_config


def verify(output, input_dir=None, script=None):
    """Verify one downloaded seed directory and return compact JSON-safe results."""
    # Import NumPy only for this read-only numerical audit.
    import numpy as np

    output = Path(output).resolve()
    report = json.loads((output / "production_report.json").read_text(encoding="utf-8"))
    status = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
    run_config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    saved_config = validate_run_metadata(report, status, run_config)

    input_dir = find_input(output, input_dir)
    # graph_production.load_verified_inputs checks payload hashes before loading
    # the immutable graph_pilot.py module.
    production = load_production_module()
    roots, nodes, edges, manifest = production.load_verified_inputs(input_dir)
    if report.get("input_scope") != manifest.get("scope"):
        raise ValueError("Report input scope differs from immutable manifest")
    if (report.get("training_horses"), report.get("graph_nodes"),
            report.get("graph_edges")) != (len(roots), len(nodes), len(edges)):
        raise ValueError("Report graph counts differ from immutable graph")
    if sha256_file(input_dir / "manifest.json") != report["input_manifest_sha256"]:
        raise ValueError("Input manifest hash mismatch")
    if (report.get("target_outcomes_read") is not False or
            report.get("outcome_columns_input") is not False or
            report.get("validation_test_roots_input") is not False):
        raise ValueError("Report does not prove outcome-free input use")

    graph_path = output / "training_graph.edg"
    graph_bytes = graph_path.read_bytes()
    if sha256_file(graph_path) != report["graph_sha256"]:
        raise ValueError("Graph hash mismatch")
    graph_edges = [tuple(line.split("\t")) for line in graph_bytes.decode("utf-8").splitlines()]
    if graph_edges != edges:
        raise ValueError("Saved graph differs from immutable reconstructed graph")

    script_path = find_script(output, script, report["script_sha256"])
    seed = int(report["settings"]["seed"])
    artifact_names = report.get("artifact_sha256", {})
    expected_artifacts = {
        f"nodevectors_seed{seed}.npz", f"cluster_centers_seed{seed}.npy",
        f"cluster_labels_seed{seed}.csv", f"cluster_diagnostics_seed{seed}.json",
        "training_graph.edg"}
    if set(artifact_names) != expected_artifacts:
        raise ValueError("Completed report artifact hashes do not match expected files")
    for name, expected in artifact_names.items():
        if sha256_file(output / name) != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")

    vector_path = output / f"nodevectors_seed{seed}.npz"
    with np.load(vector_path, allow_pickle=False) as archive:
        if set(archive.files) != {"node_ids", "vectors"}:
            raise ValueError("Unexpected Node2Vec archive fields")
        ids, vectors = archive["node_ids"].tolist(), archive["vectors"]
    if len(ids) != len(set(ids)) or set(ids) != set(nodes):
        raise ValueError("Node IDs are not unique or do not cover the graph")
    dimensions = int(report["settings"]["dimensions"])
    if vectors.shape != (len(nodes), dimensions) or vectors.dtype != np.float32:
        raise ValueError("Unexpected node-vector shape or dtype")
    if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("Node vectors contain nonfinite or zero rows")

    selected = int(report["selected_k"])
    centers = np.load(output / f"cluster_centers_seed{seed}.npy", allow_pickle=False)
    if centers.shape != (selected, dimensions) or not np.isfinite(centers).all():
        raise ValueError("Unexpected cluster-center shape or values")
    labels_path = output / f"cluster_labels_seed{seed}.csv"
    with labels_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["horse_id_netkeiba", "cluster"]:
            raise ValueError("Unexpected cluster-label columns")
        label_rows = list(reader)
    if [row["horse_id_netkeiba"] for row in label_rows] != roots:
        raise ValueError("Cluster labels do not preserve the immutable training-root order")
    labels = np.asarray([int(row["cluster"]) for row in label_rows], dtype=np.int64)
    if len(labels) != len(roots) or np.any(labels < 0) or np.any(labels >= selected):
        raise ValueError("Invalid training cluster labels")
    sizes = {str(k): int(v) for k, v in sorted(Counter(labels).items())}
    diagnostics_path = output / f"cluster_diagnostics_seed{seed}.json"
    diagnostics_file = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    candidates = diagnostics_file["candidates"]
    if diagnostics_file.get("selected_k") != selected:
        raise ValueError("Diagnostics selected_k mismatch")
    if seed == 0 and set(map(int, candidates)) != {4, 8, 16, 32, 64}:
        raise ValueError("Seed 0 is missing a frozen candidate k diagnostic")
    if seed == 0 and choose_k(candidates) != selected:
        raise ValueError("Seed 0 selected_k does not follow finite-silhouette rule")
    if seed != 0 and report["settings"].get("selected_clusters") != selected:
        raise ValueError("Repeat seed does not use the frozen selected k")
    if sizes != candidates[str(selected)]["cluster_sizes"]:
        raise ValueError("Saved label sizes differ from selected-k diagnostics")
    if report.get("cluster_diagnostics", {}).get(str(selected), {}).get("cluster_sizes") != sizes:
        raise ValueError("Report cluster sizes differ from labels")

    index = {node: i for i, node in enumerate(ids)}
    training = vectors[[index[node] for node in roots]].astype(np.float64)
    centers = centers.astype(np.float64)
    distances = ((training * training).sum(axis=1)[:, None] +
                 (centers * centers).sum(axis=1)[None, :] -
                 2 * training @ centers.T)
    mismatches = int(np.count_nonzero(np.argmin(distances, axis=1) != labels))
    if mismatches:
        raise ValueError(f"{mismatches} labels disagree with nearest saved centroid")
    result = {
        "status": "PASS",
        "seed": seed,
        "selected_k": selected,
        "counts": {"training_horses": len(roots), "graph_nodes": len(nodes), "graph_edges": len(edges)},
        "embedding_shape": list(vectors.shape),
        "embedding_dtype": str(vectors.dtype),
        "cluster_sizes": sizes,
        "nearest_centroid_label_mismatches": mismatches,
        "input_manifest_sha256": report["input_manifest_sha256"],
        "graph_sha256": report["graph_sha256"],
        "script_sha256": report["script_sha256"],
        "script_path": str(script_path),
        "scope": "Read-only production artifact integrity; no fitting or outcome evaluation.",
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Downloaded jra-production/seed-N directory")
    parser.add_argument("--input-dir", type=Path,
                        help="Immutable pilot bundle; defaults to local bundle or Kaggle input")
    parser.add_argument("--script", type=Path,
                        help="Saved graph_production.py; inferred from output root and hash")
    parser.add_argument("--output", type=Path,
                        help="Optional path for the compact JSON audit report")
    args = parser.parse_args()
    result = verify(args.output_dir, args.input_dir, args.script)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
