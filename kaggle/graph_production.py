"""Kaggle-only production Node2Vec and training-starter cluster run.

The attached dataset is deliberately treated as an immutable, outcome-free
input bundle.  Heavy imports and fitting happen only after the Kaggle guard.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time


PRODUCTION_SETTINGS = {
    "dimensions": 64,
    "num_walks": 10,
    "walk_length": 80,
    "window_size": 5,
    "epochs": 5,
    "p": 1.0,
    "q": 0.5,
    "workers": 4,
    "candidate_clusters": [4, 8, 16, 32, 64],
    "kmeans_n_init": 10,
    "kmeans_max_iter": 300,
    "silhouette_sample": 2000,
}
REQUIRED_INPUTS = ("manifest.json", "training_horses.txt", "pedigree_edges.csv",
                   "graph_pilot.py", "requirements-graph.txt")
PAYLOAD_INPUTS = tuple(name for name in REQUIRED_INPUTS if name != "manifest.json")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path, data, mode="w"):
    """Write a small text/bytes artifact then publish it with one rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if "b" in mode:
        temporary.write_bytes(data)
    else:
        temporary.write_text(data, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_helper(input_dir):
    source = Path(input_dir) / "graph_pilot.py"
    spec = importlib.util.spec_from_file_location("immutable_graph_pilot", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load immutable graph_pilot.py helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_verified_inputs(input_dir):
    """Use the pilot's verified loader without reading any outcome database."""
    input_dir = Path(input_dir).resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("purpose") != "training-only pedigree resource pilot":
        raise ValueError("Unexpected input manifest purpose")
    if manifest.get("outcome_columns_exported") is not False:
        raise ValueError("Refusing an input bundle that exports outcomes")
    if manifest.get("validation_test_roots_exported") is not False:
        raise ValueError("Refusing an input bundle that exports validation/test roots")
    if not (input_dir / "manifest.json").is_file():
        raise ValueError("Missing required input: manifest.json")
    for name in PAYLOAD_INPUTS:
        if not (input_dir / name).is_file():
            raise ValueError(f"Missing required input: {name}")
        expected = manifest.get("files", {}).get(name, {}).get("sha256")
        if expected != sha256_file(input_dir / name):
            raise ValueError(f"Input checksum mismatch: {name}")
    helper = _load_helper(input_dir)
    roots, nodes, edges, manifest = helper.load_pilot_inputs(input_dir)
    return roots, nodes, edges, manifest


def write_graph(path, edges):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, delimiter="\t", lineterminator="\n").writerows(edges)
    os.replace(temporary, path)


def _existing_config(output_dir):
    status_path = Path(output_dir) / "run_status.json"
    report_path = Path(output_dir) / "production_report.json"
    for path in (status_path, report_path):
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if value.get("status") == "complete":
                return value
    return None


def begin_run(output_dir, config):
    """Refuse completed mismatches and record stale/in-progress predecessors."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wanted = config_hash(config)
    complete = _existing_config(output_dir)
    if complete:
        if complete.get("config_hash") != wanted:
            raise RuntimeError("Refusing to overwrite complete run with different config")
        raise RuntimeError("Run is already complete for this configuration")
    prior = None
    status_path = output_dir / "run_status.json"
    if status_path.is_file():
        try:
            prior = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prior = {"status": "unreadable"}
    record = {"status": "running", "started_utc": utc_now(),
              "config_hash": wanted, "pid": os.getpid()}
    if prior and prior.get("status") in ("running", "started"):
        record["previous_status"] = "stale"
        record["previous_started_utc"] = prior.get("started_utc")
        atomic_json(output_dir / "stale_run.json", prior)
    atomic_json(status_path, record)
    atomic_json(output_dir / "run_config.json", {"config_hash": wanted, **config})
    return wanted


def finish_run(output_dir, status, config_hash_value, **extra):
    atomic_json(Path(output_dir) / "run_status.json",
                {"status": status, "finished_utc": utc_now(),
                 "config_hash": config_hash_value, **extra})


def choose_k(diagnostics):
    """Highest finite sampled silhouette, exact ties toward smaller k."""
    eligible = [(int(k), value.get("silhouette_sample"))
                for k, value in diagnostics.items()
                if value.get("silhouette_sample") is not None and
                math.isfinite(float(value["silhouette_sample"]))]
    if not eligible:
        raise ValueError("No candidate cluster has a valid silhouette score")
    return min(eligible, key=lambda pair: (-float(pair[1]), pair[0]))[0]


def _atomic_numpy_savez(path, np, **values):
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def _atomic_numpy_save(path, np, value):
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value)
    os.replace(temporary, path)


def _write_labels(path, roots, labels):
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["horse_id_netkeiba", "cluster"])
        writer.writerows(zip(roots, map(int, labels)))
    os.replace(temporary, path)


def cluster_training_vectors(train_vectors, roots, candidate_ks, seed, settings,
                             KMeans, silhouette_score, np):
    """Fit candidates on training starters and return diagnostics/models."""
    if not candidate_ks or max(candidate_ks) > len(train_vectors):
        raise ValueError("Candidate k exceeds the number of training starters")
    sample_size = min(settings["silhouette_sample"], len(train_vectors))
    sample = np.random.default_rng(0).choice(len(train_vectors), sample_size, replace=False)
    diagnostics, models = {}, {}
    for k in candidate_ks:
        model = KMeans(n_clusters=k, n_init=settings["kmeans_n_init"],
                       max_iter=settings["kmeans_max_iter"], random_state=seed)
        labels = model.fit_predict(train_vectors)
        sample_labels = labels[sample]
        score, error = None, None
        if 1 < len(np.unique(sample_labels)) < sample_size:
            try:
                score = float(silhouette_score(train_vectors[sample], sample_labels))
            except ValueError as exc:
                error = str(exc)
        elif sample_size > 1:
            error = "sample has fewer than two nonempty clusters"
        diagnostics[str(k)] = {
            "k": int(k),
            "cluster_sizes": {str(label): int(count) for label, count in
                               sorted(Counter(map(int, labels)).items())},
            "inertia": float(model.inertia_),
            "n_iter": int(model.n_iter_),
            "silhouette_sample": score,
        }
        if error:
            diagnostics[str(k)]["silhouette_error"] = error
        models[int(k)] = (model, labels)
    return diagnostics, models


def _environment(package_names):
    packages = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    memory = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                memory = line.strip()
                break
    return {"python": sys.version, "python_version": platform.python_version(),
            "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "memory_total": memory, "packages": packages}


def run(args):
    if not Path("/kaggle").is_dir():
        raise RuntimeError("Run this production graph job in Kaggle; local checks do not fit models")
    import resource
    import numpy as np
    import numba
    from pecanpy import pecanpy
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    started = time.perf_counter()
    input_dir, out = Path(args.input_dir).resolve(), Path(args.output_dir).resolve()
    if args.seed not in (0, 1, 2):
        raise ValueError("--seed must be 0, 1 or 2")
    if args.seed != 0 and args.clusters is None:
        raise ValueError("Repeat seeds require --clusters selected on seed 0")
    roots, nodes, edges, manifest = load_verified_inputs(input_dir)
    settings = dict(PRODUCTION_SETTINGS)
    settings["candidate_clusters"] = list(PRODUCTION_SETTINGS["candidate_clusters"])
    settings["seed"] = int(args.seed)
    settings["selected_clusters"] = int(args.clusters) if args.clusters is not None else None
    if args.clusters is not None and args.clusters not in settings["candidate_clusters"]:
        raise ValueError("--clusters must be one of 4, 8, 16, 32, 64")
    candidate_ks = ([args.clusters] if args.clusters is not None else
                    settings["candidate_clusters"])
    graph_path = out / "training_graph.edg"
    # Hash the deterministic edge-list bytes before publishing any output.
    graph_bytes = "".join(f"{parent}\t{child}\n" for parent, child in edges).encode("utf-8")
    graph_hash = hashlib.sha256(graph_bytes).hexdigest()
    input_hash = sha256_file(input_dir / "manifest.json")
    script_hash = sha256_file(Path(__file__).resolve())
    config = {"input_manifest_sha256": input_hash, "graph_sha256": graph_hash,
              "script_sha256": script_hash, "settings": settings,
              "candidate_clusters": candidate_ks}
    run_hash = begin_run(out, config)
    try:
        graph_started = time.perf_counter()
        write_graph(graph_path, edges)
        workers = min(settings["workers"], os.cpu_count() or 1)
        numba.set_num_threads(workers)
        report = {"status": "running", "purpose": "production training-only pedigree graph",
                  "settings": settings, "input_manifest_sha256": input_hash,
                  "graph_sha256": graph_hash, "script_sha256": script_hash,
                  "config_hash": run_hash, "input_scope": manifest["scope"],
                  "input_files_used": list(REQUIRED_INPUTS),
                  "target_outcomes_read": False, "outcome_columns_input": False,
                  "validation_test_roots_input": False,
                  "environment": _environment(("pecanpy", "numpy", "scipy", "gensim", "numba", "scikit-learn")),
                  "reproducibility": "Sorted immutable graph and fixed silhouette sample; multithreaded skip-gram is not bitwise deterministic.",
                  "started_utc": utc_now(), "numba_threads": workers,
                  "graph_build_seconds": time.perf_counter() - graph_started}
        atomic_json(out / "production_report.json", report)
        tick = time.perf_counter()
        print("embedding: starting Node2Vec", flush=True)
        graph = pecanpy.SparseOTF(p=settings["p"], q=settings["q"],
                                  workers=settings["workers"], random_state=settings["seed"])
        graph.read_edg(str(graph_path), weighted=False, directed=False)
        vectors = graph.embed(dim=settings["dimensions"], num_walks=settings["num_walks"],
                              walk_length=settings["walk_length"], window_size=settings["window_size"],
                              epochs=settings["epochs"], verbose=True)
        if set(graph.nodes) != set(nodes) or vectors.shape != (len(nodes), settings["dimensions"]):
            raise ValueError("Embedding node coverage or shape mismatch")
        if not np.isfinite(vectors).all():
            raise ValueError("Embedding contains nonfinite values")
        _atomic_numpy_savez(out / f"nodevectors_seed{args.seed}.npz", np,
                            node_ids=np.asarray(graph.nodes, dtype=str),
                            vectors=vectors.astype("float32"))
        print(f"embedding: complete in {time.perf_counter() - tick:.2f}s", flush=True)
        report.update(embedding_seconds=time.perf_counter() - tick,
                      nodevectors_file=f"nodevectors_seed{args.seed}.npz")
        atomic_json(out / "production_report.json", report)
        index = {node: i for i, node in enumerate(graph.nodes)}
        train_vectors = vectors[[index[node] for node in roots]]
        tick = time.perf_counter()
        diagnostics, models = cluster_training_vectors(
            train_vectors, roots, candidate_ks, settings["seed"], settings,
            KMeans, silhouette_score, np)
        for key in sorted(diagnostics, key=int):
            print(f"k={key}: inertia={diagnostics[key]['inertia']:.6g} "
                  f"n_iter={diagnostics[key]['n_iter']} "
                  f"silhouette={diagnostics[key]['silhouette_sample']}", flush=True)
        selected_k = int(args.clusters) if args.clusters is not None else choose_k(diagnostics)
        model, labels = models[selected_k]
        _atomic_numpy_save(out / f"cluster_centers_seed{args.seed}.npy", np,
                           model.cluster_centers_)
        _write_labels(out / f"cluster_labels_seed{args.seed}.csv", roots, labels)
        atomic_json(out / f"cluster_diagnostics_seed{args.seed}.json",
                    {"seed": args.seed, "candidates": diagnostics,
                     "selected_k": selected_k,
                     "selection_rule": "highest sampled silhouette; exact ties choose smaller k",
                     "fixed_sample_seed": 0, "fixed_sample_size": min(settings["silhouette_sample"], len(roots))})
        report.update(clustering_seconds=time.perf_counter() - tick,
                      selected_k=selected_k, cluster_diagnostics=diagnostics,
                      cluster_centers_file=f"cluster_centers_seed{args.seed}.npy",
                      cluster_labels_file=f"cluster_labels_seed{args.seed}.csv",
                      training_horses=len(roots), graph_nodes=len(nodes), graph_edges=len(edges),
                      graph_scope=manifest["scope"], total_seconds=time.perf_counter() - started,
                      peak_process_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                      finished_utc=utc_now(), status="complete")
        report["artifact_sha256"] = {
            name: sha256_file(out / name) for name in (
                f"nodevectors_seed{args.seed}.npz",
                f"cluster_centers_seed{args.seed}.npy",
                f"cluster_labels_seed{args.seed}.csv",
                f"cluster_diagnostics_seed{args.seed}.json",
                "training_graph.edg")}
        atomic_json(out / "production_report.json", report)
        finish_run(out, "complete", run_hash, selected_k=selected_k)
        return report
    except KeyboardInterrupt:
        finish_run(out, "interrupted", run_hash)
        raise
    except BaseException as exc:
        finish_run(out, "failed", run_hash, error=f"{type(exc).__name__}: {exc}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clusters", type=int, default=None,
                        help="fixed selected k for repeat seeds; omit for seed-0 selection")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({key: report[key] for key in
                      ("status", "selected_k", "training_horses", "graph_nodes",
                       "graph_edges", "embedding_seconds", "clustering_seconds",
                       "total_seconds", "peak_process_rss_gib")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
