"""Kaggle CPU resource pilot: training-only Node2Vec and K-means, no outcomes."""
import argparse
from collections import Counter, defaultdict, deque
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sqlite3
import time


def read_db(path):
    con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    con.execute("PRAGMA query_only=ON")
    return con


def training_roots(path):
    con = read_db(path)
    try:
        # Do not read finishes, odds, labels, validation roots, or test roots.
        return sorted(row[0] for row in con.execute("""
            SELECT DISTINCT n.horse_id_netkeiba FROM runners_r n
            JOIN races_r r USING(race_id)
            WHERE r.race_date BETWEEN '20100101' AND '20211231'
              AND r.race_type='flat' AND r.race_id!='201009040210'
              AND n.result_status NOT IN ('取消','除外')
              AND n.horse_id_netkeiba IS NOT NULL
        """))
    finally:
        con.close()


def load_parents(paths):
    parents = {}
    for path in paths:
        con = read_db(path)
        try:
            for child, parent, relation in con.execute(
                    "SELECT child_jbis_id,parent_jbis_id,relation FROM pedigree_edges"):
                if not child or not parent or relation not in ("sire", "dam"):
                    raise ValueError("Invalid pedigree relationship")
                key = (child, relation)
                if key in parents and parents[key] != parent:
                    raise ValueError(f"Conflicting {relation} for {child}")
                parents[key] = parent
        finally:
            con.close()
    return parents


def build_graph(roots, parents):
    """Union the three-generation paths of training starters only."""
    edges, nodes = set(), set(roots)
    for root in roots:
        frontier = [root]
        for depth in range(1, 4):
            following = []
            for child in frontier:
                for relation in ("sire", "dam"):
                    parent = parents.get((child, relation))
                    if parent is None:
                        raise ValueError(f"Incomplete generation {depth} for {root}")
                    edges.add((parent, child))
                    nodes.add(parent)
                    following.append(parent)
            frontier = following
    # Reject cycles while permitting repeated ancestors (pedigree collapse).
    degree, children = Counter(), defaultdict(list)
    for parent, child in edges:
        degree[child] += 1
        children[parent].append(child)
    queue = deque(node for node in nodes if degree[node] == 0)
    seen = 0
    while queue:
        seen += 1
        for child in children[queue.popleft()]:
            degree[child] -= 1
            if degree[child] == 0:
                queue.append(child)
    if seen != len(nodes):
        raise ValueError("Pedigree graph contains a cycle")
    return sorted(nodes), sorted(edges)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_pilot_inputs(folder):
    """Verify the exported, outcome-free inputs before constructing the graph."""
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("purpose") != "training-only pedigree resource pilot":
        raise ValueError("Unexpected input manifest")
    for name in ("training_horses.txt", "pedigree_edges.csv", "graph_pilot.py",
                 "requirements-graph.txt"):
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != manifest["files"][name]["sha256"]:
            raise ValueError(f"Input checksum mismatch: {name}")
    roots = (folder / "training_horses.txt").read_text(encoding="utf-8").splitlines()
    if not roots or roots != sorted(set(roots)) or any(not node.strip() for node in roots):
        raise ValueError("Training horse IDs must be nonempty, sorted and unique")
    parents = {}
    with (folder / "pedigree_edges.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["child", "parent", "relation"]:
            raise ValueError("Unexpected pedigree columns")
        for row in reader:
            child, parent, relation = row["child"], row["parent"], row["relation"]
            if not child or not parent or relation not in ("sire", "dam"):
                raise ValueError("Invalid pedigree relationship")
            if (child, relation) in parents and parents[child, relation] != parent:
                raise ValueError(f"Conflicting {relation} for {child}")
            parents[child, relation] = parent
    nodes, edges = build_graph(roots, parents)
    actual = dict(training_horses=len(roots), graph_nodes=len(nodes), graph_edges=len(edges))
    if actual != manifest["counts"]:
        raise ValueError("Graph counts differ from export manifest")
    if {(parent, child) for (child, _), parent in parents.items()} != set(edges):
        raise ValueError("Input contains relationships outside the training graph")
    return roots, nodes, edges, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not Path("/kaggle").is_dir():
        parser.error("Run this compute pilot in Kaggle; local checks use test_kaggle_prep.py")

    import resource
    import numpy as np
    import numba
    from pecanpy import pecanpy
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    started = time.perf_counter()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    settings = dict(seed=0, dimensions=64, num_walks=2, walk_length=20,
                    window_size=5, epochs=1, p=1.0, q=0.5,
                    workers=min(4, os.cpu_count() or 1), clusters=8,
                    kmeans_n_init=3, kmeans_max_iter=100,
                    silhouette_sample=2000)
    numba.set_num_threads(settings["workers"])
    report = {"status": "started", "purpose": "resource pilot; NOT paper results",
              "settings": settings, "python": platform.python_version(),
              "platform": platform.platform(), "cpu_count": os.cpu_count(),
              "memory_total_kib": next(line.strip() for line in
                  Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal:')),
              "packages": {p: importlib.metadata.version(p) for p in
                  ("pecanpy", "numpy", "scipy", "gensim", "numba", "scikit-learn")},
              "reproducibility": "Seed and sorted graph fixed; multithreaded training is not bitwise deterministic."}
    report["input_manifest_sha256"] = hashlib.sha256(
        (args.input_dir / "manifest.json").read_bytes()).hexdigest()
    write_json(out / "pilot_report.json", report)

    # Tiny remote smoke check catches dependency/API failures before the real graph.
    smoke_path = out / "smoke.edg"
    smoke_path.write_text("a\tb\nb\tc\nc\ta\n", encoding="utf-8")
    smoke = pecanpy.SparseOTF(p=1, q=0.5, workers=settings["workers"], random_state=0)
    smoke.read_edg(str(smoke_path), weighted=False, directed=False)
    smoke_vectors = smoke.embed(dim=8, num_walks=1, walk_length=5, epochs=1)
    assert smoke_vectors.shape == (3, 8) and np.isfinite(smoke_vectors).all()
    report["smoke_seconds"] = time.perf_counter() - started

    tick = time.perf_counter()
    roots, nodes, edges, manifest = load_pilot_inputs(args.input_dir)
    graph_path = out / "training_graph.edg"
    with graph_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, delimiter="\t", lineterminator="\n").writerows(edges)
    (out / "training_horses.txt").write_text("\n".join(roots) + "\n", encoding="utf-8")
    report.update(training_horses=len(roots), graph_nodes=len(nodes),
                  graph_edges=len(edges), graph_scope="2010-2021 eligible flat starters + three generations",
                  graph_sha256=hashlib.sha256(graph_path.read_bytes()).hexdigest(),
                  graph_build_seconds=time.perf_counter() - tick,
                  max_undirected_degree=max(Counter(node for edge in edges for node in edge).values()),
                  input_scope=manifest["scope"],
                  status="graph_ready")
    write_json(out / "pilot_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    del edges

    tick = time.perf_counter()
    graph = pecanpy.SparseOTF(p=settings["p"], q=settings["q"],
                             workers=settings["workers"], random_state=settings["seed"])
    graph.read_edg(str(graph_path), weighted=False, directed=False)
    vectors = graph.embed(dim=settings["dimensions"], num_walks=settings["num_walks"],
                          walk_length=settings["walk_length"], window_size=settings["window_size"],
                          epochs=settings["epochs"], verbose=True)
    assert set(graph.nodes) == set(nodes)
    assert vectors.shape == (len(nodes), settings["dimensions"])
    assert np.isfinite(vectors).all()
    np.savez_compressed(out / "PILOT_ONLY_embeddings.npz",
                        node_ids=np.asarray(graph.nodes, dtype=str), vectors=vectors.astype("float32"))
    report.update(embedding_seconds=time.perf_counter() - tick, status="embeddings_saved")
    write_json(out / "pilot_report.json", report)

    tick = time.perf_counter()
    index = {node: i for i, node in enumerate(graph.nodes)}
    train_vectors = vectors[[index[node] for node in roots]]
    clusters = KMeans(n_clusters=settings["clusters"], n_init=settings["kmeans_n_init"],
                      max_iter=settings["kmeans_max_iter"], random_state=settings["seed"])
    assignments = clusters.fit_predict(train_vectors)
    with (out / "PILOT_ONLY_training_clusters.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["horse_id_netkeiba", "cluster"])
        writer.writerows(zip(roots, assignments))
    np.save(out / "PILOT_ONLY_cluster_centers.npy", clusters.cluster_centers_)
    report["clustering_seconds"] = time.perf_counter() - tick
    report["cluster_sizes"] = {str(k): int(v) for k, v in sorted(Counter(assignments).items())}
    sample_size = min(settings["silhouette_sample"], len(roots))
    sample = np.random.default_rng(0).choice(len(roots), sample_size, replace=False)
    if 1 < len(np.unique(assignments[sample])) < sample_size:
        report["silhouette_sample_score"] = float(silhouette_score(train_vectors[sample], assignments[sample]))
    else:
        report["silhouette_sample_score"] = None
    report.update(status="complete", total_seconds=time.perf_counter() - started,
                  peak_process_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                  test_outcomes_read=False,
                  next_step="Review resources; then freeze production walk/cluster settings. Pilot vectors are not final features.")
    write_json(out / "pilot_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
