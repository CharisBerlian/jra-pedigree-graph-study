"""Read-only numerical/provenance audit of saved pilot files; no model fitting."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from graph_pilot import load_pilot_inputs, read_db


BASE = Path(__file__).resolve().parents[1]
INPUT = BASE / "kaggle/upload/jra-pedigree-pilot-v1"
OUTPUT = BASE / "kaggle/notebook-output/version-2/jra-pilot"


def main():
    report = json.loads((OUTPUT / "pilot_report.json").read_text())
    roots, nodes, edges, manifest = load_pilot_inputs(INPUT)
    assert report["status"] == "complete"
    assert json.loads((OUTPUT / "input_manifest.json").read_text()) == manifest
    assert hashlib.sha256((INPUT / "manifest.json").read_bytes()).hexdigest() == report["input_manifest_sha256"]
    assert (OUTPUT / "training_horses.txt").read_text().splitlines() == roots
    graph_bytes = (OUTPUT / "training_graph.edg").read_bytes()
    assert hashlib.sha256(graph_bytes).hexdigest() == report["graph_sha256"]
    assert [tuple(line.split()) for line in graph_bytes.decode().splitlines()] == edges

    with np.load(OUTPUT / "PILOT_ONLY_embeddings.npz", allow_pickle=False) as data:
        ids, vectors = data["node_ids"].tolist(), data["vectors"]
    assert len(ids) == len(set(ids)) == len(nodes)
    assert set(ids) == set(nodes)
    assert vectors.shape == (len(nodes), report["settings"]["dimensions"])
    assert vectors.dtype == np.float32 and np.isfinite(vectors).all()
    index = {node: i for i, node in enumerate(ids)}
    x = vectors[[index[node] for node in roots]].astype(np.float64)
    centers = np.load(OUTPUT / "PILOT_ONLY_cluster_centers.npy", allow_pickle=False).astype(np.float64)
    assert centers.shape == (report["settings"]["clusters"], x.shape[1])
    assert np.isfinite(centers).all()
    with (OUTPUT / "PILOT_ONLY_training_clusters.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["horse_id_netkeiba", "cluster"]
        rows = list(reader)
    assert [r["horse_id_netkeiba"] for r in rows] == roots
    labels = np.array([int(r["cluster"]) for r in rows])
    sizes = {str(k): int(v) for k, v in sorted(Counter(labels).items())}
    assert sizes == report["cluster_sizes"]
    distances2 = np.maximum((x*x).sum(1)[:, None] + (centers*centers).sum(1) - 2*x@centers.T, 0)
    mismatches = int(np.count_nonzero(distances2.argmin(1) != labels))
    assert mismatches == 0
    inertia = float(distances2[np.arange(len(x)), labels].sum())
    centered = x - x.mean(0)
    variance = float((centered*centered).sum())
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered)

    # Reproduce the exact 2,000-horse sample used remotely, without fitting anything.
    sample = np.random.default_rng(0).choice(len(x), report["settings"]["silhouette_sample"], replace=False)
    sx, sl = x[sample], labels[sample]
    d = np.sqrt(np.maximum((sx*sx).sum(1)[:, None] + (sx*sx).sum(1)[None, :] - 2*sx@sx.T, 0))
    np.fill_diagonal(d, 0)
    a, b = np.zeros(len(sx)), np.full(len(sx), np.inf)
    singleton = np.zeros(len(sx), dtype=bool)
    for label in np.unique(sl):
        members = sl == label
        sums = d[:, members].sum(1)
        if members.sum() > 1:
            a[members] = sums[members] / (members.sum() - 1)
        else:
            singleton[members] = True
        b[~members] = np.minimum(b[~members], sums[~members] / members.sum())
    silhouette = np.divide(b-a, np.maximum(a, b), out=np.zeros_like(a), where=np.maximum(a,b)>0)
    silhouette[singleton] = 0
    score = float(silhouette.mean())
    assert abs(score - report["silhouette_sample_score"]) < 1e-5

    con = read_db(BASE / "tmp/features.db")
    try:
        history = dict(con.execute("SELECT prior_race_id, COUNT(*) FROM history_long WHERE prior_race_id IN ('201009040210','201407010204') GROUP BY prior_race_id"))
        outside = con.execute("SELECT COUNT(*), MAX(prior_norm_finish) FROM history_long WHERE prior_norm_finish > 1").fetchone()
        metadata = dict(con.execute("SELECT key,value FROM meta"))
    finally:
        con.close()
    result = {
        "saved_pilot_integrity": "PASS",
        "input_checksums_and_reconstructed_graph": "PASS",
        "counts": manifest["counts"],
        "embedding_shape": list(vectors.shape),
        "embedding_dtype": str(vectors.dtype),
        "nonfinite_embedding_values": int((~np.isfinite(vectors)).sum()),
        "zero_embedding_rows": int(np.count_nonzero(np.linalg.norm(vectors, axis=1) == 0)),
        "training_vector_norm_quantiles_0_25_50_75_100": np.quantile(np.linalg.norm(x, axis=1), [0,.25,.5,.75,1]).tolist(),
        "cluster_sizes": sizes,
        "nearest_centroid_label_mismatches": mismatches,
        "inertia_from_saved_outputs": inertia,
        "variance_explained_by_centroids_fraction": 1-inertia/variance,
        "largest_centered_principal_component_variance_fraction": float(eigenvalues[-1]/eigenvalues.sum()),
        "silhouette_recomputed": score,
        "silhouette_negative_sample_fraction": float((silhouette<0).mean()),
        "history_rows_from_uncertain_races": history,
        "history_normalized_finish_above_one_count": outside[0],
        "history_normalized_finish_above_one_max": outside[1],
        "features_db_records_history_exclusion_policy": metadata.get("uncertain_history_races_excluded"),
        "prediction_training_ready": not history and outside[0] == 0,
        "scope": "Artifact integrity and small diagnostics only; no model training or target outcome evaluation."
    }
    target = BASE / "plans/kaggle_pilot_audit_2026-09-12.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
