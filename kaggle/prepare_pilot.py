"""Export a small, outcome-free Kaggle pilot bundle; does not train anything."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile

from graph_pilot import build_graph, load_parents, load_pilot_inputs, read_db, training_roots, write_json


def prepare(project, destination):
    started = time.perf_counter()
    roots = training_roots(project / "tmp/repaired_jra.db")
    # Independently match the materialized analysis cohort using IDs only.
    con = read_db(project / "tmp/features.db")
    try:
        expected = sorted(row[0] for row in con.execute(
            "SELECT DISTINCT horse_id_netkeiba FROM entries WHERE split='train'"))
    finally:
        con.close()
    if roots != expected:
        raise ValueError("Training cohort differs from features.db; reconcile before export")
    parents = load_parents([project / "local_jra_worker/data/raw_jbis_laptop.db",
                            project / "tmp/pedigree_patch.db"])
    nodes, edges = build_graph(roots, parents)
    edge_set = set(edges)
    rows = sorted((child, parent, relation) for (child, relation), parent in parents.items()
                  if (parent, child) in edge_set)
    # Refuse to overwrite a previously frozen upload.
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "training_horses.txt").write_text("\n".join(roots) + "\n", encoding="utf-8")
    with (destination / "pedigree_edges.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["child", "parent", "relation"])
        writer.writerows(rows)
    source = Path(__file__).resolve().parent
    for name in ("graph_pilot.py", "requirements-graph.txt"):
        shutil.copyfile(source / name, destination / name)
    manifest = {
        "purpose": "training-only pedigree resource pilot",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "2010-2021 eligible observed weekend JRA flat starters; union of three-generation pedigrees; undirected walks",
        "counts": dict(training_horses=len(roots), graph_nodes=len(nodes), graph_edges=len(edges)),
        "source_id_namespace": "Netkeiba (legacy pedigree database columns are named jbis_id)",
        "source_files": ["tmp/repaired_jra.db", "tmp/features.db",
                         "local_jra_worker/data/raw_jbis_laptop.db", "tmp/pedigree_patch.db"],
        "outcome_columns_exported": False,
        "validation_test_roots_exported": False,
        "root_rule": "real race_date 20100101..20211231; flat; exclude race 201009040210 and statuses 取消/除外",
        "pending_before_prediction_experiments": [
            "Reconcile incomplete races used by previous-start features (see completion_recheck_2026-09-11.md)",
            "Refresh manuscript/data counts and fix the overflowing PDF table",
            "Freeze production Node2Vec settings, cluster selection and unseen-horse fallback after resource pilot"],
        "files": {path.name: {"bytes": path.stat().st_size,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                  for path in sorted(destination.iterdir())},
    }
    write_json(destination / "manifest.json", manifest)
    load_pilot_inputs(destination)
    archive = destination.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as bundle:
        for path in sorted(destination.iterdir()):
            bundle.write(path, path.name)
    return {"counts": manifest["counts"], "folder": str(destination), "zip": str(archive),
            "zip_bytes": archive.stat().st_size,
            "zip_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "preparation_seconds": round(time.perf_counter() - started, 2)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "upload" / "jra-pedigree-pilot-v1")
    args = parser.parse_args()
    print(json.dumps(prepare(Path(__file__).resolve().parents[1], args.output_dir.resolve()), indent=2))
