"""Generate the Kaggle-only graph-table assembly notebook."""
import ast
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
HELPER = HERE / "assemble_prediction_seed.py"
OUTPUT = HERE / "graph_table_assembly.ipynb"
FROZEN_BASE_SHA256 = "98afbbdea692da9e46e723af27e4f132a14a59d59fb36e650513dc02480b4fa4"
FROZEN_BASE_UNCOMPRESSED = {
    "bytes": 981749176,
    "sha256": "d395a1802d90b0c09cc71f08b2dcbc83ca4020654797c9442969833c103ac12d",
}


def _hash(source):
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def build():
    helper_source = HELPER.read_text(encoding="utf-8")
    ast.parse(helper_source, filename=HELPER.name)
    helper_hash = _hash(helper_source)
    setup = f'''import gzip, hashlib, json, shutil, sys
from pathlib import Path

assert Path("/kaggle").is_dir(), "Run this notebook on Kaggle only."
WORK = Path("/kaggle/working/jra-graph-assembly")
WORK.mkdir(parents=True, exist_ok=True)
HELPER_SOURCE = {helper_source!r}
HELPER_SHA256 = {helper_hash!r}
helper_path = WORK / "assemble_prediction_seed.py"
helper_path.write_text(HELPER_SOURCE, encoding="utf-8")
assert hashlib.sha256(helper_path.read_bytes()).hexdigest() == HELPER_SHA256
compile(HELPER_SOURCE, str(helper_path), "exec")
sys.path.insert(0, str(WORK))
import assemble_prediction_seed as assembler

FROZEN_BASE_SHA256 = {FROZEN_BASE_SHA256!r}
FROZEN_BASE_UNCOMPRESSED = {FROZEN_BASE_UNCOMPRESSED!r}
FROZEN_GRAPH = assembler.FROZEN_PROVENANCE

def find_one(names):
    for name in names:
        found = sorted(set(path.resolve() for path in Path("/kaggle/input").rglob(name)))
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise RuntimeError(f"Expected exactly one {{name}}, found: {{found}}")
    raise RuntimeError(f"Expected one input among {{names}}")

def gzip_stats(path):
    digest = hashlib.sha256()
    size = 0
    with gzip.open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {{"bytes": size, "sha256": digest.hexdigest()}}

def raw_stats(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {{"bytes": size, "sha256": digest.hexdigest()}}

def normalize(path, metadata_path, stem, expected_base=False):
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if path.name.endswith(".gz"):
        compressed = raw_stats(path)
        if compressed["sha256"] != metadata.get("data_sha256"):
            raise ValueError(f"compressed source hash mismatch: {{path}}")
        uncompressed = gzip_stats(path)
    else:
        uncompressed = raw_stats(path)
        expected = metadata.get("uncompressed_csv", FROZEN_BASE_UNCOMPRESSED if expected_base else None)
        if expected is None or uncompressed != expected:
            raise ValueError(f"raw source hash mismatch: {{path}}")
        normalized = WORK / f"{{stem}}.csv.gz"
        with Path(path).open("rb") as source, gzip.open(normalized, "wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        runtime_meta = dict(metadata, data_sha256=raw_stats(normalized)["sha256"],
                            source_compressed_sha256=metadata.get("data_sha256"),
                            source_uncompressed_csv=uncompressed)
        runtime_sidecar = normalized.with_suffix(".json")
        runtime_sidecar.write_text(json.dumps(runtime_meta, sort_keys=True), encoding="utf-8")
        path, metadata_path = normalized, runtime_sidecar
    expected_uncompressed = metadata.get("uncompressed_csv", FROZEN_BASE_UNCOMPRESSED if expected_base else None)
    if expected_uncompressed is not None and uncompressed != expected_uncompressed:
        raise ValueError(f"uncompressed source hash mismatch: {{path}}")
    return path, metadata_path, metadata, uncompressed

BASE = find_one(("train_validation.csv.gz", "train_validation.csv"))
BASE_META_PATH = find_one(("train_validation.csv.json",))
BASE, BASE_META_RUNTIME, BASE_META, BASE_UNCOMPRESSED = normalize(BASE, BASE_META_PATH, "base", True)
assert BASE_META.get("data_sha256") == FROZEN_BASE_SHA256
print("base source:", BASE, BASE_UNCOMPRESSED)
'''
    assemble = '''import json, shutil
from pathlib import Path

def table_input(seed):
    csv_path = find_one((f"graph_features_seed{seed}.csv.gz", f"graph_features_seed{seed}.csv"))
    meta_path = find_one((f"graph_features_seed{seed}.csv.json",))
    return normalize(csv_path, meta_path, f"graph-seed{seed}")

for seed in (1, 2):
    table, table_runtime_meta, table_meta, table_uncompressed = table_input(seed)
    assembler.verify_graph_metadata(table_runtime_meta, expected_seed=seed,
                                    base_data_sha256=BASE_META["data_sha256"])
    # The helper accepts the normalized gzip sidecar and separately retains
    # the original source metadata in the output provenance.
    output = WORK / f"seed{seed}" / "train_validation.csv.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = assembler.assemble(
        BASE, table, output, graph_metadata=table_runtime_meta,
        base_metadata=BASE_META_RUNTIME, expected_seed=seed,
        source_base_sha256=BASE_META["data_sha256"],
        source_base_metadata=BASE_META)
    result["source_graph_metadata"] = table_meta
    result["source_graph_uncompressed_csv"] = table_uncompressed
    output.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\\n")
    print(json.dumps({"seed": seed, "output": str(output), "rows": result["rows"],
                      "data_sha256": result["data_sha256"],
                      "non_graph_sha256": result["assembled_non_graph_sha256"]}, sort_keys=True))
print("PASS: graph-only seed 1/2 assembly completed with frozen base preservation and provenance checks")
'''
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}},
        "cells": [
            {"cell_type": "markdown", "id": "purpose", "metadata": {},
             "source": ["# JRA graph-table assembly\n",
                         "Kaggle-only no-fit assembly. The existing private prediction bundle supplies the frozen non-graph rows; the compact graph-table dataset supplies seed-1 and seed-2 graph fields.\n"]},
            {"cell_type": "code", "id": "setup", "metadata": {}, "source": setup.splitlines(keepends=True),
             "execution_count": None, "outputs": []},
            {"cell_type": "code", "id": "assemble", "metadata": {}, "source": assemble.splitlines(keepends=True),
             "execution_count": None, "outputs": []},
        ],
    }
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{cell['id']}.py")
    OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes), helper SHA256 {helper_hash}")


if __name__ == "__main__":
    build()
