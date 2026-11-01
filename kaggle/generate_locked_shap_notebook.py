"""Generate the private validation-only grouped-SHAP notebook."""

import base64
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PLAN = HERE.parent / "plans" / "experiment_spec_2026-09-13.json"
EMBED_FILES = (
    "run_shap_validation.py", "run_test_inference.py", "evaluate_locked.py",
    "audit_experiment_outputs.py",
    "run_experiments.py", "experiment_features.py", "prepare_prediction.py",
    "graph_pilot.py",
)
DATASETS = [
    "charischerw/jra-prediction-bundle-v1",
    "charischerw/jra-locked-selection-manifest-v1",
    "charischerw/jra-scientific-tuning-xgboost-turf-output-v2",
    "charischerw/jra-scientific-tuning-xgboost-dirt-output-v2",
    "charischerw/jra-scientific-tuning-lightgbm-turf-output-v2",
    "charischerw/jra-scientific-tuning-lightgbm-dirt-output-v2",
]
KERNELS = [
    "charischerw/jra-scientific-tuning-catboost-turf",
    "charischerw/jra-scientific-tuning-catboost-dirt",
    "charischerw/jra-matched-seed-refits-xgboost-turf",
    "charischerw/jra-matched-seed-refits-xgboost-dirt",
    "charischerw/jra-matched-seed-refits-lightgbm-turf",
    "charischerw/jra-matched-seed-refits-lightgbm-dirt",
    "charischerw/jra-matched-seed-refits-catboost-turf",
    "charischerw/jra-matched-seed-refits-catboost-dirt",
]

CODE_TEMPLATE = r'''from pathlib import Path
import base64
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys

assert Path("/kaggle/input").is_dir(), "Run this notebook on Kaggle only."
os.environ["MPLBACKEND"] = "Agg"
WORK = Path("/kaggle/working/jra-locked-validation-shap")
SOURCE_ROOT = WORK / "source"
MODEL_ROOT = WORK / "models"
INPUT_ROOT = WORK / "inputs"
OUTPUT_ROOT = WORK / "outputs"
for folder in (SOURCE_ROOT, MODEL_ROOT, INPUT_ROOT, OUTPUT_ROOT):
    folder.mkdir(parents=True, exist_ok=True)
EMBEDDED = __EMBEDDED__
EXPECTED_HASHES = __HASHES__
for name, encoded in EMBEDDED.items():
    target = SOURCE_ROOT / name
    target.write_bytes(base64.b64decode(encoded))
    assert hashlib.sha256(target.read_bytes()).hexdigest() == EXPECTED_HASHES[name]
    compile(target.read_text(encoding="utf-8"), name, "exec")
SPEC_PATH = SOURCE_ROOT / "experiment_spec_2026-09-13.json"
SPEC_PATH.write_bytes(base64.b64decode(__SPEC_B64__))
MANIFEST = next(Path("/kaggle/input").rglob("experiment_selection_manifest_2026-09-14.json"))
manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
assert manifest["status"] == "locked_for_test_release"
assert manifest["test_performance_opened"] is False
assert manifest["test_evaluation_authorized"] is False
assert manifest["contains_test_predictions"] is False
assert manifest["no_test_performance"] is True
assert manifest["audited_counts"] == {"tuning_runs": 60, "candidate_runs": 240, "refit_runs": 180}

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def stage_roots(stage, family, surface):
    marker = "jra-experiment-tuning" if stage == "tune" else "jra-experiment-refits"
    candidates = []
    input_root = Path("/kaggle/input")
    for marker_path in input_root.rglob(marker):
        candidate = marker_path / "runs" / "scientific" / family / surface
        if candidate.is_dir():
            candidates.append(candidate)
    if stage == "tune":
        for mount in sorted(input_root.iterdir()):
            candidate = mount / "runs" / "scientific" / family / surface
            if candidate.is_dir():
                candidates.append(candidate)
    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    if len(unique) != 1:
        raise RuntimeError(f"expected one {stage} mount for {family}/{surface}, found {len(unique)}: {unique}")
    return unique[0]

def link_tree(source, target):
    target.mkdir(parents=True, exist_ok=True)
    for child in sorted(Path(source).iterdir()):
        destination = target / child.name
        if child.is_dir():
            link_tree(child, destination)
        else:
            destination.symlink_to(child)

families = ("xgboost", "lightgbm", "catboost")
surfaces = ("turf", "dirt")
settings = ("B1", "S1", "S2", "S3", "S4", "S5", "S6", "M2", "M3", "S7")
for family in families:
    for surface in surfaces:
        tune_root = stage_roots("tune", family, surface)
        refit_root = stage_roots("refit", family, surface)
        for setting in settings:
            link_tree(tune_root / setting / "tune", MODEL_ROOT / "scientific" / family / surface / setting / "tune")
            link_tree(refit_root / setting / "refit", MODEL_ROOT / "scientific" / family / surface / setting / "refit")

bundle_roots = [path for path in Path("/kaggle/input").rglob("jra-prediction-bundle-v1")
                if path.is_dir()]
if len(bundle_roots) != 1:
    raise RuntimeError(f"expected one prediction-bundle mount, found {len(bundle_roots)}: {bundle_roots}")
sources = [path for path in bundle_roots[0].rglob("train_validation.csv")
           if path.with_name("train_validation.csv.json").is_file()]
if len(sources) != 1:
    raise RuntimeError(f"expected one prediction-bundle validation CSV, found {len(sources)}: {sources}")
source = sources[0]
source_sha256 = sha256_file(source)
expected_csv_sha256 = "d395a1802d90b0c09cc71f08b2dcbc83ca4020654797c9442969833c103ac12d"
if source_sha256 != expected_csv_sha256:
    raise ValueError("mounted prediction-bundle CSV is not the frozen validation input")
target = INPUT_ROOT / "train_validation_seed0.csv.gz"
with source.open("rb") as source_stream, target.open("wb") as raw_output:
    with gzip.GzipFile(fileobj=raw_output, mode="wb", compresslevel=1, mtime=0) as gzip_output:
        shutil.copyfileobj(source_stream, gzip_output)
target.with_suffix(".json").write_text(json.dumps({
    "data_sha256": sha256_file(target), "uncompressed_sha256": source_sha256,
    "seed": 0, "splits": ["train", "validation"],
    "purpose": "JRA train-validation prediction inputs",
    "source_path": str(source),
    "source_sidecar": str(source.with_name("train_validation.csv.json")),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

requirements = WORK / "requirements.txt"
requirements.write_text("\n".join((
    "numpy==1.26.4", "scipy==1.13.1", "pandas==2.2.3",
    "scikit-learn==1.6.1", "xgboost==3.0.2", "lightgbm==4.6.0",
    "catboost==1.2.8", "shap==0.46.0")) + "\n", encoding="utf-8")
venv = Path("/kaggle/tmp/jra-locked-validation-shap-venv")
python = venv / "bin" / "python"
if not python.is_file():
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
pip = [sys.executable, "-m", "pip", "--python", str(python)]
subprocess.run(pip + ["install", "--disable-pip-version-check", "-r", str(requirements)], check=True)
subprocess.run(pip + ["check"], check=True)
command = [str(python), str(SOURCE_ROOT / "run_shap_validation.py"),
           "--manifest", str(MANIFEST), "--spec", str(SPEC_PATH),
           "--model-root", str(MODEL_ROOT), "--input", str(target),
           "--output-dir", str(OUTPUT_ROOT), "--source-root", str(SOURCE_ROOT)]
env = os.environ.copy()
env.update(PYTHONHASHSEED="0", MPLBACKEND="Agg", OMP_NUM_THREADS="4",
           OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
subprocess.run(command, cwd=str(SOURCE_ROOT), env=env, check=True)
report = json.loads((OUTPUT_ROOT / "shap_validation_report.json").read_text(encoding="utf-8"))
assert report["test_rows_read"] is False
assert report["test_performance_computed"] is False
assert len(report["results"]) == 4
print(json.dumps({"status": "complete", "results": len(report["results"]),
                  "manifest_sha256": report["manifest_sha256"],
                  "sample_counts": [r["validation_entry_count"] for r in report["results"]]}, sort_keys=True), flush=True)
shutil.rmtree(SOURCE_ROOT)
shutil.rmtree(MODEL_ROOT)
shutil.rmtree(INPUT_ROOT)
'''

def main():
    embedded = {name: base64.b64encode((HERE / name).read_bytes()).decode("ascii")
                for name in EMBED_FILES}
    hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
              for name in EMBED_FILES}
    spec_b64 = base64.b64encode(PLAN.read_bytes()).decode("ascii")
    code = CODE_TEMPLATE.replace("__EMBEDDED__", repr(embedded))
    code = code.replace("__HASHES__", repr(hashes))
    code = code.replace("__SPEC_B64__", repr(spec_b64))
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [
                "# JRA validation-only grouped SHAP\n\n",
                "This private notebook never reads test rows or computes test performance.\n",
            ]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
             "source": code.splitlines(keepends=True)},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    output = HERE / "locked_validation_shap.ipynb"
    output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    metadata = {"id": "charischerw/jra-locked-validation-shap-v1",
                "title": "JRA Locked Validation SHAP v1", "code_file": output.name,
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
                "machine_shape": "None", "dataset_sources": DATASETS,
                "competition_sources": [], "kernel_sources": KERNELS, "model_sources": []}
    (HERE / "locked_validation_shap-kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
