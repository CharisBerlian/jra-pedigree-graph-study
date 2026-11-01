"""Template for one private scientific seed-0 tuning notebook."""
import ast


def _setup_source(embedded, source_hashes, spec):
    return f'''from pathlib import Path
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")

assert Path("/kaggle/input").is_dir(), "Run this notebook on Kaggle CPU."
WORK = Path("/kaggle/working/jra-experiment-tuning")
SOURCE_DIR = WORK / "source"
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDED = {embedded!r}
EXPECTED_SOURCE_HASHES = {source_hashes!r}
for name, source in EMBEDDED.items():
    target = SOURCE_DIR / name
    target.write_text(source, encoding="utf-8")
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    assert actual == EXPECTED_SOURCE_HASHES[name], (name, actual, EXPECTED_SOURCE_HASHES[name])
    compile(source, name, "exec")

SPEC = {spec!r}
assert SPEC["spec_id"] == "experiment_spec_2026-09-13"
SPEC_PATH = WORK / "experiment_spec_2026-09-13.json"
SPEC_PATH.write_text(json.dumps(SPEC, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
SPEC_SHA256 = hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

matches = []
for csv_path in Path("/kaggle/input").rglob("train_validation.csv.gz"):
    metadata_path = csv_path.with_name("train_validation.csv.json")
    if metadata_path.is_file():
        matches.append((csv_path, metadata_path, "gzip"))
for csv_path in Path("/kaggle/input").rglob("train_validation.csv"):
    metadata_path = csv_path.with_name("train_validation.csv.json")
    if metadata_path.is_file():
        matches.append((csv_path, metadata_path, "csv"))
if len(matches) != 1:
    raise RuntimeError("expected exactly one frozen train-validation prediction bundle")
SOURCE, SOURCE_METADATA, source_format = matches[0]
metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
if metadata.get("purpose") != "JRA train-validation prediction inputs":
    raise ValueError("unexpected prediction input purpose")
if metadata.get("splits") != ["train", "validation"]:
    raise ValueError("prediction input splits must be exactly train and validation")
ORIGINAL_GZIP_SHA256 = "98afbbdea692da9e46e723af27e4f132a14a59d59fb36e650513dc02480b4fa4"
EXTRACTED_CSV_SHA256 = "d395a1802d90b0c09cc71f08b2dcbc83ca4020654797c9442969833c103ac12d"
source_sha256 = sha256_file(SOURCE)
if source_format == "gzip":
    if source_sha256 != ORIGINAL_GZIP_SHA256 or metadata.get("data_sha256") != source_sha256:
        raise ValueError("frozen compressed train-validation checksum mismatch")
    INPUT_DIR = WORK / "inputs"
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT = INPUT_DIR / "train_validation.csv.gz"
    shutil.copyfile(SOURCE, INPUT)
else:
    if metadata.get("data_sha256") != ORIGINAL_GZIP_SHA256 or source_sha256 != EXTRACTED_CSV_SHA256:
        raise ValueError("frozen extracted train-validation checksum mismatch")
    INPUT_DIR = WORK / "inputs"
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT = INPUT_DIR / "train_validation.csv.gz"
    with SOURCE.open("rb") as source_stream, INPUT.open("wb") as raw_output:
        with gzip.GzipFile(fileobj=raw_output, mode="wb", compresslevel=1, mtime=0) as gzip_output:
            shutil.copyfileobj(source_stream, gzip_output)
with gzip.open(INPUT, "rb") as stream:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
if digest.hexdigest() != EXTRACTED_CSV_SHA256:
    raise ValueError("frozen decompressed train-validation checksum mismatch")
actual_input_sha256 = sha256_file(INPUT)
assert INPUT.parent == INPUT_DIR, "runtime input and sidecar must be written under Kaggle working storage"
INPUT.with_suffix(".json").write_text(json.dumps(
    {{"data_sha256": actual_input_sha256, "splits": ["train", "validation"], "seed": 0,
      "purpose": "JRA train-validation prediction inputs", "source_sha256": source_sha256,
      "uncompressed_sha256": EXTRACTED_CSV_SHA256}}, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
INPUT_RECORD = {{"source": str(SOURCE), "source_sha256": source_sha256,
                 "input": str(INPUT), "input_sha256": actual_input_sha256,
                 "uncompressed_sha256": EXTRACTED_CSV_SHA256}}

runtime = SPEC.get("runtime", {{}})
package_specs = (("numpy", "numpy"), ("scipy", "scipy"), ("scikit_learn", "scikit-learn"),
                 ("pandas", "pandas"), ("lightgbm", "lightgbm"),
                 ("xgboost", "xgboost"), ("catboost", "catboost"))
requirements = [f"{{distribution}}=={{runtime[key]}}" for key, distribution in package_specs]
requirements_path = WORK / "requirements.txt"
requirements_path.write_text("\\n".join(requirements) + "\\n", encoding="utf-8")
VENV = Path("/kaggle/tmp/jra-experiment-tuning-venv")
PYTHON = VENV / "bin/python"
install_started = time.perf_counter()
subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(VENV)], check=True)
pip = [sys.executable, "-m", "pip", "--python", str(PYTHON)]
subprocess.run(pip + ["install", "--disable-pip-version-check", "-r", str(requirements_path)], check=True)
subprocess.run(pip + ["check"], check=True)
FROZEN = subprocess.check_output(pip + ["freeze"], text=True)
SETUP_REPORT = {{"purpose": "JRA scientific tuning setup", "spec_id": SPEC["spec_id"],
                "spec_sha256": SPEC_SHA256, "source_bundle": str(SOURCE_METADATA),
                "source_bundle_sha256": sha256_file(SOURCE_METADATA),
                "original_compressed_sha256": ORIGINAL_GZIP_SHA256,
                "extracted_csv_sha256": EXTRACTED_CSV_SHA256,
                "source_sha256": EXPECTED_SOURCE_HASHES, "input": INPUT_RECORD,
                "requirements": requirements,
                "python": sys.version, "pip_freeze": FROZEN,
                "installation_seconds": time.perf_counter() - install_started}}
RUN_ROOT = WORK / "runs"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
(WORK / "setup_report.json").write_text(json.dumps(SETUP_REPORT, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print("Verified frozen seed-0 input, exact embedded sources, spec, and pinned environment.", flush=True)
'''


def _run_source(family, surface):
    return f'''import json
import os
import subprocess
import time

FAMILY = {family!r}
SURFACE = {surface!r}
OUTPUT = RUN_ROOT
BATCH_DIR = RUN_ROOT / SURFACE / FAMILY
BATCH_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS = tuple(SPEC["settings"])
SUMMARY_PATH = BATCH_DIR / "batch_summary.json"
summary = {{"purpose": "JRA scientific seed-0 tuning", "family": FAMILY,
           "surface": SURFACE, "seed": 0, "settings": list(SETTINGS), "runs": []}}
for setting in SETTINGS:
    started = time.time()
    print(f"starting {{FAMILY}}/{{SURFACE}}/{{setting}} seed0", flush=True)
    command = [str(PYTHON), str(SOURCE_DIR / "run_experiments.py"),
               "--mode", "scientific", "--family", FAMILY, "--surface", SURFACE,
               "--setting", setting, "--stage", "tune", "--seed", "0",
               "--input", str(INPUT), "--spec", str(SPEC_PATH), "--output", str(OUTPUT)]
    env = os.environ.copy()
    env.update(PYTHONHASHSEED="0", MPLBACKEND="Agg", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    completed = subprocess.run(command, cwd=str(SOURCE_DIR), env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(completed.stdout, end="", flush=True)
    record = {{"setting": setting, "returncode": completed.returncode,
              "seconds": time.time() - started,
              "status": "complete" if completed.returncode == 0 else "failed"}}
    summary["runs"].append(record)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)
    print(f"complete {{FAMILY}}/{{SURFACE}}/{{setting}} seed0", flush=True)
summary["status"] = "complete"
SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print("scientific tuning batch complete:", FAMILY, SURFACE, len(SETTINGS), "settings", flush=True)
'''


def build_notebook(family, surface, embedded, source_hashes, spec):
    setup = _setup_source(embedded, source_hashes, spec)
    run = _run_source(family, surface)
    ast.parse(setup, filename="setup.py")
    ast.parse(run, filename="run.py")
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                          "language_info": {"name": "python"}},
            "cells": [
                {"cell_type": "markdown", "id": "purpose", "metadata": {},
                 "source": [f"# JRA scientific tuning: {{family}} / {{surface}}\\n\\n".format(family=family, surface=surface),
                             "Private Kaggle CPU run. This notebook tunes seed 0 for all ten declared settings with four candidates each. It opens only the train and validation feature bundle; test rows are rejected by the runner.\\n"]},
                {"cell_type": "code", "id": "setup", "metadata": {}, "source": setup.splitlines(keepends=True),
                 "execution_count": None, "outputs": []},
                {"cell_type": "code", "id": "run", "metadata": {}, "source": run.splitlines(keepends=True),
                 "execution_count": None, "outputs": []}]}
