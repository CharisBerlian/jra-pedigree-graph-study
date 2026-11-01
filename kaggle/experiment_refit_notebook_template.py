"""Template for one private matched-seed refit notebook."""
import ast


ORIGINAL_GZIP_SHA256 = "98afbbdea692da9e46e723af27e4f132a14a59d59fb36e650513dc02480b4fa4"
EXTRACTED_CSV_SHA256 = "d395a1802d90b0c09cc71f08b2dcbc83ca4020654797c9442969833c103ac12d"


def setup_source(embedded, source_hashes, spec, family, surface):
    return f'''from pathlib import Path
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

os.environ["MPLBACKEND"] = "Agg"
assert Path("/kaggle/input").is_dir(), "Run this notebook on Kaggle CPU."
WORK = Path("/kaggle/working/jra-experiment-refits")
SOURCE_DIR = WORK / "source"
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDED = {embedded!r}
EXPECTED_SOURCE_HASHES = {source_hashes!r}
for name, source in EMBEDDED.items():
    target = SOURCE_DIR / name
    target.write_text(source, encoding="utf-8")
    assert hashlib.sha256(target.read_bytes()).hexdigest() == EXPECTED_SOURCE_HASHES[name]
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

def csv_uncompressed_sha256(path):
    digest = hashlib.sha256()
    if Path(path).suffix == ".gz":
        stream = gzip.open(path, "rb")
    else:
        stream = Path(path).open("rb")
    with stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def metadata_path(path):
    return path.with_suffix(".json")

def input_candidates(seed):
    candidates = []
    for path in Path("/kaggle/input").rglob("*.csv.gz"):
        lower = str(path).lower().replace("\\\\", "/")
        if "train_validation" not in lower or "test" in lower:
            continue
        seed_token = f"seed{{seed}}"
        if seed == 0:
            if any(token in lower for token in ("seed1", "seed-1", "seed_1", "seed2", "seed-2", "seed_2")):
                continue
        elif seed_token not in lower and f"seed-{{seed}}" not in lower and f"seed_{{seed}}" not in lower:
            continue
        sidecar = metadata_path(path)
        if sidecar.is_file():
            candidates.append((path, sidecar))
    for path in Path("/kaggle/input").rglob("train_validation.csv"):
        lower = str(path).lower().replace("\\\\", "/")
        if "test" in lower:
            continue
        if seed == 0 and any(token in lower for token in ("seed1", "seed-1", "seed_1", "seed2", "seed-2", "seed_2")):
            continue
        if seed > 0 and f"seed{{seed}}" not in lower and f"seed-{{seed}}" not in lower and f"seed_{{seed}}" not in lower:
            continue
        sidecar = path.with_name("train_validation.csv.json")
        if sidecar.is_file():
            candidates.append((path, sidecar))
    return candidates

def prepare_seed_input(seed):
    candidates = input_candidates(seed)
    if seed == 0 and len(candidates) > 1:
        candidates = [item for item in candidates
                      if json.loads(item[1].read_text(encoding="utf-8")).get("data_sha256") == "{ORIGINAL_GZIP_SHA256}"]
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one mounted train-validation gzip for seed {{seed}}, found {{len(candidates)}}")
    source, source_sidecar = candidates[0]
    source_metadata = json.loads(source_sidecar.read_text(encoding="utf-8"))
    source_splits = source_metadata.get("splits") or source_metadata.get("base_metadata", {{}}).get("splits")
    if source_splits != ["train", "validation"]:
        raise ValueError(f"seed {{seed}} input is not train/validation only")
    if not str(source_metadata.get("purpose", "")).startswith("JRA train-validation prediction inputs"):
        raise ValueError(f"seed {{seed}} input provenance purpose is unexpected")
    if seed > 0 and source_metadata.get("seed") != seed:
        raise ValueError(f"seed {{seed}} input provenance is not explicitly seed-bound")
    source_sha256 = sha256_file(source)
    if source.suffix == ".gz":
        if source_metadata.get("data_sha256") != source_sha256:
            raise ValueError(f"seed {{seed}} mounted gzip checksum does not match sidecar")
    else:
        if seed == 0:
            if source_metadata.get("data_sha256") != "{ORIGINAL_GZIP_SHA256}" or source_sha256 != "{EXTRACTED_CSV_SHA256}":
                raise ValueError("seed0 extracted CSV is not the frozen prediction bundle")
        else:
            raw_stats = source_metadata.get("csv_stats", {{}}).get("uncompressed", {{}})
            if raw_stats.get("sha256") != source_sha256:
                raise ValueError(f"seed {{seed}} extracted CSV checksum does not match sidecar")
    if seed == 0 and source.suffix == ".gz" and source_sha256 != "{ORIGINAL_GZIP_SHA256}":
        raise ValueError("seed0 input is not the frozen prediction bundle")
    expected_uncompressed = source_metadata.get("csv_stats", {{}}).get("uncompressed", {{}}).get("sha256")
    if seed == 0:
        expected_uncompressed = "{EXTRACTED_CSV_SHA256}"
    if not expected_uncompressed or csv_uncompressed_sha256(source) != expected_uncompressed:
        raise ValueError(f"seed {{seed}} decompressed input checksum mismatch")
    input_dir = WORK / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    target = input_dir / f"train_validation_seed{{seed}}.csv.gz"
    if target.exists() or target.is_symlink():
        target.unlink()
    if source.suffix == ".gz":
        target.symlink_to(source)
    else:
        with source.open("rb") as source_stream, target.open("wb") as raw_output:
            with gzip.GzipFile(fileobj=raw_output, mode="wb", compresslevel=1, mtime=0) as gzip_output:
                shutil.copyfileobj(source_stream, gzip_output)
    target_sha256 = sha256_file(target)
    runtime_metadata = {{"data_sha256": target_sha256, "splits": ["train", "validation"],
                        "seed": seed, "purpose": "JRA train-validation prediction inputs",
                        "source_path": str(source), "source_sidecar": str(source_sidecar),
                        "source_sha256": source_sha256,
                        "uncompressed_sha256": expected_uncompressed}}
    metadata_path(target).write_text(json.dumps(runtime_metadata, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    return {{"source": str(source), "source_sidecar": str(source_sidecar),
            "input": str(target), "source_sha256": source_sha256,
            "input_sha256": target_sha256, "uncompressed_sha256": expected_uncompressed}}

INPUTS = {{seed: prepare_seed_input(seed) for seed in (0, 1, 2)}}

def find_tune_seed0(setting):
    matches = []
    for selected in Path("/kaggle/input").rglob("selected.json"):
        parents = selected.parents
        if len(parents) < 6 or parents[0].name != "seed0" or parents[1].name != "tune":
            continue
        if parents[2].name != setting or parents[3].name != "{surface}" or parents[4].name != "{family}" or parents[5].name != "scientific":
            continue
        complete = parents[0] / "complete.json"
        config = parents[0] / "config.json"
        if complete.is_file() and config.is_file():
            matches.append(parents[0])
    if len(matches) != 1:
        raise RuntimeError(f"expected one complete tuning checkpoint for {{setting}}, found {{len(matches)}}")
    return matches[0]

RUN_ROOT = WORK / "runs"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
for setting in SPEC["settings"]:
    source = find_tune_seed0(setting)
    target = RUN_ROOT / "scientific" / "{family}" / "{surface}" / setting / "tune" / "seed0"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not target.is_symlink() or target.resolve() != source.resolve():
            raise RuntimeError(f"tuning mount path conflicts with existing output: {{target}}")
    else:
        target.symlink_to(source, target_is_directory=True)

runtime = SPEC.get("runtime", {{}})
package_specs = (("numpy", "numpy"), ("scipy", "scipy"), ("scikit_learn", "scikit-learn"),
                 ("pandas", "pandas"), ("lightgbm", "lightgbm"),
                 ("xgboost", "xgboost"), ("catboost", "catboost"))
requirements = [f"{{distribution}}=={{runtime[key]}}" for key, distribution in package_specs]
requirements_path = WORK / "requirements.txt"
requirements_path.write_text("\\n".join(requirements) + "\\n", encoding="utf-8")
VENV = Path("/kaggle/tmp/jra-experiment-refits-venv")
PYTHON = VENV / "bin/python"
install_started = time.perf_counter()
subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(VENV)], check=True)
pip = [sys.executable, "-m", "pip", "--python", str(PYTHON)]
subprocess.run(pip + ["install", "--disable-pip-version-check", "-r", str(requirements_path)], check=True)
subprocess.run(pip + ["check"], check=True)
FROZEN = subprocess.check_output(pip + ["freeze"], text=True)
SETUP_REPORT = {{"purpose": "JRA matched-seed refit setup", "family": "{family}", "surface": "{surface}",
                "spec_id": SPEC["spec_id"], "spec_sha256": SPEC_SHA256,
                "source_sha256": EXPECTED_SOURCE_HASHES, "inputs": INPUTS,
                "tuning_mounts": {{setting: str(find_tune_seed0(setting)) for setting in SPEC["settings"]}},
                "requirements": requirements, "python": sys.version,
                "pip_freeze": FROZEN, "installation_seconds": time.perf_counter() - install_started}}
(WORK / "setup_report.json").write_text(json.dumps(SETUP_REPORT, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print("Verified frozen seed0, compact seed1/seed2 inputs, tuning checkpoints, source hashes, and exact environment.", flush=True)
'''


def run_source(family, surface):
    return f'''import json
import os
import subprocess
import time

FAMILY = {family!r}
SURFACE = {surface!r}
summary_path = WORK / "batch_summary.json"
summary = {{"purpose": "JRA matched-seed train-only refits", "family": FAMILY,
           "surface": SURFACE, "settings": list(SPEC["settings"]), "seeds": [0, 1, 2], "runs": []}}
for setting in SPEC["settings"]:
    for seed in (0, 1, 2):
        started = time.time()
        print(f"starting {{FAMILY}}/{{SURFACE}}/{{setting}} seed{{seed}}", flush=True)
        command = [str(PYTHON), str(SOURCE_DIR / "run_experiments.py"),
                   "--mode", "scientific", "--family", FAMILY, "--surface", SURFACE,
                   "--setting", setting, "--stage", "refit", "--seed", str(seed),
                   "--input", INPUTS[seed]["input"], "--spec", str(SPEC_PATH),
                   "--output", str(RUN_ROOT)]
        env = os.environ.copy()
        env.update(PYTHONHASHSEED="0", MPLBACKEND="Agg", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        completed = subprocess.run(command, cwd=str(SOURCE_DIR), env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(completed.stdout, end="", flush=True)
        record = {{"setting": setting, "seed": seed, "returncode": completed.returncode,
                  "seconds": time.time() - started,
                  "status": "complete" if completed.returncode == 0 else "failed"}}
        summary["runs"].append(record)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
        if completed.returncode:
            raise subprocess.CalledProcessError(completed.returncode, command)
        print(f"complete {{FAMILY}}/{{SURFACE}}/{{setting}} seed{{seed}}", flush=True)
summary["status"] = "complete"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print("matched-seed refit batch complete:", FAMILY, SURFACE, len(summary["runs"]), "runs", flush=True)
'''


def build_notebook(family, surface, embedded, source_hashes, spec):
    setup = setup_source(embedded, source_hashes, spec, family, surface)
    run = run_source(family, surface)
    ast.parse(setup, filename="setup.py")
    ast.parse(run, filename="run.py")
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                          "language_info": {"name": "python"}},
            "cells": [
                {"cell_type": "markdown", "id": "purpose", "metadata": {},
                 "source": [f"# JRA matched-seed refits: {family} / {surface}\\n\\n",
                             "Private Kaggle CPU run. This uses verified seed-0 tuning selections and runs fixed-round train-only refits for seeds 0, 1, and 2 across the ten declared settings. Test rows are never opened.\\n"]},
                {"cell_type": "code", "id": "setup", "metadata": {}, "source": setup.splitlines(keepends=True),
                 "execution_count": None, "outputs": []},
                {"cell_type": "code", "id": "run", "metadata": {}, "source": run.splitlines(keepends=True),
                 "execution_count": None, "outputs": []}]}
