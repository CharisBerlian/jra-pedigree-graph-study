"""Generate the Kaggle-only native-model API smoke notebook."""
import ast
import copy
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PLAN = HERE.parent / "plans" / "experiment_spec_2026-09-13.json"
RUNNER = HERE / "run_experiments.py"
OUTPUT = HERE / "experiment_api_smoke.ipynb"
EMBED_FILES = ("run_experiments.py", "experiment_features.py",
               "prepare_prediction.py", "graph_pilot.py")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def smoke_spec():
    spec = copy.deepcopy(json.loads(PLAN.read_text(encoding="utf-8")))
    spec["spec_id"] = "experiment_spec_2026-09-13_smoke"
    spec["status"] = "smoke_only"
    spec["test_inspected"] = False
    spec["tuning"]["max_rounds"] = 8
    spec["tuning"]["patience"] = 3
    spec["tuning"]["candidates_per_family_setting_surface"] = 4
    return spec


def _runner_preflight(source):
    required = ("def run(", "def fit_model(", "def fit_model_fixed(",
                "def predict_model(", "def _load_spec(path, mode)",
                '"--mode"', '"smoke"')
    missing = [marker for marker in required if marker not in source]
    if missing:
        raise RuntimeError("run_experiments.py smoke API is incomplete: " + ", ".join(missing))
    ast.parse(source, filename="run_experiments.py")


def _synthetic_rows_source():
    return r'''import csv
import hashlib
from pathlib import Path

rows = []
fields = ["race_id", "horse_id_netkeiba", "horse_number", "split", "label", "surface",
          "venue", "distance", "going", "race_class", "field_size", "bracket",
          "weight_carried", "age", "sex", "body_weight", "body_weight_change"]
for split, count in (("train", 40), ("validation", 16)):
    for race_number in range(count):
        race_id = f"smoke-{split}-{race_number}"
        for horse_number in range(1, 5):
            if split == "train":
                venue = "rare" if race_number == 0 and horse_number == 1 else "V"
                going = "G"
            else:
                venue = "UNSEEN" if horse_number == 1 else ("" if horse_number == 2 else "V")
                going = "" if horse_number == 2 else "G"
            rows.append({
                "race_id": race_id, "horse_id_netkeiba": f"{race_id}-h{horse_number}",
                "horse_number": str(horse_number), "split": split,
                "label": int(horse_number in (1, 2)), "surface": "turf",
                "venue": venue, "distance": str(1200 + 100 * race_number),
                "going": going, "race_class": "C", "field_size": "4",
                "bracket": str((horse_number + 1) // 2), "weight_carried": "55",
                "age": str(3 + horse_number % 3), "sex": "M" if horse_number % 2 else "F",
                "body_weight": "" if split == "validation" and horse_number == 2 else str(450 + horse_number),
                "body_weight_change": "0",
            })
path = WORK / "smoke.csv"
with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
data_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
(WORK / "smoke.json").write_text(json.dumps(
    {"data_sha256": data_sha256, "seed": 0,
     "splits": ["train", "validation"],
     "purpose": "experiment_api_smoke_only", "rows": len(rows)},
    indent=2, sort_keys=True), encoding="utf-8")
assert len(rows) == 224 and any(row["body_weight"] == "" for row in rows)
assert sum(row["venue"] == "rare" for row in rows if row["split"] == "train") == 1
smoke_metadata = json.loads((WORK / "smoke.json").read_text(encoding="utf-8"))
assert smoke_metadata["data_sha256"] == data_sha256
assert smoke_metadata["seed"] == 0
assert smoke_metadata["splits"] == ["train", "validation"]
print("synthetic rows:", len(rows), "train:", sum(row["split"] == "train" for row in rows))
'''


def _notebook_sources():
    runner_source = RUNNER.read_text(encoding="utf-8")
    _runner_preflight(runner_source)
    embedded = {name: (HERE / name).read_text(encoding="utf-8") for name in EMBED_FILES}
    hashes = {name: sha256_bytes(source.encode("utf-8")) for name, source in embedded.items()}
    spec = smoke_spec()
    setup = f'''import hashlib, json, os, subprocess, sys
from pathlib import Path

assert Path("/kaggle").is_dir(), "Run this notebook on Kaggle only."
os.environ["MPLBACKEND"] = "Agg"
WORK = Path("/kaggle/working/experiment-api-smoke")
WORK.mkdir(parents=True, exist_ok=True)
EMBEDDED = {embedded!r}
EXPECTED_HASHES = {hashes!r}
for name, source in EMBEDDED.items():
    target = WORK / name
    target.write_text(source, encoding="utf-8")
    assert hashlib.sha256(target.read_bytes()).hexdigest() == EXPECTED_HASHES[name]
    compile(source, name, "exec")
assert EXPECTED_HASHES["run_experiments.py"] == "{hashes['run_experiments.py']}"
assert EXPECTED_HASHES["experiment_features.py"] == "{hashes['experiment_features.py']}"
assert EXPECTED_HASHES["prepare_prediction.py"] == "{hashes['prepare_prediction.py']}"
assert EXPECTED_HASHES["graph_pilot.py"] == "{hashes['graph_pilot.py']}"
SMOKE_SPEC = {spec!r}
SPEC_PATH = WORK / "experiment_spec_smoke.json"
SPEC_PATH.write_text(json.dumps(SMOKE_SPEC, indent=2, sort_keys=True), encoding="utf-8")
requirements = WORK / "requirements-smoke.txt"
requirements.write_text("\\n".join(("numpy==1.26.4", "scipy==1.13.1", "scikit-learn==1.6.1",
                                    "pandas==2.2.3", "lightgbm==4.6.0", "xgboost==3.0.2",
                                    "catboost==1.2.8")) + "\\n", encoding="utf-8")
venv = Path("/kaggle/tmp/experiment-api-smoke-venv")
python = venv / "bin/python"
subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
pip = [sys.executable, "-m", "pip", "--python", str(python)]
subprocess.run(pip + ["install", "--disable-pip-version-check", "-r", str(requirements)], check=True)
subprocess.run(pip + ["check"], check=True)
(WORK / "environment.txt").write_text(subprocess.check_output(pip + ["freeze"], text=True), encoding="utf-8")
setup_report = {{"smoke_only": True, "spec_id": SMOKE_SPEC["spec_id"],
                 "source_sha256": EXPECTED_HASHES,
                 "spec_sha256": hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(),
                 "runner_api": "run_experiments.py --mode smoke"}}
(WORK / "setup_report.json").write_text(json.dumps(setup_report, indent=2, sort_keys=True), encoding="utf-8")
'''
    smoke_rows = _synthetic_rows_source()
    reload_source = r"""import csv, gzip, json, os, sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
import run_experiments as runner

family, input_path, refit_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
rows, columns = runner.read_rows(input_path)
preprocessing = json.loads((refit_dir / "preprocessing.json").read_text())
selected = tuple(preprocessing["features"])
maps = preprocessing["category_maps"]
validation = [row for row in rows if row["split"] == "validation"]
frame = runner.prepare_values(validation, selected, maps, family)
if family == "xgboost":
    import xgboost as xgb
    model = xgb.Booster()
    model.load_model(str(refit_dir / "model.json"))
elif family == "lightgbm":
    import lightgbm as lgb
    model = lgb.Booster(model_file=str(refit_dir / "model.txt"))
else:
    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(str(refit_dir / "model.cbm"))
scores = (model.predict(frame) if family == "lightgbm"
          else runner.predict_model(family, model, frame))
with gzip.open(refit_dir / "validation_predictions.csv.gz", "rt", encoding="utf-8", newline="") as stream:
    saved = [float(row["prediction"]) for row in csv.DictReader(stream)]
assert len(saved) == len(scores) and max(abs(float(a) - b) for a, b in zip(saved, scores)) < 1e-12
print("reload PASS", family, len(scores))
"""
    run = f'''import json, subprocess

exec(SYNTHETIC_ROWS)
runner = WORK / "run_experiments.py"
python = Path("/kaggle/tmp/experiment-api-smoke-venv/bin/python")
input_path = WORK / "smoke.csv"
output = WORK / "runs"
seed_inputs = {{}}
for seed in (0, 1, 2):
    seed_input = WORK / f"smoke-seed{{seed}}.csv"
    seed_input.write_bytes(input_path.read_bytes())
    (WORK / f"smoke-seed{{seed}}.json").write_text(json.dumps(
        {{"data_sha256": hashlib.sha256(seed_input.read_bytes()).hexdigest(),
          "splits": ["train", "validation"], "seed": seed,
          "purpose": "experiment_api_smoke_only"}}), encoding="utf-8")
    seed_inputs[seed] = seed_input
for family in ("xgboost", "lightgbm", "catboost"):
    common = [str(python), str(runner), "--mode", "smoke", "--family", family,
              "--surface", "turf", "--setting", "S1", "--seed", "0",
              "--input", str(seed_inputs[0]), "--spec", str(SPEC_PATH), "--output", str(output)]
    subprocess.run(common + ["--stage", "tune"], check=True, cwd=str(WORK))
    for seed in (0, 1, 2):
        seeded = list(common)
        seeded[seeded.index("--seed") + 1] = str(seed)
        seeded[seeded.index("--input") + 1] = str(seed_inputs[seed])
        subprocess.run(seeded + ["--stage", "refit"], check=True, cwd=str(WORK))
        second = subprocess.run(seeded + ["--stage", "refit"], check=True, cwd=str(WORK),
                                capture_output=True, text=True)
        assert json.loads(second.stdout)["status"] == "skipped"
    tune = output / "smoke" / family / "turf" / "S1" / "tune" / "seed0"
    refit = output / "smoke" / family / "turf" / "S1" / "refit" / "seed0"
    assert json.loads((tune / "complete.json").read_text())["status"] == "complete"
    assert json.loads((refit / "complete.json").read_text())["status"] == "complete"
    assert (refit / "validation_predictions.csv.gz").is_file()
    preprocessing = json.loads((tune / "preprocessing.json").read_text())
    assert "venue" in preprocessing["categorical_columns"]
    mapping = preprocessing["category_maps"]["venue"]
    assert mapping["__MISSING__"] == 0 and mapping["__UNK__"] == 1 and mapping["__RARE__"] == 2
    assert mapping["rare"] == mapping["__RARE__"] and mapping["V"] >= 3
    candidate = json.loads((tune / "candidates" / "c0" / "metrics.json").read_text())
    assert candidate["status"] == "complete" and candidate["best_round"] >= 1
    curve = json.loads((tune / "candidates" / "c0" / "ap_curve.json").read_text())
    assert curve["metric"] == "sklearn.metrics.average_precision_score" and curve["values"]
    assert (output / "smoke" / family / "turf" / "S1" / "refit" / "seed1" / "complete.json").is_file()
    assert (output / "smoke" / family / "turf" / "S1" / "refit" / "seed2" / "complete.json").is_file()

RELOAD_SOURCE = {reload_source!r}
reload_path = WORK / "reload_check.py"
reload_path.write_text(RELOAD_SOURCE, encoding="utf-8")
for family in ("xgboost", "lightgbm", "catboost"):
    refit = output / "smoke" / family / "turf" / "S1" / "refit" / "seed0"
    subprocess.run([str(python), str(reload_path), family, str(input_path), str(refit)], check=True, cwd=str(WORK))
print("PASS: smoke spec, embedded hashes, three native families, custom AP curves, rare/unknown/missing categories, numeric missingness, and refit persistence")
'''
    return setup, smoke_rows, run


def build():
    setup, synthetic, run = _notebook_sources()
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                              "language_info": {"name": "python"}},
                "cells": [
                    {"cell_type": "markdown", "id": "purpose", "metadata": {},
                     "source": ["# JRA experiment runner API smoke test\n",
                                 "Kaggle-only smoke run. This uses a copied smoke specification and synthetic complete races; it is not a scientific fit and must never use real or test data.\n",
                                 "The notebook embeds and hashes run_experiments.py, experiment_features.py, prepare_prediction.py, and graph_pilot.py, installs pinned wheels in an isolated venv, then tunes/refits S1 for all three native families.\n"]},
                    {"cell_type": "code", "id": "setup", "metadata": {}, "source": setup.splitlines(keepends=True),
                     "execution_count": None, "outputs": []},
                    {"cell_type": "code", "id": "synthetic", "metadata": {},
                     "source": ["import json\n", "SYNTHETIC_ROWS = " + repr(synthetic) + "\n"],
                     "execution_count": None, "outputs": []},
                    {"cell_type": "code", "id": "run", "metadata": {}, "source": run.splitlines(keepends=True),
                     "execution_count": None, "outputs": []},
                ]}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{cell['id']}.py")
    OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    build()
