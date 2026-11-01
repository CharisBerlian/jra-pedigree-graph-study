"""Generate six private matched-seed refit notebooks."""
import ast
import hashlib
import json
from pathlib import Path

from experiment_refit_notebook_template import build_notebook


HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE.parent / "plans" / "experiment_spec_2026-09-13.json"
SOURCE_FILES = ("run_experiments.py", "experiment_features.py", "prepare_prediction.py", "graph_pilot.py")
FAMILIES = ("xgboost", "lightgbm", "catboost")
SURFACES = ("turf", "dirt")


def load_inputs():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if spec.get("spec_id") != "experiment_spec_2026-09-13":
        raise ValueError("unexpected scientific specification")
    if tuple(spec.get("families", ())) != FAMILIES:
        raise ValueError("scientific specification family matrix changed")
    if tuple(spec.get("settings", ())) != ("B1", "S1", "S2", "S3", "S4", "S5", "S6", "M2", "M3", "S7"):
        raise ValueError("scientific specification setting matrix changed")
    if spec.get("matched_seeds") != [0, 1, 2]:
        raise ValueError("matched seed policy changed")
    embedded = {}
    for name in SOURCE_FILES:
        path = HERE / name
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=name)
        embedded[name] = source
    hashes = {name: hashlib.sha256(embedded[name].encode()).hexdigest() for name in SOURCE_FILES}
    return spec, embedded, hashes


def build():
    spec, embedded, hashes = load_inputs()
    for family in FAMILIES:
        for surface in SURFACES:
            path = HERE / f"experiment_refit_{family}_{surface}.ipynb"
            notebook = build_notebook(family, surface, embedded, hashes, spec)
            path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
            print("Wrote", path)


if __name__ == "__main__":
    build()
