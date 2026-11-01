"""Generate the six private scientific seed-0 tuning notebooks."""
import ast
import hashlib
import json
from pathlib import Path

from experiment_notebook_template import build_notebook


HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE.parent / "plans" / "experiment_spec_2026-09-13.json"
OUTPUT = HERE
SOURCE_FILES = ("run_experiments.py", "experiment_features.py", "prepare_prediction.py", "graph_pilot.py")
FAMILIES = ("xgboost", "lightgbm", "catboost")
SURFACES = ("turf", "dirt")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if spec.get("spec_id") != "experiment_spec_2026-09-13":
        raise ValueError("unexpected scientific specification")
    if tuple(spec.get("families", ())) != FAMILIES or tuple(spec.get("settings", ())) != (
            "B1", "S1", "S2", "S3", "S4", "S5", "S6", "M2", "M3", "S7"):
        raise ValueError("scientific specification family/setting matrix changed")
    if spec.get("tuning", {}).get("tuned_seed") != 0:
        raise ValueError("scientific tuning must use seed 0")
    embedded = {}
    for name in SOURCE_FILES:
        path = HERE / name
        if not path.is_file():
            raise FileNotFoundError(path)
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=name)
        embedded[name] = source
    hashes = {name: sha256_file(HERE / name) for name in SOURCE_FILES}
    return spec, embedded, hashes


def build():
    spec, embedded, hashes = load_inputs()
    written = []
    for family in FAMILIES:
        for surface in SURFACES:
            path = OUTPUT / f"experiment_tune_{family}_{surface}.ipynb"
            notebook = build_notebook(family, surface, embedded, hashes, spec)
            path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
            written.append(path)
    print("Wrote", len(written), "scientific tuning notebooks:")
    for path in written:
        print(" ", path)


if __name__ == "__main__":
    build()
