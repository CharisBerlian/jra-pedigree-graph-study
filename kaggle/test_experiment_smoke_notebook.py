"""No-fit structural check for the Kaggle API smoke notebook."""
import ast
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "experiment_api_smoke.ipynb"


def _assignment(source, name):
    tree = ast.parse(source)
    node = next(node for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name
                        for target in node.targets))
    return ast.literal_eval(node.value)


def check():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = {cell["id"]: cell for cell in notebook["cells"]}
    assert set(cells) == {"purpose", "setup", "synthetic", "run"}
    for cell in cells.values():
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{cell['id']}.py")
    setup = "".join(cells["setup"]["source"])
    embedded = _assignment(setup, "EMBEDDED")
    expected = _assignment(setup, "EXPECTED_HASHES")
    for name, source in embedded.items():
        local = (HERE / name).read_bytes()
        assert hashlib.sha256(local).hexdigest() == expected[name]
        assert hashlib.sha256(source.encode("utf-8")).hexdigest() == expected[name]
    run = "".join(cells["run"]["source"])
    assert '"--mode", "smoke"' in run
    assert '"--stage", "tune"' in run and '"--stage", "refit"' in run
    assert '"xgboost", "lightgbm", "catboost"' in run
    assert 'os.environ["MPLBACKEND"] = "Agg"' in setup
    assert 'os.environ["MPLBACKEND"] = "Agg"' in run
    assert "validation_predictions.csv.gz" in run
    assert "status\"] == \"skipped\"" in run
    assert "RELOAD_SOURCE" in run and "load_model" in run
    assert "for seed in (0, 1, 2)" in run
    synthetic = "".join(cells["synthetic"]["source"])
    assert '"data_sha256": data_sha256' in synthetic
    assert '"seed": 0' in synthetic
    assert 'smoke_metadata["splits"] == ["train", "validation"]' in synthetic
    assert 'seed_inputs[seed] = seed_input' in run
    assert '"data_sha256": hashlib.sha256(seed_input.read_bytes()).hexdigest()' in run
    assert 'seeded[seeded.index("--input") + 1] = str(seed_inputs[seed])' in run
    print("PASS: smoke notebook cells, embedded source hashes, smoke mode, three-family calls, and refit persistence checks")


if __name__ == "__main__":
    check()
