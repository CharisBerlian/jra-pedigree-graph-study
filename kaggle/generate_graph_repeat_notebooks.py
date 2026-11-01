"""Generate and verify the fixed-k graph-repeat Kaggle notebooks."""
import ast
import copy
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE_NOTEBOOK = HERE / "node2vec_production.ipynb"
RUNNER = HERE / "graph_production.py"
FROZEN_SCRIPT_SHA256 = "3a2c4e77bd70af6059a2854b601ac84a47de8ed752972bde656b97bbd162d59f"
VARIANTS = {1: HERE / "node2vec_production_seed1.ipynb",
            2: HERE / "node2vec_production_seed2.ipynb"}


def _embedded_script(notebook):
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "PRODUCTION_SOURCE" not in source:
            continue
        tree = ast.parse(source, filename="production-run-cell.py")
        assignment = next(node for node in tree.body
                           if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name)
                                   and target.id == "PRODUCTION_SOURCE"
                                   for target in node.targets))
        return ast.literal_eval(assignment.value), cell
    raise AssertionError("PRODUCTION_SOURCE assignment not found")


def _check_notebook(notebook, seed):
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "kaggle-cell.py", "exec")
    embedded, _ = _embedded_script(notebook)
    runner = RUNNER.read_text(encoding="utf-8")
    assert embedded == runner
    assert hashlib.sha256(embedded.encode("utf-8")).hexdigest() == FROZEN_SCRIPT_SHA256
    setup = next(cell for cell in notebook["cells"] if cell.get("id") == "setup")
    setup_source = "".join(setup["source"])
    assert f"SEED, SELECTED_K = {seed}, 64" in setup_source
    assert "f'/kaggle/working/jra-production/seed-{SEED}'" in setup_source
    markdown = next(cell for cell in notebook["cells"] if cell.get("id") == "purpose")
    assert f"seed-{seed}" in "".join(markdown["source"])


def build(seed):
    notebook = copy.deepcopy(json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8")))
    markdown = next(cell for cell in notebook["cells"] if cell.get("id") == "purpose")
    markdown["source"] = [
        "# JRA pedigree: production graph repeat on Kaggle CPU\n",
        "Attach the private charischerw/jra-pedigree-pilot-v1 dataset. CPU (no accelerator), Internet On. Save Version -> Save & Run All.\n",
        "\n",
        f"Frozen settings: 64 dimensions, 10 walks/node, 80 steps/walk, window 5, five epochs, p=1, q=0.5, four workers. This repeat uses Node2Vec seed {seed} and fixed selected k=64 from the completed seed-0 run.\n",
        "\n",
        f"Only the verified training-root three-generation pedigree graph is used; no race outcomes or validation/test root lists are inputs. Outputs: /kaggle/working/jra-production/seed-{seed}. These are representation artifacts, not race-prediction findings. The complete tested runner is embedded below.\n",
    ]
    setup = next(cell for cell in notebook["cells"] if cell.get("id") == "setup")
    setup_source = "".join(setup["source"])
    setup_source = setup_source.replace("SEED, SELECTED_K = 0, None",
                                        f"SEED, SELECTED_K = {seed}, 64", 1)
    setup["source"] = setup_source.splitlines(keepends=True)
    run_cell = next(cell for cell in notebook["cells"] if cell.get("id") == "production-run")
    run_source = "".join(run_cell["source"])
    marker = "SCRIPT = Path('/kaggle/working/graph_production.py')\n"
    assertion = (f"FROZEN_SCRIPT_SHA256 = '{FROZEN_SCRIPT_SHA256}'\n"
                 "assert hashlib.sha256(PRODUCTION_SOURCE.encode('utf-8')).hexdigest() == FROZEN_SCRIPT_SHA256\n")
    assert run_source.count(marker) == 1
    run_source = run_source.replace(marker, assertion + marker, 1)
    run_cell["source"] = run_source.splitlines(keepends=True)
    _check_notebook(notebook, seed)
    VARIANTS[seed].write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


def check():
    original = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    embedded, _ = _embedded_script(original)
    assert embedded == RUNNER.read_text(encoding="utf-8")
    assert hashlib.sha256(embedded.encode("utf-8")).hexdigest() == FROZEN_SCRIPT_SHA256
    for seed, path in VARIANTS.items():
        notebook = json.loads(path.read_text(encoding="utf-8"))
        _check_notebook(notebook, seed)
    print("PASS: graph repeat notebooks compile after Kaggle-style source joining, "
          "use seeds 1/2 with fixed k=64, and preserve the frozen runner hash")


if __name__ == "__main__":
    for seed in VARIANTS:
        build(seed)
    check()
