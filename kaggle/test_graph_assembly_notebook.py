"""No-fit structural check for the Kaggle graph assembly notebook."""
import ast
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "graph_table_assembly.ipynb"


def check():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = {cell["id"]: cell for cell in notebook["cells"]}
    assert set(cells) == {"purpose", "setup", "assemble"}
    for cell in cells.values():
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{cell['id']}.py")
    setup = "".join(cells["setup"]["source"])
    assert "HELPER_SOURCE" in setup and "HELPER_SHA256" in setup
    assert "FROZEN_BASE_SHA256" in setup and "FROZEN_BASE_UNCOMPRESSED" in setup
    assert "normalize(" in setup and "train_validation.csv.gz" in setup
    assemble = "".join(cells["assemble"]["source"])
    assert "for seed in (1, 2)" in assemble
    assert "verify_graph_metadata" in assemble and "source_base_sha256" in assemble
    assert "assembled_non_graph_sha256" in assemble
    assert 'WORK / f"seed{seed}" / "train_validation.csv.gz"' in assemble
    helper = (HERE / "assemble_prediction_seed.py").read_bytes()
    expected = ast.literal_eval(next(node.value for node in ast.parse(setup).body
                                     if isinstance(node, ast.Assign) and
                                     any(getattr(target, "id", None) == "HELPER_SHA256"
                                         for target in node.targets)))
    assert hashlib.sha256(helper).hexdigest() == expected
    print("PASS: assembly notebook embeds helper hash, handles raw/gzip inputs, checks provenance, and assembles seeds 1/2")


if __name__ == "__main__":
    check()
