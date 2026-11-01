"""No-fit structural checks for scientific tuning notebook generation."""
import ast
import hashlib
import json
from pathlib import Path

import generate_experiment_notebooks as generator


def check():
    spec, embedded, hashes = generator.load_inputs()
    paths = sorted(generator.OUTPUT.glob("experiment_tune_*.ipynb"))
    assert len(paths) == 6
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]
                if cell["cell_type"] == "code"}
        assert set(code) == {"setup", "run"}
        ast.parse(code["setup"], filename=str(path))
        ast.parse(code["run"], filename=str(path))
        assert "train_validation.csv.gz" in code["setup"]
        assert "INPUT_RECORD" in code["setup"]
        assert "98afbbdea692da9e46e723af27e4f132a14a59d59fb36e650513dc02480b4fa4" in code["setup"]
        assert "frozen compressed train-validation checksum mismatch" in code["setup"]
        assert "frozen extracted train-validation checksum mismatch" in code["setup"]
        assert '"--stage", "tune"' in code["run"]
        assert '"--seed", "0"' in code["run"]
        assert '"--stage", "refit"' not in code["run"]
        assert 'MPLBACKEND="Agg"' in code["run"]
        assert 'rglob("bundle_manifest.json")' not in code["setup"]
        assert all(hashlib.sha256(source.encode()).hexdigest() == hashes[name]
                   for name, source in embedded.items())
    assert tuple(spec["settings"]) == ("B1", "S1", "S2", "S3", "S4", "S5", "S6", "M2", "M3", "S7")
    print("PASS: six scientific family-surface notebooks, exact embedded hashes, frozen seed-0 checks, and tune-only flow")


if __name__ == "__main__":
    check()
