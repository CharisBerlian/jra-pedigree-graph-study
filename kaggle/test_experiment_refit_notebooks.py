"""No-fit structural checks for matched-seed refit notebooks."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

import generate_experiment_refit_notebooks as generator
from experiment_refit_notebook_template import setup_source


def check():
    spec, embedded, hashes = generator.load_inputs()
    paths = sorted(generator.HERE.glob("experiment_refit_*.ipynb"))
    assert len(paths) == 6
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = {cell["id"]: "".join(cell["source"])
                for cell in notebook["cells"] if cell["cell_type"] == "code"}
        assert set(code) == {"setup", "run"}
        ast.parse(code["setup"], filename=str(path))
        ast.parse(code["run"], filename=str(path))
        assert "find_tune_seed0" in code["setup"]
        assert 'parents[1].name != "tune"' in code["setup"]
        assert '"--stage", "refit"' in code["run"]
        assert '"--seed", str(seed)' in code["run"]
        assert '"--stage", "tune"' not in code["run"]
        assert "MPLBACKEND=\"Agg\"" in code["run"]
    assert spec["matched_seeds"] == [0, 1, 2]
    assert set(embedded) == set(hashes)
    print("PASS: six matched-seed refit notebooks, verified tune paths, seed 0/1/2 flow, and tune-only exclusion")


def _runtime_functions(root, frozen_hashes=None):
    spec, embedded, hashes = generator.load_inputs()
    source = setup_source(embedded, hashes, spec, "xgboost", "turf")
    if frozen_hashes:
        source = source.replace(
            "98afbbdea692da9e46e723af27e4f132a14a59d59fb36e650513dc02480b4fa4",
            frozen_hashes["gzip"])
        source = source.replace(
            "d395a1802d90b0c09cc71f08b2dcbc83ca4020654797c9442969833c103ac12d",
            frozen_hashes["raw"])
    tree = ast.parse(source)
    names = {"sha256_file", "csv_uncompressed_sha256", "metadata_path",
             "input_candidates", "prepare_seed_input"}
    definitions = [node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name in names]
    mount = root / "mounted"
    mount.mkdir()

    path_type = type(Path())

    class TestPath(path_type):
        def symlink_to(self, target, target_is_directory=False):
            self.write_bytes(Path(target).read_bytes())

    def path_proxy(value):
        return TestPath(mount) if str(value) == "/kaggle/input" else TestPath(value)

    namespace = {"Path": path_proxy, "gzip": gzip, "hashlib": hashlib,
                 "json": json, "shutil": __import__("shutil"),
                 "WORK": TestPath(root / "work")}
    exec(compile(ast.Module(body=definitions, type_ignores=[]),
                 "refit_input_helpers", "exec"), namespace)
    return namespace, mount


def _write_sidecar(csv_path, seed, compressed=False, data_sha_override=None,
                   uncompressed_sha_override=None):
    raw = csv_path.read_bytes()
    if compressed:
        with gzip.open(csv_path, "rb") as stream:
            uncompressed_sha = hashlib.sha256(stream.read()).hexdigest()
        data_sha = hashlib.sha256(raw).hexdigest()
    else:
        uncompressed_sha = hashlib.sha256(raw).hexdigest()
        data_sha = "not-used-for-raw-source"
    if data_sha_override is not None:
        data_sha = data_sha_override
    if uncompressed_sha_override is not None:
        uncompressed_sha = uncompressed_sha_override
    sidecar = (csv_path.with_name("train_validation.csv.json")
               if csv_path.suffix != ".gz" else csv_path.with_suffix(".json"))
    sidecar.write_text(json.dumps({
        "purpose": "JRA train-validation prediction inputs",
        "splits": ["train", "validation"], "seed": seed,
        "data_sha256": data_sha,
        "csv_stats": {"uncompressed": {"sha256": uncompressed_sha}},
    }), encoding="utf-8")


def check_input_normalization():
    payload = b"race_id,split\n1,train\n"
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        frozen = {"gzip": hashlib.sha256(b"frozen-gzip").hexdigest(),
                  "raw": hashlib.sha256(payload).hexdigest()}
        functions, mount = _runtime_functions(root, frozen)
        raw_path = mount / "train_validation.csv"
        raw_path.write_bytes(payload)
        _write_sidecar(raw_path, seed=0, data_sha_override=frozen["gzip"],
                       uncompressed_sha_override=frozen["raw"])
        record = functions["prepare_seed_input"](0)
        with gzip.open(record["input"], "rb") as stream:
            assert stream.read() == payload

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        functions, mount = _runtime_functions(root)
        raw_path = mount / "seed1" / "train_validation.csv"
        raw_path.parent.mkdir()
        raw_path.write_bytes(payload)
        _write_sidecar(raw_path, seed=1)
        record = functions["prepare_seed_input"](1)
        assert Path(record["input"]).is_file()
        with gzip.open(record["input"], "rb") as stream:
            assert stream.read() == payload

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        functions, mount = _runtime_functions(root)
        gzip_path = mount / "seed2" / "train_validation.csv.gz"
        gzip_path.parent.mkdir()
        with gzip.GzipFile(filename=str(gzip_path), mode="wb",
                           compresslevel=1, mtime=0) as stream:
            stream.write(payload)
        _write_sidecar(gzip_path, seed=2, compressed=True)
        record = functions["prepare_seed_input"](2)
        assert Path(record["input"]).is_file()
        with gzip.open(record["input"], "rb") as stream:
            assert stream.read() == payload
        assert record["uncompressed_sha256"] == hashlib.sha256(payload).hexdigest()

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        functions, mount = _runtime_functions(root)
        bad_path = mount / "seed1" / "train_validation.csv"
        bad_path.parent.mkdir()
        bad_path.write_bytes(payload)
        _write_sidecar(bad_path, seed=2)
        try:
            functions["prepare_seed_input"](1)
        except ValueError as error:
            assert "explicitly seed-bound" in str(error)
        else:
            raise AssertionError("seed-bound input metadata was not enforced")
    print("PASS: refit input resolver normalizes raw/gzip sources and enforces seed metadata")


if __name__ == "__main__":
    check()
    check_input_normalization()
