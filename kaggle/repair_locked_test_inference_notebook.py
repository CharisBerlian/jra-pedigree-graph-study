"""Repair newline escapes in the generated locked-inference notebook."""

import json
import base64
import ast
import hashlib
from pathlib import Path
import sys


def repair(path):
    path = Path(path)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = [cell for cell in notebook.get("cells", ()) if cell.get("cell_type") == "code"]
    if len(cells) != 1:
        raise ValueError("expected one code cell")
    code = "".join(cells[0].get("source", ()))
    replacements = {
        "}\\nfor name, encoded in EMBEDDED.items():":
            "}\nfor name, encoded in EMBEDDED.items():",
        "))\\nSPEC = SOURCE_ROOT / \"experiment_spec_2026-09-13.json\"":
            "))\nSPEC = SOURCE_ROOT / \"experiment_spec_2026-09-13.json\"",
    }
    for old, new in replacements.items():
        if old in code:
            if code.count(old) != 1:
                raise ValueError(f"expected exactly one generated escape: {old!r}")
            code = code.replace(old, new)
        elif new not in code:
            raise ValueError(f"generated escape is missing: {old!r}")
    sidecar_old = '    sidecar = path.with_suffix(".json")\n    assert sidecar.is_file(), sidecar'
    sidecar_new = ('    sidecars = [path.with_suffix(".json"), Path(str(path) + ".json")]\n'
                   '    sidecars = [candidate for candidate in sidecars if candidate.is_file()]\n'
                   '    if len(sidecars) != 1:\n'
                   '        raise AssertionError(f"expected one test-input sidecar for {path}: {sidecars}")\n'
                   '    sidecar = sidecars[0]')
    if sidecar_old in code:
        if code.count(sidecar_old) != 1:
            raise ValueError("expected one test-input sidecar block")
        code = code.replace(sidecar_old, sidecar_new)
    elif sidecar_new not in code:
        raise ValueError("test-input sidecar block is missing")
    stage_old = '''def stage_roots(stage, family, surface):
    marker = "jra-experiment-tuning" if stage == "tune" else "jra-experiment-refits"
    candidates = []
    for marker_path in input_root.rglob(marker):
        candidate = marker_path / "runs" / "scientific" / family / surface
        if candidate.is_dir():
            candidates.append(candidate)
    if len(candidates) != 1:
        raise RuntimeError(f"expected one {stage} mount for {family}/{surface}, found {len(candidates)}: {candidates}")
    return candidates[0]'''
    stage_new = '''def stage_roots(stage, family, surface):
    marker = "jra-experiment-tuning" if stage == "tune" else "jra-experiment-refits"
    candidates = []
    for marker_path in input_root.rglob(marker):
        candidate = marker_path / "runs" / "scientific" / family / surface
        if candidate.is_dir():
            candidates.append(candidate)
    if stage == "tune":
        for mount in sorted(input_root.iterdir()):
            candidate = mount / "runs" / "scientific" / family / surface
            if candidate.is_dir():
                candidates.append(candidate)
    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    if len(unique) != 1:
        raise RuntimeError(f"expected one {stage} mount for {family}/{surface}, found {len(unique)}: {unique}")
    return unique[0]'''
    if stage_old not in code:
        stage_intermediate = '''def stage_roots(stage, family, surface):
    marker = "jra-experiment-tuning" if stage == "tune" else "jra-experiment-refits"
    candidates = []
    for marker_path in input_root.rglob(marker):
        candidate = marker_path / "runs" / "scientific" / family / surface
        if candidate.is_dir():
            candidates.append(candidate)
    for mount in sorted(input_root.iterdir()):
        candidate = mount / "runs" / "scientific" / family / surface
        if candidate.is_dir():
            candidates.append(candidate)
    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    if len(unique) != 1:
        raise RuntimeError(f"expected one {stage} mount for {family}/{surface}, found {len(unique)}: {unique}")
    return unique[0]'''
        if stage_intermediate in code:
            code = code.replace(stage_intermediate, stage_new)
        elif stage_new not in code:
            raise ValueError("model stage-root resolver is missing")
    else:
        code = code.replace(stage_old, stage_new)
    pip_old = ('        subprocess.run(pip + ["install", "--disable-pip-version-check", "-r", "-"], '
                'input="\\n".join(requirements) + "\\n", text=True, check=True)')
    pip_new = ('        requirements_path = venv / "requirements.txt"\n'
               '        requirements_path.write_text("\\n".join(requirements) + "\\n", encoding="utf-8")\n'
               '        subprocess.run(pip + ["install", "--disable-pip-version-check", "-r", '
               'str(requirements_path)], check=True)')
    if pip_old in code:
        code = code.replace(pip_old, pip_new)
    elif pip_new not in code:
        raise ValueError("pinned-runtime install block is missing")
    lines = code.splitlines(keepends=True)
    embedded_index = next((index for index, line in enumerate(lines)
                           if line.startswith("EMBEDDED = ")), None)
    if embedded_index is None:
        raise ValueError("embedded source dictionary is missing")
    embedded = ast.literal_eval(lines[embedded_index][len("EMBEDDED = "):].strip())
    if not isinstance(embedded, dict):
        raise ValueError("embedded source dictionary is invalid")
    for name in ("prepare_prediction.py", "graph_pilot.py", "run_test_inference.py"):
        source_path = Path(__file__).with_name(name)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        embedded[name] = base64.b64encode(source_path.read_bytes()).decode("ascii")
    lines[embedded_index] = "EMBEDDED = " + repr(embedded) + "\n"
    inference_hash = hashlib.sha256(
        (Path(__file__).with_name("run_test_inference.py")).read_bytes()).hexdigest()
    hash_index = next((index for index, line in enumerate(lines)
                       if line.strip().startswith('"run_test_inference.py":')), None)
    if hash_index is None:
        raise ValueError("run_test_inference hash declaration is missing")
    lines[hash_index] = f'    "run_test_inference.py": "{inference_hash}".lower(),\n'
    code = "".join(lines)
    compile(code, str(path), "exec")
    cells[0]["source"] = code.splitlines(keepends=True)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: repair_locked_test_inference_notebook.py NOTEBOOK")
    repair(sys.argv[1])
    print("notebook_compile_ok")
