"""Small standard-library checks; never imports or runs ML libraries."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile

from graph_pilot import build_graph, load_parents, load_pilot_inputs, training_roots, write_json


def must_fail(action, text):
    try:
        action()
    except ValueError as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError("Expected rejection")


def check():
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        # Exercise a real pip-free environment, including repair of an existing
        # directory left by a failed ensurepip attempt. No ML or downloads.
        environment = folder / "pip-free-venv"
        for _ in range(2):
            subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True)
        environment_python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        identity = subprocess.check_output([str(environment_python), "-c",
            "import importlib.util,sys; assert sys.prefix != sys.base_prefix; assert importlib.util.find_spec('pip') is None; print(sys.prefix)"], text=True)
        assert Path(identity.strip()).resolve() == environment.resolve()
        empty_requirements = folder / "empty-requirements.txt"
        empty_requirements.write_text("", encoding="utf-8")
        pip_command = [sys.executable, "-m", "pip", "--python", str(environment_python)]
        subprocess.run(pip_command + ["install", "--no-index", "--disable-pip-version-check",
                                      "-r", str(empty_requirements)], check=True, capture_output=True)
        subprocess.run(pip_command + ["check"], check=True, capture_output=True)
        assert not subprocess.check_output(pip_command + ["freeze"], text=True).strip()
        source = folder / "races.db"
        con = sqlite3.connect(source)
        con.executescript("""
            CREATE TABLE races_r (race_id TEXT, race_date TEXT, race_type TEXT);
            CREATE TABLE runners_r (race_id TEXT, horse_id_netkeiba TEXT, result_status TEXT);
        """)
        cases = [("1", "20100101", "flat", "a", "finished"),
                 ("2", "20211231", "flat", "b", "中止"),
                 ("3", "20220101", "flat", "validation_only", "finished"),
                 ("4", "20240101", "flat", "test_only", "finished"),
                 ("5", "20160101", "jump", "jump_only", "finished"),
                 ("6", "20160101", "flat", "scratch", "取消"),
                 ("7", "20160101", "flat", "withdrawn", "除外"),
                 ("201009040210", "20100912", "flat", "excluded_race", "finished"),
                 ("8", "20091231", "flat", "too_early", "finished"),
                 ("9", "20210101", "flat", "c", "失格")]
        for race, date, kind, horse, status in cases:
            con.execute("INSERT INTO races_r VALUES (?,?,?)", (race, date, kind))
            con.execute("INSERT INTO runners_r VALUES (?,?,?)", (race, horse, status))
        con.commit()
        con.close()
        assert training_roots(source) == ["a", "b", "c"]

        parents, frontier = {}, ["a"]
        for depth in range(3):
            following = []
            for child in frontier:
                for relation in ("sire", "dam"):
                    parent = f"{child}_{relation}"
                    parents[child, relation] = parent
                    following.append(parent)
            frontier = following
        parents["a_sire_sire", "dam"] = parents["a_sire_sire", "sire"]  # pedigree collapse
        parents["future_only", "sire"] = "unused_parent"
        nodes, edges = build_graph(["a"], parents)
        assert "future_only" not in nodes and "unused_parent" not in nodes
        assert len(nodes) == 14 and len(edges) == 13
        broken = dict(parents)
        del broken["a", "dam"]
        must_fail(lambda: build_graph(["a"], broken), "Incomplete")
        cyclic = dict(parents)
        cyclic["a_sire_sire", "sire"] = "a"
        must_fail(lambda: build_graph(["a"], cyclic), "cycle")

        for name in ("main.db", "patch.db"):
            con = sqlite3.connect(folder / name)
            con.execute("CREATE TABLE pedigree_edges (child_jbis_id TEXT,parent_jbis_id TEXT,relation TEXT)")
            con.execute("INSERT INTO pedigree_edges VALUES ('a','b','sire')")
            con.commit()
            con.close()
        assert load_parents([folder / "main.db", folder / "patch.db"]) == {("a", "sire"): "b"}
        con = sqlite3.connect(folder / "patch.db")
        con.execute("UPDATE pedigree_edges SET parent_jbis_id='conflict'")
        con.commit()
        con.close()
        must_fail(lambda: load_parents([folder / "main.db", folder / "patch.db"]), "Conflicting")

        (folder / "training_horses.txt").write_text("a\n", encoding="utf-8")
        with (folder / "pedigree_edges.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["child", "parent", "relation"])
            writer.writerows((child, parent, relation) for (child, relation), parent in parents.items()
                             if (parent, child) in set(edges))
        for name in ("graph_pilot.py", "requirements-graph.txt"):
            (folder / name).write_text("placeholder\n", encoding="utf-8")
        manifest = {"purpose": "training-only pedigree resource pilot",
                    "counts": dict(training_horses=1, graph_nodes=14, graph_edges=13),
                    "files": {name: {"sha256": hashlib.sha256((folder / name).read_bytes()).hexdigest()}
                              for name in ("training_horses.txt", "pedigree_edges.csv", "graph_pilot.py", "requirements-graph.txt")}}
        write_json(folder / "manifest.json", manifest)
        assert load_pilot_inputs(folder)[:3] == (["a"], nodes, edges)
        (folder / "training_horses.txt").write_text("tampered\n", encoding="utf-8")
        must_fail(lambda: load_pilot_inputs(folder), "checksum")

    here = Path(__file__).resolve().parent
    notebook = json.loads((here / "node2vec_pilot.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    print("PASS: pip-free environment setup/recovery/check/freeze, chronological cohort, exclusions, three generations, shared ancestors, cycles, patch conflicts, input checksums, notebook syntax")


if __name__ == "__main__":
    check()
