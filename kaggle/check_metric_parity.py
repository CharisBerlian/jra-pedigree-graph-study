"""Check locked metric behavior in the local and pinned evaluation runtimes."""
import argparse
import importlib.metadata
import json
import math
import sys
from pathlib import Path


EXPECTED = {"numpy": "1.26.4", "scipy": "1.13.1", "scikit-learn": "1.6.1"}


def metric_record(require_pins=False):
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    labels = [1, 0, 1, 0, 1, 0, 0, 1]
    scores = [0.9, 0.9, 0.4, 0.4, 0.1, 0.1, 0.7, 0.2]
    packages = {}
    for package in EXPECTED:
        packages[package] = importlib.metadata.version(package)
    if require_pins:
        if sys.version.split()[0] != "3.12.13":
            raise RuntimeError(f"python pin mismatch: {sys.version.split()[0]}")
        if packages != EXPECTED:
            raise RuntimeError(f"metric package pin mismatch: {packages}")
    return {"python": sys.version.split()[0], "packages": packages,
            "fixture_metrics": {
                "average_precision": float(average_precision_score(labels, scores)),
                "brier": float(brier_score_loss(labels, scores)),
                "roc_auc": float(roc_auc_score(labels, scores)),
            }}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")


def compare(left_path, right_path):
    left, right = json.loads(Path(left_path).read_text()), json.loads(
        Path(right_path).read_text())
    differences = {}
    for metric, value in left["fixture_metrics"].items():
        other = right["fixture_metrics"][metric]
        differences[metric] = float(other - value)
    parity = all(math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-15)
                 for value in differences.values())
    return {"status": "complete", "parity": parity,
            "left": left, "right": right, "absolute_differences": differences,
            "fixture": "fixed labels and tied scores; sklearn AP, Brier, ROC-AUC"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-pins", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("LEFT", "RIGHT"))
    args = parser.parse_args()
    value = compare(*args.compare) if args.compare else metric_record(args.require_pins)
    if args.output:
        write(args.output, value)
    print(json.dumps(value, sort_keys=True), flush=True)
    if value.get("parity") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
