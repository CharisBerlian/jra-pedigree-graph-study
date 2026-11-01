"""Compare saved training-starter cluster labels across graph seeds."""
import argparse
from collections import Counter
import csv
import json
from math import comb
from pathlib import Path


def read_labels(path):
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames != ["horse_id_netkeiba", "cluster"]:
            raise ValueError(f"Unexpected label columns in {path}")
        labels = {}
        for row in rows:
            horse = row["horse_id_netkeiba"]
            if horse in labels:
                raise ValueError(f"Duplicate training horse ID in {path}: {horse}")
            labels[horse] = int(row["cluster"])
    if not labels:
        raise ValueError(f"No training labels in {path}")
    return labels


def adjusted_rand(labels_a, labels_b):
    """Compute adjusted Rand agreement from two label maps, without fitting."""
    ids = sorted(set(labels_a) & set(labels_b))
    if set(labels_a) != set(labels_b):
        raise ValueError("Seed label files do not contain the same training horse IDs")
    if len(ids) < 2:
        return 1.0
    contingency = Counter((labels_a[horse], labels_b[horse]) for horse in ids)
    left = Counter(labels_a.values())
    right = Counter(labels_b.values())
    pairs = comb(len(ids), 2)
    index = sum(comb(count, 2) for count in contingency.values())
    left_pairs = sum(comb(count, 2) for count in left.values())
    right_pairs = sum(comb(count, 2) for count in right.values())
    expected = left_pairs * right_pairs / pairs
    maximum = (left_pairs + right_pairs) / 2
    denominator = maximum - expected
    return 1.0 if denominator == 0 else (index - expected) / denominator


def parse_seed_dir(spec):
    seed, separator, directory = spec.partition("=")
    if not separator or not seed.isdigit() or not directory:
        raise ValueError("--seed-dir must use SEED=PATH")
    return int(seed), Path(directory)


def build_report(seed_dirs):
    labels = {seed: read_labels(directory / f"cluster_labels_seed{seed}.csv")
              for seed, directory in seed_dirs.items()}
    seed_ids = sorted(labels)
    if len(seed_ids) < 2:
        raise ValueError("Provide at least two graph seed directories")
    first_ids = set(labels[seed_ids[0]])
    if any(set(labels[seed]) != first_ids for seed in seed_ids[1:]):
        raise ValueError("Seed label files do not contain the same training horse IDs")
    pairwise = {}
    for index, left in enumerate(seed_ids):
        for right in seed_ids[index + 1:]:
            pairwise[f"seed-{left}__seed-{right}"] = adjusted_rand(labels[left], labels[right])
    return {"seeds": seed_ids, "training_horses": len(first_ids),
            "pairwise_adjusted_rand": pairwise,
            "label_files": {str(seed): str(seed_dirs[seed] / f"cluster_labels_seed{seed}.csv")
                            for seed in seed_ids}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-dir", action="append", required=True,
                        help="Saved output directory as SEED=PATH; repeat per seed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seed_dirs = dict(parse_seed_dir(spec) for spec in args.seed_dir)
    if len(seed_dirs) != len(args.seed_dir):
        raise ValueError("Each graph seed may be provided only once")
    report = build_report(seed_dirs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
