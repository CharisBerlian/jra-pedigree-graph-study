"""Read-only audit of the downloaded seed-0 validation batch."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "notebook-output" / "validation-seed0-version1" / "jra-validation-runs" / "seed-0"
DEFAULT_DB = HERE.parent / "tmp" / "features.db"
FROZEN_SCRIPT_SHA256 = "16de922f6af100caf02dac1eab1c9884ec392409bfd8d0227d88d366f8cc65e9"
EXPECTED_SETTINGS = ("S2", "S3", "S5", "S6", "M2", "M3", "S7")
SURFACES = ("turf", "dirt")
ARTIFACTS = ("complete.json", "config.json", "metrics.json", "model.txt",
             "preprocessing.json", "validation_predictions.csv.gz")
FORBIDDEN = {"finish_pos", "result_status", "time", "margin", "last_3f",
             "corner_passing", "top_three", "label", "race_id",
             "horse_id_netkeiba", "prior_race_id", "prior_race_date"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ap_score(labels, scores):
    total = sum(labels)
    if not total:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    hits = value = 0
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        positives = sum(labels[order[k]] for k in range(i, j))
        hits += positives
        if positives:
            value += positives * hits / j
        i = j
    return value / total


def auc_score(labels, scores):
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return float("nan")
    ordered = sorted(scores)
    ranks = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j] == ordered[i]:
            j += 1
        ranks[ordered[i]] = (i + j + 1) / 2.0
        i = j
    positive_rank_sum = sum(ranks[s] for s, y in zip(scores, labels) if y)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def top_three(entries):
    return sorted(entries, key=lambda x: (-x["score"], int(x["horse_number"]),
                                          x["horse_id_netkeiba"]))[:3]


def metrics(rows):
    labels = [r["label"] for r in rows]
    scores = [r["score"] for r in rows]
    groups = {}
    for row in rows:
        groups.setdefault(row["race_id"], []).append(row)
    hits = []
    for entries in groups.values():
        if len(entries) < 3:
            raise AssertionError("validation race has fewer than three entries")
        selected = top_three(entries)
        if sum(r["selected_top3"] for r in entries) != 3:
            raise AssertionError("selected_top3 is not exactly three per race")
        expected = {(r["race_id"], r["horse_id_netkeiba"], r["horse_number"])
                    for r in selected}
        actual = {(r["race_id"], r["horse_id_netkeiba"], r["horse_number"])
                  for r in entries if r["selected_top3"]}
        if actual != expected:
            raise AssertionError("selected_top3 does not match deterministic ranking")
        hits.append(sum(r["label"] for r in selected) / 3.0)
    tp = sum(y and s >= .5 for y, s in zip(labels, scores))
    fp = sum((not y) and s >= .5 for y, s in zip(labels, scores))
    fn = sum(y and s < .5 for y, s in zip(labels, scores))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"average_precision": ap_score(labels, scores), "roc_auc": auc_score(labels, scores),
            "brier": sum((s - y) ** 2 for s, y in zip(scores, labels)) / len(labels),
            "mean_hit_at_3": sum(hits) / len(hits), "complete_validation_races": len(hits),
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def read_predictions(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["race_id", "horse_id_netkeiba", "horse_number", "surface",
                                 "label", "prediction", "selected_top3"]:
            raise ValueError("unexpected prediction columns")
        rows = []
        for raw in reader:
            score = float(raw["prediction"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("prediction outside finite [0,1]")
            label = int(raw["label"])
            selected = int(raw["selected_top3"])
            if label not in (0, 1) or selected not in (0, 1):
                raise ValueError("nonbinary prediction label/flag")
            rows.append({"race_id": raw["race_id"], "horse_id_netkeiba": raw["horse_id_netkeiba"],
                         "horse_number": raw["horse_number"], "surface": raw["surface"],
                         "label": label, "score": score, "selected_top3": selected})
    keys = [(r["race_id"], r["horse_id_netkeiba"], r["horse_number"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate prediction keys")
    return rows


def expected_cohort(db_path, surface):
    con = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only=ON")
    try:
        rows = [dict(r) for r in con.execute("""
            SELECT race_id,horse_id_netkeiba,horse_number,label,field_size
            FROM entries WHERE split='validation' AND LOWER(surface)=?
            ORDER BY race_id,horse_number,horse_id_netkeiba
        """, (surface,))]
    finally:
        con.close()
    by_race = {}
    for row in rows:
        by_race.setdefault(row["race_id"], []).append(row)
    for race_id, entries in by_race.items():
        sizes = {int(e["field_size"]) for e in entries}
        if len(sizes) != 1 or len(entries) != next(iter(sizes)):
            raise ValueError("incomplete feature race %s" % race_id)
    return rows


def audit(root=DEFAULT_ROOT, db_path=DEFAULT_DB):
    errors = []
    root = Path(root)
    report = {"status": "FAIL", "root": str(root), "script_sha256": FROZEN_SCRIPT_SHA256}
    setup = {}
    setup_path = root / "setup_report.json"
    if not setup_path.is_file():
        errors.append("missing setup_report.json")
    else:
        setup = json.loads(setup_path.read_text(encoding="utf-8"))
        report["setup"] = setup
        checks = {"source_format": "csv", "source_data_sha256": "d395a1802d90b0c09cc71f08b2dcbc83ca4020654797c9442969833c103ac12d",
                  "extracted_csv_sha256": "d395a1802d90b0c09cc71f08b2dcbc83ca4020654797c9442969833c103ac12d",
                  "data_sha256": "c3902280db1aba93f6f7b3d07ccc3c9fb2d4c2b7bf0781cf3d9e3dafb9ac5b90",
                  "actual_input_data_sha256": "c3902280db1aba93f6f7b3d07ccc3c9fb2d4c2b7bf0781cf3d9e3dafb9ac5b90",
                  "script_sha256": FROZEN_SCRIPT_SHA256}
        for key, value in checks.items():
            if setup.get(key) != value:
                errors.append("setup %s mismatch" % key)
        if setup.get("source_format") == "csv" and not str(setup.get("source_path", "")).endswith("train_validation.csv"):
            errors.append("setup source path is not extracted train_validation.csv")

    try:
        if sha256_file(HERE / "predict_validation.py") != FROZEN_SCRIPT_SHA256:
            raise ValueError("local prediction helper differs from the frozen script")
        sys.path.insert(0, str(HERE))
        import predict_validation as frozen
        all_columns = set(frozen.CONTEXT + frozen.HISTORY)
        all_columns.update(f"{name}_{rank}" for name in frozen.HISTORY_FIELDS for rank in (1, 2, 3))
        all_columns.update(f"embedding_{i}" for i in range(64))
        all_columns.update(("sire", "dam", "odds", "popularity", "graph_cluster",
                            "embedding_status", "fallback_depth", "ancestor_count", "embedding_missing"))
    except Exception as exc:
        errors.append("cannot import frozen prediction helpers: %s" % exc)
        frozen = None
        all_columns = set()

    expected = {}
    for surface in SURFACES:
        try:
            expected[surface] = expected_cohort(db_path, surface)
        except Exception as exc:
            errors.append("cohort %s: %s" % (surface, exc))
            expected[surface] = []

    actual_paths = {(p.parent.parent.name, p.parent.name): p
                    for p in root.glob("*/[A-Z0-9]*/seed0")}
    expected_paths = {(surface, setting): root / surface / setting / "seed0"
                      for surface in SURFACES for setting in EXPECTED_SETTINGS}
    if set(actual_paths) != set(expected_paths):
        errors.append("run path set mismatch")
    summary = []
    hashes = {}
    rows_by_setting = {}
    for surface in SURFACES:
        baseline_keys = baseline_labels = None
        for setting in EXPECTED_SETTINGS:
            run = expected_paths[(surface, setting)]
            if not run.is_dir():
                errors.append("missing run directory %s/%s" % (surface, setting))
                continue
            for name in ARTIFACTS:
                path = run / name
                if not path.is_file() or path.stat().st_size == 0:
                    errors.append("missing or empty %s" % path)
                else:
                    hashes["%s/%s/%s" % (surface, setting, name)] = sha256_file(path)
            try:
                marker = json.loads((run / "complete.json").read_text(encoding="utf-8"))
                config = json.loads((run / "config.json").read_text(encoding="utf-8"))
                metric = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
                prep = json.loads((run / "preprocessing.json").read_text(encoding="utf-8"))
                if marker.get("status") != "complete":
                    errors.append("incomplete marker %s/%s" % (surface, setting))
                if config.get("family") != "lightgbm" or config.get("seed") != 0 or config.get("surface") != surface or config.get("setting") != setting:
                    errors.append("config identity mismatch %s/%s" % (surface, setting))
                if config.get("script_sha256") != FROZEN_SCRIPT_SHA256:
                    errors.append("script hash mismatch %s/%s" % (surface, setting))
                if config.get("input_sha256") != setup.get("actual_input_data_sha256"):
                    errors.append("input/setup hash mismatch %s/%s" % (surface, setting))
                if marker.get("metrics") != metric:
                    errors.append("completion/metrics mismatch %s/%s" % (surface, setting))
                if config.get("params") != {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "subsample": 1.0, "colsample_bytree": 1.0, "n_jobs": 4}:
                    errors.append("fixed params mismatch %s/%s" % (surface, setting))
                config_body = {k: config[k] for k in config if k not in ("input_sha256", "script_sha256")}
                rebuilt_hash = hashlib.sha256(json.dumps({"config": config_body,
                                                           "input": config.get("input_sha256"),
                                                           "script": config.get("script_sha256")}, sort_keys=True).encode()).hexdigest()
                if rebuilt_hash != config.get("config_hash", marker.get("config_hash")) or marker.get("config_hash") != metric.get("config_hash"):
                    errors.append("config hash mismatch %s/%s" % (surface, setting))
                if prep.get("features") != config.get("features") or sorted(prep.get("category_maps", {})) != sorted(config.get("categorical", [])):
                    errors.append("preprocessing/config mismatch %s/%s" % (surface, setting))
                if frozen:
                    try:
                        allow = list(frozen.feature_columns(setting, all_columns))
                        if config.get("features") != allow:
                            errors.append("feature allowlist mismatch %s/%s" % (surface, setting))
                    except Exception as exc:
                        errors.append("allowlist check %s/%s: %s" % (surface, setting, exc))
                if any(c in FORBIDDEN for c in config.get("features", [])):
                    errors.append("forbidden predictor %s/%s" % (surface, setting))
                if setting not in ("M2", "M3", "S7") and any(c in ("odds", "popularity") for c in config.get("features", [])):
                    errors.append("market leak %s/%s" % (surface, setting))
                if setting in ("M2", "M3", "S7") and not {"odds", "popularity"}.issubset(config.get("features", [])):
                    errors.append("missing market controls %s/%s" % (surface, setting))
                rows = read_predictions(run / "validation_predictions.csv.gz")
                exp = expected[surface]
                got_keys = [(r["race_id"], r["horse_id_netkeiba"], r["horse_number"]) for r in rows]
                exp_keys = [(r["race_id"], r["horse_id_netkeiba"], str(r["horse_number"])) for r in exp]
                if got_keys != exp_keys:
                    errors.append("validation key/order mismatch %s/%s" % (surface, setting))
                if [r["label"] for r in rows] != [int(r["label"]) for r in exp]:
                    errors.append("validation labels mismatch %s/%s" % (surface, setting))
                if any(r["surface"] != surface for r in rows) or len(rows) != len(exp):
                    errors.append("validation surface/count mismatch %s/%s" % (surface, setting))
                if baseline_keys is None:
                    baseline_keys = got_keys; baseline_labels = [r["label"] for r in rows]
                elif got_keys != baseline_keys or [r["label"] for r in rows] != baseline_labels:
                    errors.append("cross-setting cohort mismatch %s/%s" % (surface, setting))
                value = metrics(rows)
                for key in ("average_precision", "roc_auc", "brier", "mean_hit_at_3", "precision", "recall", "f1"):
                    if not math.isclose(value[key], float(metric[key]), rel_tol=1e-10, abs_tol=1e-10):
                        errors.append("metric mismatch %s/%s/%s" % (surface, setting, key))
                if value["complete_validation_races"] != metric.get("complete_validation_races"):
                    errors.append("race count mismatch %s/%s" % (surface, setting))
                if any(metric.get(k) != config.get(k) for k in ("surface", "setting", "seed")):
                    errors.append("metric identity mismatch %s/%s" % (surface, setting))
                if metric.get("validation_rows") != len(exp) or metric.get("train_rows") != {"turf": 271549, "dirt": 277207}[surface]:
                    errors.append("metric entry count mismatch %s/%s" % (surface, setting))
                rows_by_setting[(surface, setting)] = value
                summary.append(dict(surface=surface, setting=setting, **value))
            except Exception as exc:
                errors.append("run audit %s/%s: %s" % (surface, setting, exc))
    contrasts = {}
    for surface in SURFACES:
        for lhs, rhs in (("S6", "S2"), ("S6", "S3"), ("S6", "S5"), ("S7", "M2"), ("S7", "M3"), ("S7", "S6")):
            if (surface, lhs) in rows_by_setting and (surface, rhs) in rows_by_setting:
                contrasts["%s-%s/%s" % (lhs, rhs, surface)] = {
                    metric: rows_by_setting[(surface, lhs)][metric] - rows_by_setting[(surface, rhs)][metric]
                    for metric in ("average_precision", "mean_hit_at_3", "brier")}
    report.update({"status": "PASS" if not errors else "FAIL", "errors": errors,
                   "summary": summary, "contrasts": contrasts, "artifact_sha256": hashes,
                   "expected_runs": len(expected_paths), "audited_runs": len(rows_by_setting)})
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--features-db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--report", type=Path, default=HERE.parent / "plans" / "validation_seed0_audit_2026-09-12.json")
    args = parser.parse_args()
    result = audit(args.root, args.features_db)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "errors": result["errors"],
                      "audited_runs": result["audited_runs"], "report": str(args.report)}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
