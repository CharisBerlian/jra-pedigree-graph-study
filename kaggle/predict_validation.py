"""Run the bounded LightGBM train/validation batch.

Only the exported train and validation rows are accepted. Each surface/setting
has an independent checkpoint containing the train-fitted category mapping,
model and validation predictions. No test rows are read or written.
"""
import argparse
import csv
import gzip
import hashlib
import json
import time
from pathlib import Path


CONTEXT = ("venue", "distance", "going", "race_class", "field_size", "bracket",
           "horse_number", "weight_carried", "age", "sex", "body_weight",
           "body_weight_change")
HISTORY = ("hist_count", "has_hist1", "has_hist2", "has_hist3")
HISTORY_FIELDS = ("prior_finish_pos", "prior_norm_finish", "prior_field_size",
                  "prior_distance", "prior_surface", "prior_going",
                  "prior_race_class", "prior_weight_carried", "days_since")
CAT_BASE = ("venue", "going", "race_class", "sex")
GRAPH_META = ("embedding_status", "fallback_depth", "ancestor_count", "embedding_missing")
SETTINGS = ("S2", "S3", "S5", "S6", "M2", "M3", "S7")
SURFACES = ("turf", "dirt")
FORBIDDEN = ("finish_pos", "result_status", "time", "margin", "last_3f",
             "corner_passing", "top_three", "label", "race_id",
             "horse_id_netkeiba", "prior_race_id", "prior_race_date")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("feature export is empty")
    if any(r.get("split") not in ("train", "validation") for r in rows):
        raise ValueError("test or unknown split found; prediction batch is validation-only")
    keys = {(r["race_id"], r["horse_id_netkeiba"], r["horse_number"]) for r in rows}
    if len(keys) != len(rows):
        raise ValueError("duplicate feature keys")
    by_race = {}
    for r in rows:
        by_race.setdefault(r["race_id"], []).append(r)
    for race_id, entries in by_race.items():
        sizes = {int(r["field_size"]) for r in entries if r.get("field_size") not in (None, "")}
        if len(sizes) != 1 or len(entries) != next(iter(sizes)):
            raise ValueError("incomplete or inconsistent field_size for race %s" % race_id)
    return rows, tuple(rows[0])


def embedding_columns(columns):
    return tuple(sorted((c for c in columns if c.startswith("embedding_")
                         and c[len("embedding_"):].isdigit()),
                        key=lambda c: int(c.split("_")[1])))


def feature_columns(setting, columns):
    """Frozen predictor allowlists; the target and all IDs are rejected."""
    columns = set(columns)
    selected = list(CONTEXT)
    if setting not in SETTINGS:
        raise ValueError("unknown setting %s" % setting)
    selected += list(HISTORY) + [f"{x}_{i}" for i in (1, 2, 3) for x in HISTORY_FIELDS]
    if setting in ("S3", "M3"):
        selected += ["sire", "dam"]
    if setting in ("S5", "S6", "S7"):
        selected += [f"embedding_{i}" for i in range(64)] + list(GRAPH_META)
    if setting in ("S6", "S7"):
        selected += ["graph_cluster"]
    if setting in ("M2", "M3", "S7"):
        selected += ["odds", "popularity"]
    required = list(CONTEXT) + list(HISTORY) + [f"{x}_{i}" for i in (1, 2, 3) for x in HISTORY_FIELDS]
    if setting in ("S3", "M3"):
        required += ["sire", "dam"]
    if setting in ("S5", "S6", "S7"):
        expected_embeddings = tuple(f"embedding_{i}" for i in range(64))
        actual_embeddings = embedding_columns(columns)
        if actual_embeddings != expected_embeddings:
            raise ValueError("graph settings require exactly embedding_0..embedding_63")
        required += list(expected_embeddings) + list(GRAPH_META)
    if setting in ("S6", "S7"):
        required += ["graph_cluster"]
    if setting in ("M2", "M3", "S7"):
        required += ["odds", "popularity"]
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError("missing required feature columns for %s: %s" % (setting, missing))
    selected = [c for c in selected if c in columns]
    bad = [c for c in selected if c in FORBIDDEN]
    if bad:
        raise ValueError("target or identifier in predictor allowlist: %s" % bad)
    if not selected:
        raise ValueError("empty predictor allowlist")
    return tuple(dict.fromkeys(selected))


def fit_category_maps(rows, columns):
    maps = {}
    for col in columns:
        if col in CAT_BASE or col in ("sire", "dam", "embedding_status", "graph_cluster") or \
                col.startswith("prior_surface_") or col.startswith("prior_going_") or \
                col.startswith("prior_race_class_"):
            values = sorted({r.get(col, "") for r in rows if r.get(col, "") not in (None, "")})
            maps[col] = {value: i + 1 for i, value in enumerate(values)}
    return maps


def as_number(value):
    if value in (None, "", "NULL", "None"):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError("non-numeric value in numeric predictor: %r" % value)


def matrix(rows, columns, maps):
    out = []
    for row in rows:
        values = []
        for col in columns:
            if col in maps:
                values.append(maps[col].get(row.get(col, ""), 0))
            else:
                values.append(as_number(row.get(col)))
        out.append(values)
    return out


def labels(rows):
    return [int(r["label"]) for r in rows]


def top_three(rows, scores):
    ranked = sorted(zip(rows, scores), key=lambda x: (-x[1], int(x[0]["horse_number"]),
                                                       x[0]["horse_id_netkeiba"]))
    return ranked[:3]


def mean_hit_at_3(rows, scores):
    groups = {}
    for row, score in zip(rows, scores):
        groups.setdefault(row["race_id"], []).append((row, score))
    hits = []
    for entries in groups.values():
        if len(entries) < 3:
            continue
        selected = top_three([r for r, _ in entries], [s for _, s in entries])
        hits.append(sum(int(r["label"]) for r, _ in selected) / 3.0)
    return (sum(hits) / len(hits), len(hits)) if hits else (float("nan"), 0)


def binary_metrics(y_true, scores, threshold=0.5):
    pred = [int(s >= threshold) for s in scores]
    tp = sum(y and p for y, p in zip(y_true, pred))
    fp = sum((not y) and p for y, p in zip(y_true, pred))
    fn = sum(y and (not p) for y, p in zip(y_true, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate(rows, scores):
    y = labels(rows)
    hit3, race_count = mean_hit_at_3(rows, scores)
    try:
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    except ImportError as exc:
        raise RuntimeError("scikit-learn metrics are required on Kaggle") from exc
    values = {"average_precision": float(average_precision_score(y, scores)),
              "roc_auc": float(roc_auc_score(y, scores)) if len(set(y)) == 2 else float("nan"),
              "brier": float(brier_score_loss(y, scores)),
              "mean_hit_at_3": hit3, "complete_validation_races": race_count}
    values.update(binary_metrics(y, scores))
    return values


def _write_predictions(path, rows, scores):
    path = Path(path)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["race_id", "horse_id_netkeiba", "horse_number", "surface",
                         "label", "prediction", "selected_top3"])
        groups = {}
        for row, score in zip(rows, scores):
            groups.setdefault(row["race_id"], []).append((row, score))
        selected_keys = {(r["race_id"], r["horse_id_netkeiba"], r["horse_number"])
                         for entries in groups.values()
                         for r, _ in top_three([rr for rr, _ in entries], [ss for _, ss in entries])}
        for row, score in zip(rows, scores):
            key = (row["race_id"], row["horse_id_netkeiba"], row["horse_number"])
            writer.writerow([row["race_id"], row["horse_id_netkeiba"], row["horse_number"],
                             row["surface"], row["label"], "%.17g" % score,
                             int(key in selected_keys)])


def run_one(rows, columns, input_path, output_dir, surface, setting, seed=0):
    if surface not in SURFACES or setting not in SETTINGS:
        raise ValueError("unsupported surface/setting")
    if seed != 0:
        raise ValueError("initial validation batch is aligned to production graph seed 0")
    if not Path("/kaggle").is_dir():
        raise RuntimeError("prediction training is bounded to Kaggle; local runs only prepare/test code")
    selected = feature_columns(setting, columns)
    train = [r for r in rows if r["surface"] == surface and r["split"] == "train"]
    validation = [r for r in rows if r["surface"] == surface and r["split"] == "validation"]
    if not train or not validation:
        raise ValueError("missing %s train/validation rows" % surface)
    maps = fit_category_maps(train, selected)
    config = {"family": "lightgbm", "seed": seed, "surface": surface, "setting": setting,
              "features": selected, "categorical": sorted(maps),
              "params": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
                         "subsample": 1.0, "colsample_bytree": 1.0, "n_jobs": 4}}
    input_hash = sha256_file(input_path)
    script_hash = sha256_file(Path(__file__))
    config_hash = hashlib.sha256(json.dumps({"config": config, "input": input_hash,
                                              "script": script_hash}, sort_keys=True).encode()).hexdigest()
    run_dir = Path(output_dir) / surface / setting / ("seed%d" % seed)
    marker = run_dir / "complete.json"
    if marker.exists():
        old = json.loads(marker.read_text(encoding="utf-8"))
        if old.get("config_hash") == config_hash:
            return {"status": "skipped", "run_dir": str(run_dir), "config_hash": config_hash}
        raise FileExistsError("completed run has a different config hash: %s" % run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise RuntimeError("LightGBM is required on Kaggle; local preparation/tests do not train models") from exc
    started = time.perf_counter()
    x_train, x_val = matrix(train, selected, maps), matrix(validation, selected, maps)
    model = LGBMClassifier(objective="binary", random_state=seed, verbosity=-1, **config["params"])
    categorical = [selected.index(c) for c in maps]
    model.fit(x_train, labels(train), categorical_feature=categorical)
    scores = [float(x[1]) for x in model.predict_proba(x_val)]
    metrics = evaluate(validation, scores)
    metrics.update(surface=surface, setting=setting, seed=seed,
                   seconds=time.perf_counter() - started, validation_rows=len(validation),
                   train_rows=len(train), config_hash=config_hash)
    (run_dir / "config.json").write_text(json.dumps({**config, "input_sha256": input_hash,
                                                       "script_sha256": script_hash}, indent=2,
                                              sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "preprocessing.json").write_text(json.dumps({"features": selected, "category_maps": maps},
                                                             indent=2, sort_keys=True) + "\n", encoding="utf-8")
    model.booster_.save_model(str(run_dir / "model.txt"))
    _write_predictions(run_dir / "validation_predictions.csv.gz", validation, scores)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    marker.write_text(json.dumps({"status": "complete", "config_hash": config_hash,
                                  "metrics": metrics}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "complete", "run_dir": str(run_dir), "config_hash": config_hash,
            "metrics": metrics}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="train_validation.csv.gz")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--surfaces", nargs="+", default=list(SURFACES), choices=SURFACES)
    parser.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=SETTINGS)
    args = parser.parse_args(argv)
    if not Path("/kaggle").is_dir():
        raise RuntimeError("prediction training is bounded to Kaggle; local runs only prepare/test code")
    rows, columns = read_rows(args.input)
    results = []
    expected = len(args.surfaces) * len(args.settings)
    for surface in args.surfaces:
        for setting in args.settings:
            print(f"[{surface}/{setting}/seed{args.seed}] starting", flush=True)
            result = run_one(rows, columns, args.input, args.output_dir, surface, setting, args.seed)
            if result.get("status") not in ("complete", "skipped"):
                raise RuntimeError("run did not produce a completed checkpoint: %r" % result)
            results.append(result)
            print(f"[{surface}/{setting}/seed{args.seed}] {result['status']}", flush=True)
    if len(results) != expected:
        raise RuntimeError("checkpoint count mismatch: %d/%d" % (len(results), expected))
    print(json.dumps(results, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
