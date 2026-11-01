"""Evaluate the completed locked test-prediction bundle.

This is a post-inference reader: it never fits or changes a model.  Labels,
finish positions, and race dates are joined only after the Kaggle inference
report has declared 180 prediction outputs with no metrics.
"""

from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3

import numpy as np

import evaluate_locked as evaluation


FAMILIES = ("xgboost", "lightgbm", "catboost")
SURFACES = ("turf", "dirt")
SEEDS = (0, 1, 2)
SETTINGS = ("B1", "S1", "S2", "S3", "S4", "S5", "S6", "M2", "M3", "S7")
PRIMARY_CONTRASTS = (("S6", "S2"), ("S6", "S3"), ("S6", "S5"),
                     ("S7", "M2"), ("S7", "M3"), ("S7", "S6"))
PRIMARY_SETTINGS = tuple(dict.fromkeys(setting for pair in PRIMARY_CONTRASTS for setting in pair))
HISTORY_EXACT = ("0", "1", "2", "3+")
HISTORY_BAND = ("<3", "3+")
COVERAGE = ("direct", "ancestor_fallback", "missing")
KEY_NAMES = ("race_id", "horse_id_netkeiba", "horse_number")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def key_of(row):
    return (str(row["race_id"]), str(row["horse_id_netkeiba"]), int(row["horse_number"]))


def input_path(folder, seed):
    candidates = [Path(folder) / f"test_seed{seed}.csv.gz",
                  Path(folder) / f"test_seed{seed}.csv"]
    candidates = [path for path in candidates if path.is_file()]
    if len(candidates) != 1:
        raise ValueError(f"expected one test input for seed {seed}: {candidates}")
    return candidates[0]


def sidecar_path(path):
    candidates = [path.with_suffix(".json"), Path(str(path) + ".json")]
    candidates = [candidate for candidate in candidates if candidate.is_file()]
    if len(candidates) != 1:
        raise ValueError(f"expected one test-input sidecar for {path}: {candidates}")
    return candidates[0]


def read_input(path, expected_seed):
    metadata = read_json(sidecar_path(path))
    if metadata.get("splits") != ["test"] or metadata.get("seed") != expected_seed:
        raise ValueError(f"test input metadata is not seed-bound: {path}")
    if metadata.get("test_labels_read") is not False or metadata.get("test_metrics_computed") is not False:
        raise ValueError(f"test input sidecar is not prediction-only: {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"test input is empty: {path}")
    return rows, metadata


def _coverage(row):
    if str(row.get("embedding_missing", "0")) == "1":
        return "missing"
    status = str(row.get("embedding_status", "")).lower()
    if status == "direct":
        return "direct"
    if status == "fallback":
        return "ancestor_fallback"
    raise ValueError(f"unknown embedding coverage status: {status!r}")


def _history_groups(value):
    count = int(value)
    if count < 0:
        raise ValueError("hist_count must be nonnegative")
    exact = str(count) if count in (0, 1, 2) else "3+"
    band = "<3" if count < 3 else "3+"
    return exact, band


def build_cohorts(features_db, source_db, test_input_dir):
    input_rows = {}
    input_meta = {}
    for seed in SEEDS:
        path = input_path(test_input_dir, seed)
        rows, metadata = read_input(path, seed)
        by_key = {key_of(row): row for row in rows}
        if len(by_key) != len(rows):
            raise ValueError(f"duplicate test input keys: {path}")
        input_rows[seed] = by_key
        input_meta[seed] = metadata
    if set(input_rows[0]) != set(input_rows[1]) or set(input_rows[0]) != set(input_rows[2]):
        raise ValueError("test input seeds do not have identical row keys")

    fcon = sqlite3.connect(Path(features_db).resolve().as_uri() + "?mode=ro", uri=True)
    scon = sqlite3.connect(Path(source_db).resolve().as_uri() + "?mode=ro", uri=True)
    fcon.row_factory = sqlite3.Row
    scon.row_factory = sqlite3.Row
    try:
        entries = {}
        for row in fcon.execute("""
                SELECT race_id, horse_id_netkeiba, horse_number, split, label,
                       surface, field_size, hist_count
                FROM entries WHERE split = 'test'
                  AND LOWER(surface) IN ('turf', 'dirt')
        """):
            key = key_of(row)
            if key in entries:
                raise ValueError(f"duplicate test label key: {key}")
            entries[key] = dict(row)
    finally:
        fcon.close()

    finish = {}
    for row in scon.execute("""
            SELECT n.race_id, n.horse_id_netkeiba, n.horse_number, n.finish_pos,
                   r.race_date
            FROM runners_r n JOIN races_r r USING (race_id)
            WHERE r.race_date BETWEEN '20240101' AND '20251231'
    """):
        key = key_of(row)
        if key in entries:
            if key in finish:
                raise ValueError(f"duplicate test outcome key: {key}")
            finish[key] = (row["finish_pos"], str(row["race_date"]))
    scon.close()

    if set(entries) != set(input_rows[0]):
        raise ValueError("test feature inputs and feature labels have different keys")
    if set(entries) != set(finish):
        missing = sorted(set(entries) - set(finish))
        raise ValueError(f"test outcomes do not cover feature keys: {missing[:3]}")

    cohorts = {}
    for surface in SURFACES:
        base_keys = [key for key, row in input_rows[0].items()
                     if str(row.get("surface", "")).lower() == surface]
        if not base_keys:
            raise ValueError(f"test input has no surface rows: {surface}")
        if len({entries[key]["surface"].lower() for key in base_keys}) != 1:
            raise ValueError(f"feature label surface mismatch: {surface}")
        for seed in SEEDS[1:]:
            for key in base_keys:
                if input_rows[seed][key].get("field_size") != input_rows[0][key].get("field_size"):
                    raise ValueError(f"field size differs across test inputs: {key}")
                if _coverage(input_rows[seed][key]) != _coverage(input_rows[0][key]):
                    raise ValueError(f"coverage differs across test inputs: {key}")
        race_ids = sorted({key[0] for key in base_keys})
        race_index_by_id = {race_id: index for index, race_id in enumerate(race_ids)}
        race_index = np.asarray([race_index_by_id[key[0]] for key in base_keys], dtype=np.int32)
        labels = np.asarray([int(entries[key]["label"]) for key in base_keys], dtype=np.int8)
        field_sizes = np.asarray([int(entries[key]["field_size"]) for key in base_keys], dtype=np.int16)
        history_exact = []
        history_band = []
        coverage = []
        finish_pos = np.zeros(len(base_keys), dtype=np.int16)
        race_dates = {}
        for index, key in enumerate(base_keys):
            exact, band = _history_groups(input_rows[0][key]["hist_count"])
            history_exact.append(exact)
            history_band.append(band)
            coverage.append(_coverage(input_rows[0][key]))
            value, race_date = finish[key]
            if value not in (None, ""):
                finish_pos[index] = int(value)
            race_dates[key[0]] = race_date
            if labels[index] != int(value not in (None, "") and int(value) <= 3):
                raise ValueError(f"feature label disagrees with finish position: {key}")
        race_rows = [[] for _ in race_ids]
        for index, race_index_value in enumerate(race_index):
            race_rows[int(race_index_value)].append(index)
        for race_id, indices in zip(race_ids, race_rows):
            sizes = {int(field_sizes[index]) for index in indices}
            if len(sizes) != 1 or len(indices) != next(iter(sizes)):
                raise ValueError(f"incomplete test race: {race_id}")
        complete_graded = [index for index, indices in enumerate(race_rows)
                           if all(finish_pos[row_index] > 0 for row_index in indices)]
        graded_rows = np.asarray([row_index for race_index_value in complete_graded
                                  for row_index in race_rows[race_index_value]], dtype=np.int32)
        cohorts[surface] = {
            "surface": surface, "keys": base_keys, "race_ids": race_ids,
            "race_index": race_index, "race_rows": race_rows,
            "labels": labels, "field_sizes": field_sizes,
            "history_exact": np.asarray(history_exact, dtype=object),
            "history_band": np.asarray(history_band, dtype=object),
            "coverage": np.asarray(coverage, dtype=object),
            "finish_pos": finish_pos, "race_dates": race_dates,
            "graded_race_indices": complete_graded, "graded_rows": graded_rows,
            "input_metadata": input_meta,
        }
    return cohorts


def find_report(path):
    candidates = [Path(path)] if path else []
    if path is not None and Path(path).is_dir():
        candidates = sorted(Path(path).rglob("inference_report.json"))
    if not candidates:
        raise FileNotFoundError("inference_report.json was not found")
    if len(candidates) != 1:
        raise ValueError(f"expected one inference report: {candidates}")
    return candidates[0]


def find_prediction_files(root):
    found = {}
    for path in Path(root).rglob("*"):
        if path.name not in ("test_predictions.csv.gz", "test_predictions.csv"):
            continue
        match = re.fullmatch(r"seed([012])", path.parent.name.lower())
        if not match or len(path.parents) < 4:
            continue
        key = (path.parents[3].name.lower(), path.parents[2].name.lower(),
               path.parents[1].name, int(match.group(1)))
        if key in found:
            raise ValueError(f"duplicate prediction file: {key}")
        found[key] = path
    expected = {(family, surface, setting, seed)
                for family in FAMILIES for surface in SURFACES
                for setting in SETTINGS for seed in SEEDS}
    if set(found) != expected:
        missing = sorted(expected - set(found))
        extra = sorted(set(found) - expected)
        raise ValueError(f"prediction matrix mismatch; missing={missing[:3]}, extra={extra[:3]}")
    return found


def read_predictions(path, cohort):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        raw_rows = list(csv.DictReader(stream))
    required = set(KEY_NAMES) | {"surface", "prediction", "selected_top3"}
    if not raw_rows or not required.issubset(raw_rows[0]):
        raise ValueError(f"prediction file lacks required columns: {path}")
    position = {key: index for index, key in enumerate(cohort["keys"])}
    scores = np.full(len(position), np.nan, dtype=np.float64)
    selected = np.full(len(position), -1, dtype=np.int8)
    seen = set()
    for raw in raw_rows:
        key = key_of(raw)
        if key not in position or key in seen:
            raise ValueError(f"prediction key mismatch or duplicate: {key}")
        seen.add(key)
        index = position[key]
        if str(raw["surface"]).lower() != cohort["surface"]:
            raise ValueError(f"prediction surface mismatch: {key}")
        scores[index] = float(raw["prediction"])
        selected[index] = int(raw["selected_top3"])
    if len(seen) != len(position) or np.isnan(scores).any() or (selected < 0).any():
        raise ValueError(f"prediction file does not cover the complete cohort: {path}")
    rows = []
    for index, key in enumerate(cohort["keys"]):
        rows.append({
            "race_id": key[0], "horse_id_netkeiba": key[1], "horse_number": key[2],
            "surface": cohort["surface"], "field_size": int(cohort["field_sizes"][index]),
            "label": int(cohort["labels"][index]), "score": float(scores[index]),
            "selected_top3": int(selected[index]),
            "history_exact": cohort["history_exact"][index],
            "history_band": cohort["history_band"][index],
            "coverage": cohort["coverage"][index],
        })
    audited = evaluation.audit_selection_flags(rows)
    return scores, selected, audited["rows"]


def graded_metrics(rows, cohort):
    graded = []
    for index in cohort["graded_rows"]:
        row = dict(rows[int(index)])
        row["finish_pos"] = int(cohort["finish_pos"][int(index)])
        graded.append(row)
    return (evaluation.race_metrics(graded, require_graded=True)
            if graded else {"complete_races": 0, "ndcg_binary_at_3": None,
                            "ndcg_graded_at_3": None})


def point_metrics(rows, cohort, threshold):
    entry = evaluation.entry_metrics(rows, threshold=threshold)
    race = evaluation.race_metrics(rows)
    graded = graded_metrics(rows, cohort)
    result = {f"entry_{key}": value for key, value in entry.items()}
    result.update({f"race_{key}": value for key, value in race.items()})
    result.update({"graded_races": graded["complete_races"],
                   "ndcg_binary_at_3": graded.get("ndcg_binary_at_3"),
                   "ndcg_graded_at_3": graded.get("ndcg_graded_at_3")})
    return result


def subgroup_rows(rows, family, surface, setting, seed):
    result = []
    for field, categories in (("history_exact", HISTORY_EXACT),
                              ("history_band", HISTORY_BAND),
                              ("coverage", COVERAGE)):
        values = evaluation.subgroup_metrics(rows, field, categories)
        for category, metrics in values.items():
            result.append(dict(family=family, surface=surface, setting=setting,
                               seed=seed, subgroup_type=field,
                               subgroup=category, **metrics))
    return result


def _race_hits(cohort, selected):
    return np.bincount(cohort["race_index"],
                       weights=cohort["labels"] * selected,
                       minlength=len(cohort["race_ids"])) / 3.0


def _ap_prepare(scores, labels, race_index):
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]) + 1]
    return labels[order].astype(np.float64), race_index[order], starts


def _weighted_ap(prepared, race_weights):
    ordered_labels, ordered_races, starts = prepared
    weights = race_weights[ordered_races].astype(np.float64)
    positive = weights * ordered_labels
    total_positive = positive.sum()
    if total_positive <= 0:
        return 0.0
    group_weights = np.add.reduceat(weights, starts)
    group_positive = np.add.reduceat(positive, starts)
    cumulative_weights = np.cumsum(group_weights)
    cumulative_positive = np.cumsum(group_positive)
    precision = np.divide(cumulative_positive, cumulative_weights,
                          out=np.zeros_like(cumulative_positive),
                          where=cumulative_weights > 0)
    return float(np.dot(precision, group_positive) / total_positive)


def _race_counts(race_count, replicates, seed):
    draws = np.random.default_rng(seed).integers(0, race_count,
                                                  size=(replicates, race_count),
                                                  dtype=np.int32)
    counts = np.empty((replicates, race_count), dtype=np.int16)
    for index, draw in enumerate(draws):
        counts[index] = np.bincount(draw, minlength=race_count)
    return counts


def _ci(distribution):
    values = np.asarray(distribution, dtype=np.float64)
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _bootstrap_caches(models, cohorts, replicates, seed):
    ap_cache, hit_cache = {}, {}
    for surface in SURFACES:
        cohort = cohorts[surface]
        counts = _race_counts(len(cohort["race_ids"]), replicates, seed)
        keys = [key for key in models if key[1] == surface]
        for number, key in enumerate(sorted(keys), 1):
            scores = models[key]["scores"]
            prepared = _ap_prepare(scores, cohort["labels"], cohort["race_index"])
            ap = np.empty(replicates, dtype=np.float64)
            for index in range(replicates):
                ap[index] = _weighted_ap(prepared, counts[index])
            ap_cache[key] = ap
            hit_cache[key] = (counts @ models[key]["hit_by_race"]) / len(cohort["race_ids"])
            if number == len(keys) or number % 10 == 0:
                print(f"race bootstrap {surface}: {number}/{len(keys)} model-seeds", flush=True)
    return ap_cache, hit_cache


def _week_counts(cohort, replicates, seed):
    blocks = defaultdict(list)
    for index, race_id in enumerate(cohort["race_ids"]):
        value = str(cohort["race_dates"][race_id])
        iso_year = int(value[:4])
        from datetime import date
        observed = date.fromisoformat(f"{value[:4]}-{value[4:6]}-{value[6:]}")
        iso = observed.isocalendar()
        blocks[(int(iso.year), int(iso.week))].append(index)
    ordered = sorted(blocks)
    week_by_race = np.empty(len(cohort["race_ids"]), dtype=np.int32)
    for week_index, block in enumerate(ordered):
        week_by_race[blocks[block]] = week_index
    draws = np.random.default_rng(seed).integers(0, len(ordered),
                                                  size=(replicates, len(ordered)),
                                                  dtype=np.int32)
    counts = np.empty((replicates, len(ordered)), dtype=np.int16)
    for index, draw in enumerate(draws):
        counts[index] = np.bincount(draw, minlength=len(ordered))
    return counts, week_by_race, ordered, {f"{year}-W{week:02d}": len(blocks[(year, week)])
                                           for year, week in ordered}


def _week_bootstrap(models, cohorts, replicates, seed):
    ap_cache, hit_cache, metadata = {}, {}, {}
    for surface in SURFACES:
        cohort = cohorts[surface]
        counts, week_by_race, ordered, block_sizes = _week_counts(cohort, replicates, seed)
        metadata[surface] = {"week_count": len(ordered), "week_blocks": block_sizes}
        keys = [key for key in models if key[1] == surface and key[2] in PRIMARY_SETTINGS]
        for key in sorted(keys):
            prepared = _ap_prepare(models[key]["scores"], cohort["labels"], cohort["race_index"])
            ap = np.empty(replicates, dtype=np.float64)
            hit = np.empty(replicates, dtype=np.float64)
            for index in range(replicates):
                ap[index] = _weighted_ap(prepared, counts[index][week_by_race])
                race_weights = counts[index][week_by_race]
                hit[index] = float(np.dot(race_weights, models[key]["hit_by_race"]) /
                                   race_weights.sum())
            ap_cache[key] = ap
            hit_cache[key] = hit
    return ap_cache, hit_cache, metadata


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _contrast_records(cache, models, surface, metric_cache, metric_name, week=False):
    records = []
    for family in FAMILIES:
        for setting_a, setting_b in PRIMARY_CONTRASTS:
            values = [metric_cache[(family, surface, setting_a, seed)] -
                      metric_cache[(family, surface, setting_b, seed)]
                      for seed in SEEDS]
            distribution = sum(values) / len(values)
            point = [models[(family, surface, setting_a, seed)]["point"][metric_name] -
                     models[(family, surface, setting_b, seed)]["point"][metric_name]
                     for seed in SEEDS]
            records.append({"family": family, "surface": surface,
                            "contrast": f"{setting_a}-{setting_b}",
                            "metric": metric_name, "estimate": float(np.mean(point)),
                            "seed_point_estimates": [float(v) for v in point],
                            "seed_sd": float(np.std(point, ddof=1)),
                            "ci95": _ci(distribution), "replicates": len(distribution),
                            "seed": 20260913, "race_count": len(cache[surface]["race_ids"])
                            if isinstance(cache, dict) and surface in cache else None,
                            "distribution": [float(v) for v in distribution]})
    return records


def run(args):
    manifest = read_json(args.manifest)
    spec = read_json(args.spec)
    report_path = find_report(args.inference_report)
    inference_report = read_json(report_path)
    if inference_report.get("status") != "complete" or inference_report.get("model_outputs") != 180:
        raise ValueError("locked inference report is incomplete")
    if inference_report.get("test_labels_read") is not False or inference_report.get("test_metrics_computed") is not False:
        raise ValueError("inference report opened test performance")
    if manifest.get("status") != "locked_for_test_release" or manifest.get("test_performance_opened") is not False:
        raise ValueError("selection manifest is not the frozen validation-only manifest")
    if spec.get("status") != "locked" or spec.get("test_performance_opened") is not False:
        raise ValueError("specification is not locked")
    cohorts = build_cohorts(args.features_db, args.source_db, args.test_input_dir)
    files = find_prediction_files(args.predictions_root)
    thresholds = {(item["family"], item["surface"], item["setting"], int(item["seed"])):
                  float(item["threshold"]) for item in inference_report["outputs"]}
    if len(thresholds) != 180:
        raise ValueError("inference report threshold matrix is incomplete")

    models = {}
    model_seed_rows = []
    subgroup_output = []
    for number, key in enumerate(sorted(files), 1):
        family, surface, setting, seed = key
        cohort = cohorts[surface]
        scores, selected, rows = read_predictions(files[key], cohort)
        point = point_metrics(rows, cohort, thresholds[key])
        model = {"scores": scores, "selected": selected, "point": point,
                 "hit_by_race": _race_hits(cohort, selected)}
        models[key] = model
        model_seed_rows.append(dict(family=family, surface=surface, setting=setting,
                                    seed=seed, threshold=thresholds[key], **point))
        subgroup_output.extend(subgroup_rows(rows, family, surface, setting, seed))
        if number % 10 == 0 or number == len(files):
            print(f"audited {number}/{len(files)} prediction files", flush=True)

    replicates = int(args.bootstrap_replicates)
    ap_cache, hit_cache = _bootstrap_caches(models, cohorts, replicates, 20260913)
    model_summaries = []
    for family in FAMILIES:
        for surface in SURFACES:
            for setting in SETTINGS:
                keys = [(family, surface, setting, seed) for seed in SEEDS]
                point_keys = sorted({key for key in models[keys[0]]["point"]
                                     if key.startswith("entry_") or key.startswith("race_")})
                summary = {"family": family, "surface": surface, "setting": setting}
                for metric in point_keys:
                    values = [models[key]["point"][metric] for key in keys]
                    if all(value is not None for value in values):
                        summary[f"{metric}_mean"] = float(np.mean(values))
                        summary[f"{metric}_seed_sd"] = float(np.std(values, ddof=1))
                ap_dist = np.mean([ap_cache[key] for key in keys], axis=0)
                hit_dist = np.mean([hit_cache[key] for key in keys], axis=0)
                summary["entry_average_precision_ci95"] = _ci(ap_dist)
                summary["race_mean_hit_at_3_ci95"] = _ci(hit_dist)
                model_summaries.append(summary)

    primary = []
    for family in FAMILIES:
        for surface in SURFACES:
            race_count = len(cohorts[surface]["race_ids"])
            for setting_a, setting_b in PRIMARY_CONTRASTS:
                for metric_name, cache in (("entry_average_precision", ap_cache),
                                           ("race_mean_hit_at_3", hit_cache)):
                    distributions = np.mean([
                        cache[(family, surface, setting_a, seed)] -
                        cache[(family, surface, setting_b, seed)] for seed in SEEDS], axis=0)
                    point = [models[(family, surface, setting_a, seed)]["point"][metric_name] -
                             models[(family, surface, setting_b, seed)]["point"][metric_name]
                             for seed in SEEDS]
                    primary.append({"family": family, "surface": surface,
                                    "contrast": f"{setting_a}-{setting_b}", "metric": metric_name,
                                    "estimate": float(np.mean(point)),
                                    "seed_point_estimates": [float(v) for v in point],
                                    "seed_sd": float(np.std(point, ddof=1)),
                                    "ci95": _ci(distributions), "replicates": replicates,
                                    "seed": 20260913, "race_count": race_count,
                                    "distribution": [float(v) for v in distributions]})

    week_ap, week_hit, week_meta = _week_bootstrap(models, cohorts, replicates, 20260913)
    temporal = []
    for family in FAMILIES:
        for surface in SURFACES:
            for setting_a, setting_b in PRIMARY_CONTRASTS:
                for metric_name, cache in (("entry_average_precision", week_ap),
                                           ("race_mean_hit_at_3", week_hit)):
                    distributions = np.mean([
                        cache[(family, surface, setting_a, seed)] -
                        cache[(family, surface, setting_b, seed)] for seed in SEEDS], axis=0)
                    point = [models[(family, surface, setting_a, seed)]["point"][metric_name] -
                             models[(family, surface, setting_b, seed)]["point"][metric_name]
                             for seed in SEEDS]
                    temporal.append({"family": family, "surface": surface,
                                     "contrast": f"{setting_a}-{setting_b}", "metric": metric_name,
                                     "estimate": float(np.mean(point)),
                                     "seed_point_estimates": [float(v) for v in point],
                                     "seed_sd": float(np.std(point, ddof=1)),
                                     "ci95": _ci(distributions), "replicates": replicates,
                                     "seed": 20260913, **week_meta[surface],
                                     "distribution": [float(v) for v in distributions]})

    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite evaluation output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "model_seed_metrics.csv", model_seed_rows)
    write_csv(output / "model_summaries.csv", model_summaries)
    write_csv(output / "subgroup_metrics.csv", subgroup_output)
    final = {
        "status": "complete", "purpose": "locked test evaluation after inference",
        "test_performance_opened": True, "manifest_sha256": sha256_file(args.manifest),
        "spec_sha256": sha256_file(args.spec), "inference_report_sha256": sha256_file(report_path),
        "inference_report": str(report_path), "model_seed_count": len(model_seed_rows),
        "model_summary_count": len(model_summaries), "subgroup_row_count": len(subgroup_output),
        "bootstrap": {"replicates": replicates, "seed": 20260913,
                      "unit": "whole race", "primary_contrasts": [f"{a}-{b}" for a, b in PRIMARY_CONTRASTS]},
        "cohorts": {surface: {"rows": len(cohorts[surface]["keys"]),
                              "races": len(cohorts[surface]["race_ids"]),
                              "graded_races": len(cohorts[surface]["graded_race_indices"]),
                              "date_min": min(cohorts[surface]["race_dates"].values()),
                              "date_max": max(cohorts[surface]["race_dates"].values())}
                    for surface in SURFACES},
        "input_sidecar_sha256": {str(seed): sha256_file(sidecar_path(input_path(args.test_input_dir, seed)))
                                  for seed in SEEDS},
        "source_input_sha256": {"features_db": sha256_file(args.features_db),
                                 "source_db": sha256_file(args.source_db)},
        "model_seed_metrics": "model_seed_metrics.csv", "model_summaries": "model_summaries.csv",
        "subgroup_metrics": "subgroup_metrics.csv", "primary_contrasts": primary,
        "temporal_sensitivity": temporal,
    }
    (output / "locked_test_evaluation_report.json").write_text(
        json.dumps(final, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": final["status"], "model_seed_count": len(model_seed_rows),
                      "model_summary_count": len(model_summaries), "primary_contrasts": len(primary),
                      "temporal_contrasts": len(temporal)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--inference-report", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--features-db", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--test-input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    run(parser.parse_args())
