"""Complete reporting items omitted from the first locked test evaluator.

This is a post-test reporting supplement. It never fits a model, changes a
threshold, selects a seed, or edits the immutable validation manifest.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys

import numpy as np

import evaluate_locked as evaluation
import evaluate_locked_test_results as locked


REPLICATES = 2000
BOOTSTRAP_SEED = 20260913
METRIC_NAMES = (
    "entry_average_precision", "race_mean_hit_at_3",
    "ndcg_binary_all_races", "ndcg_graded_at_3",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    fields = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ci95(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return [float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5))]


def finite_column_mean(values):
    """Mean across model seeds without warnings for undefined subgroup metrics."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    totals = np.nansum(values, axis=0)
    counts = finite.sum(axis=0)
    return np.divide(totals, counts,
                     out=np.full(values.shape[1], np.nan, dtype=np.float64),
                     where=counts > 0)


def race_counts(race_count, replicates=REPLICATES, seed=BOOTSTRAP_SEED):
    if race_count < 1 or replicates < 1:
        raise ValueError("race_count and replicates must be positive")
    draws = np.random.default_rng(seed).integers(
        0, race_count, size=(replicates, race_count), dtype=np.int32)
    counts = np.empty((replicates, race_count), dtype=np.int16)
    for index, draw in enumerate(draws):
        counts[index] = np.bincount(draw, minlength=race_count)
    return counts


def normalized_ap_distribution(scores, labels, race_index, counts, batch_size=100):
    """Exact AP under whole-race resampling, evaluated in NumPy batches."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    race_index = np.asarray(race_index, dtype=np.int32)
    if not len(scores):
        return np.full(len(counts), np.nan, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    ordered_races = race_index[order]
    starts = np.r_[0, np.flatnonzero(
        ordered_scores[1:] != ordered_scores[:-1]) + 1]
    output = np.empty(len(counts), dtype=np.float64)
    for start in range(0, len(counts), batch_size):
        stop = min(start + batch_size, len(counts))
        weights = counts[start:stop, ordered_races].astype(np.float64)
        positive = weights * ordered_labels
        group_weights = np.add.reduceat(weights, starts, axis=1)
        group_positive = np.add.reduceat(positive, starts, axis=1)
        cumulative_weights = np.cumsum(group_weights, axis=1)
        cumulative_positive = np.cumsum(group_positive, axis=1)
        precision = np.divide(
            cumulative_positive, cumulative_weights,
            out=np.zeros_like(cumulative_positive),
            where=cumulative_weights > 0,
        )
        total_positive = positive.sum(axis=1)
        output[start:stop] = np.divide(
            (precision * group_positive).sum(axis=1), total_positive,
            out=np.zeros(stop - start, dtype=np.float64),
            where=total_positive > 0,
        )
    return output


def dcg(values):
    return sum(float(value) / math.log2(index + 2)
               for index, value in enumerate(values[:3]))


def race_ndcg_fast(scores, rows, cohort):
    """Return binary NDCG on all races and graded NDCG on graded races."""
    binary = np.full(len(cohort["race_ids"]), np.nan, dtype=np.float64)
    graded = np.full(len(cohort["race_ids"]), np.nan, dtype=np.float64)
    for race_index, indices in enumerate(cohort["race_rows"]):
        order = sorted(indices, key=lambda index: (
            -float(scores[index]), int(rows[index]["horse_number"]),
            str(rows[index]["horse_id_netkeiba"])))
        binary_rel = [int(cohort["labels"][index]) for index in order]
        binary_idcg = dcg(sorted(binary_rel, reverse=True))
        binary[race_index] = dcg(binary_rel) / binary_idcg if binary_idcg else 0.0
        if all(cohort["finish_pos"][index] > 0 for index in indices):
            graded_rel = [max(0, 4 - int(cohort["finish_pos"][index]))
                          for index in order]
            graded_idcg = dcg(sorted(graded_rel, reverse=True))
            graded[race_index] = dcg(graded_rel) / graded_idcg if graded_idcg else 0.0
    return binary, graded


def training_prevalence(features_db):
    connection = sqlite3.connect(Path(features_db).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        values = {}
        for row in connection.execute(
                "SELECT LOWER(surface), label FROM entries WHERE split = 'train' "
                "AND LOWER(surface) IN ('turf', 'dirt')"):
            surface, label = str(row[0]), int(row[1])
            total, positives = values.get(surface, (0, 0))
            values[surface] = (total + 1, positives + label)
    finally:
        connection.close()
    if set(values) != set(locked.SURFACES):
        raise ValueError(f"training prevalence surfaces are incomplete: {values}")
    return {surface: positives / total
            for surface, (total, positives) in values.items()}


def b0_rows(cohort, prevalence):
    rows = []
    for index, key in enumerate(cohort["keys"]):
        rows.append({
            "race_id": key[0], "horse_id_netkeiba": key[1],
            "horse_number": key[2], "surface": cohort["surface"],
            "field_size": int(cohort["field_sizes"][index]),
            "label": int(cohort["labels"][index]), "score": float(prevalence),
            "history_exact": cohort["history_exact"][index],
            "history_band": cohort["history_band"][index],
            "coverage": cohort["coverage"][index],
        })
    selected = set()
    for indices in cohort["race_rows"]:
        ranked = sorted(indices, key=lambda index: (
            hashlib.sha256(("b0-v1|" + str(rows[index]["race_id"]) + "|" +
                            str(rows[index]["horse_id_netkeiba"]) + "|" +
                            str(rows[index]["horse_number"])).encode("utf-8")).hexdigest()[:16],
            int(rows[index]["horse_number"]), str(rows[index]["horse_id_netkeiba"])))
        selected.update(ranked[:3])
        for rank, index in enumerate(ranked):
            rows[index]["ranking_score"] = (len(ranked) - rank) / len(ranked)
    for index, row in enumerate(rows):
        row["selected_top3"] = int(index in selected)
    return rows


def point_ndcg(rows, cohort):
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    binary, graded = race_ndcg_fast(scores, rows, cohort)
    return (float(np.nanmean(binary)), float(np.nanmean(graded)),
            int(np.isfinite(graded).sum()), binary, graded)


def b0_report(cohorts, prevalence, replicates):
    rows = []
    details = {}
    for surface in locked.SURFACES:
        cohort = cohorts[surface]
        values = b0_rows(cohort, prevalence[surface])
        entry = evaluation.entry_metrics(values, threshold=0.5)
        # Keep prevalence probabilities for entry metrics and hash order for ranking.
        ranked_values = [dict(row, score=row["ranking_score"]) for row in values]
        race = evaluation.race_metrics(ranked_values)
        binary, graded, graded_count, binary_by_race, graded_by_race = point_ndcg(ranked_values, cohort)
        expected_random = float(np.mean([
            sum(cohort["labels"][index] for index in indices) /
            len(indices) for indices in cohort["race_rows"]]))
        all_counts = race_counts(len(cohort["race_ids"]), replicates)
        graded_indices = np.asarray(cohort["graded_race_indices"], dtype=np.int32)
        graded_counts = race_counts(len(graded_indices), replicates)
        row_labels = cohort["labels"].astype(np.float64)
        selected = np.asarray([values[index]["selected_top3"]
                               for index in range(len(values))], dtype=np.float64)
        entry_positive_by_race = np.bincount(
            cohort["race_index"], weights=row_labels,
            minlength=len(cohort["race_ids"])).astype(np.float64)
        entry_count_by_race = np.bincount(
            cohort["race_index"], weights=np.ones(len(row_labels)),
            minlength=len(cohort["race_ids"])).astype(np.float64)
        hit_by_race = np.bincount(
            cohort["race_index"], weights=row_labels * selected,
            minlength=len(cohort["race_ids"])).astype(np.float64)
        ap = (all_counts @ entry_positive_by_race) / (
            all_counts @ entry_count_by_race)
        hit = (all_counts @ hit_by_race) / (
            3.0 * len(cohort["race_ids"]))
        graded_values = graded_by_race[graded_indices]
        graded_dist = (graded_counts @ graded_values) / len(graded_indices)
        binary_dist = (all_counts @ binary_by_race) / len(cohort["race_ids"])
        details[surface] = {
            "prevalence": float(prevalence[surface]),
            "expected_random_hit_at_3": expected_random,
            "entry_average_precision_ci95": ci95(ap),
            "race_mean_hit_at_3_ci95": ci95(hit),
            "ndcg_binary_all_races_ci95": ci95(binary_dist),
            "ndcg_binary_graded_races_ci95": ci95(
                (graded_counts @ binary_by_race[graded_indices]) / len(graded_indices)),
            "ndcg_graded_at_3_ci95": ci95(graded_dist),
            "replicates": replicates,
            "graded_races": graded_count,
        }
        rows.append({
            "family": "B0", "surface": surface, "setting": "B0", "seed": "reference",
            "training_prevalence": float(prevalence[surface]),
            "expected_random_hit_at_3": expected_random,
            **entry, **{f"race_{key}": value for key, value in race.items()},
            "graded_races": graded_count,
            "ndcg_binary_all_races": binary,
            "ndcg_binary_graded_races": float(np.nanmean(binary_by_race[graded_indices])),
            "ndcg_graded_at_3": graded,
            "entry_average_precision_ci95": json.dumps(ci95(ap)),
            "race_mean_hit_at_3_ci95": json.dumps(ci95(hit)),
            "ndcg_binary_all_races_ci95": json.dumps(ci95(binary_dist)),
            "ndcg_binary_graded_races_ci95": json.dumps(ci95(
                (graded_counts @ binary_by_race[graded_indices]) / len(graded_indices))),
            "ndcg_graded_at_3_ci95": json.dumps(ci95(graded_dist)),
            "bootstrap_replicates": replicates,
        })
    return rows, details


def subgroup_distribution(scores, rows, cohort, field, category, counts, replicates):
    mask = np.asarray([row.get(field) == category for row in rows], dtype=bool)
    labels = cohort["labels"].astype(np.float64)
    selected = np.asarray([int(row["selected_top3"]) for row in rows], dtype=np.float64)
    race_index = cohort["race_index"]
    subgroup_labels = labels[mask]
    subgroup_scores = np.asarray(scores, dtype=np.float64)[mask]
    subgroup_races = race_index[mask]
    ap = normalized_ap_distribution(subgroup_scores, subgroup_labels,
                                     subgroup_races, counts)
    hit_by_race = np.bincount(
        race_index, weights=labels * selected * mask,
        minlength=len(cohort["race_ids"])).astype(np.float64)
    selected_by_race = np.bincount(
        race_index, weights=selected * mask,
        minlength=len(cohort["race_ids"])).astype(np.float64)
    hit = (counts @ hit_by_race) / (3.0 * len(cohort["race_ids"]))
    selected_count = counts @ selected_by_race
    hits = counts @ hit_by_race
    precision = np.divide(hits, selected_count,
                          out=np.full(replicates, np.nan),
                          where=selected_count > 0)
    return {"entry_average_precision": ap,
            "hit_contribution": hit,
            "selection_precision": precision}


def subgroup_report(models, cohorts, point_rows, replicates):
    definitions = (("history_exact", locked.HISTORY_EXACT),
                   ("history_band", locked.HISTORY_BAND),
                   ("coverage", locked.COVERAGE))
    output = []
    completed = 0
    counts = {surface: race_counts(len(cohorts[surface]["race_ids"]), replicates)
              for surface in locked.SURFACES}
    point_by_key = {}
    for row in point_rows:
        key = (row["family"], row["surface"], row["setting"], int(row["seed"]))
        point_by_key[(key[0], key[1], key[2], key[3], row["subgroup_type"],
                      row["subgroup"])] = row
    for family in locked.FAMILIES:
        for surface in locked.SURFACES:
            for setting in locked.SETTINGS:
                keys = [(family, surface, setting, seed) for seed in locked.SEEDS]
                for field, categories in definitions:
                    for category in categories:
                        distributions = []
                        for key in keys:
                            result = subgroup_distribution(
                                models[key]["scores"], models[key]["rows"],
                                cohorts[surface], field, category,
                                counts[surface], replicates)
                            distributions.append(result)
                        values = {}
                        for metric in ("entry_average_precision", "hit_contribution",
                                       "selection_precision"):
                            stacked = np.asarray([item[metric] for item in distributions])
                            values[metric] = finite_column_mean(stacked)
                        seed_points = [point_by_key[(family, surface, setting, seed,
                                                    field, category)] for seed in locked.SEEDS]
                        ap_points = [float(row["entry_average_precision"])
                                     for row in seed_points
                                     if row["entry_average_precision"] not in ("", "None")]
                        precision_points = [float(row["selection_precision"])
                                           for row in seed_points
                                           if row["selection_precision"] not in ("", "None")]
                        output.append({
                            "family": family, "surface": surface, "setting": setting,
                            "subgroup_type": field, "subgroup": category,
                            "entry_count_mean": float(np.mean([int(row["entry_count"])
                                                                for row in seed_points])),
                            "entry_count_seed_sd": float(np.std(
                                [int(row["entry_count"]) for row in seed_points], ddof=1)),
                            "entry_average_precision_mean": float(np.mean(ap_points))
                                if ap_points else None,
                            "entry_average_precision_ci95": json.dumps(
                                ci95(values["entry_average_precision"])),
                            "full_race_selected_count_mean": float(np.mean(
                                [int(row["full_race_selected_count"])
                                 for row in seed_points])),
                            "full_race_selected_positive_hits_mean": float(np.mean(
                                [int(row["full_race_selected_positive_hits"])
                                 for row in seed_points])),
                            "hit_contribution_mean": float(np.mean(
                                [float(row["hit_contribution"]) for row in seed_points])),
                            "hit_contribution_ci95": json.dumps(
                                ci95(values["hit_contribution"])),
                            "selection_precision_mean": float(np.mean(precision_points))
                                if precision_points else None,
                            "selection_precision_ci95": json.dumps(
                                ci95(values["selection_precision"])),
                            "bootstrap_replicates": replicates,
                        })
                        completed += 1
                        if completed % 30 == 0:
                            print(json.dumps({"stage": "subgroup_bootstrap",
                                              "completed": completed,
                                              "total": 540}), flush=True)
    return output


def load_point_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def model_summary_rows(point_rows, models):
    result = []
    numeric = (
        "entry_average_precision", "entry_roc_auc", "entry_brier",
        "entry_threshold", "entry_precision", "entry_recall", "entry_f1",
        "entry_predicted_positive", "race_complete_races", "race_mean_hit_at_3",
        "race_at_least_one_hit_rate", "race_at_least_two_hit_rate",
        "race_exact_set_rate", "race_exact_set_races", "race_dead_heat_races",
        "race_dead_heat_mean_hit_at_3", "graded_races", "ndcg_binary_at_3",
        "ndcg_graded_at_3", "ndcg_binary_all_races",
        "ndcg_binary_graded_races",
    )
    by_key = {(row["family"], row["surface"], row["setting"], int(row["seed"])): row
              for row in point_rows}
    for family in locked.FAMILIES:
        for surface in locked.SURFACES:
            for setting in locked.SETTINGS:
                seed_rows = [by_key[(family, surface, setting, seed)]
                             for seed in locked.SEEDS]
                row = {"family": family, "surface": surface, "setting": setting}
                for field in numeric:
                    values = [float(seed_row[field]) for seed_row in seed_rows
                              if seed_row.get(field) not in (None, "", "None")]
                    if len(values) == len(seed_rows):
                        row[f"{field}_mean"] = float(np.mean(values))
                        row[f"{field}_seed_sd"] = float(np.std(values, ddof=1))
                for metric in ("ndcg_binary_all_races", "ndcg_binary_graded_races",
                               "ndcg_graded_at_3"):
                    dist = np.mean([models[(family, surface, setting, seed)][
                        f"{metric}_bootstrap"] for seed in locked.SEEDS], axis=0)
                    row[f"{metric}_ci95"] = json.dumps(ci95(dist))
                result.append(row)
    return result


def surface_rows(models, cohorts, base_report, replicates):
    """Create turf-minus-dirt rows from within-surface contrast distributions."""
    existing = {(row["family"], row["surface"], row["contrast"], row["metric"]): row
                for row in base_report["primary_contrasts"]}
    output = []
    ndcg_cache = {}
    for family in locked.FAMILIES:
        for surface in locked.SURFACES:
            counts = race_counts(len(cohorts[surface]["race_ids"]), replicates)
            graded = race_counts(len(cohorts[surface]["graded_race_indices"]), replicates)
            for setting_a, setting_b in locked.PRIMARY_CONTRASTS:
                contrast = f"{setting_a}-{setting_b}"
                for metric in ("ndcg_binary_all_races", "ndcg_graded_at_3"):
                    source_counts = counts if metric == "ndcg_binary_all_races" else graded
                    index = (np.asarray(cohorts[surface]["graded_race_indices"], dtype=np.int32)
                             if metric == "ndcg_graded_at_3" else None)
                    distributions = []
                    points = []
                    for seed in locked.SEEDS:
                        left = models[(family, surface, setting_a, seed)][metric + "_by_race"]
                        right = models[(family, surface, setting_b, seed)][metric + "_by_race"]
                        if index is not None:
                            left, right = left[index], right[index]
                        distributions.append(source_counts @ (left - right) / len(left))
                        points.append(models[(family, surface, setting_a, seed)][metric] -
                                      models[(family, surface, setting_b, seed)][metric])
                    ndcg_cache[(family, surface, contrast, metric)] = {
                        "estimate": float(np.mean(points)),
                        "distribution": np.mean(distributions, axis=0),
                    }
    for family in locked.FAMILIES:
        for setting_a, setting_b in locked.PRIMARY_CONTRASTS:
            contrast = f"{setting_a}-{setting_b}"
            for metric in METRIC_NAMES:
                if metric in ("ndcg_binary_all_races", "ndcg_graded_at_3"):
                    turf = ndcg_cache[(family, "turf", contrast, metric)]
                    dirt = ndcg_cache[(family, "dirt", contrast, metric)]
                else:
                    turf = existing[(family, "turf", contrast, metric)]
                    dirt = existing[(family, "dirt", contrast, metric)]
                    turf_distribution = turf["distribution"]
                    dirt_distribution = dirt["distribution"]
                    if isinstance(turf_distribution, str):
                        turf_distribution = json.loads(turf_distribution)
                    if isinstance(dirt_distribution, str):
                        dirt_distribution = json.loads(dirt_distribution)
                    turf = {"estimate": float(turf["estimate"]),
                            "distribution": np.asarray(turf_distribution)}
                    dirt = {"estimate": float(dirt["estimate"]),
                            "distribution": np.asarray(dirt_distribution)}
                output.append({
                    "family": family, "contrast": contrast, "metric": metric,
                    "estimate_turf_minus_dirt": turf["estimate"] - dirt["estimate"],
                    "ci95": json.dumps(ci95(turf["distribution"] - dirt["distribution"])),
                    "replicates": replicates,
                })
    return output


def run(args):
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite reporting supplement: {output}")
    manifest = read_json(args.manifest)
    spec = read_json(args.spec)
    inference_report = read_json(args.inference_report)
    base_report = read_json(args.base_evaluation_report)
    if manifest.get("status") != "locked_for_test_release" or manifest.get("test_performance_opened") is not False:
        raise ValueError("the immutable validation-only manifest is not intact")
    if spec.get("status") != "locked" or spec.get("test_performance_opened") is not False:
        raise ValueError("the locked specification is not intact")
    if inference_report.get("status") != "complete" or inference_report.get("model_outputs") != 180:
        raise ValueError("locked inference report is incomplete")
    if inference_report.get("test_labels_read") is not False or inference_report.get("test_metrics_computed") is not False:
        raise ValueError("locked inference report opened metrics")
    if base_report.get("status") != "complete" or base_report.get("model_seed_count") != 180:
        raise ValueError("base evaluation report is incomplete")
    if int(args.bootstrap_replicates) != REPLICATES:
        raise ValueError("the frozen reporting supplement requires 2,000 replicates")
    cohorts = locked.build_cohorts(args.features_db, args.source_db, args.test_input_dir)
    files = locked.find_prediction_files(args.predictions_root)
    thresholds = {(item["family"], item["surface"], item["setting"], int(item["seed"])):
                  float(item["threshold"]) for item in inference_report["outputs"]}
    if len(thresholds) != 180:
        raise ValueError("inference threshold matrix is incomplete")
    point_rows = load_point_rows(args.base_model_seed_metrics)
    if len(point_rows) != 180:
        raise ValueError("base model-seed table is incomplete")
    models = {}
    model_keys = sorted(files)
    for model_index, key in enumerate(model_keys, start=1):
        scores, selected, rows = locked.read_predictions(files[key], cohorts[key[1]])
        if thresholds[key] < 0 or thresholds[key] > 1:
            raise ValueError(f"invalid threshold in inference report: {key}")
        binary, graded = race_ndcg_fast(scores, rows, cohorts[key[1]])
        model = {"scores": scores, "selected": selected, "rows": rows,
                 "ndcg_binary_all_races_by_race": binary,
                 "ndcg_graded_at_3_by_race": graded}
        model["ndcg_binary_all_races"] = float(np.nanmean(binary))
        model["ndcg_graded_at_3"] = float(np.nanmean(graded))
        for metric in ("ndcg_binary_all_races", "ndcg_graded_at_3"):
            surface = key[1]
            race_indices = (np.arange(len(cohorts[surface]["race_ids"]), dtype=np.int32)
                            if metric == "ndcg_binary_all_races"
                            else np.asarray(cohorts[surface]["graded_race_indices"], dtype=np.int32))
            values = model[metric + "_by_race"][race_indices]
            counts = race_counts(len(race_indices), args.bootstrap_replicates)
            model[metric + "_bootstrap"] = (counts @ values) / len(race_indices)
        graded_indices = np.asarray(cohorts[key[1]]["graded_race_indices"], dtype=np.int32)
        binary_graded = binary[graded_indices]
        graded_counts = race_counts(len(graded_indices), args.bootstrap_replicates)
        model["ndcg_binary_graded_races"] = float(np.nanmean(binary_graded))
        model["ndcg_binary_graded_races_by_race"] = binary_graded
        model["ndcg_binary_graded_races_bootstrap"] = (
            graded_counts @ binary_graded) / len(graded_indices)
        models[key] = model
        if model_index % 30 == 0:
            print(json.dumps({"stage": "model_load_and_ndcg",
                              "completed": model_index,
                              "total": len(model_keys)}), flush=True)
    prevalence = training_prevalence(args.features_db)
    b0_rows_output, b0_details = b0_report(cohorts, prevalence, args.bootstrap_replicates)
    extended_seed_rows = []
    by_key = {(row["family"], row["surface"], row["setting"], int(row["seed"])): row
              for row in point_rows}
    for key in sorted(by_key):
        row = dict(by_key[key])
        model = models[key]
        row["ndcg_binary_all_races"] = model["ndcg_binary_all_races"]
        row["ndcg_binary_graded_races"] = float(np.nanmean(
            model["ndcg_binary_all_races_by_race"][cohorts[key[1]]["graded_race_indices"]]))
        row["ndcg_graded_at_3"] = model["ndcg_graded_at_3"]
        extended_seed_rows.append(row)
    model_summaries = model_summary_rows(extended_seed_rows, models)
    subgroup_points = load_point_rows(args.base_subgroup_metrics)
    subgroup_summaries = subgroup_report(models, cohorts, subgroup_points,
                                         args.bootstrap_replicates)
    surface_contrasts = surface_rows(models, cohorts, base_report,
                                     args.bootstrap_replicates)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "b0_metrics.csv", b0_rows_output)
    write_csv(output / "model_seed_metrics_extended.csv", extended_seed_rows)
    write_csv(output / "model_summaries_extended.csv", model_summaries)
    write_csv(output / "subgroup_metrics_with_ci.csv", subgroup_summaries)
    write_csv(output / "surface_contrasts.csv", surface_contrasts)
    report = {
        "status": "complete",
        "purpose": "post-test completion of frozen reporting requirements",
        "spec_sha256": sha256_file(args.spec),
        "manifest_sha256": sha256_file(args.manifest),
        "inference_report_sha256": sha256_file(args.inference_report),
        "base_evaluation_report_sha256": sha256_file(args.base_evaluation_report),
        "gap_audit_sha256": sha256_file(args.gap_audit),
        "bootstrap": {"replicates": args.bootstrap_replicates,
                      "seed": BOOTSTRAP_SEED, "unit": "whole race"},
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cohorts": {surface: {"rows": len(cohorts[surface]["keys"]),
                              "races": len(cohorts[surface]["race_ids"]),
                              "graded_races": len(cohorts[surface]["graded_race_indices"])}
                    for surface in locked.SURFACES},
        "b0": b0_details,
        "outputs": {
            "b0_metrics": "b0_metrics.csv",
            "model_seed_metrics_extended": "model_seed_metrics_extended.csv",
            "model_summaries_extended": "model_summaries_extended.csv",
            "subgroup_metrics_with_ci": "subgroup_metrics_with_ci.csv",
            "surface_contrasts": "surface_contrasts.csv",
        },
        "model_seed_count": len(extended_seed_rows),
        "model_summary_count": len(model_summaries),
        "subgroup_summary_count": len(subgroup_summaries),
        "surface_contrast_count": len(surface_contrasts),
        "test_labels_read": True,
        "test_performance_computed": True,
        "no_new_training": True,
    }
    write_json(output / "final_reporting_supplement_report.json", report)
    print(json.dumps({"status": report["status"],
                      "b0_surfaces": len(b0_rows_output),
                      "model_seed_count": len(extended_seed_rows),
                      "model_summary_count": len(model_summaries),
                      "subgroup_summary_count": len(subgroup_summaries),
                      "surface_contrast_count": len(surface_contrasts)}, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--inference-report", type=Path, required=True)
    parser.add_argument("--base-evaluation-report", type=Path, required=True)
    parser.add_argument("--base-model-seed-metrics", type=Path, required=True)
    parser.add_argument("--base-subgroup-metrics", type=Path, required=True)
    parser.add_argument("--gap-audit", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--features-db", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--test-input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=REPLICATES)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
