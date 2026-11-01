"""Evaluation helpers for the locked, whole-race experiment protocol.

This module does not read files or fit models. Callers provide prediction rows
and, when needed, a strict row-key metadata join supplies finish positions.
"""
import csv
import gzip
import math
from pathlib import Path


KEY = ("race_id", "horse_id_netkeiba", "horse_number")
SEEDS = (0, 1, 2)


def _key(row):
    try:
        return tuple(str(row[name]) for name in KEY)
    except KeyError as error:
        raise ValueError(f"missing prediction key column: {error.args[0]}") from error


def _score(row):
    try:
        value = float(row["score"] if "score" in row else row["prediction"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("prediction score is required and must be numeric") from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("prediction score must be finite and in [0, 1]")
    return value


def _integer(value, name):
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be integer-valued") from error
    if not math.isfinite(numeric) or numeric != parsed:
        raise ValueError(f"{name} must be integer-valued")
    return parsed


def _label(row):
    try:
        value = _integer(row["label"], "label")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("binary label is required") from error
    if value not in (0, 1):
        raise ValueError("labels must be binary")
    return value


def _group(rows, require_field_size=True):
    groups = {}
    for row in rows:
        groups.setdefault(str(row["race_id"]), []).append(row)
    if not groups:
        raise ValueError("prediction rows are empty")
    for race_id, entries in groups.items():
        sizes = {int(row["field_size"]) for row in entries
                 if row.get("field_size") not in (None, "")}
        if require_field_size and (len(sizes) != 1 or len(entries) != next(iter(sizes))):
            raise ValueError(f"race {race_id} is incomplete or has fewer than three entries")
        if len(entries) < 3:
            raise ValueError(f"race {race_id} has fewer than three entries")
    return groups


def validate_rows(rows, require_field_size=True):
    """Validate and normalize complete prediction rows without changing order."""
    normalized = []
    seen = set()
    for raw in rows:
        row = dict(raw)
        row["horse_number"] = _integer(row.get("horse_number"), "horse_number")
        key = _key(row)
        if key in seen:
            raise ValueError(f"duplicate prediction key: {key}")
        seen.add(key)
        row["race_id"], row["horse_id_netkeiba"] = key[:2]
        row["score"] = _score(row)
        row["label"] = _label(row)
        if row["horse_number"] < 1:
            raise ValueError("horse_number must be positive")
        if "field_size" in row and row["field_size"] not in (None, ""):
            row["field_size"] = _integer(row["field_size"], "field_size")
            if row["field_size"] < 3:
                raise ValueError("field_size must be at least three")
        elif require_field_size:
            raise ValueError("field_size is required to verify complete races")
        if "finish_pos" in row and row["finish_pos"] not in (None, ""):
            row["finish_pos"] = _integer(row["finish_pos"], "finish_pos")
            if row["finish_pos"] < 1:
                raise ValueError("finish_pos must be positive")
            if row["label"] != int(row["finish_pos"] <= 3):
                raise ValueError(f"label disagrees with finish_pos for {key}")
        normalized.append(row)
    _group(normalized, require_field_size=require_field_size)
    return normalized


def attach_finish_positions(predictions, metadata):
    """Join finish positions by the exact three-column prediction key.

    The metadata must cover exactly the prediction cohort. This prevents a
    missing or many-to-many join from silently changing graded NDCG or labels.
    """
    rows = validate_rows(predictions, require_field_size=False)
    metadata_by_key = {}
    for raw in metadata:
        row = dict(raw)
        row["horse_number"] = _integer(row.get("horse_number"), "horse_number")
        key = _key(row)
        if key in metadata_by_key:
            raise ValueError(f"duplicate metadata key: {key}")
        if "finish_pos" not in row or row["finish_pos"] in (None, ""):
            raise ValueError(f"metadata finish_pos is missing for {key}")
        finish_pos = _integer(row["finish_pos"], "metadata finish_pos")
        if finish_pos < 1:
            raise ValueError(f"metadata finish_pos is invalid for {key}")
        metadata_by_key[key] = (finish_pos, row)
    row_keys = {_key(row) for row in rows}
    if row_keys != set(metadata_by_key):
        missing = sorted(row_keys - set(metadata_by_key))
        extra = sorted(set(metadata_by_key) - row_keys)
        raise ValueError(f"finish_pos metadata key mismatch; missing={missing}, extra={extra}")
    joined = []
    for row in rows:
        row_key = _key(row)
        finish_pos, meta = metadata_by_key[row_key]
        if "label" in meta and int(meta["label"]) != row["label"]:
            raise ValueError(f"metadata label mismatch for {row_key}")
        copy = dict(row)
        copy["finish_pos"] = finish_pos
        if "field_size" in copy and meta.get("field_size") not in (None, ""):
            if copy["field_size"] != _integer(meta["field_size"], "metadata field_size"):
                raise ValueError(f"metadata field_size mismatch for {row_key}")
        if "field_size" not in copy and meta.get("field_size") not in (None, ""):
            copy["field_size"] = meta["field_size"]
        if "field_size" not in copy:
            raise ValueError(f"field_size metadata is missing for {row_key}")
        joined.append(copy)
    return validate_rows(joined)


def read_prediction_csv(path):
    """Read a runner prediction CSV and validate its complete-race cohort."""
    path = Path(path)
    # Kaggle may materialize a stored ``.csv.gz`` as an uncompressed ``.csv``
    # when an output dataset is mounted. Treat that transport rename as an
    # alias while keeping the bytes and schema checks strict.
    if not path.is_file() and path.name.endswith(".csv.gz"):
        alias = path.with_suffix("")
        if alias.is_file():
            path = alias
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = set(KEY) | {"surface", "label", "prediction", "selected_top3"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("prediction CSV lacks required key, label, or score columns")
        return validate_rows(list(reader), require_field_size=False)


def audit_selection_flags(rows, metadata=None):
    """Validate persisted top-three flags against the full-race ranking.

    The audit is deliberately separate from metric calculation: callers can
    verify a runner artifact's stored selections before using its scores.
    When the CSV omits field size or finish position, ``metadata`` must be an
    exact row-key join accepted by :func:`attach_finish_positions`.
    """
    strict = metadata is not None
    checked = (attach_finish_positions(rows, metadata) if strict
               else validate_rows(rows, require_field_size=False))
    summaries = []
    for race_id, entries in _group(checked, require_field_size=strict).items():
        flagged = []
        for row in entries:
            if "selected_top3" not in row:
                raise ValueError(f"selected_top3 is missing for race {race_id}")
            try:
                selected = int(row["selected_top3"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"selected_top3 is not binary for race {race_id}") from error
            if selected not in (0, 1):
                raise ValueError(f"selected_top3 is not binary for race {race_id}")
            if selected:
                flagged.append(_key(row))
        expected = {_key(row) for row in _rank(entries)[:3]}
        if set(flagged) != expected or len(flagged) != 3:
            raise ValueError(f"selected_top3 does not match deterministic ranking for race {race_id}")
        summaries.append({"race_id": str(race_id), "selected_count": len(flagged)})
    return {"complete_races": len(summaries), "races": summaries, "rows": checked}


def _sklearn_metrics():
    try:
        from sklearn.metrics import (average_precision_score, brier_score_loss,
                                     roc_auc_score)
    except ImportError as error:
        raise RuntimeError("scikit-learn metrics are required for locked evaluation") from error
    return average_precision_score, brier_score_loss, roc_auc_score


def threshold_metrics(rows, threshold=0.5):
    rows = validate_rows(rows)
    if not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1]")
    tp = sum(row["label"] == 1 and row["score"] >= threshold for row in rows)
    fp = sum(row["label"] == 0 and row["score"] >= threshold for row in rows)
    fn = sum(row["label"] == 1 and row["score"] < threshold for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": float(threshold), "precision": precision, "recall": recall,
            "f1": f1, "predicted_positive": tp + fp}


def entry_metrics(rows, threshold=0.5):
    rows = validate_rows(rows)
    average_precision_score, brier_score_loss, roc_auc_score = _sklearn_metrics()
    labels = [row["label"] for row in rows]
    scores = [row["score"] for row in rows]
    values = {
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": (float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None),
        "brier": float(brier_score_loss(labels, scores)),
    }
    values.update(threshold_metrics(rows, threshold))
    return values


def _rank(entries):
    return sorted(entries, key=lambda row: (-row["score"], row["horse_number"],
                                             row["horse_id_netkeiba"]))


def _dcg(relevances):
    return sum(float(relevance) / math.log2(index + 2)
               for index, relevance in enumerate(relevances[:3]))


def _race_summary(entries, require_finish=False):
    ranked = _rank(entries)
    selected = ranked[:3]
    positive = [row for row in entries if row["label"] == 1]
    selected_keys = {_key(row) for row in selected}
    positive_keys = {_key(row) for row in positive}
    hits = sum(row["label"] for row in selected)
    result = {
        "race_id": str(entries[0]["race_id"]),
        "selected_keys": selected_keys,
        "positive_keys": positive_keys,
        "hits": hits,
        "hit_at_3": hits / 3.0,
        "at_least_one": int(hits >= 1),
        "at_least_two": int(hits >= 2),
        "positive_count": len(positive),
        "dead_heat": len(positive) > 3,
    }
    if require_finish:
        if any(row.get("finish_pos") in (None, "") for row in entries):
            raise ValueError("graded NDCG requires finish_pos or a strict metadata join")
        binary = [int(row["finish_pos"] <= 3) for row in ranked]
        graded = [max(0, 4 - int(row["finish_pos"])) for row in ranked]
        ideal_binary = sorted(binary, reverse=True)
        ideal_graded = sorted(graded, reverse=True)
        binary_idcg, graded_idcg = _dcg(ideal_binary), _dcg(ideal_graded)
        result["ndcg_binary"] = _dcg(binary) / binary_idcg if binary_idcg else 0.0
        result["ndcg_graded"] = _dcg(graded) / graded_idcg if graded_idcg else 0.0
    return result


def race_metrics(rows, require_graded=False):
    """Compute whole-race selection, exact-set, dead-heat, and NDCG metrics."""
    rows = validate_rows(rows)
    summaries = [_race_summary(entries, require_graded)
                 for entries in _group(rows).values()]
    exact = [summary for summary in summaries if summary["positive_count"] == 3]
    dead = [summary for summary in summaries if summary["dead_heat"]]
    values = {
        "complete_races": len(summaries),
        "mean_hit_at_3": sum(x["hit_at_3"] for x in summaries) / len(summaries),
        "at_least_one_hit_rate": sum(x["at_least_one"] for x in summaries) / len(summaries),
        "at_least_two_hit_rate": sum(x["at_least_two"] for x in summaries) / len(summaries),
        "exact_set_rate": (sum(x["selected_keys"] == x["positive_keys"] for x in exact) / len(exact)
                           if exact else None),
        "exact_set_races": len(exact),
        "dead_heat_races": len(dead),
        "dead_heat_mean_hit_at_3": (sum(x["hit_at_3"] for x in dead) / len(dead)
                                     if dead else None),
    }
    if require_graded:
        values["ndcg_binary_at_3"] = sum(x["ndcg_binary"] for x in summaries) / len(summaries)
        values["ndcg_graded_at_3"] = sum(x["ndcg_graded"] for x in summaries) / len(summaries)
    return values


def evaluate(rows, threshold=0.5, metadata=None):
    """Return entry and whole-race metrics, joining finish positions strictly."""
    if metadata is not None:
        rows = attach_finish_positions(rows, metadata)
    else:
        rows = validate_rows(rows)
    entry = entry_metrics(rows, threshold)
    race = race_metrics(rows, require_graded=any("finish_pos" in row for row in rows))
    return dict(entry=entry, race=race)


def subgroup_metrics(rows, group_field, categories):
    """Report entry metrics and original full-race selection contributions."""
    rows = validate_rows(rows)
    categories = tuple(categories)
    if len(set(categories)) != len(categories):
        raise ValueError("subgroup categories must be unique")
    summaries = {str(race_id): _race_summary(entries)
                 for race_id, entries in _group(rows).items()}
    complete_races = len(summaries)
    selected = {key for summary in summaries.values() for key in summary["selected_keys"]}
    average_precision_score, _, _ = _sklearn_metrics()
    output = {}
    for category in categories:
        members = [row for row in rows if row.get(group_field) == category]
        keys = {_key(row) for row in members}
        subgroup_hits = sum(int(_key(row) in selected and row["label"] == 1) for row in members)
        selection_count = sum(int(key in selected) for key in keys)
        labels = [row["label"] for row in members]
        scores = [row["score"] for row in members]
        output[str(category)] = {
            "entry_count": len(members),
            "entry_average_precision": (float(average_precision_score(labels, scores))
                                         if members else None),
            "full_race_selected_count": selection_count,
            "full_race_selected_positive_hits": subgroup_hits,
            "hit_contribution": (subgroup_hits / (3.0 * complete_races)
                                 if complete_races else None),
            "selection_precision": (subgroup_hits / selection_count if selection_count else None),
        }
    return output


def bootstrap_race_indices(race_ids, replicates=2000, seed=20260913):
    """Create shared whole-race bootstrap indices using NumPy's default_rng."""
    if len(race_ids) == 0 or len(set(race_ids)) != len(race_ids):
        raise ValueError("race_ids must be nonempty and unique")
    replicates = _integer(replicates, "replicates")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("NumPy is required for locked bootstrap resampling") from error
    return np.asarray(np.random.default_rng(seed).integers(
        0, len(race_ids), size=(replicates, len(race_ids))))


def paired_bootstrap(seed_a, seed_b, metric, race_indices=None,
                     replicates=2000, seed=20260913):
    """Bootstrap paired A-minus-B contrasts over matched whole races.

    ``seed_a`` and ``seed_b`` map classifier seed IDs to validated prediction
    rows. AP is recomputed on concatenated sampled entries; race metrics use
    per-race values before averaging. The returned intervals are descriptive.
    """
    if set(seed_a) != set(SEEDS) or set(seed_b) != set(SEEDS):
        raise ValueError("paired bootstrap requires seed rows for 0, 1, and 2")
    if metric not in ("average_precision", "mean_hit_at_3", "ndcg_binary_at_3", "ndcg_graded_at_3"):
        raise ValueError("unsupported bootstrap metric")
    replicates = _integer(replicates, "replicates")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    per_race_metric = {"mean_hit_at_3": "hit_at_3",
                       "ndcg_binary_at_3": "ndcg_binary",
                       "ndcg_graded_at_3": "ndcg_graded"}.get(metric)
    normalized_a = {seed_id: validate_rows(rows) for seed_id, rows in seed_a.items()}
    normalized_b = {seed_id: validate_rows(rows) for seed_id, rows in seed_b.items()}
    groups_a = {s: _group(normalized_a[s]) for s in SEEDS}
    groups_b = {s: _group(normalized_b[s]) for s in SEEDS}
    races = sorted(groups_a[0])
    if any(sorted(groups_a[s]) != races or sorted(groups_b[s]) != races
           for s in SEEDS):
        raise ValueError("paired bootstrap requires identical complete race IDs")
    reference_keys = {_key(row) for row in normalized_a[0]}
    if any({_key(row) for row in normalized_a[s]} != reference_keys or
           {_key(row) for row in normalized_b[s]} != reference_keys for s in SEEDS):
        raise ValueError("paired bootstrap requires identical row keys across matched seeds")
    reference_labels = {_key(row): row["label"] for row in normalized_a[0]}
    if any({_key(row): row["label"] for row in normalized_a[s]} != reference_labels or
           {_key(row): row["label"] for row in normalized_b[s]} != reference_labels for s in SEEDS):
        raise ValueError("paired bootstrap requires identical row labels across matched seeds")
    for s in SEEDS:
        labels_a = {_key(row): row["label"] for row in normalized_a[s]}
        labels_b = {_key(row): row["label"] for row in normalized_b[s]}
        if labels_a != labels_b:
            raise ValueError("paired bootstrap requires identical row labels")
    if metric.startswith("ndcg"):
        reference_finish = {_key(row): row.get("finish_pos") for row in normalized_a[0]}
        if any(any(value in (None, "") for value in finishes.values()) or finishes != reference_finish
               for finishes in ({_key(row): row.get("finish_pos") for row in normalized_a[s]} for s in SEEDS)):
            raise ValueError("NDCG bootstrap requires identical finish positions across matched seeds")
        if any({_key(row): row.get("finish_pos") for row in normalized_b[s]} != reference_finish
               for s in SEEDS):
            raise ValueError("NDCG bootstrap requires identical finish positions across matched arms")
    if race_indices is None:
        race_indices = bootstrap_race_indices(races, replicates, seed)
    try:
        import numpy as np
    except ImportError:
        if (not isinstance(race_indices, (list, tuple)) or len(race_indices) != replicates or
                any(not isinstance(row, (list, tuple)) or len(row) != len(races)
                    for row in race_indices) or
                any(isinstance(index, bool) or not isinstance(index, int) or
                    index < 0 or index >= len(races)
                    for row in race_indices for index in row)):
            raise ValueError("race indices must be a 2D integer array with the expected shape")
    else:
        try:
            race_indices = np.asarray(race_indices)
        except (ValueError, TypeError) as error:
            raise ValueError("race indices must be a rectangular integer array") from error
        if (race_indices.ndim != 2 or race_indices.shape != (replicates, len(races)) or
                not np.issubdtype(race_indices.dtype, np.integer)):
            raise ValueError("race indices must be a 2D integer array with the expected shape")
        if race_indices.size and (race_indices.min() < 0 or race_indices.max() >= len(races)):
            raise ValueError("race indices are outside the available race range")
    summaries_a = {s: {race: _race_summary(groups_a[s][race], metric.startswith("ndcg"))
                       for race in races} for s in SEEDS}
    summaries_b = {s: {race: _race_summary(groups_b[s][race], metric.startswith("ndcg"))
                       for race in races} for s in SEEDS}
    average_precision_score, _, _ = _sklearn_metrics()
    distribution = []
    for indices in race_indices:
        per_seed = []
        for s in SEEDS:
            if metric == "average_precision":
                rows_a = [row for index in indices for row in groups_a[s][races[int(index)]]]
                rows_b = [row for index in indices for row in groups_b[s][races[int(index)]]]
                ap_a = float(average_precision_score([r["label"] for r in rows_a], [r["score"] for r in rows_a]))
                ap_b = float(average_precision_score([r["label"] for r in rows_b], [r["score"] for r in rows_b]))
                per_seed.append(ap_a - ap_b)
            else:
                per_seed.append(sum(summaries_a[s][races[int(index)]][per_race_metric]
                                    - summaries_b[s][races[int(index)]][per_race_metric]
                                    for index in indices) / len(indices))
        distribution.append(sum(per_seed) / len(per_seed))
    bootstrap_mean = sum(distribution) / len(distribution)
    ordered = sorted(float(value) for value in distribution)
    def percentile(fraction):
        position = fraction * (len(ordered) - 1)
        lower, upper = math.floor(position), math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    low, high = percentile(0.025), percentile(0.975)
    if metric == "average_precision":
        seed_point_estimates = []
        for s in SEEDS:
            a_rows, b_rows = normalized_a[s], normalized_b[s]
            a_ap = float(average_precision_score([r["label"] for r in a_rows],
                                                  [r["score"] for r in a_rows]))
            b_ap = float(average_precision_score([r["label"] for r in b_rows],
                                                  [r["score"] for r in b_rows]))
            seed_point_estimates.append(a_ap - b_ap)
    else:
        seed_point_estimates = [sum((summaries_a[s][r][per_race_metric] - summaries_b[s][r][per_race_metric])
                                    for r in races) / len(races) for s in SEEDS]
    estimate = sum(seed_point_estimates) / len(seed_point_estimates)
    return {"metric": metric, "estimate": estimate, "bootstrap_mean": bootstrap_mean,
            "ci95": [low, high],
            "distribution": distribution, "replicates": len(distribution),
            "seed": int(seed), "race_count": len(races),
            "seed_point_estimates": seed_point_estimates}
