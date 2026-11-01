"""Secondary ISO-week bootstrap sensitivity for locked test predictions.

This module is descriptive and test-only. It never reads files or fits models.
Race dates are supplied as observed metadata, and whole ISO calendar weeks are
the resampling units; every race in a sampled week is retained together.
"""
from collections import defaultdict
from datetime import date, datetime
import math
from numbers import Integral

from evaluate_locked import (_key, _race_summary, _sklearn_metrics, _group,
                             _integer, validate_rows)


SEEDS = (0, 1, 2)
METRICS = ("average_precision", "mean_hit_at_3", "ndcg_binary_at_3",
           "ndcg_graded_at_3")


def _parse_date(value):
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        text = f"{text[:4]}-{text[4:6]}-{text[6:]}"
    if (len(text) != 10 or text[4] != "-" or text[7] != "-"
            or not text[:4].isdigit() or not text[5:7].isdigit()
            or not text[8:].isdigit()):
        raise ValueError(f"race_date must be an ISO date or YYYYMMDD: {value!r}")
    try:
        return date.fromisoformat(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"race_date must be an ISO date or YYYYMMDD: {value!r}") from error


def normalize_race_dates(race_dates):
    """Return race -> ISO date from a mapping or metadata rows.

    Metadata rows must contain ``race_id`` and either ``race_date`` or
    ``date``. Duplicate race IDs are rejected even when dates agree.
    """
    if hasattr(race_dates, "items"):
        items = list(race_dates.items())
    else:
        items = []
        seen = set()
        for row in race_dates:
            race_id = str(row.get("race_id"))
            if race_id in seen:
                raise ValueError(f"duplicate race date metadata: {race_id}")
            seen.add(race_id)
            value = row.get("race_date", row.get("date"))
            items.append((race_id, value))
    result = {}
    for race_id, value in items:
        race_id = str(race_id)
        if not race_id or race_id in result:
            raise ValueError(f"duplicate/empty race date metadata: {race_id!r}")
        result[race_id] = _parse_date(value)
    if not result:
        raise ValueError("race date metadata is empty")
    return result


def _validate_rows_dates(rows, race_dates):
    normalized = validate_rows(rows)
    dates = normalize_race_dates(race_dates)
    races = {str(row["race_id"]) for row in normalized}
    if races != set(dates):
        raise ValueError("race date metadata must cover exactly the prediction races")
    for row in normalized:
        if "race_date" in row and row["race_date"] not in (None, ""):
            if _parse_date(row["race_date"]) != dates[str(row["race_id"])]:
                raise ValueError(f"race date disagrees with metadata: {row['race_id']}")
    return normalized, dates


def iso_week_blocks(race_dates):
    """Return sorted nonempty ISO blocks and their race IDs."""
    dates = normalize_race_dates(race_dates)
    blocks = defaultdict(list)
    for race_id, observed in dates.items():
        iso = observed.isocalendar()
        blocks[(int(iso.year), int(iso.week))].append(race_id)
    return {week: tuple(sorted(races)) for week, races in sorted(blocks.items())}


def bootstrap_week_indices(race_dates, replicates=2000, seed=20260913):
    """Draw week indices with replacement using NumPy default_rng."""
    blocks = iso_week_blocks(race_dates)
    replicates = _integer(replicates, "replicates")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("NumPy is required for week bootstrap resampling") from error
    return np.asarray(np.random.default_rng(seed).integers(
        0, len(blocks), size=(replicates, len(blocks))), dtype="int32")


def _checked_indices(week_indices, week_count, replicates):
    if len(week_indices) != int(replicates):
        raise ValueError("week index replicate count mismatch")
    checked = []
    for replicate in week_indices:
        values = []
        for value in replicate:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError("week indices must be integer-valued")
            value = int(value)
            if value < 0 or value >= week_count:
                raise ValueError("week index is outside the observed block range")
            values.append(value)
        if len(values) != week_count:
            raise ValueError("each replicate must draw one index per observed week")
        checked.append(tuple(values))
    return tuple(checked)


def _race_multiset(blocks, indices):
    weeks = tuple(blocks)
    return [race for index in indices for race in blocks[weeks[index]]]


def paired_week_bootstrap(seed_a, seed_b, race_dates, metric,
                          week_indices=None, replicates=2000, seed=20260913):
    """Return descriptive paired A-minus-B week-bootstrap sensitivity.

    ``seed_a`` and ``seed_b`` map classifier seeds 0, 1, 2 to prediction rows.
    One generated or supplied week-index matrix is shared across all seeds and
    both arms. Race-level metrics average the sampled race multiset; AP is
    recomputed on all sampled entries for each seed and arm.
    """
    if set(seed_a) != set(SEEDS) or set(seed_b) != set(SEEDS):
        raise ValueError("week bootstrap requires prediction rows for seeds 0, 1, and 2")
    if metric not in METRICS:
        raise ValueError(f"unsupported week bootstrap metric: {metric}")
    normalized_a = {}
    normalized_b = {}
    dates_by_seed = {}
    for classifier_seed in SEEDS:
        normalized_a[classifier_seed], dates_by_seed[classifier_seed] = _validate_rows_dates(
            seed_a[classifier_seed], race_dates)
        normalized_b[classifier_seed], dates_b = _validate_rows_dates(
            seed_b[classifier_seed], race_dates)
        if dates_b != dates_by_seed[classifier_seed]:
            raise ValueError("week bootstrap requires identical race dates across arms")
    reference_keys = {_key(row) for row in normalized_a[0]}
    reference_labels = {_key(row): row["label"] for row in normalized_a[0]}
    reference_races = {str(row["race_id"]) for row in normalized_a[0]}
    reference_finish = {_key(row): row.get("finish_pos") for row in normalized_a[0]}
    for classifier_seed in SEEDS:
        for rows in (normalized_a[classifier_seed], normalized_b[classifier_seed]):
            if {_key(row) for row in rows} != reference_keys:
                raise ValueError("week bootstrap requires identical row keys across seeds/arms")
            if {_key(row): row["label"] for row in rows} != reference_labels:
                raise ValueError("week bootstrap requires identical labels across seeds/arms")
            if {str(row["race_id"]) for row in rows} != reference_races:
                raise ValueError("week bootstrap requires identical race IDs")
            if metric == "ndcg_graded_at_3":
                finish = {_key(row): row.get("finish_pos") for row in rows}
                if any(value in (None, "") for value in finish.values()):
                    raise ValueError("graded NDCG requires finish_pos for every prediction key")
                if finish != reference_finish:
                    raise ValueError("graded NDCG requires identical finish_pos across seeds/arms")
    blocks = iso_week_blocks(dates_by_seed[0])
    if week_indices is None:
        week_indices = bootstrap_week_indices(dates_by_seed[0], replicates, seed)
    replicates = _integer(replicates, "replicates")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    week_indices = _checked_indices(week_indices, len(blocks), replicates)
    groups_a = {classifier_seed: _group(normalized_a[classifier_seed])
                for classifier_seed in SEEDS}
    groups_b = {classifier_seed: _group(normalized_b[classifier_seed])
                for classifier_seed in SEEDS}
    require_finish = metric.startswith("ndcg")
    summaries_a = {classifier_seed: {race: _race_summary(groups_a[classifier_seed][race], require_finish)
                                     for race in reference_races}
                   for classifier_seed in SEEDS}
    summaries_b = {classifier_seed: {race: _race_summary(groups_b[classifier_seed][race], require_finish)
                                     for race in reference_races}
                   for classifier_seed in SEEDS}
    average_precision_score = _sklearn_metrics()[0] if metric == "average_precision" else None
    ordered_blocks = tuple(blocks)
    per_race_key = {"mean_hit_at_3": "hit_at_3", "ndcg_binary_at_3": "ndcg_binary",
                    "ndcg_graded_at_3": "ndcg_graded"}.get(metric)
    distribution = []
    sampled_race_counts = []
    for indices in week_indices:
        sampled_races = _race_multiset(blocks, indices)
        sampled_race_counts.append(len(sampled_races))
        per_seed = []
        for classifier_seed in SEEDS:
            if metric == "average_precision":
                rows_a = [row for race in sampled_races for row in groups_a[classifier_seed][race]]
                rows_b = [row for race in sampled_races for row in groups_b[classifier_seed][race]]
                ap_a = float(average_precision_score([row["label"] for row in rows_a],
                                                      [row["score"] for row in rows_a]))
                ap_b = float(average_precision_score([row["label"] for row in rows_b],
                                                      [row["score"] for row in rows_b]))
                per_seed.append(ap_a - ap_b)
            else:
                per_seed.append(sum(summaries_a[classifier_seed][race][per_race_key]
                                    - summaries_b[classifier_seed][race][per_race_key]
                                    for race in sampled_races) / len(sampled_races))
        distribution.append(sum(per_seed) / len(per_seed))
    ordered = sorted(float(value) for value in distribution)
    def percentile(fraction):
        position = fraction * (len(ordered) - 1)
        low, high = math.floor(position), math.ceil(position)
        if low == high:
            return ordered[low]
        weight = position - low
        return ordered[low] * (1.0 - weight) + ordered[high] * weight
    point = []
    for classifier_seed in SEEDS:
        if metric == "average_precision":
            a = normalized_a[classifier_seed]
            b = normalized_b[classifier_seed]
            point.append(float(average_precision_score([row["label"] for row in a], [row["score"] for row in a]))
                         - float(average_precision_score([row["label"] for row in b], [row["score"] for row in b])))
        else:
            point.append(sum(summaries_a[classifier_seed][race][per_race_key]
                             - summaries_b[classifier_seed][race][per_race_key]
                             for race in reference_races) / len(reference_races))
    return {"unit": "ISO calendar week", "metric": metric,
            "estimate": sum(point) / len(point),
            "bootstrap_mean": sum(distribution) / len(distribution),
            "ci95": [percentile(0.025), percentile(0.975)],
            "distribution": distribution, "replicates": len(distribution),
            "seed": int(seed), "week_count": len(blocks),
            "race_count": len(reference_races),
            "sampled_race_counts": sampled_race_counts,
            "week_blocks": {f"{year}-W{week:02d}": len(blocks[(year, week)])
                            for year, week in ordered_blocks},
            "seed_point_estimates": point}
