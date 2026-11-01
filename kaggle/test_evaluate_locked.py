"""Synthetic, no-fit checks for the locked evaluation helpers."""
import sys
import types
from unittest.mock import patch

import evaluate_locked as evaluation


def _rows(scores=(.9, .8, .7, .1)):
    positions = {
        "r1": (1, 2, 3, 4),
        "r2": (1, 2, 3, 3),
        "r3": (1, 2, 3, 4),
    }
    groups = []
    for race_id, finish in positions.items():
        for number, finish_pos in enumerate(finish, 1):
            groups.append({
                "race_id": race_id, "horse_id_netkeiba": f"{race_id}-h{number}",
                "horse_number": number, "field_size": 4,
                "label": int(finish_pos <= 3), "finish_pos": finish_pos,
                "score": scores[number - 1], "history_group": "0" if race_id == "r1" else "3+",
            })
    return groups


def _fake_sklearn():
    metrics = types.ModuleType("sklearn.metrics")

    def average_precision_score(labels, scores):
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        total = sum(labels)
        hits = 0
        value = 0.0
        for rank, index in enumerate(order, 1):
            if labels[index]:
                hits += 1
                value += hits / rank
        return value / total if total else 0.0

    def brier_score_loss(labels, scores):
        return sum((score - label) ** 2 for label, score in zip(labels, scores)) / len(labels)

    def roc_auc_score(labels, scores):
        positives = [score for label, score in zip(labels, scores) if label]
        negatives = [score for label, score in zip(labels, scores) if not label]
        return sum(float(pos > neg) + .5 * float(pos == neg)
                   for pos in positives for neg in negatives) / (len(positives) * len(negatives))

    metrics.average_precision_score = average_precision_score
    metrics.brier_score_loss = brier_score_loss
    metrics.roc_auc_score = roc_auc_score
    sklearn = types.ModuleType("sklearn")
    sklearn.metrics = metrics
    return {"sklearn": sklearn, "sklearn.metrics": metrics}


def test_entry_race_and_metadata_join():
    rows = _rows()
    for row in rows:
        if row["race_id"] == "r3":
            row["score"] = {1: .1, 2: .2, 3: .3, 4: .9}[row["horse_number"]]
    prediction_only = [{key: row[key] for key in (
        "race_id", "horse_id_netkeiba", "horse_number", "label", "score", "history_group")}
                       for row in rows]
    metadata = [{key: row[key] for key in (
        "race_id", "horse_id_netkeiba", "horse_number", "field_size", "finish_pos", "label")}
                for row in rows]
    with patch.dict(sys.modules, _fake_sklearn()):
        result = evaluation.evaluate(prediction_only, threshold=.5, metadata=metadata)
        subgroup = evaluation.subgroup_metrics(evaluation.attach_finish_positions(prediction_only, metadata),
                                                "history_group", ("0", "3+", "missing"))
    audited = evaluation.audit_selection_flags(
        [dict(row, selected_top3=int(row["horse_number"] <= 3)) for row in _rows()])
    assert result["entry"]["average_precision"] > .8
    assert result["race"]["complete_races"] == 3
    assert result["race"]["exact_set_races"] == 2
    assert result["race"]["exact_set_rate"] == .5
    assert result["race"]["dead_heat_races"] == 1
    assert 0 <= result["race"]["ndcg_binary_at_3"] <= 1
    assert 0 <= result["race"]["ndcg_graded_at_3"] <= 1
    assert audited["complete_races"] == 3
    broken = [dict(row, selected_top3=0) for row in rows]
    try:
        evaluation.audit_selection_flags(broken)
    except ValueError as error:
        assert "deterministic ranking" in str(error)
    else:
        raise AssertionError("selection audit must reject incorrect persisted flags")
    assert subgroup["missing"]["entry_count"] == 0
    assert subgroup["0"]["hit_contribution"] == 3 / 9
    assert subgroup["3+"]["full_race_selected_count"] == 6


def test_strict_finish_join_and_missing_graded_position():
    rows = _rows()
    without_finish = [dict(row, **{"finish_pos": None}) for row in rows]
    try:
        evaluation.race_metrics(without_finish, require_graded=True)
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("graded NDCG must require finish positions")
    try:
        evaluation.attach_finish_positions(rows, rows[:-1])
    except ValueError as error:
        assert "key mismatch" in str(error)
    else:
        raise AssertionError("metadata join must require exact keys")

    for field, value in (("label", 1.5), ("horse_number", 1.5),
                         ("field_size", 4.5), ("finish_pos", 1.5)):
        broken = _rows()
        broken[0][field] = value
        try:
            evaluation.validate_rows(broken)
        except ValueError:
            pass
        else:
            raise AssertionError(f"fractional {field} must be rejected")


def test_paired_whole_race_bootstrap_uses_shared_indices():
    rows_a = _rows()
    rows_b = [dict(row, score=1.0 - row["score"]) for row in rows_a]
    indices = [[0, 1, 2], [2, 1, 0]]
    with patch.dict(sys.modules, _fake_sklearn()):
        result = evaluation.paired_bootstrap({seed: rows_a for seed in (0, 1, 2)},
                                             {seed: rows_b for seed in (0, 1, 2)},
                                             "mean_hit_at_3", race_indices=indices,
                                             replicates=2)
    assert result["replicates"] == 2
    assert result["race_count"] == 3
    assert len(result["distribution"]) == 2
    assert result["seed_point_estimates"] == [result["seed_point_estimates"][0]] * 3

    broken_seed = {seed: rows_a for seed in (0, 1, 2)}
    broken_seed[1] = [dict(row, label=1 - row["label"], finish_pos=None) for row in rows_a]
    try:
        evaluation.paired_bootstrap(broken_seed, {seed: rows_b for seed in (0, 1, 2)},
                                    "mean_hit_at_3", race_indices=indices, replicates=2)
    except ValueError as error:
        assert "labels" in str(error)
    else:
        raise AssertionError("paired bootstrap must reject cross-seed label drift")

    changed_finish = {seed: rows_a for seed in (0, 1, 2)}
    changed_finish[2] = [dict(row, finish_pos=(2 if row["finish_pos"] == 1 else row["finish_pos"]))
                         for row in rows_a]
    try:
        evaluation.paired_bootstrap(changed_finish, {seed: rows_a for seed in (0, 1, 2)},
                                    "ndcg_graded_at_3", race_indices=indices, replicates=2)
    except ValueError as error:
        assert "finish positions" in str(error)
    else:
        raise AssertionError("NDCG bootstrap must reject finish-position drift")

    for invalid in ([[0, 1], [2, 1]], [[0.0, 1.0, 2.0]] * 2,
                    [[0, 1, 3]] * 2, [[0, 1, 2], [2, 1]]):
        try:
            evaluation.paired_bootstrap({seed: rows_a for seed in (0, 1, 2)},
                                        {seed: rows_b for seed in (0, 1, 2)},
                                        "mean_hit_at_3", race_indices=invalid, replicates=2)
        except ValueError as error:
            assert "race indices" in str(error)
        else:
            raise AssertionError("invalid bootstrap indices must be rejected")
    try:
        evaluation.bootstrap_race_indices(("r1",), replicates=2.5)
    except ValueError:
        pass
    else:
        raise AssertionError("fractional bootstrap replicate count must be rejected")


if __name__ == "__main__":
    test_entry_race_and_metadata_join()
    test_strict_finish_join_and_missing_graded_position()
    test_paired_whole_race_bootstrap_uses_shared_indices()
    print("PASS: locked entry/race metrics, strict finish joins, subgroup denominators, and paired race bootstrap")
