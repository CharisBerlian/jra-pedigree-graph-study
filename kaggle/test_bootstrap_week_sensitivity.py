"""Synthetic no-fit checks for ISO-week paired bootstrap sensitivity."""
from bootstrap_week_sensitivity import (bootstrap_week_indices, normalize_race_dates,
                                        paired_week_bootstrap)


def rows(scores):
    races = (("r1", "2024-01-01"), ("r2", "2024-01-05"),
             ("r3", "2024-01-09"), ("r4", "2024-01-18"),
             ("r5", "2024-01-19"))
    output = []
    for race_id, race_date in races:
        for number in (1, 2, 3):
            finish = number
            output.append({"race_id": race_id, "horse_id_netkeiba": f"{race_id}-h{number}",
                           "horse_number": number, "field_size": 3,
                           "label": int(finish <= 3), "finish_pos": finish,
                           "race_date": race_date,
                           "score": scores[number - 1]})
    return output


def check():
    dates = {"r1": "2024-01-01", "r2": "2024-01-05", "r3": "2024-01-09",
             "r4": "2024-01-18", "r5": "2024-01-19"}
    indices = [[0, 0, 1], [2, 1, 0]]
    a = rows((.9, .6, .3))
    b = rows((.3, .6, .9))
    result = paired_week_bootstrap({seed: a for seed in (0, 1, 2)},
                                   {seed: b for seed in (0, 1, 2)}, dates,
                                   "mean_hit_at_3", week_indices=indices,
                                   replicates=2)
    assert result["week_count"] == 3
    assert result["week_blocks"]["2024-W01"] == 2
    assert result["week_blocks"]["2024-W02"] == 1
    assert result["week_blocks"]["2024-W03"] == 2
    assert result["sampled_race_counts"] == [5, 5]
    assert result["replicates"] == 2
    ndcg = paired_week_bootstrap({seed: a for seed in (0, 1, 2)},
                                 {seed: b for seed in (0, 1, 2)}, dates,
                                 "ndcg_graded_at_3", week_indices=indices,
                                 replicates=2)
    assert len(ndcg["distribution"]) == 2
    drifted = [dict(row) for row in b]
    drifted[0]["finish_pos"] = 2
    try:
        paired_week_bootstrap({seed: a for seed in (0, 1, 2)},
                              {0: drifted, 1: b, 2: b}, dates,
                              "ndcg_graded_at_3", week_indices=indices, replicates=2)
    except ValueError as error:
        assert "finish_pos" in str(error)
    else:
        raise AssertionError("graded finish-position drift was accepted")
    try:
        normalize_race_dates(dict(dates, r3="2024-01-09junk"))
    except ValueError as error:
        assert "race_date" in str(error)
    else:
        raise AssertionError("invalid date suffix was accepted")
    generated = bootstrap_week_indices(dates, replicates=4, seed=20260913)
    assert generated.dtype.name == "int32" and generated.shape == (4, 3)
    generated_result = paired_week_bootstrap(
        {seed: a for seed in (0, 1, 2)}, {seed: b for seed in (0, 1, 2)}, dates,
        "mean_hit_at_3", week_indices=generated, replicates=4)
    assert generated_result["replicates"] == 4
    try:
        paired_week_bootstrap({seed: a for seed in (0, 1, 2)},
                              {seed: b for seed in (0, 1, 2)},
                              dict(dates, r3="2024-01-10"), "mean_hit_at_3",
                              week_indices=indices, replicates=2)
    except ValueError as error:
        assert "date" in str(error)
    else:
        raise AssertionError("date mismatch was accepted")
    try:
        paired_week_bootstrap({seed: a for seed in (0, 1, 2)},
                              {seed: b for seed in (0, 1, 2)}, dates,
                              "mean_hit_at_3", week_indices=[[0.0, 0, 1]],
                              replicates=1)
    except ValueError as error:
        assert "integer" in str(error)
    else:
        raise AssertionError("fractional week index was accepted")
    print("PASS: ISO-week blocks, unequal race counts, shared paired draws, and strict date/index checks")


if __name__ == "__main__":
    check()
