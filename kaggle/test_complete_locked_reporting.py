"""Small deterministic checks for the post-test reporting supplement."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import complete_locked_reporting as reporting


def main():
    counts = reporting.race_counts(2, replicates=4, seed=7)
    assert counts.shape == (4, 2)
    assert np.all(counts.sum(axis=1) == 2)

    scores = np.asarray([0.9, 0.1, 0.8, 0.2])
    labels = np.asarray([1, 0, 0, 1])
    races = np.asarray([0, 0, 1, 1])
    fixed_counts = np.asarray([[1, 1], [2, 0]], dtype=np.int16)
    distribution = reporting.normalized_ap_distribution(
        scores, labels, races, fixed_counts)
    assert np.allclose(distribution, [5.0 / 6.0, 1.0])
    assert reporting.ci95([math.nan, math.nan]) is None

    rows = [
        {"horse_number": "1", "horse_id_netkeiba": "a"},
        {"horse_number": "2", "horse_id_netkeiba": "b"},
        {"horse_number": "1", "horse_id_netkeiba": "c"},
        {"horse_number": "2", "horse_id_netkeiba": "d"},
    ]
    cohort = {"race_ids": ["r1", "r2"],
              "race_rows": [[0, 1], [2, 3]],
              "labels": np.asarray([1, 0, 0, 1]),
              "finish_pos": np.asarray([1, 4, 4, 2])}
    binary, graded = reporting.race_ndcg_fast(scores, rows, cohort)
    assert np.allclose(binary, [1.0, 1.0 / math.log2(3)])
    assert np.all(np.isfinite(graded))

    # Constant prevalence must not make B0 select horse numbers 1, 2, 3.
    b0_cohort = {'surface':'dirt','keys':[('r1',str(i),i) for i in range(1,9)],
                 'race_ids':['r1'],'race_rows':[list(range(8))],
                 'race_index':np.zeros(8,dtype=int),'labels':np.array([1,1,1,0,0,0,0,0]),
                 'field_sizes':np.full(8,8),'history_exact':['3+']*8,
                 'history_band':['3+']*8,'coverage':['direct']*8,
                 'finish_pos':np.arange(1,9),'graded_race_indices':[0]}
    old_surfaces = reporting.locked.SURFACES
    reporting.locked.SURFACES = ('dirt',)
    try:
        reference = reporting.b0_rows(b0_cohort,0.2)
        expected = sum(r['selected_top3']*r['label'] for r in reference)/3
        assert expected != 1.0, 'fixture must distinguish hash from horse-number order'
        report, _ = reporting.b0_report({'dirt':b0_cohort},{'dirt':0.2},20)
        assert abs(report[0]['race_mean_hit_at_3']-expected) < 1e-12
        assert abs(report[0]['average_precision']-3/8) < 1e-12
        ranking = [dict(r,score=r['ranking_score']) for r in reference]
        b, g, _, _, _ = reporting.point_ndcg(ranking,b0_cohort)
        assert report[0]['ndcg_binary_all_races'] == b
        assert report[0]['ndcg_graded_at_3'] == g
    finally:
        reporting.locked.SURFACES = old_surfaces
    print("PASS: locked reporting bootstrap, ties, and NDCG checks")


if __name__ == "__main__":
    main()
