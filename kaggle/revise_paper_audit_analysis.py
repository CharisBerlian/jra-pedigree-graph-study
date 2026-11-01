"""Post-review diagnostics from frozen predictions; never fits or selects models."""
from pathlib import Path
import csv
import gzip
from collections import Counter
import hashlib
import json
import sqlite3
import numpy as np
from sklearn.metrics import average_precision_score
import complete_locked_reporting as reporting
import evaluate_locked_test_results as locked
import experiment_features as features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/paper_audit_revision/supplement'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tables = OUT / 'tables'
    tables.mkdir(exist_ok=True)
    cohorts = locked.build_cohorts(ROOT/'tmp/features.db', ROOT/'tmp/repaired_jra.db',
                                   ROOT/'tmp/locked_test_inputs_2026-09-15')
    baseline, details = reporting.b0_report(cohorts, reporting.training_prevalence(ROOT/'tmp/features.db'), 2000)
    reporting.write_csv(tables/'b0_metrics_corrected.csv', baseline)
    reporting.write_json(OUT/'b0_diagnostic.json', details)
    for surface, cohort in cohorts.items():
        rows = reporting.b0_rows(cohort, 0.2)
        direct_hit = sum(r['label']*r['selected_top3'] for r in rows)/(3*len(cohort['race_ids']))
        actual = next(r for r in baseline if r['surface'] == surface)['race_mean_hit_at_3']
        assert abs(direct_hit-actual) < 1e-12
        assert all(sum(rows[i]['selected_top3'] for i in ids) == 3 for ids in cohort['race_rows'])
        print('B0', surface, actual, flush=True)

    coverage = []
    columns = ['split','surface','embedding_status','fallback_depth']
    for path in [ROOT/'kaggle/upload/jra-prediction-bundle-v1/train_validation.csv.gz',
                 ROOT/'tmp/locked_test_inputs_2026-09-15/test_seed0.csv.gz']:
        with gzip.open(path,'rt',newline='') as stream:
            grouped=Counter(tuple(row[c] for c in columns) for row in csv.DictReader(stream))
        for split,surface in sorted({k[:2] for k in grouped}):
            total=sum(v for k,v in grouped.items() if k[:2] == (split,surface))
            record = dict(split=split,surface=surface,entries=total,direct=0,parent=0,generation2=0,generation3=0,missing=0)
            for (sp,sf,status,depth),count in grouped.items():
                if (sp,sf) != (split,surface):
                    continue
                category = 'direct' if status == 'direct' else 'missing' if status == 'missing' else {1:'parent',2:'generation2',3:'generation3'}[int(depth)]
                record[category] += int(count)
            assert sum(record[k] for k in ['direct','parent','generation2','generation3','missing']) == total
            coverage.append(record)
    reporting.write_csv(tables/'graph_coverage.csv', coverage)
    connection = sqlite3.connect((ROOT/'tmp/features.db').as_uri()+'?mode=ro',uri=True)
    positive_counts = connection.execute('select npositive,count(*) from (select race_id,sum(label) npositive from entries group by race_id) group by npositive').fetchall()
    connection.close()
    reporting.write_json(OUT/'positive_label_reconciliation.json', {'races_by_positive_count':positive_counts,'excess':sum((p-3)*n for p,n in positive_counts)})

    spec = json.loads((ROOT/'plans/experiment_spec_2026-09-13.json').read_text())
    manifest = json.loads((ROOT/'plans/experiment_selection_manifest_2026-09-14.json').read_text())
    selected = []
    for model in manifest['models']:
        for refit in model['refits']:
            selected.append(dict(family=model['family'],surface=model['surface'],setting=model['setting'],seed=refit['seed'],
                                 candidate=refit['selected_candidate'],rounds=refit['selected_round'],
                                 threshold=refit['validation_threshold']['selected']['threshold'],params=json.dumps(refit['params'],sort_keys=True)))
    reporting.write_csv(tables/'selected_hyperparameters.csv',selected)
    reporting.write_json(OUT/'feature_allowlists.json', {s:{'features':features.feature_columns(s),'categorical':features.categorical_columns(features.feature_columns(s)),'count':len(features.feature_columns(s))} for s in features.SETTINGS})
    reporting.write_json(OUT/'hyperparameter_search.json',spec)

    results, interactions, subgroup_counts, inventory = [], [], [], []
    for surface, cohort in cohorts.items():
        counts = reporting.race_counts(len(cohort['race_ids']), 2000)
        masks = {g:cohort['history_exact'] == g for g in ('0','1','2','3+')}
        masks['<3'] = cohort['history_exact'] != '3+'
        for g,mask in masks.items():
            subgroup_counts.append(dict(surface=surface,history=g,entries=int(mask.sum()),positives=int(cohort['labels'][mask].sum()),prevalence=float(cohort['labels'][mask].mean()),races=int(len(np.unique(cohort['race_index'][mask])))))
        for family in locked.FAMILIES:
            distributions, points = {}, {}
            for setting in ('S2','S6','M2','S7'):
                seed_dists, seed_points = [], []
                for seed in locked.SEEDS:
                    path = ROOT/f'tmp/locked_test_inference_downloads_2026-09-15/predictions/{family}/{surface}/{setting}/seed{seed}/test_predictions.csv.gz'
                    scores, _, rows = locked.read_predictions(path,cohort)
                    inventory.append(dict(path=str(path.relative_to(ROOT)),sha256=locked.sha256_file(path)))
                    ds, ps = {}, {}
                    for g,mask in masks.items():
                        labels = cohort['labels'][mask]
                        ds[g] = reporting.normalized_ap_distribution(scores[mask],labels,cohort['race_index'][mask],counts,batch_size=25)
                        ps[g] = float(average_precision_score(labels,scores[mask]))
                        # Independently check weighted bootstrap AP, including repeated races.
                        for b in (0,1):
                            weights=counts[b,cohort['race_index'][mask]]
                            expected=average_precision_score(labels,scores[mask],sample_weight=weights)
                            assert abs(expected-ds[g][b]) < 1e-10
                    seed_dists.append(ds); seed_points.append(ps)
                distributions[setting] = {g:np.mean([d[g] for d in seed_dists],axis=0) for g in masks}
                points[setting] = {g:float(np.mean([p[g] for p in seed_points])) for g in masks}
                print('subgroups',surface,family,setting,flush=True)
            for a,b in [('S6','S2'),('S7','M2')]:
                ref_dist=distributions[a]['3+']-distributions[b]['3+']
                ref_point=points[a]['3+']-points[b]['3+']
                for g in masks:
                    dist=distributions[a][g]-distributions[b][g]
                    point=points[a][g]-points[b][g]
                    lo,hi=reporting.ci95(dist)
                    results.append(dict(family=family,surface=surface,contrast=a+'-'+b,history=g,ap_first=points[a][g],ap_second=points[b][g],delta=point,ci_low=lo,ci_high=hi))
                    if g != '3+':
                        lo,hi=reporting.ci95(dist-ref_dist)
                        interactions.append(dict(family=family,surface=surface,contrast=a+'-'+b,history=g,reference='3+',difference_of_deltas=point-ref_point,ci_low=lo,ci_high=hi))
            reporting.write_csv(tables/'exploratory_history_contrasts.csv',results)
            reporting.write_csv(tables/'exploratory_history_interactions.csv',interactions)
    reporting.write_csv(tables/'history_subgroup_counts.csv',subgroup_counts)
    reporting.write_csv(tables/'analysis_prediction_inventory.csv',inventory)
    reporting.write_json(OUT/'analysis_protocol.json',dict(status='complete',exploratory=True,replicates=2000,seed=20260913,
        unit='all whole races within surface; subgroup filter after sampling',aggregation='mean of per-seed AP differences, not averaged probabilities',
        interaction='subgroup pedigree AP increment minus 3+ increment in identical bootstrap samples',models_refit=False,
        manifest_sha256=locked.sha256_file(ROOT/'plans/experiment_selection_manifest_2026-09-14.json')))
    print('COMPLETE',OUT,flush=True)


if __name__ == '__main__':
    main()
