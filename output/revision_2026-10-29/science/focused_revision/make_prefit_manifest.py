"""Freeze actual train/validation inputs and fixed 24-cell design for root review."""
from pathlib import Path
import copy,hashlib,json
import feature_adapter as features

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];PRIVATE=HERE/'private_data'
ARMS=('H','Gmixed','Gancestor','P');FAMILIES=('xgboost','lightgbm','catboost');SURFACES=('turf','dirt')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    audit_path=PRIVATE/'view_audit.json';audit=json.loads(audit_path.read_text());assert audit['status']=='pass'
    assert audit['rows']==636862 and len(audit['view_summaries'])==6
    categories=PRIVATE/'category_maps.json';assert sha(categories)==audit['category_maps_sha256']
    parent=PRIVATE/'parent_manifest.json';pm=json.loads(parent.read_text());assert pm['rows']==636862
    spec=copy.deepcopy(json.loads((ROOT/'plans/experiment_spec_2026-09-13.json').read_text()))
    spec.update(settings=list(ARMS),status='locked',preregistered=False,test_performance_opened=True,final_fit_count=72)
    spec['tuning']['fit_count']=96;spec['runtime']['python']='3.13.15'
    spec['bootstrap']={'replicates':2000,'seed':20261029,'unit':'whole_race',
                       'primary_contrasts':['Gancestor-H','Gancestor-P'],'secondary_contrasts':['Gancestor-Gmixed']}
    spec['lock']={'test_previously_seen':True,'test_features_blocked_until_root_model_freeze_review':True,
                  'no_test_retuning':True,'no_train_validation_refit':True}
    spec_path=HERE/'EXPERIMENT_SPEC.json';out=HERE/'PREFIT_MANIFEST_V4.json'
    assert not out.exists()
    spec_text=json.dumps(spec,indent=2,sort_keys=True)+'\n'
    if spec_path.exists():assert spec_path.read_text(encoding='utf-8')==spec_text
    else:spec_path.write_text(spec_text,encoding='utf-8',newline='\n')
    graphs=[]
    names=('production-seed0-version2','production-seed1-version1','production-seed2-version1')
    for seed in (0,1,2):
        d=ROOT/'kaggle/notebook-output'/names[seed]/'jra-production'/f'seed-{seed}'
        report=json.loads((d/'production_report.json').read_text());assert report['settings']['seed']==seed
        graph_files=['training_graph.edg',f'nodevectors_seed{seed}.npz',f'cluster_centers_seed{seed}.npy']
        for name in graph_files:assert sha(d/name)==report['artifact_sha256'][name]
        graphs.append({'seed':seed,'report_sha256':sha(d/'production_report.json'),
                       'graph_sha256':sha(d/graph_files[0]),'vector_sha256':sha(d/graph_files[1]),
                       'centroid_sha256':sha(d/graph_files[2]),'original_graph_python':report['environment']['python_version']})
    sources=[HERE/'PROTOCOL.md',HERE/'ancestry.py',HERE/'parent_control.py',HERE/'feature_adapter.py',
             HERE/'prepare_parent_data.py',HERE/'assemble_features.py',HERE/'audit_feature_views.py',
             HERE/'metrics.py',HERE/'test_core.py',HERE/'test_metrics.py',HERE/'build_focused_jobs.py',HERE/'build_model_transport.py',HERE/'test_transport_integration.py',Path(__file__),
             HERE/'remote_workspace.py',HERE/'remote_preparation.py',HERE/'build_remote_preparation_job.py',HERE/'verify_remote_parent.py',HERE/'verify_remote_views.py',HERE/'test_remote_workspace.py',
             HERE/'check_private_staging.py',HERE/'REMOTE_PREPARATION_PLAN.md',
             HERE.parent/'assemble_cutoff.py',
             ROOT/'kaggle/run_experiments.py',ROOT/'kaggle/experiment_features.py',ROOT/'kaggle/prepare_prediction.py',
             ROOT/'kaggle/graph_pilot.py',ROOT/'output/icdees_focused_2026-09-18/companion/code/evaluate_locked_test_results.py',
             ROOT/'output/icdees_focused_2026-09-18/companion/code/evaluate_locked.py']
    cells=[{'arm':arm,'family':family,'surface':surface,'tuning_seed':0,'candidate_ids':['c0','c1','c2','c3'],
            'refit_seeds':[0,1,2],'status':'not_launched'} for arm in ARMS for family in FAMILIES for surface in SURFACES]
    schemas={}
    for arm in ARMS:
        names=list(features.feature_columns(arm));cats=set(features.categorical_columns(names))
        schemas[arm]=[{'name':name,'type':'categorical' if name in cats else 'numeric'} for name in names]
    assert len(cells)==24 and sum(len(x['candidate_ids']) for x in cells)==96 and sum(len(x['refit_seeds']) for x in cells)==72
    data={'status':'awaiting_root_prefit_review','supersedes_prefit_sha256':sha(HERE/'PREFIT_MANIFEST_V3.json'),
          'test_previously_seen':True,'test_observations_decoded_in_this_study':False,
          'access_scope':'2010–2023 race ID/date projection and train/validation entries only; static pedigree snapshot loaded globally. An earlier preserved preparation computed an immutable whole-database checksum without decoding or selecting test outcomes.',
          'training_rows':548756,'validation_rows':88106,'race_count':44848,'cohort_and_view_audit_sha256':sha(audit_path),
          'cohort_key_sha256':audit['ordered_key_sha256'],'cohort_non_graph_and_labels_sha256':audit['non_graph_sha256'],
          'parent_feature_sha256':audit['parent_features_sha256'],'parent_manifest_sha256':sha(parent),
          'parent_map_sha256':pm['map_sha256'],'category_maps_sha256':sha(categories),'view_summaries':audit['view_summaries'],
          'staging_closure_audit_sha256':sha(PRIVATE/'staging_closure_audit.json'),
          'date_projection_sha256':sha(PRIVATE/'race_dates_train_validation.csv.gz'),
          'graph_artifacts':graphs,'feature_schemas':schemas,'spec_sha256':sha(spec_path),'candidate_grid':spec['tuning']['candidates'],
          'candidate_order':['c0','c1','c2','c3'],'tuning':spec['tuning'],'runtime':spec['runtime'],'categorical':spec['categorical'],
          'parent_feature_construction':'P uses supervised same-surface prior-outcome summaries, strictly earlier dates, focal-horse exclusion and frozen end-2021 maps. categorical.external_target_encoding=false applies only to categorical-code preprocessing; P is outcome-derived.',
          'graph_descriptor_serialization':{'status':'categorical literal fallback or missing',
            'depth':'numeric integer 1..3 or blank (NaN after numeric parsing)',
            'count':'numeric integer >=0','missing':'numeric 0 or 1',
            'cluster':'categorical decimal centroid index 0..63 or literal MISSING',
            'absent_train_categories':'unknown token code 1; blanks use missing token code 0; counts <5 use rare token code 2'},
          'metrics':{'AP':'tie-grouped entry average precision; duplicate sampled races count with multiplicity',
            'Hit@3':'sum labels of top three by score descending, horse number ascending, horse ID ascending, divided by 3; mean across all complete races',
            'invalid':'non-finite/out-of-range scores, invalid labels, duplicate keys or incomplete/<3 rosters stop',
            'bootstrap':'2000 shared whole-race draws per surface using numpy default_rng(20261029); percentile 2.5,97.5 on mean paired seed differences',
            'interpretation':'Hit@3 difference and interval multiplied by 300 equals additional correct selections per 100 races; no profit/equivalence threshold'},
          'comparisons':spec['bootstrap'],'cells':cells,'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in sources},
          'normalized_utf8_text_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_text(encoding='utf-8').encode('utf-8')).hexdigest() for p in sources},
          'synthetic_release_manifest_sha256':sha(HERE/'synthetic_release_v5/FILE_HASHES.json')}
    out.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'status':'awaiting_root_prefit_review','manifest_sha256':sha(out),'spec_sha256':sha(spec_path),'cells':len(cells)}))


if __name__=='__main__':main()
