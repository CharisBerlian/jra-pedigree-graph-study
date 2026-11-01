"""Audit 18 validation-only jobs and freeze all 54 refits before test access.

Expected harvest layout: ROOT/FAMILY/SURFACE/ARM/{spec.json,JOB_PASS.json,runs/...};
ROOT/inputs/{old,new}-seedN.csv.json are exact assembled input sidecars. Never
reads test inputs, predictions, labels, or test outcomes.
"""
from pathlib import Path
import argparse,ast,csv,gzip,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'kaggle'))
import audit_experiment_outputs as audit
import experiment_features as features
import assemble_cutoff as assembly
import audit_transport_records as transport

FAMILIES=('xgboost','lightgbm','catboost');SURFACES=('turf','dirt');ARMS=('S2','old_S6','new_S6');SEEDS=(0,1,2)
KEY=('race_id','horse_id_netkeiba','horse_number')
RUNNER_SHA='786e4db859c9a4fa17dba3c61a92bf110d3f618f6d2a0b787c47e4c6d9154b87'
HELPER_SHA='3a4fc0d2192beb39c6e9c3ed62c4528c1a84fa4d5c5c98d41e75e49e1c7ed27e'
DESIGN_SHA='6fcc6fa0f90c3e648c56b535f9d307a27746ca078e35407d40fccd1fcba57d18'
TRANSPORT_SHA='fb3184a5327a06e5189e95942ae7616f28de901a68b93a46c7b34ac523273c8c'

def sha(path):return audit._sha256(path)
def portable_relative_dir(folder,root):
    relative=folder.relative_to(root)
    assert '..' not in relative.parts
    return relative.as_posix()

def read_predictions(path,surface):
    if not path.is_file() and path.name.endswith('.gz'):path=path.with_suffix('')
    op=gzip.open if path.suffix=='.gz' else open
    with op(path,'rt',encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    assert rows and all(r['surface']==surface for r in rows)
    result={tuple(r[k] for k in KEY):int(r['label']) for r in rows}
    assert len(result)==len(rows) and set(result.values())=={0,1}
    return result

def audit_candidate_roster_and_hit(path,surface,reference,reported_hit):
    path=Path(path)
    if not path.is_file() and path.name.endswith('.gz'):path=path.with_suffix('')
    op=gzip.open if path.suffix=='.gz' else open
    with op(path,'rt',encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    actual={tuple(r[k] for k in KEY):int(r['label']) for r in rows}
    assert len(actual)==len(rows) and actual==reference and all(r['surface']==surface for r in rows)
    hits={}
    for r in rows:
        flag=int(r['selected_top3']);assert flag in (0,1)
        race=hits.setdefault(r['race_id'],[0,0]);race[0]+=flag*int(r['label']);race[1]+=flag
    assert all(n==3 for _,n in hits.values())
    value=sum(h for h,_ in hits.values())/(3*len(hits))
    assert abs(value-float(reported_hit))<1e-12
    return value

def freeze(root,out,design):
    root=Path(root);out=Path(out);design=Path(design)
    assert not out.exists() and sha(design)==DESIGN_SHA
    # Refuse to proceed if a harvested result tree contains any test artifact.
    audit._reject_test_paths(root)
    design_obj=audit._json(design)
    transport_design_path=Path(__file__).resolve().parent/'DESIGN_FREEZE_V4.json'
    assert sha(transport_design_path)==TRANSPORT_SHA
    transport_design=audit._json(transport_design_path)
    assert transport_design['v3_sha256']==DESIGN_SHA
    assert sha(Path(__file__).resolve().parent/'build_transport_adapter.py')==transport_design['adapter_sha256']
    remote=audit._json(root/'remote_jobs.json')
    expected_image=design_obj['docker_image']
    verified_transport={}
    for kind in ('graph','assembly','model'):
        expected_jobs=design_obj['job_sets'][kind]['jobs'];actual=remote[kind]
        def identity(j):return (j['seed'],) if kind!='model' else (j['family'],j['surface'],j['arm'])
        wanted={identity(j):j for j in expected_jobs};observed={identity(j):j for j in actual}
        assert len(wanted)==len(observed)==len(expected_jobs)==len(actual) and wanted.keys()==observed.keys()
        for identity_key,entry in observed.items():
            assert entry['status']=='complete' and entry['remote_source_equal'] is True and entry['private'] is True
            assert entry['image']==expected_image and int(entry['kernel_id'])>0 and int(entry['version'])>=1
            if kind=='graph':
                assert entry['source_sha256']==wanted[identity_key]['sha256']
            else:
                name=('assembly_seed'+str(identity_key[0])) if kind=='assembly' else '_'.join(identity_key)
                candidates=[x for x in transport_design['role_matrix'][kind] if
                            ((x['seed'],) if kind=='assembly' else (x['family'],x['surface'],x['arm']))==identity_key]
                assert len(candidates)==1
                frozen=candidates[0]; expected={'identity':name,'inner_sha256':frozen['inner_sha256'],'paths':frozen['paths']}
                record_path=root/'transport_records'/f'{name}.json'; source_path=root/'private_transport'/f'{name}.py'
                record=audit._json(record_path)
                assert entry['source_sha256']==record['adapter_sha256']
                assert entry['transport_record_sha256']==sha(record_path), 'prelaunch record hash mismatch'
                original=(Path(__file__).resolve().parent/wanted[identity_key]['path']).read_bytes()
                staged=transport.verify(record,expected,original,source_path.read_bytes())
                verified_transport[kind,identity_key]=(staged,sha(record_path),record)
    inputs={}
    graph_manifest_path=Path(__file__).resolve().parent/'graph_jobs_v3_manifest.json'
    assembly_manifest_path=Path(__file__).resolve().parent/'assembly_jobs_v2_manifest.json'
    assert sha(graph_manifest_path)==design_obj['job_sets']['graph']['manifest_sha256']
    assert sha(assembly_manifest_path)==design_obj['job_sets']['assembly']['manifest_sha256']
    assembly_manifest=audit._json(assembly_manifest_path)
    for regime in ('old','new'):
        for seed in SEEDS:
            p=root/'inputs'/f'{regime}-seed{seed}.csv.json';meta=audit._json(p)
            assert meta['seed']==seed and meta['regime']==regime and meta['splits']==['train','validation']
            assert meta['graph_python_version']==('3.13.15' if regime=='new' else '3.12.13')
            assert meta['base_payload_sha256']=='d395a1802d90b0c09cc71f08b2dcbc83ca4020654797c9442969833c103ac12d'
            assert meta['graph_runner_sha256']=='3a2c4e77bd70af6059a2854b601ac84a47de8ed752972bde656b97bbd162d59f'
            if regime=='new':assert meta['target_pedigree_sha256']==assembly_manifest['target_pedigree_sha256']
            assert meta['ordered_key_sha256']=='ad7abe6649728c0dec9af8350e5fadba2ce054e67b9f3d210a6bbf0930ccb62a'
            assert meta['non_graph_sha256']=='60317324596f7eb1c5a702e008ab0cb50a213f702dc2706b26a9d77d1f933f56'
            paths=[x for x in (root/'inputs'/f'{regime}-seed{seed}.csv.gz',root/'inputs'/f'{regime}-seed{seed}.csv') if x.is_file()]
            assert len(paths)==1
            assert paths[0].suffix=='.gz', 'transport contract requires gzip input'
            assembly.verify_payload(paths[0],meta['data_sha256'])
            inputs[regime,seed]=(meta,sha(p),sha(paths[0]))
    graph_manifest=audit._json(graph_manifest_path)
    for seed in SEEDS:
        graph_root=root/'graph_jobs'/f'seed{seed}';graph=graph_root/'graph'
        assembly.load_new_graph(graph,seed,graph_manifest['graph_bundle_manifest_sha256'],graph_manifest['graph_production_sha256'])
        report=audit._json(graph/'production_report.json');marker=audit._json(graph_root/'JOB_PASS.json')
        assert marker['status']=='pass' and marker['seed']==seed and marker['report_sha256']==sha(graph/'production_report.json')
        assert marker['protocol_sha256']==design_obj['protocol_sha256']
        staged=verified_transport['assembly',(seed,)][0]
        origin={x['path']:(x['origin'],x['version']) for x in verified_transport['assembly',(seed,)][2]['staged']}
        assert staged['base/train_validation.csv']=='d395a1802d90b0c09cc71f08b2dcbc83ca4020654797c9442969833c103ac12d'
        assert origin['base/train_validation.csv']==('dataset:charischerw/jra-prediction-bundle-v1:train_validation.csv',1)
        graph_job=[j for j in remote['graph'] if j['seed']==seed];assert len(graph_job)==1
        assert graph_job[0]['slug']==f'charischerw/jra-cutoff2017-v3-graph-seed{seed}'
        for filename in ('production_report.json',f'nodevectors_seed{seed}.npz',f'cluster_centers_seed{seed}.npy','training_graph.edg'):
            assert staged[f'graph/{filename}']==sha(graph/filename)
            assert origin[f'graph/{filename}']==(f"kernel:{graph_job[0]['slug']}:jra-cutoff-2017-seed{seed}/graph/{filename}",int(graph_job[0]['version']))
        if seed:
            old_file=ROOT/'kaggle/upload/jra-experiment-graph-tables-v1'/f'graph_features_seed{seed}.csv.json'
            old_spec=audit._json(old_file)
            assert staged[f'old/graph_features_seed{seed}.csv.json']==sha(old_file)
            assert staged[f'old/graph_features_seed{seed}.csv']==old_spec['uncompressed_csv']['sha256']
            for suffix in ('.csv','.csv.json'):
                name=f'graph_features_seed{seed}{suffix}'
                assert origin[f'old/{name}']==(f'dataset:charischerw/jra-experiment-graph-tables-v1:{name}',1)
        meta=inputs['new',seed][0]
        for k,v in {'graph_sha256':report['graph_sha256'],'graph_config_hash':report['config_hash'],
                    'graph_report_sha256':sha(graph/'production_report.json'),'graph_manifest_sha256':report['input_manifest_sha256'],
                    'graph_runner_sha256':report['script_sha256'],'graph_vector_sha256':report['artifact_sha256'][f'nodevectors_seed{seed}.npz'],
                    'graph_centroid_sha256':report['artifact_sha256'][f'cluster_centers_seed{seed}.npy']}.items():assert meta[k]==v,k
        old_report=root/'old_graph_reports'/f'seed{seed}'/'production_report.json';old=audit._json(old_report);oldmeta=inputs['old',seed][0]
        assert sha(old_report)==oldmeta['graph_report_sha256'] and old['settings']['seed']==seed and old['environment']['python_version']=='3.12.13'
        for k,v in {'graph_sha256':old['graph_sha256'],'graph_config_hash':old['config_hash'],
                    'graph_manifest_sha256':old['input_manifest_sha256'],'graph_runner_sha256':old['script_sha256'],
                    'graph_vector_sha256':old['artifact_sha256'][f'nodevectors_seed{seed}.npz'],
                    'graph_centroid_sha256':old['artifact_sha256'][f'cluster_centers_seed{seed}.npy']}.items():assert oldmeta[k]==v,k
        pass_record=audit._json(root/'assembly_jobs'/f'seed{seed}'/'ASSEMBLY_PASS.json')
        assert pass_record['status']=='pass' and pass_record['seed']==seed
        assert pass_record['protocol_sha256']==design_obj['protocol_sha256'] and pass_record['fingerprint_sha256']==sha(Path(__file__).resolve().parent/'cohort_fingerprint.json')
    roster=root/'inputs'/'validation_roster.csv.gz'
    assert sha(roster)=='42c3b58f5dd26889b378c21c33a3673ea325cf24569283f0e0faa52f4881b17c'
    reference={};races={}
    with gzip.open(roster,'rt',encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            surface=row['surface'];key=tuple(row[k] for k in KEY);target=reference.setdefault(surface,{})
            assert key not in target;target[key]=int(row['label'])
            race=(surface,row['race_id']);group=races.setdefault(race,[int(row['field_size']),0]);assert group[0]==int(row['field_size']);group[1]+=1
    assert all(size==n and n>=3 for size,n in races.values())
    assert {s:len(rows) for s,rows in reference.items()}=={'turf':42783,'dirt':45323}
    jobs={(j['family'],j['surface'],j['arm']):j for j in design_obj['job_sets']['model']['jobs']}
    assert len(jobs)==18
    for family,surface,arm in jobs:
        regime='new' if arm=='new_S6' else 'old'
        staged=verified_transport['model',(family,surface,arm)][0]
        origin={x['path']:(x['origin'],x['version']) for x in verified_transport['model',(family,surface,arm)][2]['staged']}
        for seed in SEEDS:
            meta,metadata_hash,transport_hash=inputs[regime,seed]
            assert staged[f'seed{seed}/{regime}-seed{seed}.csv.gz']==transport_hash
            assert staged[f'seed{seed}/{regime}-seed{seed}.csv.json']==metadata_hash
            producer=[j for j in remote['assembly'] if j['seed']==seed];assert len(producer)==1
            assert producer[0]['slug'].startswith('charischerw/')
            for suffix in ('.gz','.json'):
                filename=f'{regime}-seed{seed}.csv{suffix}'
                assert origin[f'seed{seed}/{filename}']==(f"kernel:{producer[0]['slug']}:jra-cutoff-assembly-seed{seed}/{filename}",int(producer[0]['version']))
    models=[];tunes=candidates=refits=0
    for family in FAMILIES:
      for surface in SURFACES:
       for arm in ARMS:
        setting='S2' if arm=='S2' else 'S6';regime='new' if arm=='new_S6' else 'old'
        job=root/family/surface/arm; spec=audit._json(job/'spec.json');passed=audit._json(job/'JOB_PASS.json')
        frozen_job=jobs[family,surface,arm];script=Path(__file__).resolve().parent/frozen_job['path'];assert sha(script)==frozen_job['sha256']
        tree=ast.parse(script.read_text(encoding='utf-8'))
        literal=next(x for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='spec' for t in x.targets))
        assert spec==ast.literal_eval(literal.value)
        assert sha(job/'run_experiments.py')==RUNNER_SHA and sha(job/'experiment_features.py')==HELPER_SHA
        assert passed['status']=='pass' and (passed['family'],passed['surface'],passed['arm'])==(family,surface,arm)
        assert passed['protocol_sha256']=='47fa77bce8702c4787538f68717b4c4d486576dd8e991ffe6748d703d5cce6ae'
        assert passed['spec_sha256']==sha(job/'spec.json') and spec['runtime']['python']=='3.13.15'
        assert spec['split']['graph_roots']=={'S2':'none; graph predictors excluded','old_S6':'2010-2021','new_S6':'2010-2017'}[arm]
        for seed in SEEDS: assert passed['input_metadata'][str(seed)]==inputs[regime,seed][0]
        base=job/'runs'/'scientific'/family/surface/setting;tune=base/'tune'/'seed0';marker=audit._complete(tune)
        cfg=audit._json(tune/'config.json');policy=audit._policy(spec);grid=audit._candidate_grid(spec,family)
        provenance={'spec_sha256':sha(job/'spec.json'),'script_sha256':RUNNER_SHA,'helper_sha256':HELPER_SHA}
        audit._check_config(cfg,spec,'scientific',family,surface,setting,'tune',0,policy,provenance)
        assert marker['config_hash']==audit._canonical(cfg) and cfg['candidate_grid_sha256']==audit._canonical(grid)
        assert cfg['features']==list(features.feature_columns(setting)) and cfg['categorical']==list(features.categorical_columns(cfg['features']))
        assert cfg['input_sha256']==inputs[regime,0][0]['data_sha256'] and cfg['input_metadata_sha256']==inputs[regime,0][1]
        records=[]
        for cid,params in grid.items():
            folder=tune/'candidates'/cid;cc=audit._json(folder/'config.json');assert cc['parent_config_hash']==marker['config_hash']
            record=audit._validate_curve(folder,cc,params,policy,family)
            audit_candidate_roster_and_hit(folder/'validation_predictions.csv.gz',surface,reference[surface],record['mean_hit_at_3'])
            records.append(record);candidates+=1
        assert audit._json(tune/'candidate_results.json')==records
        selected=audit._json(tune/'selected.json');winner,_=audit._winner(records,policy['ap_tie_tolerance']);assert selected==winner
        tunes+=1; refit_rows=[]
        for seed in SEEDS:
            folder=base/'refit'/f'seed{seed}'
            audit._audit_refit(folder,spec,'scientific',family,surface,setting,seed,policy,cfg,selected,provenance)
            rc=audit._json(folder/'config.json');prep=audit._json(folder/'preprocessing.json')
            assert rc['selection_sha256']==sha(tune/'selected.json') and rc['tuning_complete_sha256']==sha(tune/'complete.json') and rc['tuning_config_sha256']==sha(tune/'config.json')
            assert rc['input_sha256']==inputs[regime,seed][0]['data_sha256'] and rc['input_metadata_sha256']==inputs[regime,seed][1]
            assert prep['features']==cfg['features'] and prep['categorical_columns']==cfg['categorical'] and set(prep['category_maps'])==set(cfg['categorical'])
            rows=read_predictions(folder/'validation_predictions.csv.gz',surface)
            expected_rows={'turf':42783,'dirt':45323}[surface];assert len(rows)==expected_rows
            assert rows==reference[surface]
            marker=audit._complete(folder);refit_rows.append({'seed':seed,'relative_dir':portable_relative_dir(folder,root),
                'artifacts':marker['artifacts'],'complete_sha256':sha(folder/'complete.json'),'config_sha256':sha(folder/'config.json'),
                'model_sha256':marker['artifacts'][audit.MODEL_NAMES[family]],'preprocessing_sha256':marker['artifacts']['preprocessing.json']})
            refits+=1
        models.append(dict(family=family,surface=surface,arm=arm,setting=setting,selected_candidate=selected['candidate_id'],
            selected_round=selected['best_round'],selected_sha256=sha(tune/'selected.json'),tune_complete_sha256=sha(tune/'complete.json'),
            spec_sha256=sha(job/'spec.json'),input_regime=regime,refits=refit_rows))
    assert (tunes,candidates,refits)==(18,72,54)
    graph_fields=('graph_sha256','graph_manifest_sha256','graph_runner_sha256','graph_config_hash','graph_report_sha256','graph_vector_sha256','graph_centroid_sha256','graph_python_version')
    graph_provenance={f'{regime}/seed{seed}':{k:inputs[regime,seed][0][k] for k in graph_fields} for regime in ('old','new') for seed in SEEDS}
    source_paths={'freeze_sensitivity_models.py':Path(__file__).resolve(),'infer_sensitivity.py':Path(__file__).resolve().parent/'infer_sensitivity.py',
        'extract_test_pedigrees.py':Path(__file__).resolve().parent/'extract_test_pedigrees.py',
        'prepare_sensitivity_test.py':Path(__file__).resolve().parent/'prepare_sensitivity_test.py',
        'extract_sensitivity_outcomes.py':Path(__file__).resolve().parent/'extract_sensitivity_outcomes.py',
        'prepare_prediction.py':ROOT/'kaggle/prepare_prediction.py','graph_pilot.py':ROOT/'kaggle/graph_pilot.py',
        'evaluate_sensitivity.py':Path(__file__).resolve().parent/'evaluate_sensitivity.py','run_test_inference.py':ROOT/'kaggle/run_test_inference.py',
        'experiment_features.py':ROOT/'kaggle/experiment_features.py','audit_experiment_outputs.py':ROOT/'kaggle/audit_experiment_outputs.py',
        'complete_locked_reporting.py':ROOT/'output/icdees_focused_2026-09-18/companion/code/complete_locked_reporting.py',
        'evaluate_locked_test_results.py':ROOT/'output/icdees_focused_2026-09-18/companion/code/evaluate_locked_test_results.py',
        'evaluate_locked.py':ROOT/'output/icdees_focused_2026-09-18/companion/code/evaluate_locked.py',
        'assemble_cutoff.py':Path(__file__).resolve().parent/'assemble_cutoff.py',
        'audit_transport_records.py':Path(__file__).resolve().parent/'audit_transport_records.py',
        'build_transport_adapter.py':Path(__file__).resolve().parent/'build_transport_adapter.py',
        'build_signed_future_jobs.py':Path(__file__).resolve().parent/'execution/build_signed_future_jobs.py',
        'future_transport_runtime.py':Path(__file__).resolve().parent/'execution/future_transport_runtime.py',
        'SIGNED_TRANSPORT_POLICY_FREEZE.json':Path(__file__).resolve().parent/'execution/SIGNED_TRANSPORT_POLICY_FREEZE.json'}
    future_policy=audit._json(source_paths['SIGNED_TRANSPORT_POLICY_FREEZE.json'])
    assert future_policy['design_sha256']==DESIGN_SHA and future_policy['transport_design_sha256']==TRANSPORT_SHA
    for name,expected_hash in future_policy['source_sha256'].items():
        assert sha(Path(__file__).resolve().parent/'execution'/name)==expected_hash
    result={'status':'frozen_for_root_review','test_previously_seen':True,'test_inference_authorized':False,
        'design_sha256':DESIGN_SHA,'transport_design_sha256':TRANSPORT_SHA,
        'future_transport_policy_sha256':sha(source_paths['SIGNED_TRANSPORT_POLICY_FREEZE.json']),
        'transport_record_sha256':{kind+':'+':'.join(map(str,key)):value[1] for (kind,key),value in verified_transport.items()},
        'tuning_runs':tunes,'candidate_runs':candidates,'refit_runs':refits,
        'historical_test_input_sha256':{'0':'2190ef83d25885a8e565ec31661bdcf557d9183f84668dcbc0b8292861a0c21d',
          '1':'8273a777f2eef187ad1f5784b864fc18e3c3807663368956987801a23f2cdf33',
          '2':'b1a27eadb5e0cde3234490cec96294b6348d7fba107a7ffc97e7f93821ad29f9'},
        'historical_test_payload_sha256':{'0':'700563f4fd757f79c856d81175d6b23e1c0266fade719de59348ef63e77dddd4',
          '1':'738767c4218b66be204f70005310231b6adcc200b4b0769a7542f841c4c7047d',
          '2':'69fd5e0964d2bc0b579656d01fa6a1ebdeac36bddc330b6c2cf7fd5814c3c4bf'},
        'validation_rows':{s:len(v) for s,v in reference.items()},'validation_roster_sha256':sha(roster),
        'training_input_sidecars':{f'{regime}/seed{seed}':inputs[regime,seed][1] for regime in ('old','new') for seed in SEEDS},
        'training_graph_provenance':graph_provenance,'source_sha256':{name:sha(path) for name,path in source_paths.items()},
        'evaluation_contract':{'contrasts':['new_S6-old_S6','new_S6-S2','old_S6-S2'],
          'metrics':['entry_average_precision','race_mean_hit_at_3'],'replicates':2000,'seed':20261029,
          'unit':'whole race; shared across settings and seeds','interval_percentiles':[2.5,97.5],
          'aggregation':'mean seed-specific metric differences',
          'evaluation_python':'3.13.15','evaluation_numpy':'1.26.4',
          'outputs':{'exploratory_contrasts.csv':36,'seed_metrics.csv':54,'arm_metrics.csv':18}},'models':models}
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--design',required=True);a=p.parse_args()
    print(json.dumps(freeze(a.root,a.out,a.design),indent=2))
