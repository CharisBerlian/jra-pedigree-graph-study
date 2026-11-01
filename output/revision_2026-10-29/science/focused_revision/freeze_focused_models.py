"""Audit 24 focused validation-only cells and freeze 72 refits for root review."""
from pathlib import Path
import argparse,ast,csv,gzip,hashlib,json,subprocess,sys,tempfile
from datetime import datetime,timezone
import feature_adapter as features
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];PRIVATE=HERE/'private_data'
sys.path[:0]=[str(ROOT/'kaggle'),str(HERE.parent)]
import audit_experiment_outputs as audit
from freeze_sensitivity_models import audit_candidate_roster_and_hit,read_predictions

ARMS=('H','Gmixed','Gancestor','P');FAMILIES=('xgboost','lightgbm','catboost');SURFACES=('turf','dirt');SEEDS=(0,1,2)


def sha(path):return audit._sha256(path)


def reference():
    source=ROOT/'kaggle/upload/jra-prediction-bundle-v1/train_validation.csv.gz';out={s:{} for s in SURFACES};races={}
    assert sha(source)==audit._json(PRIVATE/'parent_manifest.json')['source_sha256']
    with gzip.open(source,'rt',encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            assert r['split'] in ('train','validation')
            if r['split']!='validation':continue
            s=r['surface'].lower();k=(r['race_id'],r['horse_id_netkeiba'],r['horse_number']);assert k not in out[s]
            out[s][k]=int(r['label']);group=races.setdefault((s,r['race_id']),[int(r['field_size']),0]);assert group[0]==int(r['field_size']);group[1]+=1
    assert {s:len(v) for s,v in out.items()}=={'turf':42783,'dirt':45323}
    assert all(n==size and n>=3 for size,n in races.values())
    return out


def verify_outer(root,arm,family,surface,cert,prefit):
    name=f'{family}_{surface}_{arm}';folder=root/'transport'/name
    record=audit._json(folder/'prelaunch_record.json');remote=audit._json(folder/'remote_ledger.json');source=folder/'source.py'
    assert record['prefit_sha256']==sha(prefit) and record['certificate_sha256']==sha(cert)
    assert record['identity']=={'arm':arm,'family':family,'surface':surface}
    assert remote['status']=='complete' and remote['private'] is True and remote['remote_source_equal'] is True
    assert remote['source_sha256']==record['script_sha256']==sha(source) and remote['prelaunch_record_sha256']==sha(folder/'prelaunch_record.json')
    assert remote['image']=='gcr.io/kaggle-images/python@sha256:17b2fc7f27c25c6c2fd4230ec8e66097ab2b89fc88e1e4d6f338b72e1ee0bc3e'
    assert int(remote['kernel_id'])>0 and int(remote['version'])>0
    tree=ast.parse(source.read_text(encoding='utf-8'))
    def literal(name):
        nodes=[n.value for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name]
        assert len(nodes)==1;return ast.literal_eval(nodes[0])
    items=literal('items');inner=literal('inner');jobroot=PRIVATE/'focused_model_jobs_v4';job=jobroot/f'{name}.py'
    assert sha(job)==record['inner_sha256'] and inner.encode('utf-8')==job.read_bytes()
    # Regenerate the exact wrapper from its private inputs and approved source.
    with tempfile.TemporaryDirectory(dir=PRIVATE,prefix='audit-transport-') as tmp:
        tmp=Path(tmp);inp=tmp/'items.json';out=tmp/'wrapper.py';inp.write_text(json.dumps(items),encoding='utf-8')
        cmd=[sys.executable,str(HERE/'build_model_transport.py'),'--family',family,'--surface',surface,'--arm',arm,
             '--certificate',str(cert),'--inputs-json',str(inp),'--out',str(out),'--jobs-dir',str(jobroot)]
        subprocess.run(cmd,check=True,capture_output=True)
        assert out.read_bytes()==source.read_bytes()
        assert audit._json(out.with_suffix('.json'))==record
    return sha(folder/'prelaunch_record.json')


def freeze(root,cert,approval,out):
    root=Path(root);cert=Path(cert);approval=Path(approval);out=Path(out)
    assert not out.exists() and all(x.resolve().is_relative_to(PRIVATE.resolve()) for x in (root,cert,approval,out))
    prefit=HERE/'PREFIT_MANIFEST_V4.json';m=audit._json(prefit);c=audit._json(cert);auth=audit._json(approval)
    assert sha(PRIVATE/'parent_manifest.json')==m['parent_manifest_sha256']
    assert c['status']=='verified' and c['prefit_sha256']==sha(prefit)
    assert auth=={'approved_by_root':True,'prefit_sha256':sha(prefit),'remote_views_certificate_sha256':sha(cert),'scope':'focused validation model fitting'}
    audit._reject_test_paths(root);spec=audit._json(HERE/'EXPERIMENT_SPEC.json');assert sha(HERE/'EXPERIMENT_SPEC.json')==m['spec_sha256']
    roster=reference();maps=audit._json(PRIVATE/'category_maps.json');assert sha(PRIVATE/'category_maps.json')==m['category_maps_sha256']
    parent_audit=audit._json(PRIVATE/'validation_frozen_parent_audit.json')
    assert parent_audit['status']=='pass' and parent_audit['validation_rows_checked']==88106
    assert parent_audit['map_sha256']==m['parent_map_sha256']==sha(PRIVATE/'parent_maps.json')
    assert parent_audit['application_source_sha256']==sha(HERE/'apply_frozen_parent.py')
    assert parent_audit['checker_source_sha256']==sha(HERE/'check_validation_parent.py')
    runner=m['normalized_utf8_text_sha256']['kaggle/run_experiments.py']
    helper=m['normalized_utf8_text_sha256']['output/revision_2026-10-29/science/focused_revision/feature_adapter.py']
    models=[];candidate_count=0;refit_count=0;cap_count=0;transport_hashes={}
    for family in FAMILIES:
      for surface in SURFACES:
       for arm in ARMS:
        job=root/family/surface/arm;passed=audit._json(job/'JOB_PASS.json');name=f'{family}_{surface}_{arm}'
        transport_hashes[name]=verify_outer(root,arm,family,surface,cert,prefit)
        assert passed['status']=='pass' and (passed['family'],passed['surface'],passed['arm'])==(family,surface,arm)
        assert passed['prefit_sha256']==sha(prefit) and passed['remote_views_certificate_sha256']==sha(cert) and passed['spec_sha256']==m['spec_sha256']
        assert sha(job/'spec.json')==m['spec_sha256'] and sha(job/'run_experiments.py')==runner and sha(job/'experiment_features.py')==helper
        view='Gancestor' if arm=='Gancestor' else 'Gmixed';inputs={s:next(x for x in c['views'] if x['seed']==s and x['view']==view) for s in SEEDS}
        base=job/'runs'/'scientific'/family/surface/arm;tune=base/'tune'/'seed0';marker=audit._complete(tune)
        cfg=audit._json(tune/'config.json');policy=audit._policy(spec);grid=audit._candidate_grid(spec,family)
        provenance={'spec_sha256':m['spec_sha256'],'script_sha256':runner,'helper_sha256':helper}
        audit._check_config(cfg,spec,'scientific',family,surface,arm,'tune',0,policy,provenance)
        assert marker['config_hash']==audit._canonical(cfg) and cfg['candidate_grid_sha256']==audit._canonical(grid)
        assert cfg['features']==list(features.feature_columns(arm)) and cfg['categorical']==list(features.categorical_columns(cfg['features']))
        assert cfg['input_sha256']==inputs[0]['payload_sha256'] and cfg['input_metadata_sha256']==inputs[0]['sidecar_sha256']
        records=[]
        for cid,params in grid.items():
            folder=tune/'candidates'/cid;cc=audit._json(folder/'config.json');assert cc['parent_config_hash']==marker['config_hash']
            result=audit._validate_curve(folder,cc,params,policy,family)
            audit_candidate_roster_and_hit(folder/'validation_predictions.csv.gz',surface,roster[surface],result['mean_hit_at_3'])
            records.append(result);candidate_count+=1;cap_count+=int(result['hit_cap'])
        assert audit._json(tune/'candidate_results.json')==records
        chosen=audit._json(tune/'selected.json');winner,_=audit._winner(records,policy['ap_tie_tolerance']);assert chosen==winner
        refits=[]
        for seed in SEEDS:
            folder=base/'refit'/f'seed{seed}';audit._audit_refit(folder,spec,'scientific',family,surface,arm,seed,policy,cfg,chosen,provenance)
            rc=audit._json(folder/'config.json');prep=audit._json(folder/'preprocessing.json');met=audit._json(folder/'metrics.json')
            assert rc['selection_sha256']==sha(tune/'selected.json') and rc['tuning_complete_sha256']==sha(tune/'complete.json')
            assert rc['input_sha256']==inputs[seed]['payload_sha256'] and rc['input_metadata_sha256']==inputs[seed]['sidecar_sha256']
            assert prep['features']==cfg['features'] and prep['categorical_columns']==cfg['categorical']
            assert prep['category_maps']==maps[f'{arm}/{surface}/seed{seed}']
            assert read_predictions(folder/'validation_predictions.csv.gz',surface)==roster[surface]
            audit_candidate_roster_and_hit(folder/'validation_predictions.csv.gz',surface,roster[surface],met['mean_hit_at_3'])
            cm=audit._complete(folder);refits.append({'seed':seed,'relative_dir':folder.relative_to(root).as_posix(),
                'artifacts':cm['artifacts'],'complete_sha256':sha(folder/'complete.json'),'config_sha256':sha(folder/'config.json'),
                'model_sha256':cm['artifacts'][audit.MODEL_NAMES[family]],'preprocessing_sha256':cm['artifacts']['preprocessing.json']})
            refit_count+=1
        models.append({'family':family,'surface':surface,'arm':arm,'selected_candidate':chosen['candidate_id'],
            'selected_round':chosen['best_round'],'selected_sha256':sha(tune/'selected.json'),'tune_complete_sha256':sha(tune/'complete.json'),
            'candidate_results_sha256':sha(tune/'candidate_results.json'),
            'candidate_evidence':[{'candidate_id':x['candidate_id'],'validation_ap':x['average_precision'],
                                   'validation_hit_at_3':x['mean_hit_at_3'],'best_round':x['best_round'],'hit_cap':x['hit_cap']} for x in records],
            'refits':refits})
    assert len(models)==24 and candidate_count==96 and refit_count==72
    sources=['freeze_focused_models.py','apply_frozen_parent.py','assemble_focused_test.py','infer_focused.py','extract_focused_outcomes.py','evaluate_focused.py','metrics.py',
             'ancestry.py','feature_adapter.py','parent_control.py','check_validation_parent.py','test_frozen_parent.py','test_future_focused.py']
    source_paths={x:HERE/x for x in sources}
    source_paths.update({'run_test_inference.py':ROOT/'kaggle/run_test_inference.py','graph_pilot.py':ROOT/'kaggle/graph_pilot.py',
        'prepare_prediction.py':ROOT/'kaggle/prepare_prediction.py','experiment_features.py':ROOT/'kaggle/experiment_features.py',
        'run_experiments.py':ROOT/'kaggle/run_experiments.py','audit_experiment_outputs.py':ROOT/'kaggle/audit_experiment_outputs.py',
        'freeze_sensitivity_models.py':HERE.parent/'freeze_sensitivity_models.py',
        'extract_sensitivity_outcomes.py':HERE.parent/'extract_sensitivity_outcomes.py',
        'evaluate_locked_test_results.py':ROOT/'output/icdees_focused_2026-09-18/companion/code/evaluate_locked_test_results.py',
        'evaluate_locked.py':ROOT/'output/icdees_focused_2026-09-18/companion/code/evaluate_locked.py'})
    parent_manifest=audit._json(PRIVATE/'parent_manifest.json')
    pedigree_hashes=[parent_manifest['source_databases_sha256'][x] for x in ('local_jra_worker\\data\\raw_jbis_laptop.db','tmp\\pedigree_patch.db')]
    result={'status':'frozen_for_root_review','created_utc':datetime.now(timezone.utc).isoformat(),
        'test_previously_seen':True,'test_inference_authorized':False,
        'prefit_sha256':sha(prefit),'remote_views_certificate_sha256':sha(cert),'model_approval_sha256':sha(approval),
        'model_count':72,'tuning_cells':24,'candidate_count':96,'candidate_cap_count':cap_count,'models':models,
        'transport_record_sha256':transport_hashes,'training_graph_provenance':m['graph_artifacts'],
        'parent_map_sha256':m['parent_map_sha256'],'pedigree_source_sha256':pedigree_hashes,'category_maps_sha256':m['category_maps_sha256'],
        'validation_frozen_parent_audit_sha256':sha(PRIVATE/'validation_frozen_parent_audit.json'),
        'historical_test_input_sha256':{'0':'2190ef83d25885a8e565ec31661bdcf557d9183f84668dcbc0b8292861a0c21d',
          '1':'8273a777f2eef187ad1f5784b864fc18e3c3807663368956987801a23f2cdf33','2':'b1a27eadb5e0cde3234490cec96294b6348d7fba107a7ffc97e7f93821ad29f9'},
        'historical_test_payload_sha256':{'0':'700563f4fd757f79c856d81175d6b23e1c0266fade719de59348ef63e77dddd4',
          '1':'738767c4218b66be204f70005310231b6adcc200b4b0769a7542f841c4c7047d','2':'69fd5e0964d2bc0b579656d01fa6a1ebdeac36bddc330b6c2cf7fd5814c3c4bf'},
        'source_sha256':{name:sha(path) for name,path in source_paths.items()},'evaluation_contract':m['metrics'],'comparisons':m['comparisons']}
    out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--certificate',required=True);p.add_argument('--approval',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    print(json.dumps({'status':freeze(a.root,a.certificate,a.approval,a.out)['status']}))
