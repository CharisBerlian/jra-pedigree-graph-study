"""Apply the frozen harvest checks to the first completed validation cell."""
import json
from pathlib import Path
import freeze_focused_models as f

root=f.PRIVATE/'model_harvest'; job=root/'xgboost'/'turf'/'H'
cert=f.PRIVATE/'remote_views_certificate.json'; prefit=f.HERE/'PREFIT_MANIFEST_V4.json'
m=f.audit._json(prefit); c=f.audit._json(cert); spec=f.audit._json(f.HERE/'EXPERIMENT_SPEC.json')
f.verify_outer(root,'H','xgboost','turf',cert,prefit)
passed=f.audit._json(job/'JOB_PASS.json')
assert passed['status']=='pass' and (passed['family'],passed['surface'],passed['arm'])==('xgboost','turf','H')
assert passed['prefit_sha256']==f.sha(prefit) and passed['remote_views_certificate_sha256']==f.sha(cert) and passed['spec_sha256']==m['spec_sha256']
assert f.sha(job/'spec.json')==m['spec_sha256']
runner=m['normalized_utf8_text_sha256']['kaggle/run_experiments.py']
helper=m['normalized_utf8_text_sha256']['output/revision_2026-10-29/science/focused_revision/feature_adapter.py']
assert f.sha(job/'run_experiments.py')==runner and f.sha(job/'experiment_features.py')==helper
inputs={s:next(x for x in c['views'] if x['seed']==s and x['view']=='Gmixed') for s in f.SEEDS}
roster=f.reference()['turf']; maps=f.audit._json(f.PRIVATE/'category_maps.json')
base=job/'runs/scientific/xgboost/turf/H'; tune=base/'tune/seed0'; marker=f.audit._complete(tune)
cfg=f.audit._json(tune/'config.json');policy=f.audit._policy(spec);grid=f.audit._candidate_grid(spec,'xgboost')
prov={'spec_sha256':m['spec_sha256'],'script_sha256':runner,'helper_sha256':helper}
f.audit._check_config(cfg,spec,'scientific','xgboost','turf','H','tune',0,policy,prov)
assert marker['config_hash']==f.audit._canonical(cfg) and cfg['candidate_grid_sha256']==f.audit._canonical(grid)
assert cfg['features']==list(f.features.feature_columns('H')) and cfg['categorical']==list(f.features.categorical_columns(cfg['features']))
assert cfg['input_sha256']==inputs[0]['payload_sha256'] and cfg['input_metadata_sha256']==inputs[0]['sidecar_sha256']
records=[]
for cid,params in grid.items():
    folder=tune/'candidates'/cid; cc=f.audit._json(folder/'config.json'); assert cc['parent_config_hash']==marker['config_hash']
    result=f.audit._validate_curve(folder,cc,params,policy,'xgboost')
    f.audit_candidate_roster_and_hit(folder/'validation_predictions.csv.gz','turf',roster,result['mean_hit_at_3'])
    records.append(result)
assert f.audit._json(tune/'candidate_results.json')==records
chosen=f.audit._json(tune/'selected.json'); winner,_=f.audit._winner(records,policy['ap_tie_tolerance']); assert chosen==winner
for seed in f.SEEDS:
    folder=base/'refit'/f'seed{seed}'
    f.audit._audit_refit(folder,spec,'scientific','xgboost','turf','H',seed,policy,cfg,chosen,prov)
    rc=f.audit._json(folder/'config.json');prep=f.audit._json(folder/'preprocessing.json');met=f.audit._json(folder/'metrics.json')
    assert rc['selection_sha256']==f.sha(tune/'selected.json') and rc['tuning_complete_sha256']==f.sha(tune/'complete.json')
    assert rc['input_sha256']==inputs[seed]['payload_sha256'] and rc['input_metadata_sha256']==inputs[seed]['sidecar_sha256']
    assert prep['features']==cfg['features'] and prep['categorical_columns']==cfg['categorical']
    assert prep['category_maps']==maps[f'H/turf/seed{seed}']
    assert f.read_predictions(folder/'validation_predictions.csv.gz','turf')==roster
    f.audit_candidate_roster_and_hit(folder/'validation_predictions.csv.gz','turf',roster,met['mean_hit_at_3'])
print(json.dumps({'status':'pass','cell':'xgboost/turf/H','candidates':len(records),'refits':3,'roster':len(roster)}))
