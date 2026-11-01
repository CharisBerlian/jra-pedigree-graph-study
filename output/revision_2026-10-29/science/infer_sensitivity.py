"""Post-review Kaggle-only inference for a frozen sensitivity model matrix.

No training, outcome labels, test metrics or model selection. This script is
prepared for later review; do not execute it before the root authorization.
"""
from pathlib import Path
import argparse,csv,gzip,hashlib,json,sys
from importlib import metadata as versions
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'kaggle'))
import run_test_inference as frozen
import experiment_features as features
import graph_pilot

FAMILIES=('xgboost','lightgbm','catboost');SURFACES=('turf','dirt');ARMS=('S2','old_S6','new_S6');SEEDS=(0,1,2)
FORBIDDEN={'label','finish_pos','result_status','time','margin','last_3f','corner_passing','top_three'}
GRAPH={'embedding_status','fallback_depth','ancestor_count','embedding_missing','graph_cluster'}|{f'embedding_{i}' for i in range(64)}

def sha(path):return frozen._sha256(path)
def key(row):return (str(row['race_id']),str(row['horse_id_netkeiba']),int(row['horse_number']))
def historical_test_source(path,seed,manifest):
    path=Path(path);payload=hashlib.sha256()
    if path.suffix=='.gz':
        assert sha(path)==manifest['historical_test_input_sha256'][str(seed)]
        stream=gzip.open(path,'rb')
    else:
        assert path.suffix=='.csv'
        stream=path.open('rb')
    with stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):payload.update(block)
    assert payload.hexdigest()==manifest['historical_test_payload_sha256'][str(seed)]
    return payload.hexdigest()
def key_digest(keys):
    digest=hashlib.sha256()
    for item in sorted(keys):digest.update((json.dumps(item,ensure_ascii=False,separators=(',',':'))+'\n').encode())
    return digest.hexdigest()

def non_graph_content(rows):
    columns=sorted(set(rows[0])-GRAPH)
    values={key(r):hashlib.sha256(json.dumps([r.get(c) for c in columns],ensure_ascii=False,separators=(',',':')).encode()).digest() for r in rows}
    assert len(values)==len(rows)
    return columns,values

def test_input(path,seed,regime,training_graph):
    path=Path(path);meta=frozen._json(frozen._test_sidecar(path))
    assert meta['regime']==regime and meta['seed']==seed and meta['splits']==['test']
    assert meta['test_labels_read'] is False and meta['test_metrics_computed'] is False
    assert meta['graph_python_version']==('3.13.15' if regime=='new' else '3.12.13')
    for name,value in training_graph.items():assert meta.get(name)==value, f'test graph mismatch: {name}'
    op=gzip.open if path.suffix=='.gz' else open
    with op(path,'rt',encoding='utf-8',newline='') as f:
        header=csv.DictReader(f).fieldnames
    assert not (set(header)&FORBIDDEN), 'outcomes present in inference input'
    rows,_=frozen.read_test_rows(path,expected_seed=seed)
    return rows,meta

def infer(manifest_path,authorization_path,model_root,input_root,out):
    assert frozen._on_kaggle(), 'inference must run on Kaggle'
    assert sys.version.split()[0]=='3.13.15'
    for name,want in {'numpy':'1.26.4','scipy':'1.13.1','pandas':'2.2.3','scikit-learn':'1.6.1','xgboost':'3.0.2','lightgbm':'4.6.0','catboost':'1.2.8'}.items():
        assert versions.version(name)==want, f'inference package mismatch: {name}'
    manifest_path=Path(manifest_path);manifest=frozen._json(manifest_path);auth=frozen._json(authorization_path)
    assert manifest['status']=='frozen_for_root_review' and manifest['test_previously_seen'] is True
    assert auth=={'authorized_by_root':True,'frozen_manifest_sha256':sha(manifest_path),'scope':'exploratory test inference and evaluation'}
    assert (manifest['tuning_runs'],manifest['candidate_runs'],manifest['refit_runs'])==(18,72,54)
    for name,path in {'infer_sensitivity.py':Path(__file__),'run_test_inference.py':Path(frozen.__file__),
        'experiment_features.py':Path(features.__file__),'prepare_prediction.py':Path(features.frozen.__file__),
        'graph_pilot.py':Path(graph_pilot.__file__)}.items():
        assert sha(path)==manifest['source_sha256'][name], f'source hash mismatch: {name}'
    assert features.frozen.load_parents is graph_pilot.load_parents
    expected={(f,s,a) for f in FAMILIES for s in SURFACES for a in ARMS}
    entries={(m['family'],m['surface'],m['arm']):m for m in manifest['models']}
    assert len(entries)==len(manifest['models'])==18 and set(entries)==expected
    inputs={};input_summary={};reference=None;content_reference=None
    for regime in ('old','new'):
        for seed in SEEDS:
            path=Path(input_root)/f'{regime}-seed{seed}.csv.gz';rows,meta=test_input(path,seed,regime,manifest['training_graph_provenance'][f'{regime}/seed{seed}'])
            keys=[key(r) for r in rows];assert len(keys)==len(set(keys))
            signature={(key(r),r['surface'],int(r['field_size'])) for r in rows}
            if reference is None:reference=signature
            else:assert signature==reference, 'test rosters differ between regimes/seeds'
            content=non_graph_content(rows)
            if content_reference is None:content_reference=content
            else:assert content==content_reference, 'non-graph test features differ between regimes/seeds'
            inputs[regime,seed]=(rows,meta,sha(path))
            coverage={}
            for surface in SURFACES:
                counts={status:sum(r['surface'].lower()==surface and r['embedding_status']==status for r in rows) for status in ('direct','fallback','missing')}
                assert sum(counts.values())=={'turf':43477,'dirt':45660}[surface]
                coverage[surface]=counts
            input_summary[f'{regime}/seed{seed}']={'sha256':sha(path),'coverage_entries':coverage}
    out=Path(out);assert not out.exists();out.mkdir(parents=True)
    roster_keys={surface:key_digest(k for k,s,field in reference if s.lower()==surface) for surface in SURFACES}
    inventory=[]
    for family,surface,arm in sorted(entries):
        entry=entries[family,surface,arm];setting='S2' if arm=='S2' else 'S6';regime='new' if arm=='new_S6' else 'old'
        assert entry['setting']==setting and entry['input_regime']==regime
        by_seed={r['seed']:r for r in entry['refits']};assert set(by_seed)==set(SEEDS)
        for seed in SEEDS:
            item=by_seed[seed];folder=Path(model_root)/item['relative_dir']
            frozen._complete(folder,item['artifacts'])
            assert sha(folder/'complete.json')==item['complete_sha256'] and sha(folder/'config.json')==item['config_sha256']
            assert sha(folder/'preprocessing.json')==item['preprocessing_sha256'] and sha(folder/frozen.MODEL_NAMES[family])==item['model_sha256']
            config=frozen._json(folder/'config.json');prep=frozen._json(folder/'preprocessing.json')
            assert (config['family'],config['surface'],config['setting'],config['seed'])==(family,surface,setting,seed)
            assert config['selected_candidate']==entry['selected_candidate'] and config['selected_round']==entry['selected_round']
            assert prep['features']==list(features.feature_columns(setting)) and prep['categorical_columns']==list(features.categorical_columns(prep['features']))
            assert set(prep['category_maps'])==set(prep['categorical_columns'])
            rows,meta,input_sha=inputs[regime,seed];surface_rows=[r for r in rows if r['surface'].lower()==surface]
            frame=frozen._prepare_frame(surface_rows,prep,family)
            scores=frozen._load_predict(family,folder/frozen.MODEL_NAMES[family],frame,entry['selected_round'])
            dest=out/family/surface/arm/f'seed{seed}'/'test_predictions.csv.gz'
            frozen._write_predictions(dest,surface_rows,scores)
            inventory.append({'family':family,'surface':surface,'arm':arm,'seed':seed,'rows':len(surface_rows),
                'path':dest.relative_to(out).as_posix(),'sha256':sha(dest),'test_input_sha256':input_sha,
                'model_sha256':item['model_sha256'],'preprocessing_sha256':item['preprocessing_sha256']})
    assert len(inventory)==54
    (out/'inference_report.json').write_text(json.dumps({'status':'complete','test_previously_seen':True,
        'test_labels_read':False,'test_metrics_computed':False,'model_manifest_sha256':sha(manifest_path),
        'authorization_sha256':sha(authorization_path),'inputs':input_summary,'roster_key_sha256':roster_keys,'predictions':inventory},indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--authorization',required=True);p.add_argument('--models',required=True);p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    infer(a.manifest,a.authorization,a.models,a.inputs,a.output)
