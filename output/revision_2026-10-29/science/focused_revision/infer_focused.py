"""After root authorization, run exactly 72 label-free focused predictions."""
from pathlib import Path
from importlib import metadata
from collections import Counter
import argparse,hashlib,json,sys
import feature_adapter as features
from assemble_focused_test import gate,GRAPH,FORBIDDEN,sha
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'kaggle'))
import run_test_inference as frozen

FAMILIES=('xgboost','lightgbm','catboost');SURFACES=('turf','dirt');ARMS=('H','Gmixed','Gancestor','P');SEEDS=(0,1,2)


def key(row):return (str(row['race_id']),str(row['horse_id_netkeiba']),int(row['horse_number']))
def digest_keys(keys):
    h=hashlib.sha256()
    for k in sorted(keys):h.update((json.dumps(k,separators=(',',':'),ensure_ascii=False)+'\n').encode())
    return h.hexdigest()


def memory_environment():
    import resource
    result={'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    mem=Path('/proc/meminfo')
    result['proc_memtotal']=next((line for line in mem.read_text().splitlines() if line.startswith('MemTotal:')),None) if mem.is_file() else None
    result['cgroup_memory_limits']={}
    for p in (Path('/sys/fs/cgroup/memory.max'),Path('/sys/fs/cgroup/memory/memory.limit_in_bytes')):
        if p.is_file():result['cgroup_memory_limits'][str(p)]=p.read_text().strip()
    return result


def infer(manifest_path,authorization_path,model_root,input_root,out):
    m=gate(manifest_path,authorization_path);assert sha(Path(__file__))==m['source_sha256']['infer_focused.py']
    assert frozen._on_kaggle() and sys.version.split()[0]=='3.13.15'
    for name,want in {'numpy':'1.26.4','scipy':'1.13.1','pandas':'2.2.3','scikit-learn':'1.6.1','xgboost':'3.0.2','lightgbm':'4.6.0','catboost':'1.2.8'}.items():assert metadata.version(name)==want
    entries={(x['family'],x['surface'],x['arm']):x for x in m['models']}
    assert len(entries)==24 and set(entries)=={(f,s,a) for f in FAMILIES for s in SURFACES for a in ARMS}
    input_root=Path(input_root);report=frozen._json(input_root/'test_view_report.json')
    assert report['status']=='complete' and report['test_labels_read'] is False and report['test_metrics_computed'] is False
    assert report['model_manifest_sha256']==sha(manifest_path) and report['authorization_sha256']==sha(authorization_path)
    views={(x['view'],x['seed']):x for x in report['views']}
    assert len(report['views'])==len(views)==6 and set(views)=={(v,s) for v in ('Gmixed','Gancestor') for s in SEEDS}
    inputs={};reference=None;content_reference=None;summary={}
    for view in ('Gmixed','Gancestor'):
      for seed in SEEDS:
        path=input_root/f'{view}-seed{seed}.csv.gz';side=path.with_suffix('.json');meta=frozen._json(side);item=views[view,seed]
        assert sha(path)==item['data_sha256']==meta['data_sha256'] and sha(side)==item['sidecar_sha256']
        assert meta['view']==view and meta['seed']==seed and meta['graph_provenance']==m['training_graph_provenance'][seed]
        assert meta['test_labels_read'] is False and meta['test_metrics_computed'] is False
        assert meta['parent_map_sha256']==m['parent_map_sha256'] and meta['model_manifest_sha256']==sha(manifest_path) and meta['authorization_sha256']==sha(authorization_path)
        rows,_=frozen.read_test_rows(path,expected_seed=seed);assert len(rows)==89137 and not (set(rows[0])&FORBIDDEN)
        roster={(key(r),r['surface'],int(r['field_size'])) for r in rows};assert len(roster)==len(rows)
        cols=sorted(set(rows[0])-GRAPH);content={key(r):hashlib.sha256(json.dumps([r[c] for c in cols],separators=(',',':'),ensure_ascii=False).encode()).digest() for r in rows}
        if reference is None:reference=roster;content_reference=content
        else:assert roster==reference and content==content_reference
        counts={}
        for surface in SURFACES:
            subgroup=[r for r in rows if r['surface'].lower()==surface];assert len(subgroup)=={'turf':43477,'dirt':45660}[surface]
            statuses=Counter(r['embedding_status'] for r in subgroup)
            counts[surface]={'entries':len(subgroup),'coverage':dict(statuses),
                'parent_zero_counts':{role:sum(int(r[role+'_prior_count'])==0 for r in subgroup) for role in ('sire','dam','damsire')}}
        inputs[view,seed]=(path,sha(path));summary[f'{view}/seed{seed}']={'sha256':sha(path),'surfaces':counts}
        del rows,content,roster
    out=Path(out);assert not out.exists();out.mkdir(parents=True);inventory=[]
    # Load one pre-verified seed/view at a time after the complete parity pass.
    # This preserves all-model/full-roster semantics without retaining six views.
    for view in ('Gmixed','Gancestor'):
      for seed in SEEDS:
        path,input_hash=inputs[view,seed];rows,_=frozen.read_test_rows(path,expected_seed=seed)
        for family,surface,arm in sorted(entries):
            if ('Gancestor' if arm=='Gancestor' else 'Gmixed')!=view:continue
            model=entries[family,surface,arm];refits={r['seed']:r for r in model['refits']};assert set(refits)==set(SEEDS)
            item=refits[seed];folder=Path(model_root)/item['relative_dir'];frozen._complete(folder,item['artifacts'])
            assert sha(folder/'complete.json')==item['complete_sha256'] and sha(folder/'config.json')==item['config_sha256']
            assert sha(folder/'preprocessing.json')==item['preprocessing_sha256'] and sha(folder/frozen.MODEL_NAMES[family])==item['model_sha256']
            config=frozen._json(folder/'config.json');prep=frozen._json(folder/'preprocessing.json')
            assert (config['family'],config['surface'],config['setting'],config['seed'])==(family,surface,arm,seed)
            assert config['selected_candidate']==model['selected_candidate'] and config['selected_round']==model['selected_round']
            assert prep['features']==list(features.feature_columns(arm)) and prep['categorical_columns']==list(features.categorical_columns(prep['features']))
            part=[r for r in rows if r['surface'].lower()==surface]
            frame=frozen._prepare_frame(part,prep,family);scores=frozen._load_predict(family,folder/frozen.MODEL_NAMES[family],frame,model['selected_round'])
            dest=out/family/surface/arm/f'seed{seed}'/'test_predictions.csv.gz';frozen._write_predictions(dest,part,scores)
            inventory.append({'family':family,'surface':surface,'arm':arm,'seed':seed,'path':dest.relative_to(out).as_posix(),
                'sha256':sha(dest),'rows':len(part),'test_input_sha256':input_hash,'model_sha256':item['model_sha256']})
            del frame,scores,part
        del rows
    assert len(inventory)==72
    keys={surface:digest_keys(k for k,s,n in reference if s.lower()==surface) for surface in SURFACES}
    (out/'inference_report.json').write_text(json.dumps({'status':'complete','test_previously_seen':True,'test_labels_read':False,'test_metrics_computed':False,
        'model_manifest_sha256':sha(manifest_path),'authorization_sha256':sha(authorization_path),'roster_key_sha256':keys,
        'inputs':summary,'memory_environment':memory_environment(),'predictions':inventory},indent=2,sort_keys=True)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--authorization',required=True);p.add_argument('--models',required=True);p.add_argument('--inputs',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    infer(a.manifest,a.authorization,a.models,a.inputs,a.out)
