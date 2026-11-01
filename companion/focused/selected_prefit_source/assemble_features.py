"""Assemble full original train/validation feature views, one original seed at a time."""
from pathlib import Path
from collections import Counter
import argparse,csv,gzip,hashlib,json,sys
import numpy as np
from ancestry import encode
from feature_adapter import P_COLUMNS

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT/'kaggle'),str(HERE.parent)]
from graph_pilot import load_parents
from assemble_cutoff import load_old

GRAPH=['embedding_status','fallback_depth','ancestor_count','embedding_missing','graph_cluster']+[f'embedding_{i}' for i in range(64)]
KEY=('race_id','horse_id_netkeiba','horse_number')
NAMES=('production-seed0-version2','production-seed1-version1','production-seed2-version1')


def sha(path,uncompressed=False):
    h=hashlib.sha256();op=gzip.open if uncompressed else open
    with op(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main(seed):
    target=HERE/'private_data';parent_file=target/'parent_features.csv.gz';parent_meta=target/'parent_manifest.json'
    meta=json.loads(parent_meta.read_text());assert sha(parent_file)==meta['output_sha256']
    source=ROOT/'kaggle/upload/jra-prediction-bundle-v1/train_validation.csv.gz';assert sha(source)==meta['source_sha256']
    paths={arm:target/f'{arm}-seed{seed}.csv.gz' for arm in ('Gmixed','Gancestor')}
    assert not any(p.exists() or p.with_suffix('.json').exists() for p in paths.values())
    folder=ROOT/'kaggle/notebook-output'/NAMES[seed]/'jra-production'/f'seed-{seed}'
    report_path=folder/'production_report.json';report=json.loads(report_path.read_text());assert report['status']=='complete' and report['settings']['seed']==seed
    graph=folder/'training_graph.edg';vf=folder/f'nodevectors_seed{seed}.npz';cf=folder/f'cluster_centers_seed{seed}.npy'
    for p in (graph,vf,cf):assert sha(p)==report['artifact_sha256'][p.name]
    assert sha(graph)==report['graph_sha256']
    with np.load(vf,allow_pickle=False) as n:
        ids=n['node_ids'].tolist();matrix=n['vectors'];assert len(ids)==len(set(ids))==93474 and matrix.shape==(93474,64) and np.isfinite(matrix).all()
        vectors={str(k):v for k,v in zip(ids,matrix.copy())}
    centers=np.load(cf,allow_pickle=False);assert centers.shape==(64,64) and np.isfinite(centers).all()
    parents=load_parents([ROOT/'local_jra_worker/data/raw_jbis_laptop.db',ROOT/'tmp/pedigree_patch.db'])
    parent={}
    with gzip.open(parent_file,'rt',encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            key=tuple(r[k] for k in KEY);assert key not in parent;parent[key]={k:r[k] for k in P_COLUMNS}
    assert len(parent)==636862
    old=None
    if seed:
        p=ROOT/'kaggle/upload/jra-experiment-graph-tables-v1'/f'graph_features_seed{seed}.csv.gz'
        if not p.exists():p=p.with_suffix('')
        m=ROOT/'kaggle/upload/jra-experiment-graph-tables-v1'/f'graph_features_seed{seed}.csv.json'
        old=load_old(p,m,seed,sha(m))
    outputs={a:gzip.open(p,'wt',encoding='utf-8',newline='',compresslevel=1) for a,p in paths.items()}
    keys=hashlib.sha256();base=hashlib.sha256();seen=set();cache={};counts=Counter();coverage=Counter()
    try:
        with gzip.open(source,'rt',encoding='utf-8',newline='') as f:
            reader=csv.DictReader(f);fields=reader.fieldnames+list(P_COLUMNS)
            assert not set(reader.fieldnames)&set(P_COLUMNS)
            writers={a:csv.DictWriter(o,fieldnames=fields) for a,o in outputs.items()}
            for w in writers.values():w.writeheader()
            base_columns=[n for n in reader.fieldnames if n not in GRAPH]
            for row in reader:
                assert row['split'] in ('train','validation')
                key=tuple(row[k] for k in KEY);assert key not in seen and key in parent;seen.add(key)
                row.update(parent[key]);counts[(row['split'],row['surface'].lower())]+=1
                keys.update((json.dumps(key,separators=(',',':'))+'\n').encode())
                base.update((json.dumps([row[n] for n in base_columns],separators=(',',':'),ensure_ascii=False)+'\n').encode())
                mixed=dict(row)
                if old is not None:mixed.update(old[row['horse_id_netkeiba']])
                writers['Gmixed'].writerow(mixed)
                horse=row['horse_id_netkeiba']
                if horse not in cache:cache[horse]=encode(horse,parents,vectors,centers)
                x=cache[horse];a=dict(row);a.update({k:x[k] for k in GRAPH[:5]})
                a.update({f'embedding_{i}':v for i,v in enumerate(x['vector'])})
                coverage[(row['split'],row['surface'].lower(),str(x['fallback_depth']))]+=1
                writers['Gancestor'].writerow(a)
    finally:
        for f in outputs.values():f.close()
    assert seen==set(parent) and len(seen)==636862
    assert counts=={('train','turf'):271549,('train','dirt'):277207,('validation','turf'):42783,('validation','dirt'):45323}
    for arm,p in paths.items():
        m={'purpose':'focused original cohort train-validation','splits':['train','validation'],'seed':seed,'arm':arm,
           'rows':len(seen),'ordered_key_sha256':keys.hexdigest(),'non_graph_sha256':base.hexdigest(),
           'data_sha256':sha(p,True),'raw_sha256':sha(p),'source_sha256':meta['source_sha256'],
           'parent_manifest_sha256':sha(parent_meta),'graph_report_sha256':sha(report_path),
           'graph_sha256':sha(graph),'graph_vector_sha256':sha(vf),'graph_centroid_sha256':sha(cf),
           'ancestor_code_sha256':sha(HERE/'ancestry.py'),'assembler_code_sha256':sha(Path(__file__)),
           'coverage':{'/'.join(k):v for k,v in sorted(coverage.items())} if arm=='Gancestor' else {}}
        p.with_suffix('.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'status':'PASS','seed':seed,'rows':len(seen),'keys':keys.hexdigest(),'non_graph':base.hexdigest()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=(0,1,2),required=True);main(p.parse_args().seed)
