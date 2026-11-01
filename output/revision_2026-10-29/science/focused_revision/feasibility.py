"""Train/validation, label-free cohort and ancestry availability only.

No test rows, targets, model predictions or outcome data are read. This is a
feasibility audit, not a scientific model fit or a feature builder.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'kaggle'))
from graph_pilot import load_parents


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parents=load_parents([ROOT/'local_jra_worker/data/raw_jbis_laptop.db',ROOT/'tmp/pedigree_patch.db'])
    graph_base=ROOT/'kaggle/notebook-output'
    vectors=[graph_base/'production-seed0-version2/jra-production/seed-0/nodevectors_seed0.npz',
             graph_base/'production-seed1-version1/jra-production/seed-1/nodevectors_seed1.npz',
             graph_base/'production-seed2-version1/jra-production/seed-2/nodevectors_seed2.npz']
    node_sets=[]
    for path in vectors:
        report=json.loads((path.parent/'production_report.json').read_text())
        assert sha(path)==report['artifact_sha256'][path.name]
        with np.load(path,allow_pickle=False) as n:
            ids=n['node_ids'].tolist();assert len(ids)==len(set(ids))
            node_sets.append(set(ids))
    assert node_sets[0]==node_sets[1]==node_sets[2]
    nodes=node_sets[0]
    cache={}
    def inspect(horse):
        if horse not in cache:
            frontier={horse};depth=0
            for d in range(1,4):
                frontier={parents.get((child,relation)) for child in frontier for relation in ('sire','dam')}
                frontier.discard(None);frontier.discard(horse)
                if frontier & nodes:depth=d;break
            sire=parents.get((horse,'sire'));dam=parents.get((horse,'dam'))
            cache[horse]=(depth,bool(sire),bool(dam),bool(dam and parents.get((dam,'sire'))))
        return cache[horse]
    source=ROOT/'kaggle/upload/jra-prediction-bundle-v1/train_validation.csv.gz'
    cols=['race_id','horse_id_netkeiba','split','surface','sire','dam']
    stats={}
    for df in pd.read_csv(source,usecols=cols,dtype=str,keep_default_na=False,chunksize=40000):
        assert set(df['split'])<= {'train','validation'}
        for row in df.itertuples(index=False):
            split,surface=row.split,row.surface.lower();assert surface in ('turf','dirt')
            c=stats.setdefault((split,surface),Counter());c['rows']+=1
            depth,sire,dam,damsire=inspect(row.horse_id_netkeiba)
            c[f'ancestor_depth_{depth}']+=1
            c['sire_available']+=sire;c['dam_available']+=dam;c['damsire_available']+=damsire
            c['sire_export_disagreement']+=bool(row.sire and row.sire!=parents.get((row.horse_id_netkeiba,'sire')))
            c['dam_export_disagreement']+=bool(row.dam and row.dam!=parents.get((row.horse_id_netkeiba,'dam')))
    expected={('train','turf'):271549,('train','dirt'):277207,('validation','turf'):42783,('validation','dirt'):45323}
    assert {k:v['rows'] for k,v in stats.items()}==expected
    result={'scope':'train and validation only; no labels/test rows/model performance read',
            'original_graph_nodes':len(nodes),'train_validation_by_split_surface':{
                '/'.join(k):dict(sorted(v.items())) for k,v in sorted(stats.items())},
            'vector_sha256':{str(p.relative_to(ROOT)):sha(p) for p in vectors},
            'source_train_validation_sha256':sha(source)}
    (OUT/'feasibility.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
