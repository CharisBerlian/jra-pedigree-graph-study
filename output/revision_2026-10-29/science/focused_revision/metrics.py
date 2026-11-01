"""Full-roster scores and race-paired bootstrap using frozen historical AP code."""
from pathlib import Path
import math,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
local=Path(__file__).resolve().parent
helper=local if (local/'evaluate_locked_test_results.py').exists() else ROOT/'output/icdees_focused_2026-09-18/companion/code'
sys.path.insert(0,str(helper))
import evaluate_locked_test_results as historical


def prepare(rows,scores):
    assert rows and len(rows)==len(scores) and all(math.isfinite(float(s)) and 0<=float(s)<=1 for s in scores)
    if any('surface' in r for r in rows):assert all('surface' in r for r in rows) and len({r['surface'] for r in rows})==1
    keys=[(r['race_id'],r['horse_id_netkeiba'],int(r['horse_number'])) for r in rows]
    assert len(set(keys))==len(keys) and all(int(r['label']) in (0,1) for r in rows)
    groups={}
    for i,r in enumerate(rows):groups.setdefault(r['race_id'],[]).append(i)
    races=sorted(groups);idx={r:i for i,r in enumerate(races)}
    for race,entries in groups.items():
        assert len(entries)>=3 and all(int(rows[i]['field_size'])==len(entries) for i in entries)
        assert len({int(rows[i]['horse_number']) for i in entries})==len(entries)
    race_index=np.array([idx[r['race_id']] for r in rows],dtype=int)
    y=np.array([int(r['label']) for r in rows],dtype=int);score=np.array(scores,dtype=float)
    hits=np.zeros(len(races),dtype=float)
    for race,entries in groups.items():
        selected=sorted(entries,key=lambda i:(-score[i],int(rows[i]['horse_number']),rows[i]['horse_id_netkeiba']))[:3]
        hits[idx[race]]=sum(y[i] for i in selected)/3.0
    return historical._ap_prepare(score,y,race_index),hits


def paired(rows,models,replicates=2000,seed=20261029):
    """models maps arm to exactly three seed-specific score arrays."""
    assert set(models)=={'H','Gmixed','Gancestor','P'} and all(len(v)==3 for v in models.values())
    prepared={arm:[prepare(rows,s) for s in values] for arm,values in models.items()}
    races=len(prepared['H'][0][1]);draws=historical._race_counts(races,replicates,seed)
    points={};samples={}
    for arm,sets in prepared.items():
        ap=[];hits=[];ap_boot=[];hit_boot=[]
        for ap_data,hit in sets:
            ap.append(historical._weighted_ap(ap_data,np.ones(races,dtype=int)))
            hits.append(float(hit.mean()))
            ap_boot.append(np.array([historical._weighted_ap(ap_data,weights) for weights in draws]))
            hit_boot.append(draws@hit/races)
        points[arm]={'ap':ap,'hit':hits}
        samples[arm]={'ap':np.mean(ap_boot,axis=0),'hit':np.mean(hit_boot,axis=0)}
    result={}
    for first,second in (('Gancestor','H'),('Gancestor','P'),('Gancestor','Gmixed')):
        for metric in ('ap','hit'):
            diff=samples[first][metric]-samples[second][metric]
            point=float(np.mean(points[first][metric])-np.mean(points[second][metric]))
            lo,hi=np.percentile(diff,[2.5,97.5]);record={'point':point,'low':float(lo),'high':float(hi)}
            if metric=='hit':record['additional_hits_per_100_races']=[300*point,300*float(lo),300*float(hi)]
            result[f'{first}-{second}/{metric}']=record
    return points,result
