"""Earlier-date, same-surface, leave-focal-horse-out parent summaries."""
from collections import defaultdict

ROLES=('sire','dam','damsire')
PRIOR=0.20
STRENGTH=20


def ids(horse,parents):
    sire=parents.get((horse,'sire'));dam=parents.get((horse,'dam'))
    return (sire,dam,parents.get((dam,'sire')) if dam else None)


def compute(rows,parents):
    """Rows are keyed dictionaries from train/validation only; never accept test."""
    assert all(r['split'] in ('train','validation') for r in rows)
    assert len({r['key'] for r in rows})==len(rows)
    for r in rows:
        assert r['horse'] and r['surface'] in ('turf','dirt')
        assert len(r['date'])==8 and r['date'].isdigit()
        assert ('20100101'<=r['date']<='20211231') if r['split']=='train' else ('20220101'<=r['date']<='20231231')
        if r['split']=='train':assert r['label'] in (0,1)
    total=defaultdict(lambda:[0,0]);byhorse=defaultdict(lambda:[0,0]);out={}
    def emit(r):
        result={}
        for role,parent in zip(ROLES,ids(r['horse'],parents)):
            key=(role,parent,r['surface']);n,s=total.get(key,(0,0)) if parent else (0,0)
            hn,hs=byhorse.get((key,r['horse']),(0,0)) if parent else (0,0)
            n-=hn;s-=hs;assert 0<=s<=n
            result[role+'_available']=int(bool(parent))
            result[role+'_prior_count']=n
            result[role+'_prior_rate']=(s+STRENGTH*PRIOR)/(n+STRENGTH)
        return result
    ordered=sorted((r for r in rows if r['split']=='train'),key=lambda r:(r['date'],r['key']))
    i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and ordered[j]['date']==ordered[i]['date']:j+=1
        for r in ordered[i:j]:out[r['key']]=emit(r)
        for r in ordered[i:j]:
            for role,parent in zip(ROLES,ids(r['horse'],parents)):
                if parent:
                    key=(role,parent,r['surface']);total[key][0]+=1;total[key][1]+=r['label']
                    pair=byhorse[(key,r['horse'])];pair[0]+=1;pair[1]+=r['label']
        i=j
    for r in rows:
        if r['split']=='validation':out[r['key']]=emit(r)
    assert len(out)==len(rows)
    frozen={'total':{k:tuple(v) for k,v in total.items()},'byhorse':{k:tuple(v) for k,v in byhorse.items()}}
    return out,frozen
