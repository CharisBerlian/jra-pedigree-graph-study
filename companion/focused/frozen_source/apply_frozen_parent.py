"""Apply the already-frozen end-2021 parent maps without updating them."""
from pathlib import Path
import hashlib,json
from parent_control import ROLES,ids,PRIOR,STRENGTH


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def load(path,expected_sha):
    assert sha(path)==expected_sha
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    assert data['cutoff']=='2021-12-31' and data['surface_specific'] is True
    assert data['prior']==PRIOR==0.20 and data['strength']==STRENGTH==20
    total={};byhorse={}
    for role,parent,surface,n,s in data['total']:
        key=(role,parent,surface);assert key not in total and role in ROLES and surface in ('turf','dirt') and 0<=s<=n
        total[key]=(n,s)
    for role,parent,surface,horse,n,s in data['byhorse']:
        key=(role,parent,surface,horse);assert key not in byhorse and role in ROLES and surface in ('turf','dirt') and 0<=s<=n
        byhorse[key]=(n,s)
    return total,byhorse


def apply(horse,surface,parents,maps):
    assert horse and surface in ('turf','dirt');total,byhorse=maps;out={}
    for role,parent in zip(ROLES,ids(horse,parents)):
        n,s=total.get((role,parent,surface),(0,0)) if parent else (0,0)
        hn,hs=byhorse.get((role,parent,surface,horse),(0,0)) if parent else (0,0)
        n-=hn;s-=hs;assert 0<=s<=n
        out[role+'_available']=int(bool(parent));out[role+'_prior_count']=n
        out[role+'_prior_rate']=(s+STRENGTH*PRIOR)/(n+STRENGTH)
    return out
