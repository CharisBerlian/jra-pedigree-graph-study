"""Path-only staging helpers for approved train/validation preparation.

No launch, network, model fitting, test inputs, or outcome DBs are used here.
"""
from pathlib import Path
import csv,gzip,hashlib,sqlite3


def digest(path,uncompressed=False):
    h=hashlib.sha256();op=gzip.open if uncompressed else open
    with op(path,'rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def surrogate(pilot,target,dates,out):
    """Write minimal local SQLite lookup tables from already verified inputs."""
    out=Path(out);raw=out/'local_jra_worker/data/raw_jbis_laptop.db';patch=out/'tmp/pedigree_patch.db';race=out/'tmp/repaired_jra.db'
    for p in (raw,patch,race):assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True)
    relations={}
    for path in (Path(pilot),Path(target)):
        op=gzip.open if path.suffix=='.gz' else open
        with op(path,'rt',encoding='utf-8',newline='') as f:
            for row in csv.DictReader(f):
                k=(row['child'],row['relation']);v=row['parent']
                assert k[0] and v and k[1] in ('sire','dam') and (k not in relations or relations[k]==v)
                relations[k]=v
    for p in (raw,patch):
        db=sqlite3.connect(p);db.execute('CREATE TABLE pedigree_edges (child_jbis_id TEXT,parent_jbis_id TEXT,relation TEXT,generation INTEGER)')
        if p==raw:db.executemany('INSERT INTO pedigree_edges VALUES (?,?,?,?)',((c,v,r,0) for (c,r),v in sorted(relations.items())))
        db.commit();db.close()
    db=sqlite3.connect(race);db.execute('CREATE TABLE races_r (race_id TEXT PRIMARY KEY,race_date TEXT)')
    seen=set()
    with gzip.open(dates,'rt',encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            rid,date=row['race_id'],row['race_date'];assert rid not in seen and '20100101'<=date<='20231231'
            seen.add(rid);db.execute('INSERT INTO races_r VALUES (?,?)',(rid,date))
    db.commit();db.close()
    return {'relationship_keys':len(relations),'races':len(seen)}


def deterministic_gzip(source,destination):
    """Same staged CSV gzip bytes across jobs in the same pinned runtime."""
    assert not Path(destination).exists()
    with open(source,'rb') as inp,open(destination,'xb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=1,mtime=0) as out:
            for b in iter(lambda:inp.read(4*1024*1024),b''):out.write(b)
