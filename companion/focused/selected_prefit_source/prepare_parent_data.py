"""Build validation-only parent summaries; no test rows or outcomes are read."""
from pathlib import Path
from collections import Counter
import csv
import gzip
import hashlib
import json
import sqlite3
import sys
from parent_control import compute,ROLES

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'kaggle'))
from graph_pilot import load_parents


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    target=HERE/'private_data';target.mkdir(exist_ok=True)
    output=target/'parent_features.csv.gz';maps=target/'parent_maps.json';manifest=target/'parent_manifest.json'
    assert not any(p.exists() for p in (output,maps,manifest)), 'refusing overwrite'
    sources=[ROOT/'local_jra_worker/data/raw_jbis_laptop.db',ROOT/'tmp/pedigree_patch.db']
    parents=load_parents(sources)
    db=sqlite3.connect((ROOT/'tmp/repaired_jra.db').as_uri()+'?mode=ro',uri=True)
    dates=dict(db.execute("SELECT race_id,race_date FROM races_r WHERE race_date BETWEEN '20100101' AND '20231231'"));db.close()
    date_hash=hashlib.sha256()
    for race_id,date in sorted(dates.items()):
        date_hash.update((json.dumps([race_id,date],separators=(',',':'))+'\n').encode('utf-8'))
    source=ROOT/'kaggle/upload/jra-prediction-bundle-v1/train_validation.csv.gz'
    rows=[];seen=set();races={};counts=Counter()
    with gzip.open(source,'rt',encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            split=r['split'];assert split in ('train','validation')
            date=dates[r['race_id']];horse=r['horse_id_netkeiba'];key=(r['race_id'],horse,r['horse_number'])
            assert key not in seen;seen.add(key)
            for role in ('sire','dam'):
                assert not r[role] or r[role]==parents.get((horse,role)), (key,role)
            surface=r['surface'].lower();assert surface in ('turf','dirt')
            label=int(r['label']) if split=='train' else None
            rows.append({'key':key,'horse':horse,'date':date,'split':split,'surface':surface,'label':label})
            group=races.setdefault(r['race_id'],[int(r['field_size']),0,split,surface])
            assert group[0]==int(r['field_size']) and group[2:]==[split,surface];group[1]+=1
            counts[(split,surface)]+=1
    assert counts=={('train','turf'):271549,('train','dirt'):277207,('validation','turf'):42783,('validation','dirt'):45323}
    assert all(n==expected and n>=3 for expected,n,_,_ in races.values())
    values,state=compute(rows,parents)
    columns=[f'{role}_{name}' for role in ROLES for name in ('available','prior_count','prior_rate')]
    with gzip.open(output,'wt',encoding='utf-8',newline='',compresslevel=1) as f:
        w=csv.DictWriter(f,fieldnames=['race_id','horse_id_netkeiba','horse_number']+columns);w.writeheader()
        for r in rows:
            k=r['key'];w.writerow(dict(zip(('race_id','horse_id_netkeiba','horse_number'),k))|values[k])
    serial={'cutoff':'2021-12-31','surface_specific':True,'prior':0.20,'strength':20,
            'total':[[*key,*value] for key,value in sorted(state['total'].items())],
            'byhorse':[[*key[0],key[1],*value] for key,value in sorted(state['byhorse'].items())]}
    maps.write_text(json.dumps(serial,separators=(',',':'))+'\n',encoding='utf-8',newline='\n')
    m={'status':'train_validation_only','source_sha256':sha(source),'source_databases_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},
       'race_date_projection_sha256':date_hash.hexdigest(),'race_date_projection_scope':'2010-2023, race_id and date only',
       'rows':len(rows),'race_count':len(races),
       'counts':{'/'.join(k):v for k,v in sorted(counts.items())},'output_sha256':sha(output),'map_sha256':sha(maps),
       'source_code_sha256':sha(Path(__file__)),'parent_code_sha256':sha(HERE/'parent_control.py')}
    manifest.write_text(json.dumps(m,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'status':'PASS','rows':len(rows),'race_count':len(races),'manifest_sha256':sha(manifest)}))


if __name__=='__main__':main()
