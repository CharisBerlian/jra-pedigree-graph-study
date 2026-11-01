"""Validate only train/validation pedigree closure and date-projection staging."""
from pathlib import Path
import csv,gzip,hashlib,json,sqlite3,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'kaggle'))
from graph_pilot import load_parents


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    pilot=ROOT/'kaggle/upload/jra-pedigree-pilot-v1/pedigree_edges.csv'
    manifest=json.loads((pilot.parent/'manifest.json').read_text());assert sha(pilot)==manifest['files'][pilot.name]['sha256']
    extra=HERE.parent/'private_inputs/target_pedigree_2018_2023.csv.gz'
    lookup={}
    for path in (pilot,extra):
        op=gzip.open if path.suffix=='.gz' else open
        with op(path,'rt',encoding='utf-8',newline='') as f:
            for r in csv.DictReader(f):
                key=(r['child'],r['relation']);assert r['child'] and r['parent'] and r['relation'] in ('sire','dam')
                assert key not in lookup or lookup[key]==r['parent'];lookup[key]=r['parent']
    local=load_parents([ROOT/'local_jra_worker/data/raw_jbis_laptop.db',ROOT/'tmp/pedigree_patch.db'])
    source=ROOT/'kaggle/upload/jra-prediction-bundle-v1/train_validation.csv.gz';horses=set();races=set()
    with gzip.open(source,'rt',encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            assert r['split'] in ('train','validation');horses.add(r['horse_id_netkeiba']);races.add(r['race_id'])
    seen=set()
    for horse in horses:
        frontier={horse}
        for depth in range(1,4):
            next_front=set()
            for child in frontier:
                for relation in ('sire','dam'):
                    key=(child,relation);assert lookup.get(key)==local.get(key),key
                    seen.add(key)
                    if local.get(key):next_front.add(local[key])
            frontier=next_front
    con=sqlite3.connect((ROOT/'tmp/repaired_jra.db').as_uri()+'?mode=ro',uri=True)
    dates=dict(con.execute("SELECT race_id,race_date FROM races_r WHERE race_date BETWEEN '20100101' AND '20231231'"));con.close()
    assert races<=dates.keys() and len(races)==44848
    dest=HERE/'private_data'/'race_dates_train_validation.csv.gz';assert not dest.exists()
    with gzip.open(dest,'wt',encoding='utf-8',newline='',compresslevel=9) as f:
        w=csv.writer(f);w.writerow(['race_id','race_date']);w.writerows((r,dates[r]) for r in sorted(races))
    record={'status':'pass','scope':'train/validation roots only; no test rows or outcomes',
            'unique_horses':len(horses),'races':len(races),'checked_relationship_slots':len(seen),
            'pilot_sha256':sha(pilot),'target_pedigree_sha256':sha(extra),'race_dates_sha256':sha(dest),
            'source_train_validation_sha256':sha(source),'checker_sha256':sha(Path(__file__))}
    out=HERE/'private_data'/'staging_closure_audit.json';assert not out.exists()
    out.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
