"""After root review only: extract exact test outcomes for aggregate evaluation."""
from pathlib import Path
import argparse,csv,gzip,hashlib,json,math,sqlite3

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def key_digest(keys):
    digest=hashlib.sha256()
    for item in sorted(keys):digest.update((json.dumps(item,ensure_ascii=False,separators=(',',':'))+'\n').encode())
    return digest.hexdigest()

def validate_prediction_file(path,surface,expected_digest,expected_entries,expected_races):
    with gzip.open(path,'rt',encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f)
        assert not set(reader.fieldnames)&{'label','finish_pos','result_status','time','margin','last_3f','corner_passing','top_three'}
        rows=list(reader)
    assert len(rows)==expected_entries
    seen=set();groups={}
    for row in rows:
        assert row['surface'].lower()==surface
        key=(row['race_id'],row['horse_id_netkeiba'],int(row['horse_number']));assert key not in seen;seen.add(key)
        score=float(row['prediction']);flag=int(row['selected_top3']);assert math.isfinite(score) and 0<=score<=1 and flag in (0,1)
        groups.setdefault(key[0],[]).append((key,score,flag))
    assert key_digest(seen)==expected_digest and len(groups)==expected_races
    for entries in groups.values():
        assert len(entries)>=3
        ranked=sorted(entries,key=lambda x:(-x[1],x[0][2],x[0][1]))[:3]
        assert {x[0] for x in ranked}=={x[0] for x in entries if x[2]==1}

def verify_predictions_before_outcomes(manifest_path,authorization,inference_dir,expected_counts=None):
    inference_dir=Path(inference_dir);report=read_json(inference_dir/'inference_report.json')
    assert report['status']=='complete' and report['test_labels_read'] is False and report['test_metrics_computed'] is False
    assert report['model_manifest_sha256']==sha(manifest_path) and report['authorization_sha256']==sha(authorization)
    expected={(f,s,a,k) for f in ('xgboost','lightgbm','catboost') for s in ('turf','dirt') for a in ('S2','old_S6','new_S6') for k in (0,1,2)}
    inv={(x['family'],x['surface'],x['arm'],x['seed']):x for x in report['predictions']}
    assert len(report['predictions'])==len(inv)==54 and set(inv)==expected
    counts=expected_counts or {'turf':(43477,3246),'dirt':(45660,3219)}
    assert set(counts)=={'turf','dirt'} and set(report['roster_key_sha256'])==set(counts)
    for (family,surface,arm,seed),item in inv.items():
        path=f'{family}/{surface}/{arm}/seed{seed}/test_predictions.csv.gz'
        assert item['path']==path and sha(inference_dir/path)==item['sha256']
        validate_prediction_file(inference_dir/path,surface,report['roster_key_sha256'][surface],*counts[surface])
    return sha(inference_dir/'inference_report.json')

def extract(manifest_path,authorization,inference_dir,features_db,source_db,out):
    manifest=read_json(manifest_path);auth=read_json(authorization)
    assert auth=={'authorized_by_root':True,'frozen_manifest_sha256':sha(manifest_path),'scope':'exploratory test inference and evaluation'}
    assert manifest['status']=='frozen_for_root_review' and sha(Path(__file__))==manifest['source_sha256']['extract_sensitivity_outcomes.py']
    inference_sha=verify_predictions_before_outcomes(manifest_path,authorization,inference_dir)
    assert sha(features_db)=='1716e13be3a61caa06374260b93326a532e72fc913a7a6c00341b621977eb962'
    assert sha(source_db)=='2c388401d2be9af182f98d393fb1040b5d875e3fcff7fcc33a2dee2a0d303752'
    con=sqlite3.connect(Path(features_db).resolve().as_uri()+'?mode=ro',uri=True)
    con.execute('ATTACH DATABASE ? AS source',(Path(source_db).resolve().as_uri()+'?mode=ro',));con.execute('PRAGMA query_only=ON')
    rows=con.execute('''SELECT e.race_id,e.horse_id_netkeiba,e.horse_number,lower(e.surface),e.field_size,e.label,
        r.race_date,n.finish_pos FROM entries e JOIN source.races_r r USING(race_id)
        JOIN source.runners_r n ON n.race_id=e.race_id AND n.horse_id_netkeiba=e.horse_id_netkeiba AND n.horse_number=e.horse_number
        WHERE e.split='test' AND r.race_date BETWEEN '20240101' AND '20251231' ORDER BY e.race_id,e.horse_number,e.horse_id_netkeiba''').fetchall()
    con.close();assert len(rows)==89137 and len({r[:3] for r in rows})==len(rows)
    for row in rows:assert row[5]==int(row[7] in (1,2,3))
    out=Path(out);assert not out.exists()
    with gzip.open(out,'wt',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['race_id','horse_id_netkeiba','horse_number','surface','field_size','label','race_date','finish_pos']);writer.writerows(rows)
    out.with_suffix('.json').write_text(json.dumps({'status':'complete','test_labels_read':True,'test_metrics_computed':False,
        'rows':len(rows),'data_sha256':sha(out),'model_manifest_sha256':sha(manifest_path),'authorization_sha256':sha(authorization),
        'inference_report_sha256':inference_sha,
        'features_db_sha256':sha(features_db),'source_db_sha256':sha(source_db)},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--authorization',required=True);p.add_argument('--inference-dir',required=True);p.add_argument('--features-db',required=True);p.add_argument('--source-db',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    extract(a.manifest,a.authorization,a.inference_dir,a.features_db,a.source_db,a.out)
