"""Open outcomes only after all 72 label-free predictions pass verification."""
from pathlib import Path
import argparse,csv,gzip,json,sqlite3,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE.parent))
import extract_sensitivity_outcomes as historical
from assemble_focused_test import gate,sha


def verify_predictions(manifest_path,authorization,inference_dir,expected_counts=None):
    folder=Path(inference_dir);report=historical.read_json(folder/'inference_report.json');manifest=historical.read_json(manifest_path)
    assert report['status']=='complete' and report['test_labels_read'] is False and report['test_metrics_computed'] is False
    assert report['model_manifest_sha256']==sha(manifest_path) and report['authorization_sha256']==sha(authorization)
    expected={(f,s,a,k) for f in ('xgboost','lightgbm','catboost') for s in ('turf','dirt') for a in ('H','Gmixed','Gancestor','P') for k in (0,1,2)}
    inventory={(x['family'],x['surface'],x['arm'],x['seed']):x for x in report['predictions']}
    assert len(report['predictions'])==len(inventory)==72 and set(inventory)==expected
    models={(x['family'],x['surface'],x['arm']):x for x in manifest['models']}
    assert len(models)==len(manifest['models'])==24
    assert set(report['inputs'])=={f'{view}/seed{seed}' for view in ('Gmixed','Gancestor') for seed in (0,1,2)}
    counts=expected_counts or {'turf':(43477,3246),'dirt':(45660,3219)}
    assert set(counts)=={'turf','dirt'} and set(report['roster_key_sha256'])==set(counts)
    for (family,surface,arm,seed),item in inventory.items():
        relative=f'{family}/{surface}/{arm}/seed{seed}/test_predictions.csv.gz';assert item['path']==relative and sha(folder/relative)==item['sha256']
        refits={x['seed']:x for x in models[family,surface,arm]['refits']};assert set(refits)=={0,1,2}
        assert item['model_sha256']==refits[seed]['model_sha256']
        view='Gancestor' if arm=='Gancestor' else 'Gmixed'
        assert item['test_input_sha256']==report['inputs'][f'{view}/seed{seed}']['sha256']
        historical.validate_prediction_file(folder/relative,surface,report['roster_key_sha256'][surface],*counts[surface])
    return sha(folder/'inference_report.json')


def extract(manifest_path,authorization,inference_dir,features_db,source_db,out):
    m=gate(manifest_path,authorization);assert sha(Path(__file__))==m['source_sha256']['extract_focused_outcomes.py']
    assert sha(Path(historical.__file__))==m['source_sha256']['extract_sensitivity_outcomes.py']
    report_sha=verify_predictions(manifest_path,authorization,inference_dir) # Must finish before either outcome DB is opened or hashed.
    assert sha(features_db)=='1716e13be3a61caa06374260b93326a532e72fc913a7a6c00341b621977eb962'
    assert sha(source_db)=='2c388401d2be9af182f98d393fb1040b5d875e3fcff7fcc33a2dee2a0d303752'
    con=sqlite3.connect(Path(features_db).resolve().as_uri()+'?mode=ro',uri=True)
    con.execute('ATTACH DATABASE ? AS source',(Path(source_db).resolve().as_uri()+'?mode=ro',));con.execute('PRAGMA query_only=ON')
    rows=con.execute('''SELECT e.race_id,e.horse_id_netkeiba,e.horse_number,lower(e.surface),e.field_size,e.label,
        r.race_date,n.finish_pos FROM entries e JOIN source.races_r r USING(race_id)
        JOIN source.runners_r n ON n.race_id=e.race_id AND n.horse_id_netkeiba=e.horse_id_netkeiba AND n.horse_number=e.horse_number
        WHERE e.split='test' AND r.race_date BETWEEN '20240101' AND '20251231' ORDER BY e.race_id,e.horse_number,e.horse_id_netkeiba''').fetchall()
    con.close();assert len(rows)==89137 and len({r[:3] for r in rows})==len(rows)
    for r in rows:assert r[5]==int(r[7] in (1,2,3))
    out=Path(out);assert not out.exists()
    with gzip.open(out,'wt',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['race_id','horse_id_netkeiba','horse_number','surface','field_size','label','race_date','finish_pos']);w.writerows(rows)
    out.with_suffix('.json').write_text(json.dumps({'status':'complete','test_labels_read':True,'test_metrics_computed':False,
        'rows':len(rows),'data_sha256':sha(out),'model_manifest_sha256':sha(manifest_path),'authorization_sha256':sha(authorization),
        'inference_report_sha256':report_sha},indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--authorization',required=True);p.add_argument('--inference-dir',required=True);p.add_argument('--features-db',required=True);p.add_argument('--source-db',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    extract(a.manifest,a.authorization,a.inference_dir,a.features_db,a.source_db,a.out)
