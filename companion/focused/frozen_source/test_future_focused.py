"""Synthetic 72-prediction outcome-access gate; no real inputs or outcomes."""
from pathlib import Path
import csv,gzip,hashlib,json,tempfile
from unittest.mock import patch
import extract_focused_outcomes as extractor
from extract_focused_outcomes import verify_predictions
from infer_focused import digest_keys


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);manifest=root/'manifest.json';auth=root/'authorization.json';manifest.write_text('{}');auth.write_text('{}')
        fake={'source_sha256':{'extract_focused_outcomes.py':sha(Path(extractor.__file__)),
             'extract_sensitivity_outcomes.py':sha(Path(extractor.historical.__file__))}}
        with patch.object(extractor,'gate',return_value=fake):
            try:extractor.extract(manifest,auth,root,root/'DO_NOT_OPEN_FEATURES',root/'DO_NOT_OPEN_SOURCE',root/'out.gz')
            except FileNotFoundError as exc:assert 'inference_report.json' in str(exc)
            else:raise AssertionError('missing report reached outcome database access')
        models=[{'family':family,'surface':surface,'arm':arm,
                 'refits':[{'seed':seed,'model_sha256':'a'*64} for seed in (0,1,2)]}
                for family in ('xgboost','lightgbm','catboost') for surface in ('turf','dirt') for arm in ('H','Gmixed','Gancestor','P')]
        manifest.write_text(json.dumps({'models':models}));inv=[];roster={};counts={}
        for surface in ('turf','dirt'):
            keys=[(f'{surface}{race}',f'h{race}{horse}',horse) for race in (1,2) for horse in (1,2,3,4)]
            roster[surface]=digest_keys(keys);counts[surface]=(8,2)
            for family in ('xgboost','lightgbm','catboost'):
              for arm in ('H','Gmixed','Gancestor','P'):
               for seed in (0,1,2):
                rel=f'{family}/{surface}/{arm}/seed{seed}/test_predictions.csv.gz';p=root/rel;p.parent.mkdir(parents=True,exist_ok=True)
                with gzip.open(p,'wt',encoding='utf-8',newline='') as f:
                    w=csv.writer(f);w.writerow(['race_id','horse_id_netkeiba','horse_number','surface','prediction','selected_top3'])
                    for race,horse,num in keys:w.writerow([race,horse,num,surface,1-num/10,int(num<=3)])
                inv.append({'family':family,'surface':surface,'arm':arm,'seed':seed,'path':rel,'sha256':sha(p),
                            'model_sha256':'a'*64,'test_input_sha256':'b'*64})
        report={'status':'complete','test_labels_read':False,'test_metrics_computed':False,'model_manifest_sha256':sha(manifest),
                'authorization_sha256':sha(auth),'roster_key_sha256':roster,'predictions':inv,
                'inputs':{f'{view}/seed{seed}':{'sha256':'b'*64} for view in ('Gmixed','Gancestor') for seed in (0,1,2)}}
        rp=root/'inference_report.json';rp.write_text(json.dumps(report));assert verify_predictions(manifest,auth,root,counts)==sha(rp)
        for field in ('model_sha256','test_input_sha256'):
            original=inv[0][field];inv[0][field]='f'*64;rp.write_text(json.dumps(report))
            try:verify_predictions(manifest,auth,root,counts)
            except AssertionError:pass
            else:raise AssertionError('changed model/input binding passed')
            inv[0][field]=original
        report['predictions']=inv[:-1];rp.write_text(json.dumps(report))
        try:verify_predictions(manifest,auth,root,counts)
        except AssertionError:pass
        else:raise AssertionError('incomplete 72-model inventory passed')
        report['predictions']=inv;rp.write_text(json.dumps(report));(root/inv[0]['path']).write_bytes(b'corrupt')
        try:verify_predictions(manifest,auth,root,counts)
        except AssertionError:pass
        else:raise AssertionError('corrupt prediction passed')
        # Even a self-consistent changed file/hash cannot redefine the race ranking.
        first=inv[0];path=root/first['path'];surface=first['surface']
        keys=[(f'{surface}{race}',f'h{race}{horse}',horse) for race in (1,2) for horse in (1,2,3,4)]
        with gzip.open(path,'wt',encoding='utf-8',newline='') as f:
            w=csv.writer(f);w.writerow(['race_id','horse_id_netkeiba','horse_number','surface','prediction','selected_top3'])
            for race,horse,num in keys:w.writerow([race,horse,num,surface,1-num/10,int(num in (1,2,4))])
        first['sha256']=sha(path);rp.write_text(json.dumps(report))
        try:verify_predictions(manifest,auth,root,counts)
        except AssertionError:pass
        else:raise AssertionError('incorrect full-race ranking passed')
    print('PASS synthetic complete 72-prediction gate and incomplete/corruption rejection')


if __name__=='__main__':main()
