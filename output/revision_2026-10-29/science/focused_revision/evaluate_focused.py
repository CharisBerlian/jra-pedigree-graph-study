"""Exploratory 72-model evaluation; never select or fit models."""
from pathlib import Path
import argparse,csv,gzip,json,statistics
import metrics,extract_focused_outcomes as extractor
from assemble_focused_test import gate,sha
from infer_focused import key

FAMILIES=('xgboost','lightgbm','catboost');SURFACES=('turf','dirt');ARMS=('H','Gmixed','Gancestor','P');SEEDS=(0,1,2)


def write(path,rows):
    assert rows
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def evaluate(manifest_path,authorization,inference_dir,outcomes,out):
    m=gate(manifest_path,authorization)
    for name,path in {'evaluate_focused.py':Path(__file__),'metrics.py':Path(metrics.__file__),
                      'extract_focused_outcomes.py':Path(extractor.__file__),
                      'evaluate_locked_test_results.py':Path(metrics.historical.__file__),
                      'evaluate_locked.py':Path(metrics.historical.evaluation.__file__)}.items():assert sha(path)==m['source_sha256'][name]
    assert m['comparisons']=={'replicates':2000,'seed':20261029,'unit':'whole_race',
        'primary_contrasts':['Gancestor-H','Gancestor-P'],'secondary_contrasts':['Gancestor-Gmixed']}
    report_sha=extractor.verify_predictions(manifest_path,authorization,inference_dir)
    side=json.loads(Path(outcomes).with_suffix('.json').read_text());assert side['status']=='complete' and side['test_labels_read'] is True
    assert side['data_sha256']==sha(outcomes) and side['model_manifest_sha256']==sha(manifest_path) and side['authorization_sha256']==sha(authorization)
    assert side['inference_report_sha256']==report_sha
    with gzip.open(outcomes,'rt',encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==89137 and len({key(r) for r in rows})==len(rows)
    output=Path(out);assert not output.exists();output.mkdir(parents=True);seed_rows=[];arm_rows=[];contrast_rows=[]
    for surface in SURFACES:
      cohort=sorted((r for r in rows if r['surface'].lower()==surface),key=key);assert len(cohort)=={'turf':43477,'dirt':45660}[surface]
      for family in FAMILIES:
        models={}
        for arm in ARMS:
          models[arm]=[]
          for seed in SEEDS:
            path=Path(inference_dir)/family/surface/arm/f'seed{seed}'/'test_predictions.csv.gz'
            with gzip.open(path,'rt',encoding='utf-8',newline='') as f:prediction=list(csv.DictReader(f))
            lookup={key(r):float(r['prediction']) for r in prediction};assert len(lookup)==len(prediction)==len(cohort)
            assert set(lookup)=={key(r) for r in cohort};models[arm].append([lookup[key(r)] for r in cohort])
        points,comparisons=metrics.paired(cohort,models,2000,20261029)
        for arm in ARMS:
            for seed in SEEDS:seed_rows.append({'surface':surface,'family':family,'arm':arm,'seed':seed,'average_precision':points[arm]['ap'][seed],'hit_at_3':points[arm]['hit'][seed]})
            arm_rows.append({'surface':surface,'family':family,'arm':arm,'ap_mean':statistics.mean(points[arm]['ap']),
                'ap_seed_sd':statistics.stdev(points[arm]['ap']),'hit_mean':statistics.mean(points[arm]['hit']),'hit_seed_sd':statistics.stdev(points[arm]['hit'])})
        for label,value in comparisons.items():
            contrast,metric=label.split('/');r={'surface':surface,'family':family,'contrast':contrast,'metric':metric,
                'difference':value['point'],'ci_low':value['low'],'ci_high':value['high'],
                'correct_per_100_races':'','correct_per_100_low':'','correct_per_100_high':''}
            if metric=='hit':r['correct_per_100_races'],r['correct_per_100_low'],r['correct_per_100_high']=value['additional_hits_per_100_races']
            contrast_rows.append(r)
    assert (len(seed_rows),len(arm_rows),len(contrast_rows))==(72,24,36)
    write(output/'seed_metrics.csv',seed_rows);write(output/'arm_metrics.csv',arm_rows);write(output/'contrasts.csv',contrast_rows)
    inference=json.loads((Path(inference_dir)/'inference_report.json').read_text())
    result={'status':'complete','exploratory':True,'model_manifest_sha256':sha(manifest_path),'authorization_sha256':sha(authorization),
            'inference_report_sha256':report_sha,'outcomes_sha256':sha(outcomes),'coverage_and_parent_zero_counts':inference['inputs'],
            'outputs':{name:sha(output/name) for name in ('seed_metrics.csv','arm_metrics.csv','contrasts.csv')}}
    (output/'evaluation_report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--authorization',required=True);p.add_argument('--inference-dir',required=True);p.add_argument('--outcomes',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    evaluate(a.manifest,a.authorization,a.inference_dir,a.outcomes,a.out)
