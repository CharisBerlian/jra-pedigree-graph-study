"""Generate 24 private validation-only wrappers; never submit or fit."""
from pathlib import Path
import argparse,hashlib,json

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];PRIVATE=HERE/'private_data'
ARMS=('H','Gmixed','Gancestor','P');FAMILIES=('xgboost','lightgbm','catboost');SURFACES=('turf','dirt')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(certificate,target):
    manifest=HERE/'PREFIT_MANIFEST_V4.json';m=json.loads(manifest.read_text());assert m['status']=='awaiting_root_prefit_review'
    assert certificate.resolve().is_relative_to(PRIVATE.resolve())
    cert=json.loads(certificate.read_text());assert cert['status']=='verified' and cert['prefit_sha256']==sha(manifest)
    assert cert['cohort_key_sha256']==m['cohort_key_sha256'] and cert['cohort_non_graph_and_labels_sha256']==m['cohort_non_graph_and_labels_sha256']
    assert cert['parent_feature_sha256']==m['parent_feature_sha256']
    assert len(cert['views'])==6 and {(x['view'],x['seed']) for x in cert['views']}=={(arm,s) for arm in ('Gmixed','Gancestor') for s in (0,1,2)}
    for row in cert['views']:
        local=next(x for x in m['view_summaries'] if x['view']==row['view'] and x['seed']==row['seed'])
        assert row['payload_sha256']==local['payload_sha256']
    assert m['source_sha256'][Path(__file__).resolve().relative_to(ROOT).as_posix()]==sha(Path(__file__))
    spec=(HERE/'EXPERIMENT_SPEC.json').read_text(encoding='utf-8');assert sha(HERE/'EXPERIMENT_SPEC.json')==m['spec_sha256']
    sources={name:(ROOT/'kaggle'/path).read_text(encoding='utf-8') for name,path in
             (('run_experiments.py','run_experiments.py'),('experiment_features_original.py','experiment_features.py'),
              ('prepare_prediction.py','prepare_prediction.py'),('graph_pilot.py','graph_pilot.py'))}
    sources['experiment_features.py']=(HERE/'feature_adapter.py').read_text(encoding='utf-8')
    source_hashes={k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()}
    origins={'run_experiments.py':ROOT/'kaggle/run_experiments.py','experiment_features_original.py':ROOT/'kaggle/experiment_features.py',
             'prepare_prediction.py':ROOT/'kaggle/prepare_prediction.py','graph_pilot.py':ROOT/'kaggle/graph_pilot.py',
             'experiment_features.py':HERE/'feature_adapter.py'}
    for name,p in origins.items():
        key=p.relative_to(ROOT).as_posix();assert sha(p)==m['source_sha256'][key]
        assert source_hashes[name]==m['normalized_utf8_text_sha256'][key]
    maps=json.loads((PRIVATE/'category_maps.json').read_text());assert sha(PRIVATE/'category_maps.json')==m['category_maps_sha256']
    assert target.resolve().is_relative_to(PRIVATE.resolve()) and not target.exists();target.mkdir(parents=True)
    records=[]
    for arm in ARMS:
      view='Gancestor' if arm=='Gancestor' else 'Gmixed';alias='new' if arm=='Gancestor' else 'old'
      for family in FAMILIES:
       for surface in SURFACES:
        expected_maps={seed:maps[f'{arm}/{surface}/seed{seed}'] for seed in (0,1,2)}
        inputs={s:next(x for x in cert['views'] if x['seed']==s and x['view']==view) for s in (0,1,2)}
        code=f'''from pathlib import Path
import hashlib,json,os,subprocess,sys
assert Path('/kaggle/input').is_dir()
os.environ.update(MPLBACKEND='Agg',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0')
w=Path('/kaggle/working/focused-{family}-{surface}-{arm.lower()}');w.mkdir(parents=True,exist_ok=True)
sources={sources!r}; hashes={source_hashes!r}
for name,source in sources.items():
    assert hashlib.sha256(source.encode()).hexdigest()==hashes[name]
    (w/name).write_text(source,encoding='utf-8')
spec={spec!r};(w/'spec.json').write_text(spec,encoding='utf-8')
assert hashlib.sha256((w/'spec.json').read_bytes()).hexdigest()=={m['spec_sha256']!r}
expected={inputs!r};input_paths={{}}
for seed in (0,1,2):
    found=list(Path('/kaggle/input').rglob(f'{alias}-seed{{seed}}.csv.gz'));assert len(found)==1
    path=found[0];meta_path=path.with_suffix('.json')
    assert hashlib.sha256(meta_path.read_bytes()).hexdigest()==expected[seed]['sidecar_sha256']
    meta=json.loads(meta_path.read_text());assert meta['seed']==seed and meta['arm']=={view!r}
    assert meta['data_sha256']==expected[seed]['payload_sha256']
    input_paths[seed]=path
venv=Path('/kaggle/tmp/focused-model-venv');python=venv/'bin/python'
subprocess.run([sys.executable,'-m','venv','--without-pip',str(venv)],check=True)
pip=[sys.executable,'-m','pip','--python',str(python)]
subprocess.run(pip+['install','--disable-pip-version-check','numpy==1.26.4','scipy==1.13.1','scikit-learn==1.6.1','pandas==2.2.3','lightgbm==4.6.0','xgboost==3.0.2','catboost==1.2.8'],check=True)
subprocess.run(pip+['check'],check=True)
assert sys.version.split()[0]=='3.13.15'
out=w/'runs';common=[str(python),str(w/'run_experiments.py'),'--mode','scientific','--family',{family!r},'--surface',{surface!r},'--setting',{arm!r},'--spec',str(w/'spec.json'),'--output',str(out)]
for stage,seed in [('tune',0),('refit',0),('refit',1),('refit',2)]:
    subprocess.run(common+['--stage',stage,'--seed',str(seed),'--input',str(input_paths[seed])],check=True,cwd=w)
run=out/'scientific'/{family!r}/{surface!r}/{arm!r};expected_maps={expected_maps!r}
for stage,seed in [('tune',0),('refit',0),('refit',1),('refit',2)]:
    d=run/stage/f'seed{{seed}}';assert json.loads((d/'complete.json').read_text())['status']=='complete'
    assert json.loads((d/'preprocessing.json').read_text())['category_maps']==expected_maps[seed]
(w/'JOB_PASS.json').write_text(json.dumps({{'status':'pass','arm':{arm!r},'family':{family!r},'surface':{surface!r},'prefit_sha256':{sha(manifest)!r},'remote_views_certificate_sha256':{sha(certificate)!r},'spec_sha256':{m['spec_sha256']!r}}}),encoding='utf-8')
'''
        compile(code,'<focused-model-job>','exec')
        p=target/f'{family}_{surface}_{arm}.py';p.write_text(code,encoding='utf-8',newline='\n')
        records.append({'arm':arm,'family':family,'surface':surface,'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size})
    assert len(records)==24 and all(x['bytes']<1_000_000 for x in records)
    (target/'job_sources.json').write_text(json.dumps({'prefit_sha256':sha(manifest),'remote_views_certificate_sha256':sha(certificate),'jobs':records},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'prepared_not_launched','jobs':len(records),'maximum_bytes':max(x['bytes'] for x in records)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--certificate',required=True,type=Path)
    p.add_argument('--output-dir',type=Path,default=PRIVATE/'focused_model_jobs_v4');a=p.parse_args();main(a.certificate,a.output_dir)
