"""Generate a private feature-preparation wrapper; never submit or run it."""
from pathlib import Path
import argparse,base64,hashlib,io,json,re,zipfile
from urllib.parse import urlsplit

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];PRIVATE=HERE/'private_data'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('parent','seed0','seed1','seed2'),required=True)
    parser.add_argument('--inputs-json',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--parent-certificate',type=Path);args=parser.parse_args()
    assert args.inputs_json.resolve().is_relative_to(PRIVATE.resolve()) and args.output.resolve().is_relative_to(PRIVATE.resolve())
    record_path=args.output.with_suffix('.json');assert not args.output.exists() and not record_path.exists()
    prefit=HERE/'PREFIT_MANIFEST_V4.json';m=json.loads(prefit.read_text())
    assert m['status']=='awaiting_root_prefit_review'
    assert sha(Path(__file__))==m['source_sha256'][Path(__file__).resolve().relative_to(ROOT).as_posix()]
    audit=json.loads((PRIVATE/'staging_closure_audit.json').read_text())
    assert sha(PRIVATE/'staging_closure_audit.json')==m['staging_closure_audit_sha256'] and audit['status']=='pass'
    stage=args.stage;common={'train_validation.csv':'d395a1802d90b0c09cc71f08b2dcbc83ca4020654797c9442969833c103ac12d',
      'pedigree_edges.csv':audit['pilot_sha256'],'target_pedigree.csv.gz':audit['target_pedigree_sha256']}
    expected=dict(common)
    parent_cert=None
    if stage!='parent':
        seed=int(stage[-1]);g=m['graph_artifacts'][seed];assert g['seed']==seed
        expected.update({'production_report.json':g['report_sha256'],'training_graph.edg':g['graph_sha256'],
                         f'nodevectors_seed{seed}.npz':g['vector_sha256'],f'cluster_centers_seed{seed}.npy':g['centroid_sha256']})
        assert args.parent_certificate and args.parent_certificate.resolve().is_relative_to(PRIVATE.resolve())
        parent_cert=json.loads(args.parent_certificate.read_text());assert parent_cert['status']=='verified'
        assert parent_cert['prefit_sha256']==sha(prefit)
        assert parent_cert['parent_features_payload_sha256']=='ab732ff06a8e6d23f99d23f9ed2dcddeb0685f2b1b868be43af626a4380e30da'
        assert parent_cert['parent_map_sha256']=='f07bbf61e92d67a70de28f5832751a0a27d5d6675d2111ceccd1cf10b2c63774'
        expected.update({'parent_features.csv.gz':parent_cert['parent_features_raw_sha256'],
                         'parent_manifest.json':parent_cert['parent_manifest_sha256']})
        if seed:
            meta=ROOT/'kaggle/upload/jra-experiment-graph-tables-v1'/f'graph_features_seed{seed}.csv.json'
            expected_meta={1:'115510f5d420f75330c2d2d7ead53f91c911864674b8163d26fc6c7e4c3b1f9a',2:'4456fa6f9cfdc60b88fb1e5001b3a9a869f86fb4094b37c88de6c938967da32e'}
            assert sha(meta)==expected_meta[seed]
            old=json.loads(meta.read_text());expected[f'graph_features_seed{seed}.csv']=old['uncompressed_csv']['sha256']
            expected[f'graph_features_seed{seed}.csv.json']=sha(meta)
    else:assert args.parent_certificate is None
    items=json.loads(args.inputs_json.read_text());assert {x['path'] for x in items}==set(expected) and len(items)==len(expected)
    for x in items:
        assert set(x)=={'path','url','sha256','origin','version'} and Path(x['path']).name==x['path']
        assert re.fullmatch('[0-9a-f]{64}',x['sha256']) and (expected[x['path']] is None or x['sha256']==expected[x['path']])
        u=urlsplit(x['url']);assert u.scheme=='https' and u.netloc==u.hostname and u.hostname in ('storage.googleapis.com','www.kaggleusercontent.com') and not u.fragment
        assert not any(ord(c)<33 or ord(c)>126 for c in x['url'])
        assert isinstance(x['origin'],str) and x['origin'] and '://' not in x['origin'] and isinstance(x['version'],int) and x['version']>0
        if parent_cert and x['path'] in ('parent_features.csv.gz','parent_manifest.json'):
            assert x['origin']==parent_cert['origins'][x['path']] and x['version']==parent_cert['version']
    paths={'remote_preparation.py':HERE/'remote_preparation.py','remote_workspace.py':HERE/'remote_workspace.py',
           'parent_control.py':HERE/'parent_control.py','prepare_parent_data.py':HERE/'prepare_parent_data.py',
           'ancestry.py':HERE/'ancestry.py','feature_adapter.py':HERE/'feature_adapter.py','assemble_features.py':HERE/'assemble_features.py',
           'assemble_cutoff.py':HERE.parent/'assemble_cutoff.py','graph_pilot.py':ROOT/'kaggle/graph_pilot.py',
           'prepare_prediction.py':ROOT/'kaggle/prepare_prediction.py','experiment_features.py':ROOT/'kaggle/experiment_features.py'}
    for name,p in paths.items():assert sha(p)==m['source_sha256'][p.relative_to(ROOT).as_posix()]
    dates=PRIVATE/'race_dates_train_validation.csv.gz';assert sha(dates)==m['date_projection_sha256']
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,p in paths.items():z.writestr(name,p.read_bytes())
        z.writestr('race_dates.csv.gz',dates.read_bytes())
    archive=data.getvalue();bundle64=base64.b64encode(archive).decode('ascii')
    code_hashes={name:sha(p) for name,p in paths.items() if name not in ('remote_preparation.py','remote_workspace.py')}
    file_hashes={x['path']:x['sha256'] for x in items};file_hashes['race_dates.csv.gz']=sha(dates)
    cfg={'stage':stage,'input_dir':'/kaggle/tmp/focused-stage-input','code_dir':'/kaggle/tmp/focused-stage-code',
         'workspace':'/kaggle/tmp/focused-stage-workspace','output_dir':'/kaggle/working/focused-preparation','files':file_hashes,
         'required_files':sorted(file_hashes),'code_sha256':code_hashes}
    if stage!='parent':
        seed=int(stage[-1]);cfg['golden_payload_sha256']={arm:next(x['payload_sha256'] for x in m['view_summaries'] if x['seed']==seed and x['view']==arm) for arm in ('Gmixed','Gancestor')}
        cfg['golden_key_sha256']=m['cohort_key_sha256'];cfg['golden_non_graph_sha256']=m['cohort_non_graph_and_labels_sha256']
    source=f'''from pathlib import Path
import base64,hashlib,io,json,subprocess,sys,urllib.request,zipfile
inp=Path('/kaggle/tmp/focused-stage-input');bundle=Path('/kaggle/tmp/focused-stage-code');assert not inp.exists() and not bundle.exists();inp.mkdir();bundle.mkdir()
items={items!r}
for item in items:
    target=inp/item['path'];h=hashlib.sha256()
    try:
        with urllib.request.urlopen(item['url'],timeout=120) as r,target.open('xb') as f:
            while True:
                block=r.read(4*1024*1024)
                if not block:break
                h.update(block);f.write(block)
    except Exception as exc:
        print('TRANSPORT_FAILED',item['path'],type(exc).__name__,flush=True);raise SystemExit(1) from None
    assert h.hexdigest()==item['sha256']
archive=base64.b64decode({bundle64!r});assert hashlib.sha256(archive).hexdigest()=={hashlib.sha256(archive).hexdigest()!r}
with zipfile.ZipFile(io.BytesIO(archive)) as z:
    assert set(z.namelist())==set({list(paths)+['race_dates.csv.gz']!r})
    for name in z.namelist():
        dest=inp/name if name=='race_dates.csv.gz' else bundle/name;dest.write_bytes(z.read(name))
assert hashlib.sha256((inp/'race_dates.csv.gz').read_bytes()).hexdigest()=={sha(dates)!r}
config={cfg!r};(bundle/'config.json').write_text(json.dumps(config,sort_keys=True),encoding='utf-8')
venv=Path('/kaggle/tmp/focused-prep-venv');python=venv/'bin/python'
subprocess.run([sys.executable,'-m','venv','--without-pip',str(venv)],check=True)
pip=[sys.executable,'-m','pip','--python',str(python)];subprocess.run(pip+['install','--disable-pip-version-check','numpy==1.26.4'],check=True);subprocess.run(pip+['check'],check=True)
subprocess.run([str(python),str(bundle/'remote_preparation.py'),'--config',str(bundle/'config.json')],check=True,cwd=bundle)
'''
    code=source.encode('utf-8');compile(source,'<private-preparation>','exec');assert len(code)<1_000_000
    record={'stage':stage,'prefit_sha256':sha(prefit),'script_sha256':hashlib.sha256(code).hexdigest(),
            'source_bytes':len(code),'archive_sha256':hashlib.sha256(archive).hexdigest(),
            'staged':[ {k:x[k] for k in ('path','sha256','origin','version')} for x in items],
            'embedded_date_sha256':sha(dates),'status':'frozen_not_submitted'}
    if parent_cert:record['parent_certificate_sha256']=sha(args.parent_certificate)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_bytes(code)
    record_path.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'stage':stage,'source_bytes':len(code),'script_sha256':record['script_sha256'],'record_sha256':sha(record_path)}))


if __name__=='__main__':main()
