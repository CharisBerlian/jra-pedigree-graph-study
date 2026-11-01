"""Local-only mocked certificates and URL roles; no submission or model fit."""
from pathlib import Path
import ast,hashlib,json,subprocess,sys,tempfile
HERE=Path(__file__).resolve().parent;PRIVATE=HERE/'private_data'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prefit=HERE/'PREFIT_MANIFEST_V4.json';m=json.loads(prefit.read_text())
    with tempfile.TemporaryDirectory(dir=PRIVATE,prefix='synthetic-transport-') as t:
        root=Path(t);cert=root/'fake_certificate.json';views=[]
        for item in m['view_summaries']:
            views.append({'seed':item['seed'],'view':item['view'],'payload_sha256':item['payload_sha256'],
                          'raw_sha256':item['raw_sha256'],'sidecar_sha256':item['sidecar_sha256'],'version':1,
                          'origins':{'data':f"synthetic/{item['view']}/{item['seed']}/data",'sidecar':f"synthetic/{item['view']}/{item['seed']}/meta"}})
        cert.write_text(json.dumps({'status':'verified','prefit_sha256':sha(prefit),'views':views,
            'cohort_key_sha256':m['cohort_key_sha256'],'cohort_non_graph_and_labels_sha256':m['cohort_non_graph_and_labels_sha256'],
            'parent_feature_sha256':m['parent_feature_sha256']}),encoding='utf-8')
        jobs=root/'jobs';subprocess.run([sys.executable,str(HERE/'build_focused_jobs.py'),'--certificate',str(cert),'--output-dir',str(jobs)],check=True,capture_output=True)
        info=json.loads((jobs/'job_sources.json').read_text());assert len(info['jobs'])==24
        for arm in ('H','Gmixed','Gancestor','P'):
            view='Gancestor' if arm=='Gancestor' else 'Gmixed';alias='new' if view=='Gancestor' else 'old';items=[]
            for seed in (0,1,2):
                row=next(x for x in views if x['view']==view and x['seed']==seed)
                for kind,suffix in (('data','.gz'),('sidecar','.json')):
                    items.append({'path':f'seed{seed}/{alias}-seed{seed}.csv{suffix}',
                      'url':'https://www.kaggleusercontent.com/synthetic','sha256':row['raw_sha256' if kind=='data' else 'sidecar_sha256'],
                      'origin':row['origins'][kind],'version':1})
            input_path=root/f'{arm}_inputs.json';input_path.write_text(json.dumps(items));out=root/f'{arm}.py'
            cmd=[sys.executable,str(HERE/'build_model_transport.py'),'--family','xgboost','--surface','turf','--arm',arm,
                 '--certificate',str(cert),'--inputs-json',str(input_path),'--out',str(out),'--jobs-dir',str(jobs)]
            result=subprocess.run(cmd,capture_output=True,text=True,check=True);record=json.loads(result.stdout)
            assert record['status']=='frozen_not_submitted' and out.stat().st_size<1_000_000
            tree=ast.parse(out.read_text());assert len(tree.body)>0
            items[0]['sha256']='0'*64;input_path.write_text(json.dumps(items))
            bad=cmd.copy();bad[bad.index(str(out))]=str(root/f'{arm}-bad.py')
            assert subprocess.run(bad,capture_output=True).returncode!=0
    print('PASS synthetic 24 model sources and all four certificate-bound transport roles; corrupt hash rejected')


if __name__=='__main__':main()
