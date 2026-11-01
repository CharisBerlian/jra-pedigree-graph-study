"""Build one private, certificate-bound model transport wrapper; no submission."""
from pathlib import Path
import argparse,ast,hashlib,json,re
from urllib.parse import urlsplit
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];PRIVATE=HERE/'private_data'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--family',required=True);p.add_argument('--surface',required=True);p.add_argument('--arm',required=True)
    p.add_argument('--certificate',required=True,type=Path);p.add_argument('--inputs-json',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--jobs-dir',type=Path,default=PRIVATE/'focused_model_jobs_v4');a=p.parse_args()
    assert a.family in ('xgboost','lightgbm','catboost') and a.surface in ('turf','dirt') and a.arm in ('H','Gmixed','Gancestor','P')
    for path in (a.certificate,a.inputs_json,a.out,a.jobs_dir):assert path.resolve().is_relative_to(PRIVATE.resolve())
    record_path=a.out.with_suffix('.json');assert not a.out.exists() and not record_path.exists()
    prefit=HERE/'PREFIT_MANIFEST_V4.json';m=json.loads(prefit.read_text());assert m['status']=='awaiting_root_prefit_review'
    assert m['source_sha256'][Path(__file__).resolve().relative_to(ROOT).as_posix()]==sha(Path(__file__))
    cert=json.loads(a.certificate.read_text());assert cert['status']=='verified' and cert['prefit_sha256']==sha(prefit)
    assert cert['cohort_key_sha256']==m['cohort_key_sha256'] and cert['cohort_non_graph_and_labels_sha256']==m['cohort_non_graph_and_labels_sha256']
    assert cert['parent_feature_sha256']==m['parent_feature_sha256']
    jobs=a.jobs_dir;jobinfo=json.loads((jobs/'job_sources.json').read_text())
    assert jobinfo['prefit_sha256']==sha(prefit) and jobinfo['remote_views_certificate_sha256']==sha(a.certificate)
    found=[x for x in jobinfo['jobs'] if (x['family'],x['surface'],x['arm'])==(a.family,a.surface,a.arm)];assert len(found)==1
    job=jobs/found[0]['path'];assert sha(job)==found[0]['sha256'];inner=job.read_bytes().decode('utf-8')
    view='Gancestor' if a.arm=='Gancestor' else 'Gmixed';alias='new' if view=='Gancestor' else 'old'
    expected={}
    for seed in (0,1,2):
        c=[x for x in cert['views'] if x['seed']==seed and x['view']==view];assert len(c)==1;c=c[0]
        local=next(x for x in m['view_summaries'] if x['seed']==seed and x['view']==view)
        assert c['payload_sha256']==local['payload_sha256']
        expected[f'seed{seed}/{alias}-seed{seed}.csv.gz']=(c['raw_sha256'],c['origins']['data'],c['version'])
        expected[f'seed{seed}/{alias}-seed{seed}.csv.json']=(c['sidecar_sha256'],c['origins']['sidecar'],c['version'])
    items=json.loads(a.inputs_json.read_text());assert len(items)==6 and {x['path'] for x in items}==set(expected)
    for x in items:
        assert set(x)=={'path','url','sha256','origin','version'} and (x['sha256'],x['origin'],x['version'])==expected[x['path']]
        assert re.fullmatch('[0-9a-f]{64}',x['sha256']);u=urlsplit(x['url'])
        assert u.scheme=='https' and u.netloc==u.hostname and u.hostname in ('storage.googleapis.com','www.kaggleusercontent.com') and not u.fragment
        assert not any(ord(c)<33 or ord(c)>126 for c in x['url'])
    original=ast.parse(inner)
    paths=[n for n in ast.walk(original) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='Path' and len(n.args)==1 and not n.keywords and isinstance(n.args[0],ast.Constant) and n.args[0].value=='/kaggle/input']
    assert len(paths)==2
    script=f'''from pathlib import Path
import ast,copy,hashlib,urllib.request
root=Path('/kaggle/tmp/focused-model-input');assert not root.exists();root.mkdir(parents=True)
items={items!r}
for item in items:
    dest=root.joinpath(*item['path'].split('/'));dest.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256()
    try:
        with urllib.request.urlopen(item['url'],timeout=120) as r,dest.open('xb') as f:
            while True:
                block=r.read(4*1024*1024)
                if not block:break
                h.update(block);f.write(block)
    except Exception as exc:
        print('TRANSPORT_FAILED',item['path'],type(exc).__name__,flush=True);raise SystemExit(1) from None
    assert h.hexdigest()==item['sha256']
inner={inner!r};assert hashlib.sha256(inner.encode('utf-8')).hexdigest()=={sha(job)!r}
class Map(ast.NodeTransformer):
    def __init__(self,src,dst):self.src=src;self.dst=dst;self.n=0
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id=='Path' and len(node.args)==1 and not node.keywords and isinstance(node.args[0],ast.Constant) and node.args[0].value==self.src:
            node.args[0].value=self.dst;self.n+=1
        return node
original=ast.parse(inner);mapped=ast.parse(inner);forward=Map('/kaggle/input','/kaggle/tmp/focused-model-input');mapped=forward.visit(mapped);assert forward.n==2
check=copy.deepcopy(mapped);reverse=Map('/kaggle/tmp/focused-model-input','/kaggle/input');check=reverse.visit(check)
assert reverse.n==2 and ast.dump(check)==ast.dump(original)
exec(compile(mapped,'<focused-model-path-adapter>','exec'))
'''
    code=script.encode();compile(script,'<focused-model-transport>','exec');assert len(code)<1_000_000
    record={'status':'frozen_not_submitted','identity':{'family':a.family,'surface':a.surface,'arm':a.arm},
            'prefit_sha256':sha(prefit),'certificate_sha256':sha(a.certificate),'inner_sha256':sha(job),
            'script_sha256':hashlib.sha256(code).hexdigest(),'bytes':len(code),
            'staged':[{k:x[k] for k in ('path','sha256','origin','version')} for x in items]}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_bytes(code)
    record_path.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'status':'frozen_not_submitted','script_sha256':record['script_sha256'],'record_sha256':sha(record_path),'bytes':len(code)}))


if __name__=='__main__':main()
