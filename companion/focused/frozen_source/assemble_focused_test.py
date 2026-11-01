"""After root model-freeze authorization, create six label-free test views."""
from pathlib import Path
import argparse,csv,gzip,hashlib,json,math,sys
from datetime import datetime,timezone
import numpy as np
import ancestry,apply_frozen_parent,feature_adapter,parent_control
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'kaggle'))
import graph_pilot,run_test_inference as frozen

FORBIDDEN={'label','finish_pos','result_status','time','margin','last_3f','corner_passing','top_three'}
GRAPH={'embedding_status','fallback_depth','ancestor_count','embedding_missing','graph_cluster'}|{f'embedding_{i}' for i in range(64)}
KEY=('race_id','horse_id_netkeiba','horse_number')


def sha(path):return frozen._sha256(path)


def gate(manifest_path,authorization_path):
    manifest=frozen._json(manifest_path);auth=frozen._json(authorization_path)
    assert manifest['status']=='frozen_for_root_review' and manifest['model_count']==72 and manifest['test_inference_authorized'] is False
    assert set(auth)=={'authorized_by_root','frozen_manifest_sha256','scope','authorized_at_utc'}
    assert auth['authorized_by_root'] is True and auth['frozen_manifest_sha256']==sha(manifest_path)
    assert auth['scope']=='exploratory test inference and evaluation'
    authorized=datetime.fromisoformat(auth['authorized_at_utc'].replace('Z','+00:00'))
    created=datetime.fromisoformat(manifest['created_utc'].replace('Z','+00:00'))
    assert authorized.tzinfo is not None and created.tzinfo is not None and authorized>=created
    for name,path in {'assemble_focused_test.py':Path(__file__),'ancestry.py':Path(ancestry.__file__),
        'apply_frozen_parent.py':Path(apply_frozen_parent.__file__),'feature_adapter.py':Path(feature_adapter.__file__),
        'parent_control.py':Path(parent_control.__file__),
        'experiment_features.py':Path(feature_adapter.original.__file__),'prepare_prediction.py':Path(feature_adapter.original.frozen.__file__),
        'run_test_inference.py':Path(frozen.__file__),'graph_pilot.py':Path(graph_pilot.__file__)}.items():
        assert sha(path)==manifest['source_sha256'][name],name
    assert sha(Path(frozen.features.__file__))==manifest['source_sha256']['experiment_features.py']
    assert sha(Path(frozen.features.frozen.__file__))==manifest['source_sha256']['prepare_prediction.py']
    return manifest


def prepare(manifest_path,authorization_path,test_dir,graph_root,pedigree_paths,map_path,out):
    m=gate(manifest_path,authorization_path) # No test-path or test-data access before this check.
    assert len(pedigree_paths)==2
    for path,expected in zip(pedigree_paths,m['pedigree_source_sha256']):assert sha(path)==expected
    parents=graph_pilot.load_parents(pedigree_paths);maps=apply_frozen_parent.load(map_path,m['parent_map_sha256'])
    out=Path(out);assert not out.exists();out.mkdir(parents=True)
    shared=None;outputs=[]
    for seed in (0,1,2):
        choices=[p for p in (Path(test_dir)/f'test_seed{seed}.csv.gz',Path(test_dir)/f'test_seed{seed}.csv') if p.is_file()]
        assert len(choices)==1;source=choices[0]
        if source.suffix=='.gz':assert sha(source)==m['historical_test_input_sha256'][str(seed)]
        h=hashlib.sha256();op=gzip.open if source.suffix=='.gz' else open
        with op(source,'rb') as f:
            for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
        assert h.hexdigest()==m['historical_test_payload_sha256'][str(seed)]
        with op(source,'rt',encoding='utf-8',newline='') as f:header=csv.DictReader(f).fieldnames
        assert not set(header)&FORBIDDEN and not set(header)&set(feature_adapter.P_COLUMNS)
        rows,meta=frozen.read_test_rows(source,expected_seed=seed);assert len(rows)==89137 and meta['test_labels_read'] is False and meta['test_metrics_computed'] is False
        expected=m['training_graph_provenance'][seed];original=meta['graph'];assert original['seed']==seed
        assert original['graph_sha256']==expected['graph_sha256'] and original['report_sha256']==expected['report_sha256']
        assert original['artifact_sha256'][f'nodevectors_seed{seed}.npz']==expected['vector_sha256']
        assert original['artifact_sha256'][f'cluster_centers_seed{seed}.npy']==expected['centroid_sha256']
        folder=Path(graph_root)/f'seed-{seed}';vf=folder/f'nodevectors_seed{seed}.npz';cf=folder/f'cluster_centers_seed{seed}.npy'
        assert sha(vf)==expected['vector_sha256'] and sha(cf)==expected['centroid_sha256']
        with np.load(vf,allow_pickle=False) as a:
            ids=a['node_ids'].tolist();matrix=a['vectors'];assert len(ids)==len(set(ids))==93474 and matrix.shape==(93474,64) and np.isfinite(matrix).all()
            vectors={str(k):v for k,v in zip(ids,matrix.copy())}
        centers=np.load(cf,allow_pickle=False);assert centers.shape==(64,64) and np.isfinite(centers).all()
        columns=sorted(set(header)-GRAPH);content={};cache={}
        for row in rows:
            horse=row['horse_id_netkeiba']
            for role in ('sire','dam'):
                assert not row.get(role) or row[role]==parents.get((horse,role)),(horse,role)
            row.update(apply_frozen_parent.apply(horse,row['surface'].lower(),parents,maps))
            key=tuple(str(row[k]) for k in KEY);assert key not in content
            content[key]=hashlib.sha256(json.dumps([row.get(c) for c in columns]+[row[c] for c in feature_adapter.P_COLUMNS],separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
        if shared is None:shared=content
        else:assert content==shared,'non-graph/P test values differ across seeds'
        for arm in ('Gmixed','Gancestor'):
            path=out/f'{arm}-seed{seed}.csv.gz'
            with gzip.open(path,'wt',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=header+list(feature_adapter.P_COLUMNS));w.writeheader()
                for original_row in rows:
                    row=dict(original_row)
                    if arm=='Gancestor':
                        horse=row['horse_id_netkeiba']
                        if horse not in cache:cache[horse]=ancestry.encode(horse,parents,vectors,centers)
                        x=cache[horse];row.update({k:x[k] for k in ('embedding_status','fallback_depth','ancestor_count','embedding_missing','graph_cluster')})
                        row.update({f'embedding_{i}':v for i,v in enumerate(x['vector'])})
                    w.writerow(row)
            side={'seed':seed,'view':arm,'splits':['test'],'test_labels_read':False,'test_metrics_computed':False,
                  'data_sha256':sha(path),'model_manifest_sha256':sha(manifest_path),'authorization_sha256':sha(authorization_path),
                  'graph_provenance':expected,'parent_map_sha256':m['parent_map_sha256']}
            path.with_suffix('.json').write_text(json.dumps(side,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
            outputs.append({'seed':seed,'view':arm,'data_sha256':sha(path),'sidecar_sha256':sha(path.with_suffix('.json'))})
    (out/'test_view_report.json').write_text(json.dumps({'status':'complete','test_labels_read':False,'test_metrics_computed':False,
        'model_manifest_sha256':sha(manifest_path),'authorization_sha256':sha(authorization_path),'views':outputs},indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--authorization',required=True);p.add_argument('--test-dir',required=True);p.add_argument('--graph-root',required=True);p.add_argument('--pedigree',required=True,nargs=2);p.add_argument('--maps',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    prepare(a.manifest,a.authorization,a.test_dir,a.graph_root,a.pedigree,a.maps,a.out)
