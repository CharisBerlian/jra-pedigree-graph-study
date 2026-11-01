"""Assemble matched train/validation features; no training and no test rows."""
from pathlib import Path
from collections import Counter
import argparse,csv,gzip,hashlib,json,math,sys
import numpy as np
from prepare_prediction import resolve_embedding

GRAPH=['embedding_status','fallback_depth','ancestor_count','embedding_missing','graph_cluster']+[f'embedding_{i}' for i in range(64)]
KEY=['race_id','horse_id_netkeiba','horse_number','split']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def opener(path): return gzip.open if str(path).endswith('.gz') else open

def verify_payload(path,expected):
    h=hashlib.sha256()
    with opener(path)(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    assert h.hexdigest()==expected, 'base payload hash mismatch'
    return h.hexdigest()

def load_new_graph(folder, seed, manifest_hash, runner_hash):
    folder=Path(folder); report=json.loads((folder/'production_report.json').read_text()); s=report['settings']
    assert report['status']=='complete' and report['target_outcomes_read'] is False and report['validation_test_roots_input'] is False
    assert report['input_manifest_sha256']==manifest_hash and report['script_sha256']==runner_hash
    assert report['environment']['python_version']=='3.13.15'
    assert (report['training_horses'],report['graph_nodes'],report['graph_edges'],report['selected_k'])==(43152,71438,130550,64)
    assert (s['seed'],s['dimensions'],s['num_walks'],s['walk_length'],s['epochs'],s['window_size'],s['selected_clusters'],s['kmeans_n_init'],s['kmeans_max_iter'])==(seed,64,10,80,5,5,64,10,300)
    assert (s['p'],s['q'],s['workers'])==(1.0,0.5,4)
    for name in (f'nodevectors_seed{seed}.npz',f'cluster_centers_seed{seed}.npy','training_graph.edg'):
        assert sha(folder/name)==report['artifact_sha256'][name],name
    assert report['artifact_sha256']['training_graph.edg']==report['graph_sha256']
    with np.load(folder/f'nodevectors_seed{seed}.npz',allow_pickle=False) as a:
        ids=a['node_ids'].tolist();matrix=a['vectors']
    assert len(ids)==len(set(ids))==71438 and matrix.shape==(71438,64) and np.isfinite(matrix).all()
    centers=np.load(folder/f'cluster_centers_seed{seed}.npy',allow_pickle=False)
    assert centers.shape==(64,64) and np.isfinite(centers).all()
    return {str(k):tuple(map(float,v)) for k,v in zip(ids,matrix)},centers.tolist()

def load_parents(path, expected_hash):
    assert sha(path)==expected_hash, 'target pedigree hash mismatch'
    result={}
    with opener(path)(path,'rt',encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            key=(r['child'],r['relation']); parent=r['parent']
            assert key[1] in ('sire','dam') and key[0] and parent
            assert key not in result or result[key]==parent
            result[key]=parent
    return result

def load_old(path, meta_path, seed, expected_meta_hash):
    assert sha(meta_path)==expected_meta_hash, 'original graph metadata hash mismatch'
    meta=json.loads(Path(meta_path).read_text(encoding='utf-8')); graph=meta['graph']
    assert meta['seed']==seed and graph['seed']==seed and graph['selected_k']==64 and graph['dimensions']==64
    assert graph['graph_sha256']=='5cdd9acf5bd3ad3dc749300cc2fb985163233fc4d30faae0910b2b9193976acc'
    assert graph['input_manifest_sha256']=='a2b316aee5fa255df908450ac5f90960c1a05b329a1a93609d55579cb3d26447'
    assert graph['script_sha256']=='3a2c4e77bd70af6059a2854b601ac84a47de8ed752972bde656b97bbd162d59f'
    h=hashlib.sha256()
    with opener(path)(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    assert h.hexdigest()==meta['uncompressed_csv']['sha256'], 'original graph table hash mismatch'
    result={}
    with opener(path)(path,'rt',encoding='utf-8',newline='') as f:
        r=csv.DictReader(f);assert r.fieldnames==['horse_id_netkeiba']+GRAPH
        for row in r:
            horse=row['horse_id_netkeiba'];assert horse and horse not in result
            result[horse]={c:row[c] for c in GRAPH}
    return result

def assemble(base,out,regime,seed,fingerprint,old=None,new=None,parents=None):
    assert not Path(out).exists() and not Path(out).with_suffix('.json').exists(), 'refusing to overwrite prior assembly'
    expected=json.loads(Path(fingerprint).read_text(encoding='utf-8'))
    key_hash=hashlib.sha256(); content_hash=hashlib.sha256();counts=Counter(); race={};seen=set();cache={};n=0
    with opener(base)(base,'rt',encoding='utf-8',newline='') as inp, gzip.open(out,'wt',encoding='utf-8',newline='') as sink:
        reader=csv.DictReader(inp);columns=reader.fieldnames
        assert [c for c in columns if c not in GRAPH]==expected['non_graph_columns']
        writer=csv.DictWriter(sink,fieldnames=columns);writer.writeheader()
        for row in reader:
            year=int(row['race_id'][:4])
            if not 2018<=year<=2023: continue
            split=row['split'];assert (year<=2021 and split=='train') or (year>=2022 and split=='validation')
            key=tuple(row[k] for k in KEY);assert key not in seen;seen.add(key)
            key_hash.update((json.dumps(list(key),separators=(',',':'))+'\n').encode())
            content_hash.update((json.dumps([row[c] for c in expected['non_graph_columns']],ensure_ascii=False,separators=(',',':'))+'\n').encode())
            horse=row['horse_id_netkeiba']
            if regime=='old' and seed>0: row.update(old[horse])
            if regime=='new':
                if horse not in cache:
                    vector,status,depth,count,missing,cluster=resolve_embedding(horse,new[0],parents,new[1])
                    assert len(vector)==64 and all(math.isfinite(x) for x in vector)
                    cache[horse]=dict(zip(GRAPH,[status,'' if depth is None else depth,count,missing,cluster,*vector]))
                row.update(cache[horse])
            assert row['embedding_status'] in ('direct','fallback','missing')
            counts[split,row['surface'].lower(),row['embedding_status']]+=1
            rid=row['race_id'];field=int(row['field_size'])
            if rid not in race:race[rid]=[field,0]
            assert race[rid][0]==field;race[rid][1]+=1
            writer.writerow(row);n+=1
    assert n==expected['rows'] and key_hash.hexdigest()==expected['ordered_key_sha256'] and content_hash.hexdigest()==expected['non_graph_sha256']
    assert all(field==n and n>=3 for field,n in race.values())
    if regime=='new':
        target={('train','turf','direct'):27242,('train','turf','fallback'):59891,('train','dirt','direct'):29808,('train','dirt','fallback'):61765,('validation','turf','direct'):712,('validation','turf','fallback'):42071,('validation','dirt','direct'):772,('validation','dirt','fallback'):44551}
        assert counts==Counter(target),(counts,target)
    if regime=='old': assert counts['train','turf','direct']==87133 and counts['train','dirt','direct']==91573
    with gzip.open(out,'rb') as f:
        h=hashlib.sha256()
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    sidecar={'purpose':'JRA train-validation prediction inputs; cutoff sensitivity','splits':['train','validation'],'seed':seed,
        'data_sha256':h.hexdigest(),'raw_sha256':sha(out),'rows':n,'regime':regime,
        'ordered_key_sha256':key_hash.hexdigest(),'non_graph_sha256':content_hash.hexdigest(),
        'coverage':{'/'.join(k):v for k,v in sorted(counts.items())}}
    Path(out).with_suffix('.json').write_text(json.dumps(sidecar,indent=2)+'\n',encoding='utf-8')
    return sidecar

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--base-payload-hash',required=True);p.add_argument('--out',required=True);p.add_argument('--regime',choices=('old','new'),required=True);p.add_argument('--seed',type=int,choices=(0,1,2),required=True);p.add_argument('--fingerprint',required=True);p.add_argument('--old-table');p.add_argument('--old-meta');p.add_argument('--old-meta-hash');p.add_argument('--graph');p.add_argument('--parents');p.add_argument('--parent-hash');p.add_argument('--manifest-hash');p.add_argument('--runner-hash');a=p.parse_args()
    verify_payload(a.base,a.base_payload_hash)
    old=load_old(a.old_table,a.old_meta,a.seed,a.old_meta_hash) if a.regime=='old' and a.seed>0 else None
    new=load_new_graph(a.graph,a.seed,a.manifest_hash,a.runner_hash) if a.regime=='new' else None
    parents=load_parents(a.parents,a.parent_hash) if a.regime=='new' else None
    result=assemble(a.base,a.out,a.regime,a.seed,a.fingerprint,old,new,parents)
    result['base_sha256']=sha(a.base); result['base_payload_sha256']=a.base_payload_hash
    result['fingerprint_sha256']=sha(a.fingerprint)
    if a.regime=='new':
        report=json.loads((Path(a.graph)/'production_report.json').read_text())
        result.update(graph_report_sha256=sha(Path(a.graph)/'production_report.json'),
                      graph_manifest_sha256=a.manifest_hash,graph_runner_sha256=a.runner_hash,
                      graph_config_hash=report['config_hash'],graph_sha256=report['graph_sha256'],
                      graph_python_version=report['environment']['python_version'],
                      graph_vector_sha256=report['artifact_sha256'][f'nodevectors_seed{a.seed}.npz'],
                      graph_centroid_sha256=report['artifact_sha256'][f'cluster_centers_seed{a.seed}.npy'],
                      target_pedigree_sha256=a.parent_hash)
    elif a.seed>0:
        meta=json.loads(Path(a.old_meta).read_text());graph=meta['graph']
        result.update(old_graph_metadata_sha256=a.old_meta_hash,graph_sha256=graph['graph_sha256'],
            graph_manifest_sha256=graph['input_manifest_sha256'],graph_runner_sha256=graph['script_sha256'],graph_report_sha256=graph['report_sha256'],
            graph_config_hash={1:'729eb50fe0da694d8828d5447b98bc3341369685def41648b564ae6059367290',2:'c6550cd20de40ceb948f3a4ea8f9f37493111eb1d9567c29cdecaa610ec5002c'}[a.seed],
            graph_vector_sha256=graph['artifact_sha256'][f'nodevectors_seed{a.seed}.npz'],
            graph_centroid_sha256=graph['artifact_sha256'][f'cluster_centers_seed{a.seed}.npy'])
        result['graph_python_version']='3.12.13'
    else:
        result['graph_sha256']='5cdd9acf5bd3ad3dc749300cc2fb985163233fc4d30faae0910b2b9193976acc'
        result['graph_vector_sha256']='7451b79270c8f272e57f93b6fa91ba6bac977e75280da21c8bc46320aa0faa74'
        result['graph_centroid_sha256']='f72fe56a676d15cf33a1ffa71739bff52bef4f179290f955fb65f7b780b964d8'
        result['graph_config_hash']='ab5a1fffe5757eda198fdd98b2de2624fdb387676573b2bb199287865b53551c'
        result['graph_report_sha256']='5e358dbebee6c3d12b1ef03467d7dd9e7b1ebe4e95c47a4a90505c4947272cd4'
        result['graph_manifest_sha256']='a2b316aee5fa255df908450ac5f90960c1a05b329a1a93609d55579cb3d26447'
        result['graph_runner_sha256']='3a2c4e77bd70af6059a2854b601ac84a47de8ed752972bde656b97bbd162d59f'
        result['graph_python_version']='3.12.13'
    Path(a.out).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
