"""Reproduce focused train/validation views from verified staged artifacts.

This is an execution source, not a submitter. Its caller stages the exact input
files and code bundle privately and supplies a reviewed hash-bound config.
"""
from pathlib import Path
import argparse,json,shutil,subprocess,sys
import numpy as np
from remote_workspace import surrogate,deterministic_gzip,digest

BASE='d395a1802d90b0c09cc71f08b2dcbc83ca4020654797c9442969833c103ac12d'
PARENT='ab732ff06a8e6d23f99d23f9ed2dcddeb0685f2b1b868be43af626a4380e30da'
MAP='f07bbf61e92d67a70de28f5832751a0a27d5d6675d2111ceccd1cf10b2c63774'
REL='output/revision_2026-10-29/science/focused_revision'


def run(config_path):
    cfg=json.loads(Path(config_path).read_text());stage=cfg['stage'];assert stage in ('parent','seed0','seed1','seed2')
    assert sys.version.split()[0]=='3.13.15' and np.__version__=='1.26.4'
    inputs=Path(cfg['input_dir']);bundle=Path(cfg['code_dir']);work=Path(cfg['workspace']);assert not work.exists()
    assert set(cfg['files'])==set(cfg['required_files']) and len(cfg['files'])==len(cfg['required_files'])
    for name,expected in cfg['files'].items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts and digest(inputs/name)==expected
    assert cfg['files']['train_validation.csv']==BASE
    assert cfg['files']['pedigree_edges.csv']=='7c9fae4cf81b4a65b2ed8100b46bbd016a3e9ef198bca4a334bb14d841cfe6e7'
    assert cfg['files']['target_pedigree.csv.gz']=='0bb528f1fe3f4413742dbb2f32b1b29c3b720cc808ff7da131abf53aeddf9a6b'
    assert cfg['files']['race_dates.csv.gz']=='f5d5486195e05dbf7ddf6b019035856ce1628dd20b1b6e898a92f568cd046ae2'
    expected_code={'parent_control.py':f'{REL}/parent_control.py','prepare_parent_data.py':f'{REL}/prepare_parent_data.py',
       'ancestry.py':f'{REL}/ancestry.py','feature_adapter.py':f'{REL}/feature_adapter.py',
       'assemble_features.py':f'{REL}/assemble_features.py','assemble_cutoff.py':'output/revision_2026-10-29/science/assemble_cutoff.py',
       'graph_pilot.py':'kaggle/graph_pilot.py','prepare_prediction.py':'kaggle/prepare_prediction.py',
       'experiment_features.py':'kaggle/experiment_features.py'}
    assert set(cfg['code_sha256'])==set(expected_code)
    for name,dest in expected_code.items():
        src=bundle/name;assert digest(src)==cfg['code_sha256'][name]
        dst=work/dest;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    info=surrogate(inputs/'pedigree_edges.csv',inputs/'target_pedigree.csv.gz',inputs/'race_dates.csv.gz',work)
    assert info['races']==44848
    data=work/'kaggle/upload/jra-prediction-bundle-v1/train_validation.csv.gz';data.parent.mkdir(parents=True)
    deterministic_gzip(inputs/'train_validation.csv',data);assert digest(data,True)==BASE
    focused=work/REL;private=focused/'private_data';private.mkdir()
    if stage=='parent':
        subprocess.run([sys.executable,str(focused/'prepare_parent_data.py')],check=True,cwd=work)
        assert digest(private/'parent_features.csv.gz',True)==PARENT and digest(private/'parent_maps.json')==MAP
        outputs=['parent_features.csv.gz','parent_maps.json','parent_manifest.json']
    else:
        seed=int(stage[-1]);assert seed in (0,1,2)
        for name in ('parent_features.csv.gz','parent_manifest.json'):shutil.copyfile(inputs/name,private/name)
        parent=json.loads((private/'parent_manifest.json').read_text());assert digest(private/'parent_features.csv.gz',True)==PARENT
        assert digest(private/'parent_features.csv.gz')==parent['output_sha256'] and parent['map_sha256']==MAP
        folder=work/'kaggle/notebook-output'/('production-seed0-version2' if seed==0 else f'production-seed{seed}-version1')/'jra-production'/f'seed-{seed}';folder.mkdir(parents=True)
        for name in ('production_report.json','training_graph.edg',f'nodevectors_seed{seed}.npz',f'cluster_centers_seed{seed}.npy'):
            shutil.copyfile(inputs/name,folder/name)
        if seed:
            old=work/'kaggle/upload/jra-experiment-graph-tables-v1';old.mkdir(parents=True)
            for suffix in ('.csv','.csv.json'):
                name=f'graph_features_seed{seed}{suffix}';shutil.copyfile(inputs/name,old/name)
        subprocess.run([sys.executable,str(focused/'assemble_features.py'),'--seed',str(seed)],check=True,cwd=work)
        outputs=[]
        for arm in ('Gmixed','Gancestor'):
            name=f'{arm}-seed{seed}.csv.gz';meta=private/name.replace('.gz','.json')
            side=json.loads(meta.read_text());assert digest(private/name,True)==cfg['golden_payload_sha256'][arm]==side['data_sha256']
            assert side['ordered_key_sha256']==cfg['golden_key_sha256'] and side['non_graph_sha256']==cfg['golden_non_graph_sha256']
            outputs.extend((name,meta.name))
    published=Path(cfg['output_dir']);assert not published.exists();published.mkdir(parents=True)
    for name in outputs:shutil.copyfile(private/name,published/name)
    result={'status':'pass','stage':stage,'scope':'train_validation_only','input_hashes':cfg['files'],
            'outputs':{name:digest(published/name) for name in outputs},'source_config_sha256':digest(Path(config_path))}
    (published/'REMOTE_PREPARATION_PASS.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(p.parse_args().config)
