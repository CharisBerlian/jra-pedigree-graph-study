"""Certify an already completed private P job; no network or fitting."""
from pathlib import Path
import argparse,json
from remote_workspace import digest

HERE=Path(__file__).resolve().parent;PRIVATE=HERE/'private_data'
PARENT='ab732ff06a8e6d23f99d23f9ed2dcddeb0685f2b1b868be43af626a4380e30da'
MAP='f07bbf61e92d67a70de28f5832751a0a27d5d6675d2111ceccd1cf10b2c63774'


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True,type=Path);p.add_argument('--record',required=True,type=Path)
    p.add_argument('--remote-ledger',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    assert all(x.resolve().is_relative_to(PRIVATE.resolve()) for x in (a.folder,a.record,a.remote_ledger,a.out)) and not a.out.exists()
    prefit=HERE/'PREFIT_MANIFEST_V4.json';record=json.loads(a.record.read_text());remote=json.loads(a.remote_ledger.read_text())
    assert record['stage']=='parent' and record['prefit_sha256']==digest(prefit)
    assert remote['status']=='complete' and remote['private'] is True and remote['remote_source_equal'] is True
    assert remote['source_sha256']==record['script_sha256'] and remote['prelaunch_record_sha256']==digest(a.record)
    assert int(remote['version'])>0 and remote['slug'].startswith('charischerw/')
    marker=json.loads((a.folder/'REMOTE_PREPARATION_PASS.json').read_text());assert marker['status']=='pass' and marker['stage']=='parent'
    assert marker['scope']=='train_validation_only'
    expected_inputs={x['path']:x['sha256'] for x in record['staged']};expected_inputs['race_dates.csv.gz']=record['embedded_date_sha256']
    assert marker['input_hashes']==expected_inputs
    feature=a.folder/'parent_features.csv.gz';maps=a.folder/'parent_maps.json';meta=a.folder/'parent_manifest.json'
    assert digest(feature,True)==PARENT and digest(maps)==MAP
    assert marker['outputs']=={p.name:digest(p) for p in (feature,maps,meta)}
    parent=json.loads(meta.read_text());assert parent['output_sha256']==digest(feature) and parent['map_sha256']==MAP
    prefix=f"kernel:{remote['slug']}:{remote['output_prefix']}"
    result={'status':'verified','prefit_sha256':digest(prefit),'parent_features_payload_sha256':PARENT,
            'parent_features_raw_sha256':digest(feature),'parent_manifest_sha256':digest(meta),'parent_map_sha256':MAP,
            'version':int(remote['version']),'origins':{name:prefix+'/'+name for name in ('parent_features.csv.gz','parent_manifest.json')},
            'remote_marker_sha256':digest(a.folder/'REMOTE_PREPARATION_PASS.json'),'record_sha256':digest(a.record)}
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'status':'verified','certificate_sha256':digest(a.out)}))


if __name__=='__main__':main()
