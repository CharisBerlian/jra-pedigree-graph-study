"""Certify three completed private seed view jobs against the local pre-fit freeze."""
from pathlib import Path
import argparse,json
from remote_workspace import digest

HERE=Path(__file__).resolve().parent;PRIVATE=HERE/'private_data'


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True,type=Path);p.add_argument('--parent-certificate',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    assert all(x.resolve().is_relative_to(PRIVATE.resolve()) for x in (a.folder,a.parent_certificate,a.out)) and not a.out.exists()
    prefit=HERE/'PREFIT_MANIFEST_V4.json';m=json.loads(prefit.read_text());parent=json.loads(a.parent_certificate.read_text())
    assert parent['status']=='verified' and parent['prefit_sha256']==digest(prefit)
    outputs=[]
    for seed in (0,1,2):
        root=a.folder/f'seed{seed}';record_path=root/'prelaunch_record.json';record=json.loads(record_path.read_text())
        remote=json.loads((root/'remote_ledger.json').read_text());marker_path=root/'REMOTE_PREPARATION_PASS.json';marker=json.loads(marker_path.read_text())
        assert record['stage']==f'seed{seed}' and record['prefit_sha256']==digest(prefit)
        assert record['parent_certificate_sha256']==digest(a.parent_certificate)
        assert remote['status']=='complete' and remote['private'] is True and remote['remote_source_equal'] is True
        assert remote['source_sha256']==record['script_sha256'] and remote['prelaunch_record_sha256']==digest(record_path)
        assert int(remote['version'])>0 and remote['slug'].startswith('charischerw/')
        assert marker['status']=='pass' and marker['stage']==f'seed{seed}' and marker['scope']=='train_validation_only'
        expected_inputs={x['path']:x['sha256'] for x in record['staged']};expected_inputs['race_dates.csv.gz']=record['embedded_date_sha256']
        assert marker['input_hashes']==expected_inputs
        expected_names={f'{arm}-seed{seed}.csv{suffix}' for arm in ('Gmixed','Gancestor') for suffix in ('.gz','.json')}
        assert set(marker['outputs'])==expected_names
        for arm in ('Gmixed','Gancestor'):
            data=root/f'{arm}-seed{seed}.csv.gz';side=root/f'{arm}-seed{seed}.csv.json';meta=json.loads(side.read_text())
            golden=next(x for x in m['view_summaries'] if x['seed']==seed and x['view']==arm)
            assert digest(data,True)==golden['payload_sha256']==meta['data_sha256']
            assert digest(data)==marker['outputs'][data.name] and digest(side)==marker['outputs'][side.name]
            assert meta['seed']==seed and meta['arm']==arm and meta['rows']==636862
            assert meta['ordered_key_sha256']==m['cohort_key_sha256'] and meta['non_graph_sha256']==m['cohort_non_graph_and_labels_sha256']
            assert meta['parent_manifest_sha256']==parent['parent_manifest_sha256']
            graph=m['graph_artifacts'][seed]
            assert meta['graph_report_sha256']==graph['report_sha256'] and meta['graph_sha256']==graph['graph_sha256']
            assert meta['graph_vector_sha256']==graph['vector_sha256'] and meta['graph_centroid_sha256']==graph['centroid_sha256']
            origin=f"kernel:{remote['slug']}:{remote['output_prefix']}"
            outputs.append({'seed':seed,'view':arm,'payload_sha256':golden['payload_sha256'],'raw_sha256':digest(data),
                'sidecar_sha256':digest(side),'version':int(remote['version']),'origins':{'data':origin+'/'+data.name,'sidecar':origin+'/'+side.name}})
    result={'status':'verified','prefit_sha256':digest(prefit),'parent_certificate_sha256':digest(a.parent_certificate),
            'cohort_key_sha256':m['cohort_key_sha256'],'cohort_non_graph_and_labels_sha256':m['cohort_non_graph_and_labels_sha256'],
            'parent_feature_sha256':m['parent_feature_sha256'],'views':outputs}
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'status':'verified','views':len(outputs),'certificate_sha256':digest(a.out)}))


if __name__=='__main__':main()
