"""Independently re-read six train/validation views and freeze categorical maps."""
from pathlib import Path
from collections import Counter
import csv,gzip,hashlib,json
import feature_adapter as f
from assemble_features import GRAPH,KEY,sha

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];PRIVATE=HERE/'private_data'


def upd(h,values):h.update((json.dumps(values,separators=(',',':'),ensure_ascii=False)+'\n').encode())


def main():
    paths=[PRIVATE/f'{arm}-seed{s}.csv.gz' for s in (0,1,2) for arm in ('Gmixed','Gancestor')]
    assert all(p.exists() and p.with_suffix('.json').exists() for p in paths)
    output=PRIVATE/'view_audit.json';map_path=PRIVATE/'category_maps.json';assert not output.exists() and not map_path.exists()
    original=ROOT/'kaggle/upload/jra-prediction-bundle-v1/train_validation.csv.gz'
    with gzip.open(original,'rt',encoding='utf-8',newline='') as h:
        reader=csv.DictReader(h);base_columns=[n for n in reader.fieldnames if n not in GRAPH]
        k=hashlib.sha256();b=hashlib.sha256();n=0
        for row in reader:
            assert row['split'] in ('train','validation');upd(k,[row[x] for x in KEY]);upd(b,[row[x] for x in base_columns]);n+=1
    assert n==636862
    expected_key=k.hexdigest();expected_base=b.hexdigest();expected_parent=None;all_maps={};summaries=[];parent_coverage=Counter()
    for path in paths:
        meta=json.loads(path.with_suffix('.json').read_text());assert meta['rows']==n
        assert sha(path,True)==meta['data_sha256'] and sha(path)==meta['raw_sha256']
        seed=meta['seed'];view=meta['arm'];arms=('H','P','Gmixed') if view=='Gmixed' else ('Gancestor',)
        maps={(surface,arm):{c:Counter() for c in f.categorical_columns(f.feature_columns(arm))}
              for surface in ('turf','dirt') for arm in arms}
        k=hashlib.sha256();b=hashlib.sha256();ph=hashlib.sha256();nview=0;counts=Counter()
        with gzip.open(path,'rt',encoding='utf-8',newline='') as inp:
            for row in csv.DictReader(inp):
                assert row['split'] in ('train','validation') and row['label'] in ('0','1')
                surface=row['surface'].lower();assert surface in ('turf','dirt')
                upd(k,[row[x] for x in KEY]);upd(b,[row[x] for x in base_columns]);upd(ph,[row[x] for x in KEY]+[row[x] for x in f.P_COLUMNS])
                counts[(row['split'],surface)]+=1;nview+=1
                if seed==0 and view=='Gmixed':
                    for role in ('sire','dam','damsire'):
                        parent_coverage[(row['split'],surface,role,'rows')]+=1
                        parent_coverage[(row['split'],surface,role,'zero_prior')]+=int(int(row[role+'_prior_count'])==0)
                        parent_coverage[(row['split'],surface,role,'missing_id')]+=int(int(row[role+'_available'])==0)
                if row['split']=='train':
                    for arm in arms:
                        for column,counter in maps[surface,arm].items():
                            token=f.original._category_token(row[column])
                            assert token not in f.original.RESERVED_TOKENS or token==f.original.MISSING_TOKEN
                            counter[token]+=1
        assert nview==n and k.hexdigest()==expected_key==meta['ordered_key_sha256'] and b.hexdigest()==expected_base==meta['non_graph_sha256']
        if expected_parent is None:expected_parent=ph.hexdigest()
        assert ph.hexdigest()==expected_parent
        assert counts=={('train','turf'):271549,('train','dirt'):277207,('validation','turf'):42783,('validation','dirt'):45323}
        for (surface,arm),columns in maps.items():
            result={}
            for name,counter in columns.items():
                mapping={f.original.MISSING_TOKEN:0,f.original.UNKNOWN_TOKEN:1,f.original.RARE_TOKEN:2};next_code=3
                for token,count in sorted(counter.items()):
                    if token==f.original.MISSING_TOKEN:continue
                    if count<f.original.MIN_TRAIN_FREQUENCY:mapping[token]=2
                    else:mapping[token]=next_code;next_code+=1
                result[name]=mapping
            all_maps[f'{arm}/{surface}/seed{seed}']=result
        summaries.append({'seed':seed,'view':view,'raw_sha256':sha(path),'payload_sha256':meta['data_sha256'],
                          'sidecar_sha256':sha(path.with_suffix('.json')),'rows':nview,'parent_feature_sha256':ph.hexdigest()})
        print('audited',path.name,flush=True)
    assert len(all_maps)==24
    map_path.write_text(json.dumps(all_maps,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8',newline='\n')
    data={'status':'pass','scope':'train_validation_only','rows':n,'ordered_key_sha256':expected_key,
          'non_graph_sha256':expected_base,'parent_features_sha256':expected_parent,'view_summaries':summaries,
          'parent_coverage':{'/'.join(k):v for k,v in sorted(parent_coverage.items())},
          'category_maps_sha256':sha(map_path),'auditor_sha256':sha(Path(__file__))}
    output.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print('PASS six feature views and 24 surface/arm/seed category maps')


if __name__=='__main__':main()
