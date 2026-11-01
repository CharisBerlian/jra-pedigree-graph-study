"""Compare frozen map application to all existing validation P features."""
from pathlib import Path
import csv,gzip,hashlib,json,math,sys
import pandas as pd
from apply_frozen_parent import load,apply,sha

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];PRIVATE=HERE/'private_data'
sys.path.insert(0,str(ROOT/'kaggle'))
from graph_pilot import load_parents


def main():
    target=PRIVATE/'validation_frozen_parent_audit.json';assert not target.exists()
    manifest=json.loads((PRIVATE/'parent_manifest.json').read_text());source=ROOT/'kaggle/upload/jra-prediction-bundle-v1/train_validation.csv.gz'
    assert sha(source)==manifest['source_sha256']
    file=PRIVATE/'parent_features.csv.gz';assert sha(file)==manifest['output_sha256']
    maps=load(PRIVATE/'parent_maps.json',manifest['map_sha256'])
    parents=load_parents([ROOT/'local_jra_worker/data/raw_jbis_laptop.db',ROOT/'tmp/pedigree_patch.db'])
    cols=['race_id','horse_id_netkeiba','horse_number','split','surface']
    data=pd.read_csv(source,usecols=cols,dtype=str,keep_default_na=False)
    assert len(data)==636862 and set(data['split'])=={'train','validation'}
    checked=0
    with gzip.open(file,'rt',encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f)
        for row,entry in zip(reader,data.itertuples(index=False)):
            assert (row['race_id'],row['horse_id_netkeiba'],row['horse_number'])==(entry.race_id,entry.horse_id_netkeiba,entry.horse_number)
            if entry.split!='validation':continue
            got=apply(entry.horse_id_netkeiba,entry.surface.lower(),parents,maps)
            for name,value in got.items():assert math.isclose(float(row[name]),value,rel_tol=0,abs_tol=1e-15),(entry.race_id,name)
            checked+=1
        assert next(reader,None) is None
    assert checked==88106
    result={'status':'pass','validation_rows_checked':checked,'scope':'train/validation only; no test input',
            'map_sha256':manifest['map_sha256'],'parent_feature_raw_sha256':sha(file),'source_sha256':sha(source),
            'application_source_sha256':sha(HERE/'apply_frozen_parent.py'),'checker_source_sha256':sha(Path(__file__))}
    target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result))


if __name__=='__main__':main()
