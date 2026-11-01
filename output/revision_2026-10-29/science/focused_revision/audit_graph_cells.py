"""Apply the already-reviewed first-cell audit to the two completed graph arms."""
from pathlib import Path

source=(Path(__file__).parent/'audit_first_cell.py').read_text(encoding='utf-8')
for arm in ('Gmixed','Gancestor'):
    replacements={
        "root/'xgboost'/'turf'/'H'":f"root/'xgboost'/'turf'/'{arm}'",
        "verify_outer(root,'H','xgboost','turf'":f"verify_outer(root,'{arm}','xgboost','turf'",
        "('xgboost','turf','H')":f"('xgboost','turf','{arm}')",
        "x['view']=='Gmixed'":f"x['view']=='{arm}'",
        "job/'runs/scientific/xgboost/turf/H'":f"job/'runs/scientific/xgboost/turf/{arm}'",
        "'xgboost','turf','H','tune'":f"'xgboost','turf','{arm}','tune'",
        "feature_columns('H')":f"feature_columns('{arm}')",
        "'xgboost','turf','H',seed":f"'xgboost','turf','{arm}',seed",
        "f'H/turf/seed{seed}'":f"f'{arm}/turf/seed{{seed}}'",
        "'cell':'xgboost/turf/H'":f"'cell':'xgboost/turf/{arm}'",
    }
    script=source
    for before,after in replacements.items():
        assert script.count(before)==1,(before,script.count(before))
        script=script.replace(before,after)
    exec(compile(script,str(Path(__file__).parent/'audit_first_cell.py'),'exec'),{'__name__':'__main__'})
