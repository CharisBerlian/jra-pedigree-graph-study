"""Four-arm allowlists using the original categorical policy unchanged.

When deployed, this file is named experiment_features.py next to the original
module saved as experiment_features_original.py and the frozen exporter.
"""
import importlib.util
from pathlib import Path
import sys

_path=Path(__file__).with_name('experiment_features_original.py')
if not _path.exists():_path=Path(__file__).resolve().parents[4]/'kaggle/experiment_features.py'
sys.path.insert(0,str(_path.parent))
_spec=importlib.util.spec_from_file_location('focused_original_features',_path)
original=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(original)
SETTINGS=('H','Gmixed','Gancestor','P')
P_COLUMNS=tuple(f'{role}_{field}' for role in ('sire','dam','damsire') for field in ('available','prior_count','prior_rate'))


def feature_columns(setting,available_columns=None):
    assert setting in SETTINGS
    cols=original.feature_columns('S6' if setting in ('Gmixed','Gancestor') else 'S2')
    if setting=='P':cols+=P_COLUMNS
    assert len(cols)==({'H':43,'Gmixed':112,'Gancestor':112,'P':52}[setting])
    if available_columns is not None:
        missing=set(cols)-set(available_columns)
        if missing:raise ValueError('missing feature columns: '+str(sorted(missing)))
    return cols


categorical_columns=original.categorical_columns
fit_category_maps=original.fit_category_maps
transform_rows=original.transform_rows


def preprocessing_metadata(setting,selected,category_maps):
    assert tuple(selected)==feature_columns(setting)
    cats=categorical_columns(selected);assert set(category_maps)==set(cats)
    return {'spec_id':original.SPEC_ID,'setting':setting,'features':list(selected),
            'categorical_columns':list(cats),'category_maps':category_maps,
            'category_policy':{'min_train_frequency':original.MIN_TRAIN_FREQUENCY,
                'missing_token':original.MISSING_TOKEN,'unknown_token':original.UNKNOWN_TOKEN,
                'rare_token':original.RARE_TOKEN,'fit_scope':'train_rows_only'},
            'frozen_helper':str(original.HELPER_PATH),'frozen_helper_sha256':original.HELPER_SHA256}
