"""Frozen feature allowlists and train-only categorical preprocessing policy.

This module does no fitting.  It builds only the feature names and category
vocabularies consumed by later native XGBoost, LightGBM, and CatBoost wrappers.
"""
from collections import Counter
import hashlib
import math
from pathlib import Path

import prepare_prediction as frozen


SPEC_ID = "experiment_spec_2026-09-13"
MIN_TRAIN_FREQUENCY = 5
MISSING_TOKEN = "__MISSING__"
UNKNOWN_TOKEN = "__UNK__"
RARE_TOKEN = "__RARE__"
RESERVED_TOKENS = frozenset((MISSING_TOKEN, UNKNOWN_TOKEN, RARE_TOKEN))
DIMENSIONS = 64
EMBEDDING_COLUMNS = tuple(f"embedding_{i}" for i in range(DIMENSIONS))
SETTINGS = ("B1", "S1", "S2", "S3", "S4", "S5", "S6", "M2", "M3", "S7")
SURFACES = frozen.SURFACES

# Keep these definitions imported from the audited exporter.  The hash is
# recorded by downstream checkpoints so a changed helper cannot go unnoticed.
CONTEXT = frozen.CONTEXT
HISTORY = frozen.HISTORY
HISTORY_FIELDS = frozen.HISTORY_FIELDS
PEDIGREE_CATS = frozen.PEDIGREE_CATS
FROZEN_GRAPH_META = frozen.GRAPH_META
GRAPH_CLUSTER = ("graph_cluster",)
GRAPH_META = tuple(column for column in FROZEN_GRAPH_META if column not in GRAPH_CLUSTER)
HELPER_PATH = Path(frozen.__file__).resolve()
HELPER_SHA256 = hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest()

HISTORY_COLUMNS = tuple(f"{field}_{rank}"
                       for rank in (1, 2, 3)
                       for field in HISTORY_FIELDS)
MARKET = ("odds", "popularity")
CAT_BASE = frozenset(("venue", "going", "race_class", "sex"))
CATEGORICAL_PREFIXES = ("prior_surface_", "prior_going_", "prior_race_class_")
CATEGORICAL_COLUMNS = CAT_BASE | frozenset(PEDIGREE_CATS) | frozenset(("embedding_status", "graph_cluster"))
# ``horse_number`` is intentionally part of the existing CONTEXT baseline;
# the other row keys remain identifiers or split/target fields.
FORBIDDEN = frozenset(frozen.TARGET_COLUMNS) | frozenset((
    "race_id", "horse_id_netkeiba", "split", "label", "surface",
    "prior_race_id", "prior_race_date"))


def _embedding_columns(columns):
    return tuple(sorted((column for column in columns
                         if column.startswith("embedding_")
                         and column[len("embedding_"):].isdigit()),
                        key=lambda column: int(column.split("_", 1)[1])))


def feature_columns(setting, available_columns=None):
    """Return the exact predictor allowlist for one proposed setting."""
    if setting not in SETTINGS:
        raise ValueError(f"unknown setting {setting}")
    if setting == "B1":
        selected = list(MARKET[:1])
    else:
        selected = list(CONTEXT)
        if setting != "S1":
            selected += list(HISTORY) + list(HISTORY_COLUMNS)
        if setting in ("S3", "M3"):
            selected += list(PEDIGREE_CATS)
        if setting in ("S4", "S5", "S6", "S7"):
            selected += list(GRAPH_META)
        if setting in ("S5", "S6", "S7"):
            selected += list(EMBEDDING_COLUMNS)
        if setting in ("S4", "S6", "S7"):
            selected += list(GRAPH_CLUSTER)
        if setting in ("M2", "M3", "S7"):
            selected += list(MARKET)
    selected = tuple(dict.fromkeys(selected))
    bad = sorted(set(selected) & FORBIDDEN)
    if bad:
        raise ValueError(f"target or identifier in predictor allowlist: {bad}")
    if available_columns is not None:
        available = set(available_columns)
        if setting in ("S5", "S6", "S7") and _embedding_columns(available) != EMBEDDING_COLUMNS:
            raise ValueError("graph settings require exactly embedding_0..embedding_63")
        missing = [column for column in selected if column not in available]
        if missing:
            raise ValueError(f"missing required feature columns for {setting}: {missing}")
    return selected


def categorical_columns(columns):
    """Return columns to pass as categorical to native model wrappers."""
    return tuple(column for column in columns
                 if column in CATEGORICAL_COLUMNS or
                 column.startswith(CATEGORICAL_PREFIXES))


def _category_token(value):
    if value is None or value == "" or value == "NULL" or value == "None":
        return MISSING_TOKEN
    if isinstance(value, float) and math.isnan(value):
        return MISSING_TOKEN
    return str(value)


def fit_category_maps(train_rows, columns, min_frequency=MIN_TRAIN_FREQUENCY):
    """Fit deterministic rare/unknown/missing maps using training rows only.

    Codes are implementation values; callers must retain ``categorical_columns``
    separately so native wrappers treat them as categories rather than ordered
    numeric predictors.
    """
    if min_frequency < 1:
        raise ValueError("min_frequency must be positive")
    columns = tuple(categorical_columns(columns))
    counts = {column: Counter() for column in columns}
    for row in train_rows:
        split = row.get("split")
        if split not in (None, "train"):
            raise ValueError("category maps must be fitted on train rows only")
        for column in columns:
            token = _category_token(row.get(column))
            if token in RESERVED_TOKENS and token != MISSING_TOKEN:
                raise ValueError(f"input category uses reserved token: {token}")
            counts[column][token] += 1
    maps = {}
    for column in columns:
        mapping = {MISSING_TOKEN: 0, UNKNOWN_TOKEN: 1, RARE_TOKEN: 2}
        next_code = 3
        for token, count in sorted(counts[column].items()):
            if token == MISSING_TOKEN:
                continue
            if count < min_frequency:
                mapping[token] = mapping[RARE_TOKEN]
            else:
                mapping[token] = next_code
                next_code += 1
        maps[column] = mapping
    return maps


def encode_category(value, mapping):
    """Encode missing, rare, known, and unseen values with separate codes."""
    token = _category_token(value)
    return mapping.get(token, mapping[UNKNOWN_TOKEN])


def transform_rows(rows, columns, category_maps):
    """Yield transformed row copies while retaining categorical metadata."""
    columns = tuple(columns)
    categories = set(categorical_columns(columns))
    if categories != set(category_maps):
        raise ValueError("category maps must cover exactly the selected categorical columns")
    for row in rows:
        transformed = dict(row)
        for column in categories:
            transformed[column] = encode_category(row.get(column), category_maps[column])
        yield transformed


def preprocessing_metadata(setting, selected, category_maps):
    """Return JSON-ready policy metadata for a model checkpoint."""
    if tuple(selected) != feature_columns(setting):
        raise ValueError("selected columns do not match the frozen setting allowlist")
    categories = categorical_columns(selected)
    if set(category_maps) != set(categories):
        raise ValueError("category maps do not match categorical columns")
    return {"spec_id": SPEC_ID, "setting": setting,
            "features": list(selected), "categorical_columns": list(categories),
            "category_maps": category_maps,
            "category_policy": {"min_train_frequency": MIN_TRAIN_FREQUENCY,
                                 "missing_token": MISSING_TOKEN,
                                 "unknown_token": UNKNOWN_TOKEN,
                                 "rare_token": RARE_TOKEN,
                                 "fit_scope": "train_rows_only"},
            "frozen_helper": str(HELPER_PATH),
            "frozen_helper_sha256": HELPER_SHA256}
