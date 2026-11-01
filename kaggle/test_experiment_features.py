"""Small no-fit checks for the full experiment feature policy."""
from experiment_features import (
    CATEGORICAL_COLUMNS, CONTEXT, DIMENSIONS, EMBEDDING_COLUMNS, GRAPH_CLUSTER, GRAPH_META,
    HISTORY, HISTORY_COLUMNS, MARKET, SETTINGS, categorical_columns,
    feature_columns, fit_category_maps, preprocessing_metadata, transform_rows,
)


def _all_columns():
    return set(CONTEXT + HISTORY + HISTORY_COLUMNS + MARKET + GRAPH_META + GRAPH_CLUSTER +
               EMBEDDING_COLUMNS + ("sire", "dam"))


def check_allowlists():
    assert SETTINGS == ("B1", "S1", "S2", "S3", "S4", "S5", "S6", "M2", "M3", "S7")
    available = _all_columns()
    assert feature_columns("B1", available) == ("odds",)
    assert feature_columns("S1", available) == CONTEXT
    assert feature_columns("S2", available) == CONTEXT + HISTORY + HISTORY_COLUMNS
    assert feature_columns("S4", available) == CONTEXT + HISTORY + HISTORY_COLUMNS + GRAPH_META + GRAPH_CLUSTER
    assert len([c for c in feature_columns("S5", available) if c in EMBEDDING_COLUMNS]) == DIMENSIONS
    assert set(feature_columns("S5", available)) == set(CONTEXT + HISTORY + HISTORY_COLUMNS + GRAPH_META + EMBEDDING_COLUMNS)
    assert set(feature_columns("S6", available)) == set(CONTEXT + HISTORY + HISTORY_COLUMNS + GRAPH_META + EMBEDDING_COLUMNS + GRAPH_CLUSTER)
    assert set(feature_columns("M2", available)) == set(CONTEXT + HISTORY + HISTORY_COLUMNS + MARKET)
    assert set(feature_columns("M3", available)) == set(CONTEXT + HISTORY + HISTORY_COLUMNS + ("sire", "dam") + MARKET)
    assert set(feature_columns("S7", available)) == set(CONTEXT + HISTORY + HISTORY_COLUMNS + GRAPH_META + EMBEDDING_COLUMNS + GRAPH_CLUSTER + MARKET)
    assert set(feature_columns("S6", available)) - set(feature_columns("S5", available)) == set(GRAPH_CLUSTER)
    try:
        feature_columns("S5", available - {"embedding_63"})
    except ValueError as error:
        assert "embedding_0..embedding_63" in str(error)
    else:
        raise AssertionError("incomplete graph dimensions must be rejected")
    assert "odds" not in feature_columns("S2", available)
    assert "popularity" not in feature_columns("S6", available)


def check_train_only_categories():
    columns = ("sire", "venue", "embedding_status", "graph_cluster")
    rows = [{"split": "train", "sire": "common", "venue": "V", "embedding_status": "direct", "graph_cluster": "1"}
            for _ in range(5)]
    rows += [{"split": "train", "sire": "rare", "venue": "V", "embedding_status": "direct", "graph_cluster": "1"}]
    maps = fit_category_maps(rows, columns)
    assert maps["sire"]["common"] >= 3
    assert maps["sire"]["rare"] == maps["sire"]["__RARE__"]
    transformed = list(transform_rows([
        {"sire": "common", "venue": None, "embedding_status": "new", "graph_cluster": "1"},
        {"sire": "unseen", "venue": "V", "embedding_status": "direct", "graph_cluster": "1"},
    ], columns, maps))
    assert transformed[0]["sire"] == maps["sire"]["common"]
    assert transformed[0]["venue"] == maps["venue"]["__MISSING__"]
    assert transformed[0]["embedding_status"] == maps["embedding_status"]["__UNK__"]
    assert transformed[1]["sire"] == maps["sire"]["__UNK__"]
    assert tuple(categorical_columns(columns)) == columns
    metadata = preprocessing_metadata("S3", feature_columns("S3"),
                                      fit_category_maps(rows, categorical_columns(feature_columns("S3"))))
    assert metadata["category_policy"]["fit_scope"] == "train_rows_only"
    try:
        fit_category_maps(rows + [{"split": "validation", "sire": "leak"}], ("sire",))
    except ValueError as error:
        assert "train rows only" in str(error)
    else:
        raise AssertionError("validation rows must not fit category maps")


if __name__ == "__main__":
    check_allowlists()
    check_train_only_categories()
    print("PASS: ten setting allowlists, graph dimensions, train-only rare/unknown/missing category policy, and categorical metadata")
