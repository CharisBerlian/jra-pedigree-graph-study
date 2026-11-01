# Supplement S1: Methods, results, and audit corrections

Version: 2026-09-16 audit revision. Local author-review supplement; no public DOI or repository exists yet.

The 180 model predictions and original six main contrasts are unchanged. New history contrasts are explicitly exploratory. B0 ranking summaries correct a reporting bug; no model was retrained.

## Exact feature definitions and preprocessing

| Setting | Total columns | Categorical columns |
|---|---:|---:|
| B1 | 1 | 0 |
| M2 | 45 | 13 |
| M3 | 47 | 15 |
| S1 | 12 | 4 |
| S2 | 43 | 13 |
| S3 | 45 | 15 |
| S4 | 48 | 15 |
| S5 | 111 | 14 |
| S6 | 112 | 15 |
| S7 | 114 | 15 |

### B1

Ordered predictors: `odds`.

Categorical predictors: None.

### M2

Ordered predictors: `venue`, `distance`, `going`, `race_class`, `field_size`, `bracket`, `horse_number`, `weight_carried`, `age`, `sex`, `body_weight`, `body_weight_change`, `hist_count`, `has_hist1`, `has_hist2`, `has_hist3`, `prior_finish_pos_1`, `prior_norm_finish_1`, `prior_field_size_1`, `prior_distance_1`, `prior_surface_1`, `prior_going_1`, `prior_race_class_1`, `prior_weight_carried_1`, `days_since_1`, `prior_finish_pos_2`, `prior_norm_finish_2`, `prior_field_size_2`, `prior_distance_2`, `prior_surface_2`, `prior_going_2`, `prior_race_class_2`, `prior_weight_carried_2`, `days_since_2`, `prior_finish_pos_3`, `prior_norm_finish_3`, `prior_field_size_3`, `prior_distance_3`, `prior_surface_3`, `prior_going_3`, `prior_race_class_3`, `prior_weight_carried_3`, `days_since_3`, `odds`, `popularity`.

Categorical predictors: `venue`, `going`, `race_class`, `sex`, `prior_surface_1`, `prior_going_1`, `prior_race_class_1`, `prior_surface_2`, `prior_going_2`, `prior_race_class_2`, `prior_surface_3`, `prior_going_3`, `prior_race_class_3`.

### M3

Ordered predictors: `venue`, `distance`, `going`, `race_class`, `field_size`, `bracket`, `horse_number`, `weight_carried`, `age`, `sex`, `body_weight`, `body_weight_change`, `hist_count`, `has_hist1`, `has_hist2`, `has_hist3`, `prior_finish_pos_1`, `prior_norm_finish_1`, `prior_field_size_1`, `prior_distance_1`, `prior_surface_1`, `prior_going_1`, `prior_race_class_1`, `prior_weight_carried_1`, `days_since_1`, `prior_finish_pos_2`, `prior_norm_finish_2`, `prior_field_size_2`, `prior_distance_2`, `prior_surface_2`, `prior_going_2`, `prior_race_class_2`, `prior_weight_carried_2`, `days_since_2`, `prior_finish_pos_3`, `prior_norm_finish_3`, `prior_field_size_3`, `prior_distance_3`, `prior_surface_3`, `prior_going_3`, `prior_race_class_3`, `prior_weight_carried_3`, `days_since_3`, `sire`, `dam`, `odds`, `popularity`.

Categorical predictors: `venue`, `going`, `race_class`, `sex`, `prior_surface_1`, `prior_going_1`, `prior_race_class_1`, `prior_surface_2`, `prior_going_2`, `prior_race_class_2`, `prior_surface_3`, `prior_going_3`, `prior_race_class_3`, `sire`, `dam`.

### S1

Ordered predictors: `venue`, `distance`, `going`, `race_class`, `field_size`, `bracket`, `horse_number`, `weight_carried`, `age`, `sex`, `body_weight`, `body_weight_change`.

Categorical predictors: `venue`, `going`, `race_class`, `sex`.

### S2

Ordered predictors: `venue`, `distance`, `going`, `race_class`, `field_size`, `bracket`, `horse_number`, `weight_carried`, `age`, `sex`, `body_weight`, `body_weight_change`, `hist_count`, `has_hist1`, `has_hist2`, `has_hist3`, `prior_finish_pos_1`, `prior_norm_finish_1`, `prior_field_size_1`, `prior_distance_1`, `prior_surface_1`, `prior_going_1`, `prior_race_class_1`, `prior_weight_carried_1`, `days_since_1`, `prior_finish_pos_2`, `prior_norm_finish_2`, `prior_field_size_2`, `prior_distance_2`, `prior_surface_2`, `prior_going_2`, `prior_race_class_2`, `prior_weight_carried_2`, `days_since_2`, `prior_finish_pos_3`, `prior_norm_finish_3`, `prior_field_size_3`, `prior_distance_3`, `prior_surface_3`, `prior_going_3`, `prior_race_class_3`, `prior_weight_carried_3`, `days_since_3`.

Categorical predictors: `venue`, `going`, `race_class`, `sex`, `prior_surface_1`, `prior_going_1`, `prior_race_class_1`, `prior_surface_2`, `prior_going_2`, `prior_race_class_2`, `prior_surface_3`, `prior_going_3`, `prior_race_class_3`.

### S3

Ordered predictors: `venue`, `distance`, `going`, `race_class`, `field_size`, `bracket`, `horse_number`, `weight_carried`, `age`, `sex`, `body_weight`, `body_weight_change`, `hist_count`, `has_hist1`, `has_hist2`, `has_hist3`, `prior_finish_pos_1`, `prior_norm_finish_1`, `prior_field_size_1`, `prior_distance_1`, `prior_surface_1`, `prior_going_1`, `prior_race_class_1`, `prior_weight_carried_1`, `days_since_1`, `prior_finish_pos_2`, `prior_norm_finish_2`, `prior_field_size_2`, `prior_distance_2`, `prior_surface_2`, `prior_going_2`, `prior_race_class_2`, `prior_weight_carried_2`, `days_since_2`, `prior_finish_pos_3`, `prior_norm_finish_3`, `prior_field_size_3`, `prior_distance_3`, `prior_surface_3`, `prior_going_3`, `prior_race_class_3`, `prior_weight_carried_3`, `days_since_3`, `sire`, `dam`.

Categorical predictors: `venue`, `going`, `race_class`, `sex`, `prior_surface_1`, `prior_going_1`, `prior_race_class_1`, `prior_surface_2`, `prior_going_2`, `prior_race_class_2`, `prior_surface_3`, `prior_going_3`, `prior_race_class_3`, `sire`, `dam`.

### S4

Ordered predictors: `venue`, `distance`, `going`, `race_class`, `field_size`, `bracket`, `horse_number`, `weight_carried`, `age`, `sex`, `body_weight`, `body_weight_change`, `hist_count`, `has_hist1`, `has_hist2`, `has_hist3`, `prior_finish_pos_1`, `prior_norm_finish_1`, `prior_field_size_1`, `prior_distance_1`, `prior_surface_1`, `prior_going_1`, `prior_race_class_1`, `prior_weight_carried_1`, `days_since_1`, `prior_finish_pos_2`, `prior_norm_finish_2`, `prior_field_size_2`, `prior_distance_2`, `prior_surface_2`, `prior_going_2`, `prior_race_class_2`, `prior_weight_carried_2`, `days_since_2`, `prior_finish_pos_3`, `prior_norm_finish_3`, `prior_field_size_3`, `prior_distance_3`, `prior_surface_3`, `prior_going_3`, `prior_race_class_3`, `prior_weight_carried_3`, `days_since_3`, `embedding_status`, `fallback_depth`, `ancestor_count`, `embedding_missing`, `graph_cluster`.

Categorical predictors: `venue`, `going`, `race_class`, `sex`, `prior_surface_1`, `prior_going_1`, `prior_race_class_1`, `prior_surface_2`, `prior_going_2`, `prior_race_class_2`, `prior_surface_3`, `prior_going_3`, `prior_race_class_3`, `embedding_status`, `graph_cluster`.

### S5

Ordered predictors: `venue`, `distance`, `going`, `race_class`, `field_size`, `bracket`, `horse_number`, `weight_carried`, `age`, `sex`, `body_weight`, `body_weight_change`, `hist_count`, `has_hist1`, `has_hist2`, `has_hist3`, `prior_finish_pos_1`, `prior_norm_finish_1`, `prior_field_size_1`, `prior_distance_1`, `prior_surface_1`, `prior_going_1`, `prior_race_class_1`, `prior_weight_carried_1`, `days_since_1`, `prior_finish_pos_2`, `prior_norm_finish_2`, `prior_field_size_2`, `prior_distance_2`, `prior_surface_2`, `prior_going_2`, `prior_race_class_2`, `prior_weight_carried_2`, `days_since_2`, `prior_finish_pos_3`, `prior_norm_finish_3`, `prior_field_size_3`, `prior_distance_3`, `prior_surface_3`, `prior_going_3`, `prior_race_class_3`, `prior_weight_carried_3`, `days_since_3`, `embedding_status`, `fallback_depth`, `ancestor_count`, `embedding_missing`, `embedding_0`, `embedding_1`, `embedding_2`, `embedding_3`, `embedding_4`, `embedding_5`, `embedding_6`, `embedding_7`, `embedding_8`, `embedding_9`, `embedding_10`, `embedding_11`, `embedding_12`, `embedding_13`, `embedding_14`, `embedding_15`, `embedding_16`, `embedding_17`, `embedding_18`, `embedding_19`, `embedding_20`, `embedding_21`, `embedding_22`, `embedding_23`, `embedding_24`, `embedding_25`, `embedding_26`, `embedding_27`, `embedding_28`, `embedding_29`, `embedding_30`, `embedding_31`, `embedding_32`, `embedding_33`, `embedding_34`, `embedding_35`, `embedding_36`, `embedding_37`, `embedding_38`, `embedding_39`, `embedding_40`, `embedding_41`, `embedding_42`, `embedding_43`, `embedding_44`, `embedding_45`, `embedding_46`, `embedding_47`, `embedding_48`, `embedding_49`, `embedding_50`, `embedding_51`, `embedding_52`, `embedding_53`, `embedding_54`, `embedding_55`, `embedding_56`, `embedding_57`, `embedding_58`, `embedding_59`, `embedding_60`, `embedding_61`, `embedding_62`, `embedding_63`.

Categorical predictors: `venue`, `going`, `race_class`, `sex`, `prior_surface_1`, `prior_going_1`, `prior_race_class_1`, `prior_surface_2`, `prior_going_2`, `prior_race_class_2`, `prior_surface_3`, `prior_going_3`, `prior_race_class_3`, `embedding_status`.

### S6

Ordered predictors: `venue`, `distance`, `going`, `race_class`, `field_size`, `bracket`, `horse_number`, `weight_carried`, `age`, `sex`, `body_weight`, `body_weight_change`, `hist_count`, `has_hist1`, `has_hist2`, `has_hist3`, `prior_finish_pos_1`, `prior_norm_finish_1`, `prior_field_size_1`, `prior_distance_1`, `prior_surface_1`, `prior_going_1`, `prior_race_class_1`, `prior_weight_carried_1`, `days_since_1`, `prior_finish_pos_2`, `prior_norm_finish_2`, `prior_field_size_2`, `prior_distance_2`, `prior_surface_2`, `prior_going_2`, `prior_race_class_2`, `prior_weight_carried_2`, `days_since_2`, `prior_finish_pos_3`, `prior_norm_finish_3`, `prior_field_size_3`, `prior_distance_3`, `prior_surface_3`, `prior_going_3`, `prior_race_class_3`, `prior_weight_carried_3`, `days_since_3`, `embedding_status`, `fallback_depth`, `ancestor_count`, `embedding_missing`, `embedding_0`, `embedding_1`, `embedding_2`, `embedding_3`, `embedding_4`, `embedding_5`, `embedding_6`, `embedding_7`, `embedding_8`, `embedding_9`, `embedding_10`, `embedding_11`, `embedding_12`, `embedding_13`, `embedding_14`, `embedding_15`, `embedding_16`, `embedding_17`, `embedding_18`, `embedding_19`, `embedding_20`, `embedding_21`, `embedding_22`, `embedding_23`, `embedding_24`, `embedding_25`, `embedding_26`, `embedding_27`, `embedding_28`, `embedding_29`, `embedding_30`, `embedding_31`, `embedding_32`, `embedding_33`, `embedding_34`, `embedding_35`, `embedding_36`, `embedding_37`, `embedding_38`, `embedding_39`, `embedding_40`, `embedding_41`, `embedding_42`, `embedding_43`, `embedding_44`, `embedding_45`, `embedding_46`, `embedding_47`, `embedding_48`, `embedding_49`, `embedding_50`, `embedding_51`, `embedding_52`, `embedding_53`, `embedding_54`, `embedding_55`, `embedding_56`, `embedding_57`, `embedding_58`, `embedding_59`, `embedding_60`, `embedding_61`, `embedding_62`, `embedding_63`, `graph_cluster`.

Categorical predictors: `venue`, `going`, `race_class`, `sex`, `prior_surface_1`, `prior_going_1`, `prior_race_class_1`, `prior_surface_2`, `prior_going_2`, `prior_race_class_2`, `prior_surface_3`, `prior_going_3`, `prior_race_class_3`, `embedding_status`, `graph_cluster`.

### S7

Ordered predictors: `venue`, `distance`, `going`, `race_class`, `field_size`, `bracket`, `horse_number`, `weight_carried`, `age`, `sex`, `body_weight`, `body_weight_change`, `hist_count`, `has_hist1`, `has_hist2`, `has_hist3`, `prior_finish_pos_1`, `prior_norm_finish_1`, `prior_field_size_1`, `prior_distance_1`, `prior_surface_1`, `prior_going_1`, `prior_race_class_1`, `prior_weight_carried_1`, `days_since_1`, `prior_finish_pos_2`, `prior_norm_finish_2`, `prior_field_size_2`, `prior_distance_2`, `prior_surface_2`, `prior_going_2`, `prior_race_class_2`, `prior_weight_carried_2`, `days_since_2`, `prior_finish_pos_3`, `prior_norm_finish_3`, `prior_field_size_3`, `prior_distance_3`, `prior_surface_3`, `prior_going_3`, `prior_race_class_3`, `prior_weight_carried_3`, `days_since_3`, `embedding_status`, `fallback_depth`, `ancestor_count`, `embedding_missing`, `embedding_0`, `embedding_1`, `embedding_2`, `embedding_3`, `embedding_4`, `embedding_5`, `embedding_6`, `embedding_7`, `embedding_8`, `embedding_9`, `embedding_10`, `embedding_11`, `embedding_12`, `embedding_13`, `embedding_14`, `embedding_15`, `embedding_16`, `embedding_17`, `embedding_18`, `embedding_19`, `embedding_20`, `embedding_21`, `embedding_22`, `embedding_23`, `embedding_24`, `embedding_25`, `embedding_26`, `embedding_27`, `embedding_28`, `embedding_29`, `embedding_30`, `embedding_31`, `embedding_32`, `embedding_33`, `embedding_34`, `embedding_35`, `embedding_36`, `embedding_37`, `embedding_38`, `embedding_39`, `embedding_40`, `embedding_41`, `embedding_42`, `embedding_43`, `embedding_44`, `embedding_45`, `embedding_46`, `embedding_47`, `embedding_48`, `embedding_49`, `embedding_50`, `embedding_51`, `embedding_52`, `embedding_53`, `embedding_54`, `embedding_55`, `embedding_56`, `embedding_57`, `embedding_58`, `embedding_59`, `embedding_60`, `embedding_61`, `embedding_62`, `embedding_63`, `graph_cluster`, `odds`, `popularity`.

Categorical predictors: `venue`, `going`, `race_class`, `sex`, `prior_surface_1`, `prior_going_1`, `prior_race_class_1`, `prior_surface_2`, `prior_going_2`, `prior_race_class_2`, `prior_surface_3`, `prior_going_3`, `prior_race_class_3`, `embedding_status`, `graph_cluster`.

Race context has 12 columns. For each of the three prior observed starts, history has nine fields plus the capped count and three availability indicators. Historical surface may include jump starts; current target races are turf or dirt. The simple-pedigree setting uses categorical sire/dam IDs only, without genetic summaries. Graph coverage descriptors accompany S4/S5/S6/S7, so S6-S2 measures the combined representation and its availability descriptors.

Category maps use only each surface's training rows. Missing = code 0, unseen = 1, rare (<5 training entries) = 2; other tokens are sorted and numbered from 3. Codes are native categories in XGBoost/LightGBM and strings in CatBoost, not ordered numeric predictors. Numeric missing values remain NaN. No scaling, numeric imputation, external target encoding, or oversampling is used. Native CatBoost categorical statistics are part of its algorithm. Future-category values use the explicit unknown category, including unseen sire/dam IDs. Missing graph vectors use zeros and explicit missing representation/cluster categories.

## Hyperparameter search and selected fits

The following are four fixed candidate vectors per family, not a Cartesian product. All fits allow at most 1,200 rounds, learning rates as shown, patience 75 on validation AP, and four threads. Selection uses AP (tie tolerance 1e-6), then higher Hit@3, fewer rounds, and candidate ID. Seeds 0/1/2 reuse the selected seed-0 parameter vector and round count in training-only refits. Each graph seed is paired with the same classifier seed; randomness sources cannot be separated in this design.


### catboost

| Candidate | depth | l2_leaf_reg | learning_rate | random_strength | rsm |
|---|---|---|---|---|---|
| c0 | 5 | 5.0 | 0.05 | 1.0 | 0.8 |
| c1 | 6 | 10.0 | 0.03 | 1.0 | 0.8 |
| c2 | 7 | 20.0 | 0.03 | 1.0 | 0.7 |
| c3 | 4 | 20.0 | 0.06 | 2.0 | 1.0 |

### lightgbm

| Candidate | colsample_bytree | learning_rate | max_depth | min_child_samples | num_leaves | reg_alpha | reg_lambda | subsample | subsample_freq |
|---|---|---|---|---|---|---|---|---|---|
| c0 | 0.8 | 0.05 | 5 | 50 | 15 | 0.0 | 5.0 | 0.8 | 1 |
| c1 | 0.8 | 0.03 | 7 | 50 | 31 | 0.1 | 10.0 | 0.8 | 1 |
| c2 | 0.7 | 0.03 | 9 | 100 | 63 | 0.5 | 20.0 | 0.7 | 1 |
| c3 | 1.0 | 0.06 | 4 | 100 | 15 | 1.0 | 20.0 | 1.0 | 0 |

### xgboost

| Candidate | colsample_bytree | learning_rate | max_depth | min_child_weight | reg_alpha | reg_lambda | subsample |
|---|---|---|---|---|---|---|---|
| c0 | 0.8 | 0.05 | 4 | 5 | 0.0 | 5.0 | 0.8 |
| c1 | 0.8 | 0.03 | 6 | 10 | 0.1 | 10.0 | 0.8 |
| c2 | 0.7 | 0.03 | 8 | 20 | 0.5 | 20.0 | 0.7 |
| c3 | 1.0 | 0.06 | 3 | 1 | 1.0 | 20.0 | 1.0 |

Native categorical configuration (in addition to candidate parameters):

```json
{
  "catboost": {
    "bagging_temperature": 1,
    "bootstrap_type": "Bayesian",
    "max_ctr_complexity": 1,
    "one_hot_max_size": 32
  },
  "catboost_native_ordered_statistics": true,
  "external_target_encoding": false,
  "lightgbm": {
    "cat_l2": 10,
    "cat_smooth": 20,
    "categorical_feature": "native",
    "max_cat_threshold": 64,
    "min_data_per_group": 20
  },
  "missing_token": "__MISSING__",
  "rare_min_train_rows": 5,
  "rare_token": "__RARE__",
  "unseen_token": "__UNK__",
  "xgboost": {
    "enable_categorical": true,
    "max_cat_threshold": 64,
    "max_cat_to_onehot": 32,
    "tree_method": "hist"
  }
}
```

### Selected configurations

The table gives candidate and frozen round count, shared across the three refit seeds. All 180 per-seed validation thresholds and expanded parameter vectors are in `tables/selected_hyperparameters.csv`.

| Family | Surface | Setting | Candidate | Rounds |
|---|---|---|---|---:|
| catboost | dirt | B1 | c2 | 735 |
| catboost | dirt | M2 | c2 | 399 |
| catboost | dirt | M3 | c1 | 96 |
| catboost | dirt | S1 | c2 | 101 |
| catboost | dirt | S2 | c2 | 1130 |
| catboost | dirt | S3 | c0 | 563 |
| catboost | dirt | S4 | c2 | 1150 |
| catboost | dirt | S5 | c2 | 1057 |
| catboost | dirt | S6 | c1 | 1070 |
| catboost | dirt | S7 | c1 | 828 |
| catboost | turf | B1 | c3 | 283 |
| catboost | turf | M2 | c1 | 837 |
| catboost | turf | M3 | c2 | 80 |
| catboost | turf | S1 | c3 | 845 |
| catboost | turf | S2 | c2 | 1192 |
| catboost | turf | S3 | c3 | 1166 |
| catboost | turf | S4 | c2 | 950 |
| catboost | turf | S5 | c2 | 1170 |
| catboost | turf | S6 | c2 | 1195 |
| catboost | turf | S7 | c0 | 628 |
| lightgbm | dirt | B1 | c0 | 141 |
| lightgbm | dirt | M2 | c3 | 250 |
| lightgbm | dirt | M3 | c0 | 34 |
| lightgbm | dirt | S1 | c2 | 120 |
| lightgbm | dirt | S2 | c1 | 749 |
| lightgbm | dirt | S3 | c2 | 50 |
| lightgbm | dirt | S4 | c2 | 493 |
| lightgbm | dirt | S5 | c1 | 932 |
| lightgbm | dirt | S6 | c1 | 640 |
| lightgbm | dirt | S7 | c3 | 274 |
| lightgbm | turf | B1 | c1 | 73 |
| lightgbm | turf | M2 | c0 | 462 |
| lightgbm | turf | M3 | c1 | 16 |
| lightgbm | turf | S1 | c2 | 130 |
| lightgbm | turf | S2 | c1 | 831 |
| lightgbm | turf | S3 | c2 | 28 |
| lightgbm | turf | S4 | c2 | 394 |
| lightgbm | turf | S5 | c1 | 651 |
| lightgbm | turf | S6 | c2 | 393 |
| lightgbm | turf | S7 | c3 | 155 |
| xgboost | dirt | B1 | c0 | 240 |
| xgboost | dirt | M2 | c0 | 199 |
| xgboost | dirt | M3 | c0 | 19 |
| xgboost | dirt | S1 | c2 | 84 |
| xgboost | dirt | S2 | c1 | 764 |
| xgboost | dirt | S3 | c2 | 80 |
| xgboost | dirt | S4 | c2 | 371 |
| xgboost | dirt | S5 | c1 | 988 |
| xgboost | dirt | S6 | c2 | 401 |
| xgboost | dirt | S7 | c3 | 396 |
| xgboost | turf | B1 | c1 | 161 |
| xgboost | turf | M2 | c0 | 448 |
| xgboost | turf | M3 | c0 | 31 |
| xgboost | turf | S1 | c2 | 126 |
| xgboost | turf | S2 | c0 | 1034 |
| xgboost | turf | S3 | c2 | 33 |
| xgboost | turf | S4 | c1 | 626 |
| xgboost | turf | S5 | c3 | 1172 |
| xgboost | turf | S6 | c0 | 828 |
| xgboost | turf | S7 | c0 | 155 |

The classification threshold maximizes validation F1 on 0.01, 0.02, ..., 0.99. Ties choose closest to 0.5, then the larger threshold. Three selections per race always use ranked probabilities independently of that threshold.

## Graph construction, software, and coverage

Pedigree records were collected retrospectively; the availability of each record at its historical graph cutoff was not independently verified. Earlier classifier-training entries may also use topology contributed by later training-period roots, so the original graph is not a rolling historical simulation. The training-root graph has 62,270 distinct focal horses, 93,474 nodes, and 173,412 deduplicated edges. The full-cohort collection (116,266 nodes / 217,704 edges) is not the graph used for fitting. Edges are stored parent-to-child but projected to undirected walks. This projection and ancestor averaging do not preserve explicit sire/dam role or directional generation information.

The 64-dimensional representation was frozen after resource-pilot review before production, not optimized on predictive outcomes. Node2Vec uses PecanPy 2.0.9 SparseOTF, Gensim 4.4.0, 10 walks/node, 80 steps/walk, window 5, 5 epochs, p=1, q=0.5, and 4 workers. K-means uses scikit-learn 1.6.1, K-means++ initialization, 10 initializations, maximum 300 iterations, and raw vectors of training focal horses only. Candidates k=4,8,16,32,64 were compared by silhouette on a fixed NumPy default_rng(0) sample of 2,000 training horses. Highest finite silhouette selects k; exact ties favor smaller k. k=64 won at 0.090921 and was retained for seeds 1/2. This is not evidence for 64 natural lineages. Four-worker skip-gram is not bitwise deterministic.

Training versions: Python 3.12.13, NumPy 1.26.4, SciPy 1.13.1, pandas 2.2.3, scikit-learn 1.6.1, XGBoost 3.0.2, LightGBM 4.6.0, CatBoost 1.2.8. Graph additionally uses Numba 0.61.2. Validation SHAP uses SHAP 0.46.0. New reporting uses Python 3.14.3, NumPy 2.4.4, scikit-learn 1.8.0; metric parity and weighted-AP checks accompany the reporting source.

`tables/graph_coverage.csv` gives exact entry counts by split/surface and generation. Direct coverage is followed by averaging unique embedded ancestors at the first available generation (1, then 2, then 3). Available vectors are assigned to the saved seed-specific centroids. Graph fitting never uses later roots or outcomes.

## Metric and uncertainty definitions

Primary AP is non-interpolated average precision, not trapezoidal PR-AUC. Point estimates average metrics across the three seeds, rather than averaging probabilities first. Race ranking is descending probability, ascending horse number, then lexicographic horse ID. All eligible entrants, including documented non-finishers, stay in race-level evaluation. Each race selects exactly three entries. Mean Hit@3 divides correct selections by three even when official dead heats create more than three positive labels. At-least-one/two-hit rates use all eligible races. Exact-set rate uses only races with exactly three positives; >3-positive races are reported separately.

Binary relevance is the official top-three label. Graded relevance is max(0,4-finish). DCG@3 sums relevance/log2(rank+1), using direct gains, not 2^relevance-1. NDCG divides by the ideal DCG in the same race; zero ideal DCG gives zero. All-race binary NDCG uses all eligible races. Complete-finish NDCG requires a non-null positive official finish for every eligible entry, without requiring exactly three positives. Shared official positions have the same relevance.

For each surface, sample the same number of races as observed with replacement, 2,000 times using NumPy default_rng(20260913). Use identical race multiplicities across settings and seeds; repeated draws retain their weight. Compute each metric within each seed, subtract paired settings, then average the three differences within each replicate. Intervals are the 2.5th/97.5th percentiles of that averaged distribution, conditional on the three fitted seed pairs. No seed resampling, formal hypothesis tests, or multiplicity adjustment is used. Seed SDs are separate. Week-block sensitivities supplement but do not remove repeated-horse dependence.

For new exploratory history analyses, sample all races in a surface and then select subgroup entries. The same draws are reused across subgroups, permitting a paired difference of AP increments versus 3+. Report all 0/1/2/3+ strata and the pooled <3 stratum; the latter AP is recomputed on the pooled entries, not averaged from stratum APs. Counts and positive prevalence are in `tables/history_subgroup_counts.csv`. Sixty paired AP increments and forty-eight subgroup-minus-3+ differences appear in the exploratory tables. This extends the post-test reporting; it is not a predeclared confirmatory test. No predictions or model choices changed.

## Random-baseline correction and positive-label reconciliation

The previous release passed tied training-prevalence scores into the generic race evaluator, which ranked by horse number. Its Hit@3 bootstrap instead used the intended SHA-256 selection. Therefore its point estimate and interval described different rankings. The correction retains prevalence for entry AP, Brier, and ROC-AUC, but supplies within-race ordinal scores from the frozen hash order for every ranking metric. The hash is SHA256(`b0-v1|race_id|horse_id|horse_number`)[:16], ascending, then horse number and ID. Exactly three selections per race and agreement between direct hit counting and the generic evaluator are verified. A regression fixture fails under the previous code.

Corrected deterministic B0 Hit@3: turf 0.2430683919, dirt 0.2161126644. Analytic random-ranking expectations (mean positive fraction across races) remain turf 0.2432507692 and dirt 0.2188228904 and are the primary random reference. Corrected NDCG, selection rates, and intervals are in `tables/b0_metrics_corrected.csv`; this supersedes the older B0 table only. The 180 fitted-model metrics and main contrasts remain unchanged.

Across all 51,313 races, 51,180 have 3 positives, 132 have 4, and one has 5: 51,180*3 + 132*4 + 1*5 = 154,073. This is exactly 134 above 3*51,313. All labels were checked against official normalized finishes in the local source; there were zero mismatches. The label convention includes every official rank <=3, including shared positions; it is not an error to have more than three positives.

## SHAP definition

Models are selected on validation mean AP, then Hit@3, lower round, and fixed family order. Explain the selected seed-0 training-only S6/S7 refits: CatBoost turf S6/S7, LightGBM dirt S6, CatBoost dirt S7. Shuffle whole validation races using seed 20260913 until at least 2,000 entries are reached: turf 2,010 entries/154 races; dirt 2,003/142. The same sample serves both settings within a surface. TreeExplainer uses tree_path_dependent and raw output. The reference is stored training path counts; there is no separate background matrix. Group importance is mean_i(abs(sum_{j in G} phi_ij)), allowing within-group cancellation. It is not sum_j(mean_i(abs(phi_ij))). Zero group importance for coverage descriptors does not follow solely from the absence of missing representations.

## Literature verification and limits of access

Descriptions below are limited to positively confirmed study characteristics. We do not infer that a method was absent from unavailable text.

- [Oda and Onogi, publisher full text](https://onlinelibrary.wiley.com/doi/full/10.1111/jbg.12822): JRA turf/good ground at four courses; pedigree breeding values estimated using historical data through 2010 and evaluated for later horses. Targets are velocity and graded-winner status. Full methods checked.
- [Pudaruth et al., original full text](https://research.ijcaonline.org/volume72/number5/pxc3889048.pdf): nine factors include odds and exactly three recent performances; 240 calibration races, 24 evaluation races. Bibliography completed: volume 72, issue 5, pages 39-42. Full methods checked.
- [Silverman and Suchard, original journal](https://www.ubplj.org/index.php/jpm/article/view/595) and [author abstract indexed in RePEc](https://ideas.repec.org/a/buc/jpredm/v7y2013i1p43-52.html): Hong Kong data, four development years, held-out year, regularized conditional logistic regression with frailty exploiting winning dividends. Subscription full text was not accessible; claims are confined to the author abstract. Pages 43-52 added.
- [Gulum, university thesis record and author abstract](https://ir.library.louisville.edu/etd/2953/): 2015-2017 race data, directed horse graph features, neural-network/logistic comparisons, paired-horse tests, and Kentucky Derby checks. Full thesis download was unavailable; no claim about unreported pedigree or validation procedures is made.
- [Gupta and Singh, publisher author abstract](https://ijritcc.org/index.php/ijritcc/article/view/8119): Indian racing data 2017-2019, classifiers and sampling comparisons including SMOTE. Full-text endpoint was inaccessible in this pass; the table does not make an unsupported claim that chronological validation or pedigree was absent.
- [An et al., publisher abstract/highlights](https://www.sciencedirect.com/science/article/pii/S1090023326000638): retrospective selected veterinary cohort, pedigree/surgery variables, 75 treated horses and 257 maternal siblings, CatBoost/XGBoost and SHAP. Full text was not accessible; no strong absence claim is made.

This is a comparison of the reviewed studies, not a systematic-review claim or a universal first-study claim. Full-text verification of the four access-limited studies remains desirable if stronger comparisons are planned.

## Data provenance and reproducibility boundaries

Identifiers in legacy *_jbis_id columns are Netkeiba IDs; the Netkeiba-to-JBIS mapping table is empty. The prior collection repaired calendar-date extraction, odds parsing, race classes/statuses, jump classification, and duplicated spotlight rows. These implementation-history details were removed from bibliography entries and condensed in the main article. The study observes weekend starts only; history is left-censored at 2010 and excludes uncollected weekday, NAR, and overseas starts. History zero is not necessarily a debut. Field size reflects eligible entrants and requires a final-field availability assumption.

Market experiments use retrospective final odds/popularity. No timestamped pre-start deployment horizon was tested. Separate turf/dirt classifiers are a study design; without a pooled comparator they cannot be claimed superior. Undirected embeddings and ancestor averaging cannot be read as causal genetic effects.

The package includes aggregate tables, figure source, reporting/evaluation code, feature/graph/training code, configurations and provenance. Tables and figures can be regenerated from included aggregate records. Rebuilding features requires original race/pedigree data; graph/model fitting requires these inputs and the documented environments. Rerunning prediction-level evaluation additionally requires the private frozen prediction files. Databases, prediction rows, fitted model binaries, credentials and category vocabularies derived from private data are not redistributed. This is reporting reproducibility plus documented experiment implementation, not self-contained full-experiment reproducibility.

Public hosting and a persistent URL/DOI remain pending. The target ISRITI venue page states a fee threshold beyond six pages, but its maximum and supplement policy require confirmation. No publication or submission was performed.
