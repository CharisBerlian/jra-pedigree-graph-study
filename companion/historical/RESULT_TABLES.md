# Aggregate result tables

## model_summaries_extended.csv

| family | surface | setting | entry_average_precision_mean | race_mean_hit_at_3_mean |
|---|---|---|---|---|
| xgboost | turf | B1 | 0.567909 | 0.536318 |
| xgboost | turf | S1 | 0.342822 | 0.321113 |
| xgboost | turf | S2 | 0.470998 | 0.458000 |
| xgboost | turf | S3 | 0.431265 | 0.435750 |
| xgboost | turf | S4 | 0.463591 | 0.461012 |
| xgboost | turf | S5 | 0.469969 | 0.460875 |
| xgboost | turf | S6 | 0.463090 | 0.459951 |
| xgboost | turf | M2 | 0.579203 | 0.540049 |
| xgboost | turf | M3 | 0.570839 | 0.538304 |
| xgboost | turf | S7 | 0.578085 | 0.538167 |
| xgboost | dirt | B1 | 0.550713 | 0.526147 |
| xgboost | dirt | S1 | 0.292070 | 0.296814 |
| xgboost | dirt | S2 | 0.456108 | 0.463360 |
| xgboost | dirt | S3 | 0.442218 | 0.452556 |
| xgboost | dirt | S4 | 0.455968 | 0.461358 |
| xgboost | dirt | S5 | 0.457507 | 0.463878 |
| xgboost | dirt | S6 | 0.454066 | 0.461565 |
| xgboost | dirt | M2 | 0.558190 | 0.528494 |
| xgboost | dirt | M3 | 0.550599 | 0.528459 |
| xgboost | dirt | S7 | 0.557428 | 0.528701 |
| lightgbm | turf | B1 | 0.566760 | 0.536010 |
| lightgbm | turf | S1 | 0.343125 | 0.324297 |
| lightgbm | turf | S2 | 0.473761 | 0.458547 |
| lightgbm | turf | S3 | 0.434826 | 0.435442 |
| lightgbm | turf | S4 | 0.465810 | 0.463613 |
| lightgbm | turf | S5 | 0.472394 | 0.465873 |
| lightgbm | turf | S6 | 0.466283 | 0.464503 |
| lightgbm | turf | M2 | 0.578387 | 0.538680 |
| lightgbm | turf | M3 | 0.570767 | 0.538577 |
| lightgbm | turf | S7 | 0.577523 | 0.538646 |
| lightgbm | dirt | B1 | 0.549555 | 0.526872 |
| lightgbm | dirt | S1 | 0.293462 | 0.297159 |
| lightgbm | dirt | S2 | 0.458501 | 0.464982 |
| lightgbm | dirt | S3 | 0.438970 | 0.448448 |
| lightgbm | dirt | S4 | 0.457893 | 0.461358 |
| lightgbm | dirt | S5 | 0.460075 | 0.462739 |
| lightgbm | dirt | S6 | 0.456706 | 0.461358 |
| lightgbm | dirt | M2 | 0.557723 | 0.529150 |
| lightgbm | dirt | M3 | 0.550103 | 0.526216 |
| lightgbm | dirt | S7 | 0.557119 | 0.528252 |
| catboost | turf | B1 | 0.567566 | 0.536489 |
| catboost | turf | S1 | 0.337343 | 0.322551 |
| catboost | turf | S2 | 0.472532 | 0.460122 |
| catboost | turf | S3 | 0.452511 | 0.446498 |
| catboost | turf | S4 | 0.466068 | 0.461731 |
| catboost | turf | S5 | 0.468413 | 0.462758 |
| catboost | turf | S6 | 0.466846 | 0.462723 |
| catboost | turf | M2 | 0.579284 | 0.539467 |
| catboost | turf | M3 | 0.574622 | 0.537790 |
| catboost | turf | S7 | 0.578341 | 0.538030 |
| catboost | dirt | B1 | 0.551117 | 0.526492 |
| catboost | dirt | S1 | 0.289368 | 0.294501 |
| catboost | dirt | S2 | 0.452254 | 0.460288 |
| catboost | dirt | S3 | 0.440781 | 0.451831 |
| catboost | dirt | S4 | 0.455882 | 0.462842 |
| catboost | dirt | S5 | 0.453027 | 0.462048 |
| catboost | dirt | S6 | 0.452500 | 0.461876 |
| catboost | dirt | M2 | 0.558209 | 0.528563 |
| catboost | dirt | M3 | 0.554356 | 0.526457 |
| catboost | dirt | S7 | 0.558918 | 0.527528 |

## exploratory_history_contrasts.csv

| family | surface | history | contrast | delta | ci_low | ci_high |
|---|---|---|---|---|---|---|
| xgboost | turf | 0 | S6-S2 | 0.029863 | 0.016355 | 0.043785 |
| xgboost | turf | 1 | S6-S2 | -0.012899 | -0.024545 | -0.002185 |
| xgboost | turf | 2 | S6-S2 | -0.006938 | -0.019101 | 0.004794 |
| xgboost | turf | 3+ | S6-S2 | -0.007264 | -0.011378 | -0.003146 |
| xgboost | turf | <3 | S6-S2 | -0.005081 | -0.012866 | 0.002515 |
| xgboost | turf | 0 | S7-M2 | 0.000490 | -0.003069 | 0.003659 |
| xgboost | turf | 1 | S7-M2 | -0.002668 | -0.006540 | 0.001023 |
| xgboost | turf | 2 | S7-M2 | 0.000653 | -0.004647 | 0.005881 |
| xgboost | turf | 3+ | S7-M2 | -0.001541 | -0.003390 | 0.000314 |
| xgboost | turf | <3 | S7-M2 | -0.000605 | -0.002945 | 0.001620 |
| lightgbm | turf | 0 | S6-S2 | 0.031088 | 0.015603 | 0.046847 |
| lightgbm | turf | 1 | S6-S2 | -0.011534 | -0.023045 | -0.000838 |
| lightgbm | turf | 2 | S6-S2 | -0.005796 | -0.018073 | 0.007510 |
| lightgbm | turf | 3+ | S6-S2 | -0.006861 | -0.011192 | -0.002565 |
| lightgbm | turf | <3 | S6-S2 | -0.003353 | -0.011237 | 0.004900 |
| lightgbm | turf | 0 | S7-M2 | 0.000817 | -0.003424 | 0.004599 |
| lightgbm | turf | 1 | S7-M2 | -0.001800 | -0.005659 | 0.001936 |
| lightgbm | turf | 2 | S7-M2 | -0.000537 | -0.006141 | 0.005712 |
| lightgbm | turf | 3+ | S7-M2 | -0.001351 | -0.003246 | 0.000470 |
| lightgbm | turf | <3 | S7-M2 | -0.000226 | -0.002846 | 0.002234 |
| catboost | turf | 0 | S6-S2 | 0.032624 | 0.018345 | 0.047483 |
| catboost | turf | 1 | S6-S2 | -0.007672 | -0.018187 | 0.002209 |
| catboost | turf | 2 | S6-S2 | -0.000443 | -0.011183 | 0.010352 |
| catboost | turf | 3+ | S6-S2 | -0.006337 | -0.009981 | -0.002821 |
| catboost | turf | <3 | S6-S2 | 0.000401 | -0.006449 | 0.007544 |
| catboost | turf | 0 | S7-M2 | -0.001064 | -0.003348 | 0.001393 |
| catboost | turf | 1 | S7-M2 | -0.001840 | -0.004726 | 0.000902 |
| catboost | turf | 2 | S7-M2 | 0.000984 | -0.002822 | 0.004766 |
| catboost | turf | 3+ | S7-M2 | -0.000947 | -0.002154 | 0.000340 |
| catboost | turf | <3 | S7-M2 | -0.000777 | -0.002425 | 0.000850 |
| xgboost | dirt | 0 | S6-S2 | 0.014353 | -0.006633 | 0.034660 |
| xgboost | dirt | 1 | S6-S2 | 0.001660 | -0.009352 | 0.014157 |
| xgboost | dirt | 2 | S6-S2 | -0.001563 | -0.011339 | 0.007625 |
| xgboost | dirt | 3+ | S6-S2 | -0.002199 | -0.005352 | 0.001063 |
| xgboost | dirt | <3 | S6-S2 | -0.000369 | -0.007805 | 0.006812 |
| xgboost | dirt | 0 | S7-M2 | -0.004466 | -0.011042 | 0.002558 |
| xgboost | dirt | 1 | S7-M2 | 0.002428 | -0.003064 | 0.007568 |
| xgboost | dirt | 2 | S7-M2 | 0.001608 | -0.002795 | 0.005876 |
| xgboost | dirt | 3+ | S7-M2 | -0.001228 | -0.002868 | 0.000440 |
| xgboost | dirt | <3 | S7-M2 | 0.000074 | -0.002993 | 0.003287 |
| lightgbm | dirt | 0 | S6-S2 | 0.011240 | -0.009223 | 0.030870 |
| lightgbm | dirt | 1 | S6-S2 | 0.001069 | -0.011062 | 0.013756 |
| lightgbm | dirt | 2 | S6-S2 | 0.000234 | -0.009389 | 0.008096 |
| lightgbm | dirt | 3+ | S6-S2 | -0.002170 | -0.005361 | 0.000826 |
| lightgbm | dirt | <3 | S6-S2 | 0.000121 | -0.007217 | 0.006946 |
| lightgbm | dirt | 0 | S7-M2 | -0.003989 | -0.011230 | 0.003280 |
| lightgbm | dirt | 1 | S7-M2 | 0.000199 | -0.005603 | 0.005990 |
| lightgbm | dirt | 2 | S7-M2 | -0.001068 | -0.005902 | 0.003513 |
| lightgbm | dirt | 3+ | S7-M2 | -0.000559 | -0.002431 | 0.001290 |
| lightgbm | dirt | <3 | S7-M2 | -0.001338 | -0.004768 | 0.001910 |
| catboost | dirt | 0 | S6-S2 | 0.019738 | 0.003601 | 0.035494 |
| catboost | dirt | 1 | S6-S2 | -0.003101 | -0.012217 | 0.007159 |
| catboost | dirt | 2 | S6-S2 | -0.000618 | -0.008880 | 0.006526 |
| catboost | dirt | 3+ | S6-S2 | -0.000591 | -0.003094 | 0.002025 |
| catboost | dirt | <3 | S6-S2 | 0.001843 | -0.003565 | 0.007404 |
| catboost | dirt | 0 | S7-M2 | -0.000456 | -0.004699 | 0.003946 |
| catboost | dirt | 1 | S7-M2 | 0.002900 | -0.000677 | 0.006453 |
| catboost | dirt | 2 | S7-M2 | 0.000995 | -0.002208 | 0.004047 |
| catboost | dirt | 3+ | S7-M2 | 0.000077 | -0.000992 | 0.001251 |
| catboost | dirt | <3 | S7-M2 | 0.001867 | -0.000335 | 0.003792 |

## exploratory_history_interactions.csv

| family | surface | history | contrast | difference_of_deltas | ci_low | ci_high |
|---|---|---|---|---|---|---|
| xgboost | turf | 0 | S6-S2 | 0.037126 | 0.022567 | 0.052256 |
| xgboost | turf | 1 | S6-S2 | -0.005636 | -0.017932 | 0.005970 |
| xgboost | turf | 2 | S6-S2 | 0.000326 | -0.012070 | 0.012945 |
| xgboost | turf | <3 | S6-S2 | 0.002183 | -0.006396 | 0.010565 |
| xgboost | turf | 0 | S7-M2 | 0.002031 | -0.001918 | 0.005539 |
| xgboost | turf | 1 | S7-M2 | -0.001127 | -0.005477 | 0.003070 |
| xgboost | turf | 2 | S7-M2 | 0.002194 | -0.003514 | 0.007841 |
| xgboost | turf | <3 | S7-M2 | 0.000936 | -0.001954 | 0.003738 |
| lightgbm | turf | 0 | S6-S2 | 0.037950 | 0.021866 | 0.054034 |
| lightgbm | turf | 1 | S6-S2 | -0.004673 | -0.016702 | 0.006715 |
| lightgbm | turf | 2 | S6-S2 | 0.001065 | -0.011639 | 0.014951 |
| lightgbm | turf | <3 | S6-S2 | 0.003508 | -0.005283 | 0.012409 |
| lightgbm | turf | 0 | S7-M2 | 0.002168 | -0.002472 | 0.006356 |
| lightgbm | turf | 1 | S7-M2 | -0.000449 | -0.004864 | 0.003712 |
| lightgbm | turf | 2 | S7-M2 | 0.000814 | -0.004840 | 0.007244 |
| lightgbm | turf | <3 | S7-M2 | 0.001125 | -0.002016 | 0.004194 |
| catboost | turf | 0 | S6-S2 | 0.038961 | 0.024144 | 0.054153 |
| catboost | turf | 1 | S6-S2 | -0.001335 | -0.012382 | 0.009025 |
| catboost | turf | 2 | S6-S2 | 0.005894 | -0.005392 | 0.017016 |
| catboost | turf | <3 | S6-S2 | 0.006738 | -0.001053 | 0.014325 |
| catboost | turf | 0 | S7-M2 | -0.000117 | -0.002685 | 0.002719 |
| catboost | turf | 1 | S7-M2 | -0.000893 | -0.003994 | 0.002013 |
| catboost | turf | 2 | S7-M2 | 0.001931 | -0.001897 | 0.005959 |
| catboost | turf | <3 | S7-M2 | 0.000170 | -0.001886 | 0.002207 |
| xgboost | dirt | 0 | S6-S2 | 0.016552 | -0.003624 | 0.037099 |
| xgboost | dirt | 1 | S6-S2 | 0.003859 | -0.008114 | 0.016428 |
| xgboost | dirt | 2 | S6-S2 | 0.000636 | -0.009836 | 0.009940 |
| xgboost | dirt | <3 | S6-S2 | 0.001830 | -0.006064 | 0.009445 |
| xgboost | dirt | 0 | S7-M2 | -0.003238 | -0.009911 | 0.004007 |
| xgboost | dirt | 1 | S7-M2 | 0.003656 | -0.002237 | 0.008938 |
| xgboost | dirt | 2 | S7-M2 | 0.002836 | -0.001808 | 0.007644 |
| xgboost | dirt | <3 | S7-M2 | 0.001302 | -0.002147 | 0.004855 |
| lightgbm | dirt | 0 | S6-S2 | 0.013410 | -0.007030 | 0.032750 |
| lightgbm | dirt | 1 | S6-S2 | 0.003239 | -0.009776 | 0.016417 |
| lightgbm | dirt | 2 | S6-S2 | 0.002404 | -0.007858 | 0.010556 |
| lightgbm | dirt | <3 | S6-S2 | 0.002291 | -0.005386 | 0.009727 |
| lightgbm | dirt | 0 | S7-M2 | -0.003430 | -0.010978 | 0.004194 |
| lightgbm | dirt | 1 | S7-M2 | 0.000758 | -0.005601 | 0.006596 |
| lightgbm | dirt | 2 | S7-M2 | -0.000509 | -0.005667 | 0.004312 |
| lightgbm | dirt | <3 | S7-M2 | -0.000778 | -0.004764 | 0.002999 |
| catboost | dirt | 0 | S6-S2 | 0.020329 | 0.003564 | 0.036477 |
| catboost | dirt | 1 | S6-S2 | -0.002510 | -0.012017 | 0.007879 |
| catboost | dirt | 2 | S6-S2 | -0.000027 | -0.008932 | 0.007639 |
| catboost | dirt | <3 | S6-S2 | 0.002434 | -0.003642 | 0.008512 |
| catboost | dirt | 0 | S7-M2 | -0.000532 | -0.004887 | 0.004067 |
| catboost | dirt | 1 | S7-M2 | 0.002823 | -0.000994 | 0.006544 |
| catboost | dirt | 2 | S7-M2 | 0.000919 | -0.002509 | 0.004212 |
| catboost | dirt | <3 | S7-M2 | 0.001790 | -0.000763 | 0.003986 |

## history_subgroup_counts.csv

| surface | history | entries | positives | prevalence |
|---|---|---|---|---|
| turf | 0 | 5917 | 1287 | 0.217509 |
| turf | 1 | 4577 | 1048 | 0.228971 |
| turf | 2 | 3836 | 852 | 0.222106 |
| turf | 3+ | 29147 | 6559 | 0.225032 |
| turf | <3 | 14330 | 3187 | 0.222401 |
| dirt | 0 | 4004 | 683 | 0.170579 |
| dirt | 1 | 4629 | 830 | 0.179304 |
| dirt | 2 | 4341 | 872 | 0.200875 |
| dirt | 3+ | 32686 | 7283 | 0.222817 |
| dirt | <3 | 12974 | 2385 | 0.183829 |

## graph_coverage.csv

| split | surface | entries | direct | parent | generation2 | generation3 | missing |
|---|---|---|---|---|---|---|---|
| train | dirt | 277207 | 277207 | 0 | 0 | 0 | 0 |
| train | turf | 271549 | 271549 | 0 | 0 | 0 | 0 |
| validation | dirt | 45323 | 24398 | 20683 | 242 | 0 | 0 |
| validation | turf | 42783 | 21708 | 20872 | 202 | 1 | 0 |
| test | dirt | 45660 | 5506 | 39107 | 1039 | 8 | 0 |
| test | turf | 43477 | 5641 | 37225 | 596 | 15 | 0 |

## b0_metrics_corrected.csv

| surface | expected_random_hit_at_3 | race_mean_hit_at_3 | ndcg_binary_all_races |
|---|---|---|---|
| turf | 0.243251 | 0.243068 | 0.243179 |
| dirt | 0.218823 | 0.216113 | 0.215869 |
