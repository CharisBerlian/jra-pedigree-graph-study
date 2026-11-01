# Aggregate and source companion — local only

This companion contains no raw race/pedigree records, identified per-entry scores or outcomes, graph vectors, category/parent maps, fitted model weights, private payloads or access credentials. It is not a public archive, a guaranteed reviewer-access route or a self-contained retraining release. Source and data rights, venue delivery and anonymity requirements require separate confirmation. No prediction-level or full-data reproduction is claimed from this folder alone.

## New four-arm study

- `focused/config/` contains the reviewed pre-fit specification and protocol snapshots. Their original status wording is historical, not the current execution ledger. The 24 cells, 96 candidate fits and 72 selected refits were frozen under the model-manifest hash recorded in the validation and test reports.
- `focused/validation/` contains all 96 candidate rows, all 24 selected settings and cap diagnostics. Curves reaching 1,200 rounds are distinguished from the selected best/refit rounds.
- `focused/test/` contains the verified aggregate evaluation report, all 72 seed-specific score rows, 24 arm/family/surface score summaries, and all 36 predeclared AP/Hit@3 contrast rows. Reported intervals are 2,000-draw paired whole-race percentile intervals, conditional on the fitted seeds and unadjusted for multiple comparisons. The test was previously inspected, making the new findings exploratory.
- `focused/cohort/` contains the verified six-view graph status/depth/ancestor-count distributions, complete test race and entry denominators, parent-ID availability and zero-prior counts, and the separately checked train/validation feasibility counts. These are label-free aggregate features; common construction does not equate coverage or distributions across splits.
- `focused/frozen_source/` contains byte-exact copies of the 23 source files bound to the reviewed model freeze; additional selected pre-fit sources, the source-level synthetic fixture v5 and reporting generators are included. The fixture tests actual feature/metric source behavior using fabricated records, not scientific training or private predictions. From `focused/fixture/`, `python -B test_core.py` and `python -B test_metrics.py` require NumPy.

To regenerate the new main-paper table and figure from the aggregate outputs, from `focused/reporting/` use a fresh output directory whose parent exists:

```text
python -B generate_focused_ieee.py --report ../test/evaluation_report.json --seed-metrics ../test/seed_metrics.csv --arm-metrics ../test/arm_metrics.csv --contrasts ../test/contrasts.csv --out <new-output-directory>
```

The generator checks file hashes and exact row keys, means, seed SDs, contrast points and coverage totals. Compare the resulting TeX files to the included paper inputs. This regenerates displays from aggregate results; it is not independent evaluation of predictions. The complete primary and secondary Hit@3 intervals remain in `focused/test/contrasts.csv` even though the figure shows AP.

## Historical context

`historical/tables/` retains the complete historical seed/summary, parent-ID, market, graph coverage and exploratory outputs. The original A–F score table and full prior contrast figure are retained as reference artifacts. Final odds and popularity are retrospective and enter together; the new four-arm experiment does not test its ancestry or parent comparator beyond market information. `b0_metrics_corrected.csv` supersedes older random-ranking releases; historical fitted-model results and selection were not altered.

Four scripts in `historical/code/` regenerate the historical score table, coverage table, contrast figure and `RESULT_TABLES.md` from aggregate files. Use a disposable copy of `historical/` with an empty sibling `paper/figures/` directory; the generated outputs are checked against `historical/reference_artifacts/`. The historical input contract and accompanying scripts describe additional authorized prerequisites and stage commands; they do not supply the repaired data, graph bundle, original fitted models or prediction rows.

One historical feature-building source has a public web address removed from a comment in this package to keep the companion free of addresses. `historical/COMMENT_SANITIZATION.json` records both byte hashes and confirms its Python syntax tree is unchanged; the original source was preserved. All model-freeze-bound scientific/evaluation sources included under `focused/frozen_source/` remain byte-exact.

## Prediction-level checks and limitations

The 72 focused test prediction files were checked against complete rosters before outcomes were opened; an internal independent calculation checked all 72 AP and Hit@3 point scores and aggregate arithmetic. A separate restricted original-identifier packet was replay-checked in the pinned scientific environment, but is not included here and does not establish external reviewer access. Historical post-test subgroup point checks cover all 24 rows, while only one of 24 subgroup intervals was independently rerun. Full training still requires the original data, source repair, model/graph artifacts or reconstruction, appropriate runtime and access permissions. The historical scientific runtime and new pinned runtime differ, and local synthetic checks in another runtime do not prove end-to-end replication.
