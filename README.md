# Evaluating Pedigree Graph Features Beyond Recent Form and Market Information in Horse Racing

Research data: [charischerw/jra-pedigree-graph-study-2010-2025](https://www.kaggle.com/datasets/charischerw/jra-pedigree-graph-study-2010-2025) (paper release v1).
Code and manuscript: [https://github.com/CharisBerlian/jra-pedigree-graph-study](https://github.com/CharisBerlian/jra-pedigree-graph-study).

Current paper: [v1.0.2-paper](https://github.com/CharisBerlian/jra-pedigree-graph-study/releases/tag/v1.0.2-paper), five A4 pages. Figure 2 is beside the historical results on page 4; references are in first-citation order. The [manuscript review](docs/MANUSCRIPT_REVIEW.md) records changes, numerical checks and remaining boundaries.

This release contains the historical experiment and the October 2026 focused four-arm study. The fixed title is retained. The four-arm comparisons use H, Gmixed, Gancestor and P, with no market predictors. Historical market comparisons use retrospectively recorded final odds/popularity. The test cohort had previously been examined, so the new comparisons are exploratory.

## Contents

- `paper/`: current five-page IEEE manuscript and its LaTeX sources, figures and licensed fonts.
- `docs/TABLE_LAYOUT_CHECKS.json`: numerical parity checks for the compact table layout in v1.0.1-paper.
- `docs/MANUSCRIPT_REVIEW_CHECKS.json`: checks for all four tables, both figures, reported numerical claims, citation order and the v1.0.2-paper PDF.
- `companion/focused/`: all 96 validation candidates, 24 selections, 72 test seed scores, 24 arm summaries and 36 AP/Hit@3 contrasts; frozen scientific source and synthetic fixtures.
- `companion/historical/`: historical aggregate results, source and reporting generators.
- `kaggle/`: historical graph, training, inference, evaluation and test sources.
- `output/revision_2026-10-29/science/`: focused preparation, fitting orchestration, feature, inference and evaluation source in its original relative layout.
- `docs/`: input schemas, reconstruction instructions and release scope.
- `notebooks/`: a small public example displaying the supplied results.

## Quick checks

Use Python with NumPy installed. These synthetic tests use the actual feature and metric implementation on fabricated examples:

```sh
python -B companion/focused/fixture/test_core.py
python -B companion/focused/fixture/test_metrics.py
```

To regenerate the focused manuscript table and figure from the released aggregate results, run from `companion/focused/reporting/` (the output directory must be new):

```sh
python -B generate_focused_ieee.py --report ../test/evaluation_report.json --seed-metrics ../test/seed_metrics.csv --arm-metrics ../test/arm_metrics.csv --contrasts ../test/contrasts.csv --out ../../../../regenerated-focused
```

To regenerate the current column-width historical Figure 2, run from the repository root:

```sh
python -B companion/focused/reporting/render_historical_contrasts.py --source companion/historical/tables/primary_contrasts.csv --out regenerated-historical-figure.tex
```

This also writes a JSON coordinate audit. The original wide historical renderer is retained as a dated artifact under `companion/historical/code/`.

## Data, models and prediction replay

Download version 1 of the linked Kaggle dataset. Its README maps every archive to the original relative workspace paths. Extract archives at the repository root, preserving paths. The dataset includes the race/pedigree inputs, prepared train/validation/test views, original three-seed graph spaces, parent/category maps, fitted artifacts and complete test predictions. `docs/REPRODUCING.md` explains the levels of reproduction and environment requirements. This release does not claim a fresh external retraining replication.

The historical scientific environment used Python 3.12.13. The focused scientific environment used Python 3.13.15; exact package pins are in `companion/focused/config/EXPERIMENT_SPEC.json`. Four-worker graph training is not bitwise deterministic. The frozen source retains execution checks and original Kaggle/layout assumptions.

## Licenses and attribution

Research data and aggregate result tables: CC BY 4.0, as confirmed by the depositing author. Attribute the study and acknowledge Netkeiba as the race/pedigree record source. Source code authored for this study: MIT. Included IEEEtran and font files retain their original notices and licenses; these are not relicensed by the repository license. The depositor confirmed permission for public redistribution and CC BY 4.0 reuse. No account credentials, temporary signed links or operational payload-bearing wrappers are distributed.

Earlier companion documents are preserved as dated methodological evidence. Their descriptions of private/local access and pending work describe their historical snapshots and are superseded by this release's README and public dataset inventory. Their scientific caveats remain applicable.

## Published Kaggle layout

Kaggle expands both ZIP and GZIP files during ingestion and adds each ZIP basename as a top-level directory. The published dataset has 4,921 files (about 12.74 GB expanded); source `.csv.gz` files appear as `.csv`. The archive/member inventories record the original uploaded containers, whose compressed-byte hashes cannot be applied directly to those expanded CSVs. Preserve the published CSV values and original split assignments. The example notebook supports this expanded layout and recomputes all 72 point scores.

For the byte-exact, self-contained focused replay packet, download [focused-prediction-replay.zip](https://github.com/CharisBerlian/jra-pedigree-graph-study/releases/download/v1.0.0-paper/focused-prediction-replay.zip) from the GitHub release. That asset preserves the original GZIP members and all hash bindings. Extract it into an empty directory and use its recorded Python 3.13.15 environment to run `verify.py`. Its earlier private-access labels describe the original snapshot and are superseded by this authorized CC BY 4.0 public release.

For a new full training run, restore the workspace paths by removing the top-level archive basenames, rebuild/export inputs with the provided source and create new run metadata. Do not substitute recompressed CSV files under original compressed-file hashes or disable frozen provenance guards. Publishing the data does not claim bitwise-identical retraining.
