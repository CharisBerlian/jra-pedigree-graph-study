# Reproduction and scope

Current paper release: v1.0.2-paper, 1 November 2026. Data: https://www.kaggle.com/datasets/charischerw/jra-pedigree-graph-study-2010-2025 (version 1). The original exact replay packet remains attached to v1.0.0-paper.

1. Aggregate reporting: regenerate tables and plots from `companion/` using the README commands.
2. Synthetic implementation checks: run both fixture scripts. These are fabricated examples, separate from the real data.
3. Real focused prediction replay: the Kaggle `focused-prediction-replay.zip` contains 72 prediction files, all 89,137 keyed outcomes, exact evaluation source, references and `verify.py`. Extract it into a separate directory and run `python -B verify.py`. The archive preserves the original packet bytes for verification; its earlier private-access wording is historical, superseded by the public release and CC BY 4.0 authorization.
4. Frozen model inference and full training: the dataset supplies the original input views and fitted artifacts. The scripts retain their original Kaggle runtime and hash checks. The historical execution map under `docs/historical/` records the stage commands; the focused sources/spec record its four-arm experiment. Restore original paths using the dataset inventory. Use the recorded scientific versions; do not silently disable guards or substitute a new dataset under an old manifest.

The release has been packaged from frozen artifacts. The complete scientific training run has not been rerun as part of publication. Report any new execution as a replication attempt with its own environment and outputs. Test reuse, repeated horses, unadjusted intervals and historical graph/market timing limitations remain unchanged.

Source access was confirmed by the depositing author. Operational transport wrappers, expired signed URLs, credentials and failed partial transfers are excluded. Rejected tuning model binaries are included where present in the frozen model tree; all candidate aggregate evidence is in the companion. A fresh website scrape is not an exact replacement for these frozen inputs.

## Published Kaggle layout

Kaggle expands both ZIP and GZIP files during ingestion and adds each ZIP basename as a top-level directory. The published dataset has 4,921 files (about 12.74 GB expanded); source `.csv.gz` files appear as `.csv`. The archive/member inventories record the original uploaded containers, whose compressed-byte hashes cannot be applied directly to those expanded CSVs. Preserve the published CSV values and original split assignments. The example notebook supports this expanded layout and recomputes all 72 point scores.

For the byte-exact, self-contained focused replay packet, download [focused-prediction-replay.zip](https://github.com/CharisBerlian/jra-pedigree-graph-study/releases/download/v1.0.0-paper/focused-prediction-replay.zip) from the GitHub release. That asset preserves the original GZIP members and all hash bindings. Extract it into an empty directory and use its recorded Python 3.13.15 environment to run `verify.py`. Its earlier private-access labels describe the original snapshot and are superseded by this authorized CC BY 4.0 public release.

For a new full training run, restore the workspace paths by removing the top-level archive basenames, rebuild/export inputs with the provided source and create new run metadata. Do not substitute recompressed CSV files under original compressed-file hashes or disable frozen provenance guards. Publishing the data does not claim bitwise-identical retraining.
