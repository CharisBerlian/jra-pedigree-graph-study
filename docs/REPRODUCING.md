# Reproduction and scope

Paper release: v1.0.0-paper, 1 November 2026. Data: https://www.kaggle.com/datasets/charischerw/jra-pedigree-graph-study-2010-2025.

1. Aggregate reporting: regenerate tables and plots from `companion/` using the README commands.
2. Synthetic implementation checks: run both fixture scripts. These are fabricated examples, separate from the real data.
3. Real focused prediction replay: the Kaggle `focused-prediction-replay.zip` contains 72 prediction files, all 89,137 keyed outcomes, exact evaluation source, references and `verify.py`. Extract it into a separate directory and run `python -B verify.py`. The archive preserves the original packet bytes for verification; its earlier private-access wording is historical, superseded by the public release and CC BY 4.0 authorization.
4. Frozen model inference and full training: the dataset supplies the original input views and fitted artifacts. The scripts retain their original Kaggle runtime and hash checks. The historical execution map under `docs/historical/` records the stage commands; the focused sources/spec record its four-arm experiment. Restore original paths using the dataset inventory. Use the recorded scientific versions; do not silently disable guards or substitute a new dataset under an old manifest.

The release has been packaged from frozen artifacts. The complete scientific training run has not been rerun as part of publication. Report any new execution as a replication attempt with its own environment and outputs. Test reuse, repeated horses, unadjusted intervals and historical graph/market timing limitations remain unchanged.

Source access was confirmed by the depositing author. Operational transport wrappers, expired signed URLs, credentials and failed partial transfers are excluded. Rejected tuning model binaries are included where present in the frozen model tree; all candidate aggregate evidence is in the companion. A fresh website scrape is not an exact replacement for these frozen inputs.
