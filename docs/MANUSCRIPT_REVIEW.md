# Manuscript review, 1 November 2026

Current update: **v1.0.4-paper** adds the user-approved workflow as Figure 1 on page 2. The paper now has six A4 pages. Existing result plots are Figures 2 and 3 on page 4; all tables and quantitative figure files are byte-identical to v1.0.3-paper. All six rendered pages were checked, with the availability statement together on page 6 and the reference columns rebalanced. See `WORKFLOW_FIGURE_CHECKS.json` for the current PDF hash and checks, and `WORKFLOW_FIGURE.md` for the approved asset and generation prompt. The reports below describe the earlier releases explicitly named in each paragraph.

Release: **v1.0.2-paper**. The paper remains five A4 pages. This is a manuscript and released-artifact consistency audit, not a new training experiment or a full bootstrap replay.

Author update: **v1.0.3-paper** adds the user-supplied Charis Berlian author block, department/school, Bina Nusantara University, Jakarta, Indonesia and email. All five pages were rendered and checked again; Figure 2 remains on page 4, both public links remain on page 5, and the reference columns were rebalanced. Scientific text and table/figure data are unchanged. See `AUTHOR_DETAILS_CHECKS.json` for the new PDF hash and verification; the 89-check report below describes v1.0.2-paper.

## Changes

- Figure 2 was deferred to the final page because its previous `figure*` required a full-width float slot. It is now a column-width figure on page 4, beside Table IV and the historical-results discussion. Its four panels are stacked, with readable family labels and separate turf/dirt marker shapes. All 24 estimates and 48 interval endpoints come from the same frozen historical CSV. Panels (a) and (b) retain a shared horizontal scale; the other scales are explicitly labeled.
- Removed the forced bibliography page break and balanced the final reference columns. The conclusion and availability statement remain before the References heading in reading order.
- Reordered all 18 bibliography entries by first citation, as required by the numbered IEEE style. Citation keys and their associated claims are preserved.
- Corrected the Pudaruth, Medard and Dookhun (2013) page range from 39-42 to **37-42**, confirmed against the [publisher record](https://www.ijcaonline.org/archives/volume72/number5/12493-9048/).
- Updated the availability statement to identify this paper release. The dataset remains version 1 and the code/data licenses remain unchanged.

## Verification

`MANUSCRIPT_REVIEW_CHECKS.json` records 89 checks and the final PDF hash. The source CSV hashes are recorded so the audit can be distinguished from a later data version.

- Table I: chronological cohort counts, test race counts and arithmetic totals checked against released coverage/representation records. Training/validation race totals are retained from the existing cohort table; raw training/validation race rosters were not recounted in this pass.
- Table II: all six entry/direct/depth rows, zero-missing status, parent-identifier availability, zero-prior counts and printed percentages checked against released cohort tables, including consistency across seeds.
- Table III: all **96 displayed means and sample SDs** checked; exact regeneration from the hash-bound 72 seed rows and 24 arm summaries passed.
- Table IV: all **36 displayed historical scores** checked against the historical model summaries.
- Figure 1: exact regeneration of all **18 AP estimates and intervals** passed. Figure 2: all **24 estimates and intervals** checked against the frozen historical contrast table and the new renderer's coordinate audit.
- Checked the reported negative turf primary AP intervals, zero-containing dirt primary AP and primary Hit@3 intervals, secondary CatBoost dirt findings, historical market ranges and intervals, parent-category rates, baseline scores and focused tuning-cap counts.
- All 18 bibliography entries are cited; every citation and table/figure reference resolves; no duplicate keys/labels or draft placeholders were found. First-citation numbering is now sequential.
- The seven aggregate-reporting validation tests and both independent synthetic feature/metric fixtures passed.
- The public [Kaggle notebook](https://www.kaggle.com/code/charischerw/jra-pedigree-graph-study-reproduction), version 2, now reports **COMPLETE**. Its execution log states: "PASS: all 72 seed-level AP and Hit@3 point scores match the released references." This checks supplied predictions and outcomes; it does not refit the models or rerun the interval procedure.
- All five final PDF pages rendered and visually inspected. No clipping, overlapping elements or missing glyphs observed. All 14 font subsets are embedded. No overfull boxes, undefined references/citations or missing-character warnings remain. One expected LaTeX notice identifies the float-only left column on page 4; the right column contains the historical discussion and limitations.

## Bibliographic review and remaining boundaries

Article metadata was cross-checked against Crossref and accessible primary records, including the [Silverman publisher](https://www.ubplj.org/index.php/jpm/article/view/595), [Bailey publisher](https://www.annualreviews.org/content/journals/10.1146/annurev-animal-020420-035235), [Gulum university record](https://ir.library.louisville.edu/etd/2953/), [Sugiura full article](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1896192/full), [LightGBM proceedings](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html), [LightGBM author repository](https://github.com/lightgbm-org/LightGBM), [CatBoost proceedings](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html), and the two cited JRA statistics pages. The online NeurIPS BibTeX records do not include pagination; existing page ranges were retained, with LightGBM's range also supported by its author repository. The Japanese report's bibliographic record/abstract supports the cautious description already in the paper; its detailed target and controls were not verified from full text.

The author block was blank in v1.0.2-paper. It is populated in v1.0.3-paper following the user's explicit choice and supplied details. This pass does not establish venue compliance, a new submission deadline, external scientific replication, untouched-test inference, causal effects or betting profitability. Existing scientific caveats remain in the manuscript.
