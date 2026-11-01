# Focused revision research brief — for design review

## Question and contribution

For this retrospective JRA cohort, do pedigree-graph features add predictive information beyond recent history when training and evaluation consistently use ancestor-derived vectors, and how do they compare with a transparent parent-performance control? The contribution is a controlled empirical comparison, not a new graph algorithm or a claim that pedigree graphs always improve prediction. The already-seen 2024–2025 test makes every new result exploratory. The primary interpretation is based on all predeclared families, surfaces and seeds, not the best-looking cell.

Hold the original 2010–2021 graph, centroids, 2010–2021 classifier training cohort and 2022–2023 validation cohort fixed. Compare: **H**, history/context; **Gmixed**, the original direct-else-nearest-ancestor S6 representation; **Gancestor**, an ancestor-only S6 representation applied to every entry; and **P**, history/context plus strictly prior sire, dam and damsire performance summaries. All four are fitted with the same classifier rows, family-specific search grids, selection rule, seeds and software environment. Gmixed and Gancestor share the identical original seed-specific graph and centroids. No new graph is fitted. Compare Gancestor−H and Gancestor−P first; Gancestor−Gmixed is a secondary representation-protocol comparison.

## Reviewer-facing logic

The review PDF supplies ratings, not detailed reasons; this mapping addresses plausible weaknesses rather than attributing specific objections to reviewers.

| Criterion | Action and evidence | Honest interpretation / remaining gate |
|---|---|---|
| Relevance | Target an operational question: whether an ancestry signal improves over a strong historical predictor in a chronological JRA cohort. | Retrospective fields and final entry information are not verified at an operational decision time; no deployment or betting-profit claim. |
| Originality | Treat the contribution as a controlled evaluation of consistent ancestor-derived representation and a simple ancestry comparator, situated against existing studies. | Neither Node2Vec nor parent-performance summaries are claimed as inventions; literature and venue positioning must be checked separately. |
| Significance | Report absolute AP and full-roster Hit@3 plus complete paired differences for H, Gmixed, Gancestor and P. | A small, negative or inconclusive incremental result is still an empirical result; it is not evidence of practical value or generality across countries. |
| Methods | Reuse the full original cohorts and same graph space; construct ancestor vectors identically for all periods; construct the P features with strictly earlier dates and frozen post-2021 maps. | Future topology within the original training graph and retrospective pedigree availability remain; changing representation does not establish historical point-in-time validity. |
| Evaluation | Four matched arms, identical tuning budgets, three paired seeds, two surfaces, all three model families, race-resampled paired intervals and full rosters. | The test is seen; intervals are conditional on these fitted models and do not address all repeated-horse dependence. No confirmatory significance claim. |
| Reproducibility | Freeze exact code, graph/input provenance, category maps, feature formulas, fits, checkpoints and output hashes; provide a small synthetic reproduction fixture. | Source data and fitted artifacts are private; full independent reproduction still requires authorized access and a durable archive. |
| Related work | Position against prior pedigree, boosted-history and market work, with careful comparison of what each actually tests. | Separate literature review needed; no unverified first-study, superiority or exhaustive-search claim. |
| Writing | Explain representation mismatch, transparent comparator, negative outcomes and constraints directly; fit the required IEEE template. | Formatting and a stronger narrative do not themselves fix unavailable data, seen-test status or unresolved empirical findings. |

## Possible outcomes and stop rules

If Gancestor exceeds H consistently, report that it adds information for this cohort and these implementation choices; if it does not, say no consistent gain was established. If it also exceeds P, the comparison supports a gain over **this** parent-performance implementation, not all pedigree baselines. If P performs as well or better, say a simple control was competitive; neither outcome identifies why. Gancestor−Gmixed changes a group of graph-derived columns and the learned model, so it tests the complete representation protocol rather than the isolated causal effect of fallback.

Stop before fitting if complete-roster joins, graph/centroid provenance, earlier-date construction, non-outcome predictors, fixed category policies or train/validation-only preparation cannot be verified. Stop before test feature generation or prediction unless root has reviewed the complete actual model/selection freeze. If that gate fails, preserve the current paper's results and limitations and report the additional study as unavailable. Do not choose variants or omit unfavorable arms after inspecting scores. No acceptance or rating improvement is promised.

## Feasibility established so far

`feasibility.py` reads only identifiers, surface and pedigree fields from train/validation inputs, and verifies all three historical graph vector hashes. It does not read labels, test rows or model predictions. All three original graphs contain the same **93,474** node IDs. Every training entry has an embedded first-generation ancestor (271,549 turf and 277,207 dirt). Validation first-generation coverage is 42,580/42,783 turf and 45,081/45,323 dirt; all others resolve by generation 2 or 3. Sire, dam and damsire links exist for all training/validation entries, with no disagreement between nonempty exported parent IDs and pedigree edges. These are entry-weighted availability checks, not evidence of historical timestamp availability or future/test coverage. Exact provenance and counts are in `feasibility.json`.

All training entries resolving at depth 1 also means training cannot learn effects specific to depth 2/3 or missing status. The proposed rule aligns how vectors are constructed; it does not eliminate every coverage or descriptor distribution shift.

The previous 2017-cutoff sensitivity changed both graph vintage and classifier period and cannot isolate fallback. Its three graph runs and three accepted assembly outputs are preserved as superseded exploratory preparation. Do not launch its 18 model jobs. The new design cannot repair the original graph's later-training topology or make the previously inspected test untouched.
