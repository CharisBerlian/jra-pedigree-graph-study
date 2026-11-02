# Workflow figure

Current release: **v1.0.5-paper**, corrected on 2 November 2026. Figure 1 is entirely within one column on page 2. It uses [native vector source](../paper/figures/study_workflow_diagram.tex) and the [single-column wrapper and caption](../paper/figures/study_workflow.tex). Main labels are 8-point text; math subscripts are smaller. The four feature arms have independent input/output branches and are fitted separately. The paper has five pages. Existing numerical tables and result plots are unchanged.

All five pages were rendered and visually checked. The diagram extends to x=271.987 PDF points, within the left-column boundary at x=291.64. See [SINGLE_COLUMN_WORKFLOW_CHECKS.json](SINGLE_COLUMN_WORKFLOW_CHECKS.json) for the current PDF hash and audit.

## Prior wide preview and v1.0.4 integration

The following record describes the earlier six-page release. Its approved raster is retained as provenance; the current PDF uses the vector source above.

Release: v1.0.4-paper. The user approved the landscape preview and requested its insertion into the manuscript.

- Asset: [paper/figures/study_workflow.png](../paper/figures/study_workflow.png), 1536 x 1024 pixels.
- Inclusion and caption: [paper/figures/study_workflow.tex](../paper/figures/study_workflow.tex).
- Generation mode: built-in imagegen tool; no CLI/API fallback.
- Approved PNG SHA-256: `5eab65b831d2c2cae19f6a47a2fdc4a504095ecbdd37ac57bf658e004b69f9aa`.
- Integration: Figure 1 on page 2, spanning both columns. Existing AP figures are now Figures 2 and 3. The six-page PDF embeds the approved pixels without alteration. Scientific table/plot data are unchanged.
- Verification: [WORKFLOW_FIGURE_CHECKS.json](WORKFLOW_FIGURE_CHECKS.json).

## Original preview generation prompt

```text
Use case: scientific-educational. Create a new high-resolution academic workflow diagram on white, landscape 3:2 composition, crisp flat vector-like linework, Times New Roman-like serif text, thin black outlines, pale blue process boxes and pale teal feature boxes. No illustrations, texture, shadows, figure numbers or outer caption. This is a preview for an IEEE research paper.

The critical scientific structure is FOUR INDEPENDENT PARALLEL EXPERIMENTAL ARMS. Arrange the four feature boxes in ONE HORIZONTAL ROW, not a two-by-two grid. No arrows between feature boxes. A common branching rail above the row gives each box its own downward input arrow. A collecting rail below the row receives one downward output arrow from each box, followed by one arrow into the model-fitting stage. Label that collection rail "Each pipeline fitted separately" to make clear it is routing to separate experiments, not concatenating or ensembling the four arms.

Top-to-bottom flow, all boxes centered or symmetrically aligned:
1. Wide top box, exact text: "Race and pedigree records" / "Observed JRA flat-race cohort".
2. Wide box below, exact text: "Chronological partitions" / "Training: 2010–2021  |  Validation: 2022–2023  |  Test: 2024–2025".
3. Heading above the common branching rail: "Four feature pipelines".
4. ONE ROW of four equally sized feature boxes, with readable wrapped text:
   Box 1 heading "H"; body "Context + recent history"; footer "43 predictors".
   Box 2 heading "Gmixed" with mixed as subscript; body "H + original graph features" / "Direct vector; ancestor fallback"; footer "112 predictors".
   Box 3 heading "Gancestor" with ancestor as subscript; body "H + ancestor-rule graph features" / "Ancestor construction for every entry"; footer "112 predictors".
   Box 4 heading "P"; body "H + prior parent-performance features" / "Training outcomes only"; footer "52 predictors".
A small note next to the feature stage, not on any arrow: "Graph arms share the fixed training-root graph."
5. A wide box below the four separate outputs, exact text: "Independent model fitting" / "XGBoost · LightGBM · CatBoost" / "Separate fits by pipeline and surface" / "Validation tuning → train-only refits".
6. Next box, exact text: "Test predictions" / "Three paired seeds · turf and dirt".
7. Bottom box, exact text: "Evaluation" / "Average precision (AP) · Hit@3" / "Paired whole-race bootstrap intervals".
Small footer outside the boxes: "Four-arm study: no market predictors. Historical market comparisons are reported separately."

Priorities: all four arms explicitly parallel, no serial H-to-Gancestor or Gmixed-to-P arrows; all words readable without overlap; balanced compact spacing. Only permitted arrows are data → split → each of four arms → separate fitting stage → predictions → evaluation. No other arrows, no data leakage claims, no accuracy claims, no final-market predictors in any arm. The split is chronological; the graph is fixed training-root topology, not rolling. Retain every exact scientific label and count.
```
