# Validation-only focused tuning report

`report_focused_tuning.py` summarizes the 24-cell validation-only model freeze after its manifest exists and has been independently reviewed. It neither authorizes test access nor reads any prediction files, labels, private feature rows or test data. Do not run it against partial cell artifacts or report its output as test performance.

From this directory after review, provide the reviewed manifest hash and actual frozen paths:

```text
python -B report_focused_tuning.py --manifest <frozen-manifest.json> --expected-sha256 <reviewed-64-digit-sha256> --harvest-root <model-harvest-directory> --out <new-output-directory>
```

The output directory must not already exist and its parent must exist. The tool verifies the supplied manifest hash, frozen design/status and exact 24-cell matrix; each manifest-bound candidate-results file and selected record; all 96 candidate metrics and corresponding AP curves; original validation-AP/Hit@3/best-round/candidate-ID selection; and hashes and selected-round identities of all 72 refit configuration files. It verifies curve cap using the curve length and the earliest best AP round using the frozen criterion. These checks are validation-only reporting checks; they do not independently audit the raw validation predictions or prove that the supplied manifest itself was independently reviewed.

The output consists of `validation_candidates.csv` (all 96 candidate rows), `validation_selections.csv` (all 24 selections with seed-specific refit-round agreement), and `SUMMARY.json` (manifest/source/output hashes and cap counts). `curve_hit_cap` means a candidate's tuning curve ran 1,200 rounds; `selected_best_round_at_cap` separately means the chosen best round used for the refits is 1,200. A tuning curve can hit the cap even if its best/refit round is earlier. Keep all rows, including unfavorable candidates, and distinguish validation metrics from the as-yet-unavailable test results. Do not publish without independent review and authorization.

Synthetic validation of the reporting code, using temporary fabricated files and no scientific data:

```text
python -B test_focused_tuning_report.py
```
