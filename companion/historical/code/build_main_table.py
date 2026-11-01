"""Generate the paper's main table from the frozen aggregate scores."""
import csv
from pathlib import Path

root = Path(__file__).resolve().parents[1]
with (root / 'tables/model_summaries_extended.csv').open(encoding='utf-8', newline='') as f:
    rows = list(csv.DictReader(f))
values = {(r['setting'], r['family'], r['surface']): r for r in rows}
assert len(rows) == len(values) == 60
order = [('A', 'S2'), ('B', 'S3'), ('C', 'S6'), ('D', 'M2'), ('E', 'S7'), ('F', 'B1')]
families = ('xgboost', 'lightgbm', 'catboost')
surfaces = ('turf', 'dirt')
lines = [r'\begin{table*}[!t]', r'\caption{Test Scores for the Main Settings and Odds-Only Baseline (Three-Seed Means)}', r'\label{tab:locked_summary}', r'\centering\footnotesize', r'\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.08}', r'\begin{tabular}{lrrrrrr|rrrrrr}', r'\hline', r' & \multicolumn{6}{c|}{Entry AP} & \multicolumn{6}{c}{Mean Hit@3} \\', r' & \multicolumn{2}{c}{XGBoost} & \multicolumn{2}{c}{LightGBM} & \multicolumn{2}{c|}{CatBoost} & \multicolumn{2}{c}{XGBoost} & \multicolumn{2}{c}{LightGBM} & \multicolumn{2}{c}{CatBoost} \\', r'Setting & Turf & Dirt & Turf & Dirt & Turf & Dirt & Turf & Dirt & Turf & Dirt & Turf & Dirt \\', r'\hline']
for label, setting in order:
    nums = [float(values[setting, family, surface][metric]) for metric in ('entry_average_precision_mean', 'race_mean_hit_at_3_mean') for family in families for surface in surfaces]
    assert all(0 <= n <= 1 for n in nums)
    lines.append(label + ' & ' + ' & '.join(f'{n:.4f}' for n in nums) + r' \\')
lines += [r'\hline', r'\end{tabular}', r'\par\smallskip\footnotesize Metrics are averaged after evaluation of each seed, not from an ensemble. Fig.~\ref{fig:primary_ap_contrasts} shows paired AP contrasts.', r'\end{table*}']
(root.parent / 'paper/results_table.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('Generated 72 displayed metrics from aggregate scores.')
