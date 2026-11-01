"""Render all-family historical market settings from frozen aggregate scores."""
import argparse
import csv
import hashlib
from pathlib import Path

FAMILIES = ('xgboost', 'lightgbm', 'catboost')
LABELS = {'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'catboost': 'CatBoost'}
SURFACES = ('turf', 'dirt')
SETTINGS = ('M2', 'S7', 'B1')  # historical D, E, F; fixed before new outcomes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    with a.source.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    values = {(r['setting'], r['family'], r['surface']): r for r in rows}
    if len(rows) != 60 or len(values) != 60:
        raise ValueError('historical score table is incomplete or duplicated')
    lines = [r'\begin{table*}[!t]',
             r'\caption{Historical Market-Information Scores Across All Families and Surfaces}',
             r'\label{tab:historical_market}', r'\centering', r'\begin{tabular}{llrrrrrr}',
             r'\hline', r'& & \multicolumn{3}{c}{Entry AP} & \multicolumn{3}{c}{Mean Hit@3} \\',
             r'Family & Surface & D & E & F & D & E & F \\', r'\hline']
    for family in FAMILIES:
        for surface in SURFACES:
            nums = [float(values[setting, family, surface][metric])
                    for metric in ('entry_average_precision_mean', 'race_mean_hit_at_3_mean')
                    for setting in SETTINGS]
            if not all(0 <= n <= 1 for n in nums):
                raise ValueError('invalid historical score')
            lines.append(f'{LABELS[family]} & {surface.title()} & ' + ' & '.join(f'{n:.4f}' for n in nums) + r' \\')
    lines += [r'\hline', r'\end{tabular}',
              r'\par\smallskip Historical three-seed means: D adds final odds and popularity to history; '
              r'E adds them to the original graph pipeline; F uses final odds alone. '
              r'The market fields are retrospective and are not inputs in the new four-arm study.',
              r'\end{table*}', '']
    if a.out.exists():
        raise FileExistsError('preserving existing output')
    a.out.write_text('\n'.join(lines), encoding='utf-8')
    print('source_sha256=' + hashlib.sha256(a.source.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
