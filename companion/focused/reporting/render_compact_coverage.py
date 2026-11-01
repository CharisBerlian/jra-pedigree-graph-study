"""Render a column-width coverage table from the frozen aggregate companion."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def render(companion):
    history = {(r['split'], r['surface']): r for r in rows(companion/'historical/tables/graph_coverage.csv')}
    feasibility = {(r['split'], r['surface']): r for r in rows(companion/'focused/cohort/train_validation_feasibility.csv')}
    test = {(r['view'], r['seed'], r['surface']): r for r in rows(companion/'focused/cohort/test_representation.csv')}
    parents = {(r['view'], r['seed'], r['surface'], r['role']): r for r in rows(companion/'focused/cohort/test_parent_availability.csv')}
    lines = [r'\begin{table}[!t]', r'\caption{Graph representation coverage}',
             r'\label{tab:focused_coverage}', r'\centering', r'\footnotesize',
             r'\setlength{\tabcolsep}{2pt}', r'\renewcommand{\arraystretch}{1.12}',
             r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrr@{}}',
             r'\hline', r'& & $G_{\mathrm{mixed}}$ & \multicolumn{3}{c}{$G_{\mathrm{ancestor}}$ depth} \\',
             r'\cline{4-6}', r'Split & Entries & Direct & 1 & 2 & 3 \\', r'\hline']
    for surface in ('turf', 'dirt'):
        lines.append(r'\multicolumn{6}{@{}l}{\textit{' + surface.title() + r'}} \\')
        for split, label in (('train', 'Training'), ('validation', 'Validation'), ('test', 'Test')):
            original = history[split, surface]
            if split == 'test':
                ancestor = test['Gancestor', '0', surface]
                depths = [int(ancestor[f'depth_{i}']) for i in (1, 2, 3)]
                assert int(original['direct']) == int(test['Gmixed', '0', surface]['direct'])
            else:
                ancestor = feasibility[split, surface]
                depths = [int(ancestor[f'ancestor_depth_{i}']) for i in (1, 2, 3)]
            entries = int(original['entries'])
            assert int(ancestor['entries']) == entries and sum(depths) == entries
            assert int(original['missing']) == 0
            values = [entries, int(original['direct']), *depths]
            lines.append(label + ' & ' + ' & '.join(f'{n:,}' for n in values) + r' \\')
        lines.append(r'\hline')
    zero = {}
    for surface in ('turf', 'dirt'):
        counts = []
        for role in ('sire', 'dam', 'damsire'):
            row = parents['Gmixed', '0', surface, role]
            assert int(row['id_missing']) == 0
            counts.append(f"{int(row['all_zero_prior']):,}")
        zero[surface] = '/'.join(counts)
    lines += [r'\end{tabular*}',
              r'\par\smallskip\begin{minipage}{\columnwidth}\footnotesize\raggedright '
              r'Counts are entries, identical across graph seeds. No representation is missing; '
              r'$G_{\mathrm{mixed}}$ uses ancestor fallback when no direct vector exists. '
              r'All test entries have sire/dam/damsire IDs. P zero-prior counts, in that order: turf '
              + zero['turf'] + '; dirt ' + zero['dirt'] + r'.\end{minipage}',
              r'\end{table}', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--companion', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('preserving existing output')
    args.out.write_text(render(args.companion), encoding='utf-8')
