"""Render all 24 locked historical AP contrasts as four column-width panels."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

FAMILIES = (('xgboost', 'XGBoost'), ('lightgbm', 'LightGBM'), ('catboost', 'CatBoost'))
PANELS = (
    ('S6-S2', '(a) Pedigree beyond recent history', 'C--A', (-0.015, 0.045), (-0.01, 0, 0.02, 0.04)),
    ('S6-S3', '(b) Graph versus simple pedigree', 'C--B', (-0.015, 0.045), (-0.01, 0, 0.02, 0.04)),
    ('S7-M2', '(c) Pedigree beyond history and market', 'E--D', (-0.004, 0.004), (-0.004, -0.002, 0, 0.002, 0.004)),
    ('S7-S6', '(d) Adding odds and popularity', 'E--C', (-0.005, 0.13), (0, 0.04, 0.08, 0.12)),
)


def render(source, out):
    with source.open(encoding='utf-8', newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['metric'] == 'entry_average_precision']
    records = {(r['contrast'], r['family'], r['surface']): r for r in rows}
    assert len(records) == len(rows) == 36
    lines = [r'% Column-width layout; all coordinates come from the frozen aggregate CSV.',
             '% Source SHA-256: ' + hashlib.sha256(source.read_bytes()).hexdigest(),
             r'\begin{tikzpicture}[x=1pt,y=1pt,every node/.style={font=\footnotesize,inner sep=0pt}]',
             r'\useasboundingbox (0,-376) rectangle (248,23);',
             r'\fill[blue!70!black] (85,18) circle (1.8pt);',
             r'\node[anchor=west] at (91,18) {Turf};',
             r'\fill[red!70!black] (135,16.2) rectangle (138.6,19.8);',
             r'\node[anchor=west] at (144,18) {Dirt};']
    audit = []
    for index, (contrast, title, display, limits, ticks) in enumerate(PANELS):
        top = -91 * index
        lo, hi = limits
        def xpos(value):
            return 55 + (float(value) - lo) / (hi - lo) * 181
        lines += [f'\\node[anchor=west,font=\\footnotesize\\bfseries] at (0,{top}) {{{title}}};',
                  f'\\node[anchor=east] at (248,{top}) {{{display}}};',
                  f'\\draw[gray,dashed] ({xpos(0):.5f},{top-9}) -- ({xpos(0):.5f},{top-63});']
        for family_index, (family, label) in enumerate(FAMILIES):
            y = top - 16 - family_index * 17
            lines.append(f'\\node[anchor=west] at (0,{y-3.0}) {{{label}}};')
            for surface, offset, color in (('turf', 0, 'blue!70!black'), ('dirt', -6, 'red!70!black')):
                row = records[contrast, family, surface]
                lower, point, upper = (float(row[k]) for k in ('ci95_lower', 'estimate', 'ci95_upper'))
                assert lo <= lower <= point <= upper <= hi
                py, px = y + offset, xpos(point)
                lines.append(f'\\draw[{color},line width=0.8pt] ({xpos(lower):.5f},{py}) -- ({xpos(upper):.5f},{py});')
                if surface == 'turf':
                    lines.append(f'\\fill[{color}] ({px:.5f},{py}) circle (1.8pt);')
                else:
                    lines.append(f'\\fill[{color}] ({px-1.8:.5f},{py-1.8:.1f}) rectangle ({px+1.8:.5f},{py+1.8:.1f});')
                audit.append({'contrast': contrast, 'family': family, 'surface': surface,
                              'estimate': point, 'lower': lower, 'upper': upper,
                              'axis_min': lo, 'axis_max': hi, 'point_x': round(px, 5),
                              'lower_x': round(xpos(lower), 5), 'upper_x': round(xpos(upper), 5)})
        lines.append(f'\\draw[gray] (55,{top-63}) -- (236,{top-63});')
        for tick in ticks:
            label = '0' if tick == 0 else f'{tick:.{3 if index == 2 else 2}f}'.replace('-', r'\textminus{}')
            lines += [f'\\draw[gray] ({xpos(tick):.5f},{top-63}) -- ({xpos(tick):.5f},{top-66});',
                      f'\\node[anchor=north] at ({xpos(tick):.5f},{top-69}) {{{label}}};']
    lines += [r'\node at (124,-367) {Difference in AP (first setting minus second)};', r'\end{tikzpicture}', '']
    out.write_text('\n'.join(lines), encoding='utf-8')
    out.with_suffix('.json').write_text(json.dumps({'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'contrasts_plotted': len(audit), 'points': audit}, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    render(args.source, args.out)
