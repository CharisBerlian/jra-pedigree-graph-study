"""Generate split coverage table from aggregate counts."""
import csv
from pathlib import Path
root = Path(__file__).resolve().parents[1]
with (root / 'tables/graph_coverage.csv').open(newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
items = {(r['split'], r['surface']): r for r in rows}
assert len(items) == len(rows) == 6
out = [r'\begin{tabular}{llrrr}', r'\hline', r'Split & Surface & Entries & Direct & Fallback \\', r'\hline']
for split in ('train', 'validation', 'test'):
    for surface in ('turf', 'dirt'):
        r = items[split, surface]
        n, direct, missing = (int(r[k]) for k in ('entries', 'direct', 'missing'))
        fallback = sum(int(r[k]) for k in ('parent', 'generation2', 'generation3'))
        assert direct + fallback + missing == n and n > 0 and missing == 0
        out.append(f"{split.title()} & {surface.title()} & {n:,} & {100*direct/n:.2f}\\% & {100*fallback/n:.2f}\\% " + r'\\')
out += [r'\hline', r'\end{tabular}']
(root.parent / 'paper/coverage_table.tex').write_text('\n'.join(out)+'\n', encoding='utf-8')
print('Generated coverage table from six split/surface count rows.')
