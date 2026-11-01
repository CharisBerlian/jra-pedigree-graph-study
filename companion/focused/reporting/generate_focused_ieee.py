"""Generate staged IEEE assets only from the complete, hash-bound evaluation.

This does not authorize test access, perform evaluation, or establish an audit.
Inputs are the exact outputs of science/focused_revision/evaluate_focused.py.
"""
import argparse
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import statistics

ARMS = ('H', 'Gmixed', 'Gancestor', 'P')
FAMILIES = ('xgboost', 'lightgbm', 'catboost')
SURFACES = ('turf', 'dirt')
SEEDS = (0, 1, 2)
CONTRASTS = ('Gancestor-H', 'Gancestor-P', 'Gancestor-Gmixed')
METRICS = ('ap', 'hit')
FILES = ('seed_metrics.csv', 'arm_metrics.csv', 'contrasts.csv')
HEADERS = {
    FILES[0]: ('surface', 'family', 'arm', 'seed', 'average_precision', 'hit_at_3'),
    FILES[1]: ('surface', 'family', 'arm', 'ap_mean', 'ap_seed_sd', 'hit_mean', 'hit_seed_sd'),
    FILES[2]: ('surface', 'family', 'contrast', 'metric', 'difference', 'ci_low', 'ci_high',
               'correct_per_100_races', 'correct_per_100_low', 'correct_per_100_high'),
}
SURFACE_ENTRIES = {'turf': 43477, 'dirt': 45660}
FAMILY_LABELS = {'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'catboost': 'CatBoost'}
ARM_LABELS = {'H': 'H', 'Gmixed': r'$G_{\mathrm{mixed}}$',
              'Gancestor': r'$G_{\mathrm{ancestor}}$', 'P': 'P'}
PANEL_LABELS = {
    'Gancestor-H': r'Primary: $G_{\mathrm{ancestor}}-H$',
    'Gancestor-P': r'Primary: $G_{\mathrm{ancestor}}-P$',
    'Gancestor-Gmixed': r'Secondary: $G_{\mathrm{ancestor}}-G_{\mathrm{mixed}}$',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def number(value, where, low=None, high=None):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{where}: invalid number') from error
    require(math.isfinite(result), f'{where}: nonfinite number')
    require(low is None or result >= low, f'{where}: below range')
    require(high is None or result <= high, f'{where}: above range')
    return result


def near(actual, expected, where):
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
            f'{where}: inconsistent with seed metrics')


def load_csv(data, name):
    try:
        text = data.decode('utf-8')
        reader = csv.DictReader(io.StringIO(text, newline=''))
        require(tuple(reader.fieldnames or ()) == HEADERS[name], f'{name}: incorrect header')
        rows = list(reader)
    except (UnicodeError, csv.Error) as error:
        raise ValueError(f'{name}: invalid CSV') from error
    require(all(None not in row and all(v is not None for v in row.values()) for row in rows),
            f'{name}: malformed row')
    return rows


def indexed(rows, keys, expected, name):
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        require(key not in result, f'{name}: duplicate key {key}')
        result[key] = row
    require(set(result) == expected, f'{name}: missing or unexpected keys')
    return result


def validate_coverage(report):
    coverage = report.get('coverage_and_parent_zero_counts')
    expected = {f'{v}/seed{s}' for v in ('Gmixed', 'Gancestor') for s in SEEDS}
    require(isinstance(coverage, dict) and set(coverage) == expected, 'coverage: wrong views')
    output = []
    reference = {}
    for view, seed in itertools.product(('Gmixed', 'Gancestor'), SEEDS):
        item = coverage[f'{view}/seed{seed}']
        require(isinstance(item, dict) and isinstance(item.get('surfaces'), dict)
                and set(item['surfaces']) == set(SURFACES), 'coverage: wrong surfaces')
        for surface in SURFACES:
            value = item['surfaces'][surface]
            entries = value['entries']
            require(type(entries) is int and entries == SURFACE_ENTRIES[surface], 'coverage: entry count')
            status = value['coverage']
            require(isinstance(status, dict) and set(status) <= {'direct', 'fallback', 'missing'}
                    and all(type(v) is int and v >= 0 for v in status.values())
                    and sum(status.values()) == entries, 'coverage: invalid status counts')
            zeros = value['parent_zero_counts']
            require(isinstance(zeros, dict) and set(zeros) == {'sire', 'dam', 'damsire'}
                    and all(type(n) is int and 0 <= n <= entries for n in zeros.values()),
                    'coverage: invalid parent zero counts')
            if surface in reference:
                require(zeros == reference[surface], 'parent counts differ across views or seeds')
            else:
                reference[surface] = zeros
            output.append({'view': view, 'seed': seed, 'surface': surface, 'entries': entries,
                           'direct': status.get('direct', 0), 'fallback': status.get('fallback', 0),
                           'missing': status.get('missing', 0), **{f'{k}_zero': v for k, v in zeros.items()}})
    return output


def validate(report, payloads):
    require(isinstance(report, dict) and report.get('status') == 'complete'
            and report.get('exploratory') is True, 'evaluation report is not complete/exploratory')
    hashes = report.get('outputs')
    require(isinstance(hashes, dict) and set(hashes) == set(FILES), 'report: wrong output inventory')
    for name in FILES:
        require(isinstance(hashes[name], str) and digest(payloads[name]) == hashes[name].lower(),
                f'{name}: SHA-256 mismatch')
    seeds = indexed(load_csv(payloads[FILES[0]], FILES[0]), ('surface', 'family', 'arm', 'seed'),
                    set(itertools.product(SURFACES, FAMILIES, ARMS, map(str, SEEDS))), FILES[0])
    arms = indexed(load_csv(payloads[FILES[1]], FILES[1]), ('surface', 'family', 'arm'),
                   set(itertools.product(SURFACES, FAMILIES, ARMS)), FILES[1])
    contrasts = indexed(load_csv(payloads[FILES[2]], FILES[2]), ('surface', 'family', 'contrast', 'metric'),
                        set(itertools.product(SURFACES, FAMILIES, CONTRASTS, METRICS)), FILES[2])
    for surface, family, arm in arms:
        row = arms[surface, family, arm]
        for original, mean, sd in (('average_precision', 'ap_mean', 'ap_seed_sd'),
                                   ('hit_at_3', 'hit_mean', 'hit_seed_sd')):
            values = [number(seeds[surface, family, arm, str(s)][original], original, 0, 1) for s in SEEDS]
            near(number(row[mean], mean, 0, 1), statistics.mean(values), mean)
            near(number(row[sd], sd, 0, 1), statistics.stdev(values), sd)
    for surface, family, contrast, metric in contrasts:
        row = contrasts[surface, family, contrast, metric]
        point = number(row['difference'], 'difference', -1, 1)
        lo = number(row['ci_low'], 'ci_low', -1, 1)
        hi = number(row['ci_high'], 'ci_high', -1, 1)
        require(lo <= hi, 'confidence interval endpoints reversed')
        left, right = contrast.split('-')
        field = 'ap_mean' if metric == 'ap' else 'hit_mean'
        near(point, float(arms[surface, family, left][field]) - float(arms[surface, family, right][field]), 'contrast')
        extra = ('correct_per_100_races', 'correct_per_100_low', 'correct_per_100_high')
        if metric == 'hit':
            for field, value in zip(extra, (point, lo, hi)):
                near(number(row[field], field), 300 * value, field)
        else:
            require(all(row[field] == '' for field in extra), 'AP row has unexpected Hit@3 scaling')
    return seeds, arms, contrasts, validate_coverage(report)


def table(arms):
    lines = [r'\begin{table*}[!t]', r'\caption{Four-arm test performance}',
             r'\label{tab:focused_scores}', r'\centering', r'\footnotesize',
             r'\setlength{\tabcolsep}{3pt}', r'\renewcommand{\arraystretch}{1.12}',
             r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcccc@{}}', r'\hline',
             r'& & \multicolumn{2}{c}{Turf} & \multicolumn{2}{c}{Dirt} \\',
             r'\cline{3-4}\cline{5-6}',
             r'Family & Arm & AP & Hit@3 & AP & Hit@3 \\', r'\hline']
    for family in FAMILIES:
        for index, arm in enumerate(ARMS):
            values = []
            for surface in SURFACES:
                row = arms[surface, family, arm]
                values += [f"{float(row[mean]):.4f} $\\pm$ {float(row[sd]):.4f}"
                           for mean, sd in (('ap_mean', 'ap_seed_sd'), ('hit_mean', 'hit_seed_sd'))]
            family_label = FAMILY_LABELS[family] if index == 0 else ''
            lines.append(f'{family_label} & {ARM_LABELS[arm]} & ' + ' & '.join(values) + r' \\')
        lines.append(r'\hline')
    lines += [r'\end{tabular*}',
              r'\par\smallskip\begin{minipage}{\textwidth}\footnotesize\raggedright '
              r'Mean $\pm$ sample SD across three paired seeds. Scores are fractions; '
              r'the test was previously examined. Cohort/coverage: '
              r'Tables~\ref{tab:cohort} and \ref{tab:focused_coverage}.\end{minipage}',
              r'\end{table*}', '']
    return '\n'.join(lines)


def plot(contrasts):
    lines = [r'\begin{figure}[!t]', r'\centering', r'\begin{tikzpicture}[x=1cm,y=1cm]',
             r'\small']
    for panel, contrast in enumerate(CONTRASTS):
        rows = [contrasts[s, f, contrast, 'ap'] for s, f in itertools.product(SURFACES, FAMILIES)]
        vals = [100 * float(r[k]) for r in rows for k in ('difference', 'ci_low', 'ci_high')]
        bound = max(max(abs(v) for v in vals), 1e-8)
        magnitude = 10 ** math.floor(math.log10(bound))
        step = next(x * magnitude for x in (1, 2, 5, 10) if x * magnitude >= bound / 2)
        extent = math.ceil(bound / step) * step
        def x(value):
            return 2.45 + 4.6 * (value + extent) / (2 * extent)
        top = -3.40 * panel
        lines.append(r'\node[anchor=west] at (0,' + f'{top:.3f}' + ') {' + PANEL_LABELS[contrast] + '};')
        for index, (surface, family) in enumerate(itertools.product(SURFACES, FAMILIES)):
            y = top - 0.42 - index * 0.33
            row = contrasts[surface, family, contrast, 'ap']
            lo, hi, pt = (100 * float(row[k]) for k in ('ci_low', 'ci_high', 'difference'))
            label = ('T' if surface == 'turf' else 'D') + ' ' + FAMILY_LABELS[family]
            lines.append(f'\\node[anchor=east] at (2.32,{y:.3f}) {{{label}}};')
            lines.append(f'\\draw ({x(lo):.4f},{y:.3f}) -- ({x(hi):.4f},{y:.3f});')
            for endpoint in (lo, hi):
                lines.append(f'\\draw ({x(endpoint):.4f},{y-0.06:.3f}) -- ({x(endpoint):.4f},{y+0.06:.3f});')
            lines.append(f'\\fill ({x(pt):.4f},{y:.3f}) circle (1.4pt);')
        axis = top - 2.50
        lines.append(f'\\draw[densely dotted] ({x(0):.4f},{top-0.30:.3f}) -- ({x(0):.4f},{axis:.3f});')
        lines.append(f'\\draw (2.45,{axis:.3f}) -- (7.05,{axis:.3f});')
        for tick in range(-int(round(extent / step)), int(round(extent / step)) + 1):
            value = tick * step
            label = '0' if tick == 0 else f'{value:.3g}'
            lines.append(f'\\draw ({x(value):.4f},{axis:.3f}) -- ({x(value):.4f},{axis-0.06:.3f}) '
                         f'node[below] {{{label}}};')
    lines += [r'\end{tikzpicture}',
              r'\caption{All prespecified AP differences, in percentage points (100 times the fraction difference). '
              r'T and D denote turf and dirt. Points average paired seed-specific differences; whiskers are '
              r'descriptive 95\% paired whole-race bootstrap percentile intervals, conditional on the fitted models '
              r'and unadjusted for multiple comparisons. Panels use their indicated scales. The test was previously examined. Denominators and '
              r'coverage are in Tables~\ref{tab:cohort} and \ref{tab:focused_coverage}.}',
              r'\label{fig:focused_contrasts}', r'\end{figure}', '']
    return '\n'.join(lines)


def write_coverage(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate(report_path, inputs, out):
    report_bytes = Path(report_path).read_bytes()
    report = json.loads(report_bytes)
    payloads = {name: Path(inputs[name]).read_bytes() for name in FILES}
    _, arms, contrasts, coverage = validate(report, payloads)
    out = Path(out)
    out.mkdir(parents=False, exist_ok=False)
    (out / 'focused_scores.tex').write_text(table(arms), encoding='utf-8')
    (out / 'focused_contrasts.tex').write_text(plot(contrasts), encoding='utf-8')
    companion = out / 'companion'
    companion.mkdir()
    for name, data in payloads.items():
        (companion / name).write_bytes(data)
    write_coverage(companion / 'coverage_summary.csv', coverage)
    (out / 'PROVENANCE.json').write_text(json.dumps({
        'evaluation_report_sha256': digest(report_bytes),
        'input_sha256': {name: digest(data) for name, data in payloads.items()},
        'generator_sha256': digest(Path(__file__).read_bytes()),
        'note': 'Hash agreement is not an independent scientific audit or authorization.'
    }, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, help='evaluation_report.json')
    parser.add_argument('--seed-metrics', required=True, help='hash-bound seed_metrics.csv')
    parser.add_argument('--arm-metrics', required=True, help='hash-bound arm_metrics.csv')
    parser.add_argument('--contrasts', required=True, help='hash-bound contrasts.csv')
    parser.add_argument('--out', required=True, help='new output directory; parent must exist')
    args = parser.parse_args()
    generate(args.report, dict(zip(FILES, (args.seed_metrics, args.arm_metrics, args.contrasts))), args.out)


if __name__ == '__main__':
    main()
