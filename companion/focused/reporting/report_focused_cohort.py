"""Aggregate six reviewed label-free test views; never open predictions or outcomes."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
from importlib import metadata
import itertools
import json
import math
from pathlib import Path
import sys

VIEWS = ('Gmixed', 'Gancestor')
SEEDS = (0, 1, 2)
SURFACES = ('turf', 'dirt')
ROLES = ('sire', 'dam', 'damsire')
ENTRY_COUNTS = {'turf': 43477, 'dirt': 45660}
RACE_COUNTS = {'turf': 3246, 'dirt': 3219}
FREEZE_SHA = '09200d2194e71836e6093c03c4e5d35a395ab859ad2488633e7cbc0657c7715e'
VIEW_REPORT_SHA = '774929c47352c3d808aca60804ab6583e6fbc3aa7e1b343372295ac704ff15ad'
QUALITY_SHA = '21c08cee7b9d28cecb4eef8a12eb3b932d99faccd2bf835da5226df58b7a5fc2'
GRAPH_QUALITY_SHA = '7445914ef8b24f971dc78cda06d4e7fabc79cb67dab471695daa7dfcf3951a50'
INFERENCE_REPORT_SHA = '225f1bdb391f5ffeff06242ec550bdf96e551b273aceda3118add74c78433cd9'
FEASIBILITY_SHA = '66c92c84c7fbb246b2528d7f47dad003244a249390a7b228ef3db8b08bbd7ecb'
FORBIDDEN = {'label', 'finish_pos', 'result_status', 'time', 'margin', 'last_3f', 'corner_passing', 'top_three'}
P_FIELDS = tuple(f'{r}_{suffix}' for r in ROLES for suffix in ('available', 'prior_count', 'prior_rate'))


def check(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path, expected):
    check(digest(path) == expected, f'input checksum mismatch: {path.name}')
    return json.loads(path.read_text(encoding='utf-8'))


def validate_quality(views, quality, graph, inference):
    check(views['status'] == 'complete' and not views['test_labels_read']
          and not views['test_metrics_computed'] and views['model_manifest_sha256'] == FREEZE_SHA,
          'view-report status')
    check(quality['status'] == graph['status'] == 'pass'
          and not quality['test_labels_read'] and not quality['test_metrics_computed']
          and not graph['test_labels_read'] and not graph['test_metrics_computed']
          and quality['model_manifest_sha256'] == graph['model_manifest_sha256'] == FREEZE_SHA
          and quality['test_view_report_sha256'] == VIEW_REPORT_SHA
          and graph['view_verification_sha256'] == QUALITY_SHA, 'prior audit binding')
    check(quality['authorization_sha256'] == graph['authorization_sha256'] == views['authorization_sha256'],
          'authorization binding')
    check(inference['status'] == 'complete' and not inference['test_labels_read']
          and not inference['test_metrics_computed'] and inference['model_manifest_sha256'] == FREEZE_SHA
          and inference['authorization_sha256'] == views['authorization_sha256'], 'inference-report binding')
    items = {(r['view'], r['seed']): r for r in views['views']}
    qa = {(r['view'], r['seed']): r for r in quality['views']}
    wanted = set(itertools.product(VIEWS, SEEDS))
    check(len(views['views']) == len(items) == len(quality['views']) == len(qa) == 6
          and set(items) == set(qa) == wanted, 'missing or duplicate views')
    check(len(graph['views']) == 3 and {x['seed'] for x in graph['views']} == set(SEEDS)
          and all(x['rows'] == 89137 for x in graph['views']), 'all-row graph audit coverage')
    for key, item in items.items():
        check(item['data_sha256'] == qa[key]['data_sha256']
              and item['sidecar_sha256'] == qa[key]['sidecar_sha256'] and qa[key]['rows'] == 89137,
              'prior view-check binding')
        check(inference['inputs'][f'{key[0]}/seed{key[1]}']['sha256'] == item['data_sha256'],
              'inference view binding')
    return items


def read_view(path, view, seed, reference, inference, build_reference=False):
    # Keys and P fields exist in memory only and are not written to outputs.
    seen = set()
    groups = {}
    count = Counter()
    joint = Counter()
    parents = Counter()
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames
        check(header and len(header) == len(set(header)) and not (set(header) & FORBIDDEN), 'invalid label-free schema')
        required = {'race_id', 'horse_id_netkeiba', 'horse_number', 'surface', 'field_size', 'split',
                    'embedding_status', 'fallback_depth', 'ancestor_count', 'embedding_missing'} | set(P_FIELDS)
        check(required <= set(header), 'missing feature columns')
        for row in reader:
            check(None not in row and all(v is not None for v in row.values()) and row['split'] == 'test', 'malformed row')
            surface = row['surface'].lower()
            check(surface in SURFACES, 'invalid surface')
            number = int(row['horse_number'])
            key = (row['race_id'], row['horse_id_netkeiba'], number)
            check(all(key) and key not in seen, 'duplicate or empty roster key')
            seen.add(key)
            field_size = int(row['field_size'])
            check(field_size >= 3, 'small race')
            group = groups.setdefault((surface, key[0]), [field_size, 0, set()])
            check(group[0] == field_size and number not in group[2], 'roster conflict')
            group[1] += 1
            group[2].add(number)
            values = (surface, field_size, *(row[name] for name in P_FIELDS))
            if build_reference:
                reference[key] = values
            else:
                check(reference.get(key) == values, 'key, surface, field size or P-feature mismatch across views')
            count[surface] += 1
            status = row['embedding_status']
            check(status in ('direct', 'fallback', 'missing') and (view != 'Gancestor' or status != 'direct'), 'graph status')
            missing = int(row['embedding_missing'])
            n = int(row['ancestor_count'])
            depth = row['fallback_depth']
            check(missing == int(status == 'missing') and n >= 0, 'invalid graph missingness')
            if status == 'direct':
                check(depth == '0' and n == 0, 'invalid direct representation')
            elif status == 'fallback':
                check(depth in ('1', '2', '3') and n > 0, 'invalid ancestor representation')
            else:
                check(depth == '' and n == 0, 'invalid missing representation')
            joint[surface, status, depth, n, missing] += 1
            for role in ROLES:
                available = int(row[role + '_available'])
                prior = int(row[role + '_prior_count'])
                rate = float(row[role + '_prior_rate'])
                check(available in (0, 1) and prior >= 0 and math.isfinite(rate) and 0 <= rate <= 1,
                      'invalid parent feature')
                if not available:
                    check(prior == 0 and math.isclose(rate, 0.2, abs_tol=1e-12), 'invalid missing-parent prior')
                parents[surface, role, 'available'] += available
                parents[surface, role, 'missing'] += 1 - available
                parents[surface, role, 'known_zero'] += int(available == 1 and prior == 0)
                parents[surface, role, 'zero'] += int(prior == 0)
    check(len(seen) == 89137 and dict(count) == ENTRY_COUNTS, 'incomplete surface roster')
    check(all(x[0] == x[1] for x in groups.values()), 'incomplete race roster')
    check({s: sum(1 for k in groups if k[0] == s) for s in SURFACES} == RACE_COUNTS, 'race counts mismatch')
    check(len(reference) == len(seen), 'cross-view roster mismatch')
    for s in SURFACES:
        summary = inference['inputs'][f'{view}/seed{seed}']['surfaces'][s]
        status_counts = {status: sum(v for (surface, st, depth, n, missing), v in joint.items()
                                     if surface == s and st == status) for status in ('direct', 'fallback', 'missing')}
        check(summary['entries'] == count[s]
              and all(summary['coverage'].get(k, 0) == v for k, v in status_counts.items()), 'inference coverage mismatch')
        check(all(summary['parent_zero_counts'][r] == parents[s, r, 'zero'] for r in ROLES), 'inference parent zero mismatch')
    return joint, parents


def write(path, rows):
    check(rows, 'empty output')
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('views', 'views-quality', 'graph-quality', 'inference-report', 'feasibility', 'out'):
        parser.add_argument('--' + option, required=True, type=Path)
    args = parser.parse_args()
    view_report = load(args.views / 'test_view_report.json', VIEW_REPORT_SHA)
    quality = load(args.views_quality, QUALITY_SHA)
    graph = load(args.graph_quality, GRAPH_QUALITY_SHA)
    inference = load(args.inference_report, INFERENCE_REPORT_SHA)
    feasibility = load(args.feasibility, FEASIBILITY_SHA)
    items = validate_quality(view_report, quality, graph, inference)
    baseline = {}
    representation, distribution, parent_output = [], [], []
    first = True
    for view, seed in itertools.product(VIEWS, SEEDS):
        item = items[view, seed]
        path = args.views / f'{view}-seed{seed}.csv.gz'
        side = args.views / f'{view}-seed{seed}.csv.json'
        check(digest(path) == item['data_sha256'] and digest(side) == item['sidecar_sha256'], 'view or sidecar hash')
        sidecar = json.loads(side.read_text(encoding='utf-8'))
        check(sidecar['view'] == view and sidecar['seed'] == seed and sidecar['data_sha256'] == item['data_sha256']
              and sidecar['model_manifest_sha256'] == FREEZE_SHA
              and sidecar['authorization_sha256'] == view_report['authorization_sha256']
              and sidecar['splits'] == ['test'] and not sidecar['test_labels_read']
              and not sidecar['test_metrics_computed'], 'invalid sidecar')
        joint, parents = read_view(path, view, seed, baseline, inference, build_reference=first)
        first = False
        for surface in SURFACES:
            def sum_filter(**wanted):
                return sum(v for (s, status, depth, n, missing), v in joint.items()
                           if s == surface and all({'status': status, 'depth': depth, 'missing': missing}[k] == x
                                                   for k, x in wanted.items()))
            representation.append({'view': view, 'seed': seed, 'surface': surface,
                'entries': ENTRY_COUNTS[surface], 'races': RACE_COUNTS[surface],
                'direct': sum_filter(status='direct'), 'fallback': sum_filter(status='fallback'),
                'missing': sum_filter(status='missing'),
                'depth_1': sum_filter(status='fallback', depth='1'),
                'depth_2': sum_filter(status='fallback', depth='2'),
                'depth_3': sum_filter(status='fallback', depth='3')})
            for (s, status, depth, n, missing), entries in sorted(joint.items()):
                if s == surface:
                    distribution.append({'view': view, 'seed': seed, 'surface': s, 'status': status,
                        'depth': depth, 'ancestor_count': n, 'missing': missing, 'entries': entries,
                        'fraction_of_surface_entries': entries / ENTRY_COUNTS[s]})
            for role in ROLES:
                counts = {k: parents[surface, role, k] for k in ('available', 'missing', 'known_zero', 'zero')}
                check(counts['available'] + counts['missing'] == ENTRY_COUNTS[surface]
                      and counts['known_zero'] + counts['missing'] == counts['zero'], 'parent count identity')
                parent_output.append({'view': view, 'seed': seed, 'surface': surface, 'role': role,
                    'entries': ENTRY_COUNTS[surface], 'id_available': counts['available'],
                    'id_available_fraction': counts['available'] / ENTRY_COUNTS[surface],
                    'id_missing': counts['missing'], 'id_missing_fraction': counts['missing'] / ENTRY_COUNTS[surface],
                    'known_id_zero_prior': counts['known_zero'],
                    'all_zero_prior': counts['zero'], 'zero_prior_fraction': counts['zero'] / ENTRY_COUNTS[surface],
                    'known_id_zero_prior_fraction': counts['known_zero'] / ENTRY_COUNTS[surface],
                    'known_id_zero_given_available_fraction':
                        counts['known_zero'] / counts['available'] if counts['available'] else ''})
    trainval = []
    expected = {('train', 'turf'): 271549, ('train', 'dirt'): 277207,
                ('validation', 'turf'): 42783, ('validation', 'dirt'): 45323}
    for split, surface in itertools.product(('train', 'validation'), SURFACES):
        row = feasibility['train_validation_by_split_surface'][f'{split}/{surface}']
        entries = expected[split, surface]
        depths = [row.get(f'ancestor_depth_{i}', 0) for i in (1, 2, 3)]
        check(row['rows'] == entries and sum(depths) == entries, 'train/validation feasibility mismatch')
        trainval.append({'split': split, 'surface': surface, 'entries': entries,
                         'ancestor_depth_1': depths[0], 'ancestor_depth_2': depths[1], 'ancestor_depth_3': depths[2],
                         'sire_id_available': row['sire_available'], 'dam_id_available': row['dam_available'],
                         'damsire_id_available': row['damsire_available']})
    out = args.out
    out.mkdir(parents=False, exist_ok=False)
    outputs = {'test_representation.csv': representation, 'test_joint_distribution.csv': distribution,
               'test_parent_availability.csv': parent_output, 'train_validation_feasibility.csv': trainval}
    for name, rows in outputs.items():
        write(out / name, rows)
    summary = {'status': 'label_free_aggregate_complete', 'test_labels_read': False, 'test_metrics_read': False,
        'views': 6, 'rows_per_view': 89137, 'entries_by_surface': ENTRY_COUNTS, 'races_by_surface': RACE_COUNTS,
        'test_view_report_sha256': VIEW_REPORT_SHA, 'views_quality_sha256': QUALITY_SHA,
        'graph_parent_quality_sha256': GRAPH_QUALITY_SHA, 'inference_report_sha256': INFERENCE_REPORT_SHA,
        'train_validation_feasibility_sha256': FEASIBILITY_SHA, 'model_freeze_sha256': FREEZE_SHA,
        'view_file_sha256': {f'{view}/seed{seed}': items[view, seed]['data_sha256'] for view, seed in items},
        'reporter_sha256': digest(Path(__file__)), 'qa_python': sys.version.split()[0],
        'qa_numpy': metadata.version('numpy'), 'native_view_runtime_python': '3.13.15',
        'native_view_runtime_numpy': '1.26.4',
        'scope': 'Counts of label-free entries; cross-view P parity, complete race rosters, graph coverage and parent availability. No predictions, outcome labels or test performance were read.',
        'outputs': {name: digest(out / name) for name in outputs}}
    (out / 'SUMMARY.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'views': 6, 'rows_per_view': 89137,
                      'output_rows': {name: len(rows) for name, rows in outputs.items()}}, sort_keys=True))


if __name__ == '__main__':
    main()
