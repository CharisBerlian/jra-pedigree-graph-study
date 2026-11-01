"""Summarize frozen validation-only candidate and refit configuration evidence.

No prediction files, source rows, test features, labels, or test metrics are read.
The expected digest must be independently supplied from the reviewed freeze.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

FAMILIES = ('xgboost', 'lightgbm', 'catboost')
SURFACES = ('turf', 'dirt')
ARMS = ('H', 'Gmixed', 'Gancestor', 'P')
SEEDS = (0, 1, 2)
CANDIDATES = ('c0', 'c1', 'c2', 'c3')
PREFIT_SHA = '512e0e87831f9b8b30b28ba1e52ff1327e5e3101039e542f54ea37430a7bf7fa'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path, digest=None):
    data = path.read_bytes()
    if digest is not None:
        require(isinstance(digest, str) and sha(data) == digest.lower(), f'hash mismatch: {path.name}')
    return json.loads(data)


def fraction(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
            f'{name}: invalid fraction')
    return float(value)


def inspect(manifest, root):
    require(manifest.get('status') == 'frozen_for_root_review'
            and manifest.get('prefit_sha256') == PREFIT_SHA
            and manifest.get('test_previously_seen') is True
            and manifest.get('test_inference_authorized') is False, 'incorrect freeze status or design')
    require((manifest.get('tuning_cells'), manifest.get('candidate_count'), manifest.get('model_count'))
            == (24, 96, 72), 'wrong freeze totals')
    models = manifest.get('models')
    require(isinstance(models, list) and len(models) == 24, 'wrong model cell count')
    cells = {}
    for item in models:
        cell = (item['family'], item['surface'], item['arm'])
        require(cell not in cells, 'duplicate model cell')
        cells[cell] = item
    require(set(cells) == set(itertools.product(FAMILIES, SURFACES, ARMS)), 'wrong model cell keys')
    root = Path(root).resolve(strict=True)
    candidates, selections = [], []
    for family, surface, arm in itertools.product(FAMILIES, SURFACES, ARMS):
        model = cells[family, surface, arm]
        base = root / family / surface / arm / 'runs' / 'scientific' / family / surface / arm
        require(base.resolve(strict=True).is_relative_to(root), 'invalid cell path')
        tune = base / 'tune' / 'seed0'
        read_json(tune / 'complete.json', model['tune_complete_sha256'])
        selected = read_json(tune / 'selected.json', model['selected_sha256'])
        records = read_json(tune / 'candidate_results.json', model['candidate_results_sha256'])
        evidence = model['candidate_evidence']
        require(isinstance(records, list) and len(records) == 4 and isinstance(evidence, list)
                and len(evidence) == 4, 'wrong candidate count')
        byid = {r['candidate_id']: r for r in records}
        evbyid = {r['candidate_id']: r for r in evidence}
        require(set(byid) == set(evbyid) == set(CANDIDATES), 'wrong candidate keys')
        for cid in CANDIDATES:
            record, ev = byid[cid], evbyid[cid]
            folder = tune / 'candidates' / cid
            require(read_json(folder / 'metrics.json') == record, 'candidate metrics disagreement')
            curvefile = read_json(folder / 'ap_curve.json')
            curve = curvefile.get('values')
            require(curvefile.get('metric') == 'sklearn.metrics.average_precision_score'
                    and isinstance(curve, list) and 1 <= len(curve) <= 1200
                    and curve == record.get('curve'), 'candidate curve disagreement')
            values = [fraction(v, 'curve AP') for v in curve]
            best = max(values)
            earliest = next(i for i, v in enumerate(values, 1) if abs(v - best) <= 1e-15)
            cap = len(values) >= 1200
            require(record.get('status') == 'complete' and type(record.get('hit_cap')) is bool
                    and record['hit_cap'] == cap and type(record.get('best_round')) is int
                    and record['best_round'] == earliest
                    and abs(fraction(record.get('average_precision'), 'validation AP') - best) <= 1e-12,
                    'candidate AP, best round or cap mismatch')
            hit = fraction(record.get('mean_hit_at_3'), 'validation Hit@3')
            require(ev == {'candidate_id': cid, 'validation_ap': record['average_precision'],
                           'validation_hit_at_3': record['mean_hit_at_3'],
                           'best_round': record['best_round'], 'hit_cap': cap}, 'freeze evidence mismatch')
            candidates.append({'family': family, 'surface': surface, 'arm': arm, 'candidate_id': cid,
                               'validation_ap': record['average_precision'], 'validation_hit_at_3': hit,
                               'curve_rounds': len(curve), 'curve_hit_cap': cap, 'best_round': earliest,
                               'selected': cid == model['selected_candidate']})
        maximum = max(r['average_precision'] for r in records)
        eligible = [r for r in records if maximum - r['average_precision'] <= 1e-6]
        winner = min(eligible, key=lambda r: (-r['mean_hit_at_3'], r['best_round'], r['candidate_id']))
        require(selected == winner and model['selected_candidate'] == winner['candidate_id']
                and model['selected_round'] == winner['best_round'], 'selected candidate mismatch')
        refits = model['refits']
        require(isinstance(refits, list) and len(refits) == 3
                and {r['seed'] for r in refits} == set(SEEDS), 'wrong refit seed set')
        for refit in refits:
            seed = refit['seed']
            relative = (base / 'refit' / f'seed{seed}').relative_to(root).as_posix()
            require(refit['relative_dir'] == relative, 'unexpected refit path')
            folder = root / relative
            read_json(folder / 'complete.json', refit['complete_sha256'])
            config = read_json(folder / 'config.json', refit['config_sha256'])
            require(config.get('family') == family and config.get('surface') == surface
                    and config.get('setting') == arm and config.get('seed') == seed
                    and config.get('selected_candidate') == winner['candidate_id']
                    and config.get('selected_round') == winner['best_round'], 'refit configuration mismatch')
        selections.append({'family': family, 'surface': surface, 'arm': arm,
                           'selected_candidate': winner['candidate_id'],
                           'validation_ap': winner['average_precision'],
                           'validation_hit_at_3': winner['mean_hit_at_3'],
                           'selected_curve_rounds': len(winner['curve']),
                           'selected_curve_hit_cap': winner['hit_cap'],
                           'selected_best_round': winner['best_round'],
                           'selected_best_round_at_cap': winner['best_round'] == 1200,
                           'refit_seeds': '0,1,2', 'all_refit_rounds': winner['best_round']})
    require(len(candidates) == 96 and len(selections) == 24, 'incorrect output totals')
    require(sum(r['curve_hit_cap'] for r in candidates) == manifest['candidate_cap_count'], 'cap count mismatch')
    return candidates, selections


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--expected-sha256', required=True, help='independently reviewed manifest SHA-256')
    parser.add_argument('--harvest-root', required=True)
    parser.add_argument('--out', required=True, help='new output directory; parent must exist')
    args = parser.parse_args()
    require(len(args.expected_sha256) == 64 and all(c in '0123456789abcdef' for c in args.expected_sha256.lower()),
            'expected manifest hash must be 64 hex digits')
    manifest = read_json(Path(args.manifest), args.expected_sha256)
    candidates, selections = inspect(manifest, args.harvest_root)
    out = Path(args.out)
    out.mkdir(parents=False, exist_ok=False)
    write_csv(out / 'validation_candidates.csv', candidates)
    write_csv(out / 'validation_selections.csv', selections)
    summary = {'status': 'complete_validation_only', 'manifest_sha256': args.expected_sha256.lower(),
               'source_sha256': sha(Path(__file__).read_bytes()),
               'candidate_count': len(candidates), 'tuning_cell_count': len(selections),
               'candidate_curves_at_cap': sum(r['curve_hit_cap'] for r in candidates),
               'selected_curves_at_cap': sum(r['selected_curve_hit_cap'] for r in selections),
               'selected_best_rounds_at_cap': sum(r['selected_best_round_at_cap'] for r in selections),
               'refit_configs_checked': 3 * len(selections),
               'outputs': {name: sha((out / name).read_bytes()) for name in
                           ('validation_candidates.csv', 'validation_selections.csv')},
               'limitations': 'Validation-only accounting; no test data or prediction rows were read.'}
    (out / 'SUMMARY.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
