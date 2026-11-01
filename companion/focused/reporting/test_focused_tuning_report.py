"""Synthetic validation-accounting fixtures; no experimental data or fits."""
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import report_focused_tuning as report


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True) + '\n').encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def fixture(root):
    manifest = {'status': 'frozen_for_root_review', 'prefit_sha256': report.PREFIT_SHA,
                'test_previously_seen': True, 'test_inference_authorized': False,
                'tuning_cells': 24, 'candidate_count': 96, 'model_count': 72,
                'candidate_cap_count': 1, 'models': []}
    for index, (family, surface, arm) in enumerate(itertools.product(report.FAMILIES, report.SURFACES, report.ARMS)):
        base = root / family / surface / arm / 'runs' / 'scientific' / family / surface / arm
        tune = base / 'tune' / 'seed0'
        records, evidence = [], []
        for j, cid in enumerate(report.CANDIDATES):
            capped = index == 0 and cid == 'c0'
            curve = [0.3 if cid == 'c0' else 0.2 + j / 100] * (1200 if capped else 2)
            record = {'candidate_id': cid, 'status': 'complete', 'average_precision': max(curve),
                      'mean_hit_at_3': 0.4, 'best_round': 1, 'hit_cap': capped, 'curve': curve}
            folder = tune / 'candidates' / cid
            put(folder / 'metrics.json', record)
            put(folder / 'ap_curve.json', {'metric': 'sklearn.metrics.average_precision_score', 'values': curve})
            records.append(record)
            evidence.append({'candidate_id': cid, 'validation_ap': record['average_precision'],
                             'validation_hit_at_3': record['mean_hit_at_3'], 'best_round': 1, 'hit_cap': capped})
        model = {'family': family, 'surface': surface, 'arm': arm, 'selected_candidate': 'c0',
                 'selected_round': 1, 'candidate_evidence': evidence,
                 'tune_complete_sha256': put(tune / 'complete.json', {'status': 'complete'}),
                 'selected_sha256': put(tune / 'selected.json', records[0]),
                 'candidate_results_sha256': put(tune / 'candidate_results.json', records), 'refits': []}
        for seed in report.SEEDS:
            folder = base / 'refit' / f'seed{seed}'
            config = {'family': family, 'surface': surface, 'setting': arm, 'seed': seed,
                      'selected_candidate': 'c0', 'selected_round': 1}
            model['refits'].append({'seed': seed, 'relative_dir': folder.relative_to(root).as_posix(),
                                    'complete_sha256': put(folder / 'complete.json', {'status': 'complete'}),
                                    'config_sha256': put(folder / 'config.json', config)})
        manifest['models'].append(model)
    return manifest


class TuningReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_full_matrix_and_curve_cap_is_not_best_round(self):
        candidates, selections = report.inspect(self.manifest, self.root)
        self.assertEqual((len(candidates), len(selections)), (96, 24))
        self.assertEqual(sum(r['curve_hit_cap'] for r in candidates), 1)
        self.assertTrue(selections[0]['selected_curve_hit_cap'])
        self.assertEqual(selections[0]['selected_best_round'], 1)
        self.assertFalse(selections[0]['selected_best_round_at_cap'])
        self.assertEqual(selections[0]['all_refit_rounds'], 1)

    def test_missing_or_duplicate_cell_rejected(self):
        self.manifest['models'][-1] = self.manifest['models'][0]
        with self.assertRaises(ValueError):
            report.inspect(self.manifest, self.root)

    def test_changed_refit_config_rejected(self):
        model = self.manifest['models'][0]
        path = self.root / model['refits'][0]['relative_dir'] / 'config.json'
        put(path, {'selected_round': 1200})
        with self.assertRaises(ValueError):
            report.inspect(self.manifest, self.root)

    def test_changed_curve_rejected(self):
        path = self.root / 'xgboost/turf/H/runs/scientific/xgboost/turf/H/tune/seed0/candidates/c0/ap_curve.json'
        put(path, {'metric': 'sklearn.metrics.average_precision_score', 'values': [0.4]})
        with self.assertRaises(ValueError):
            report.inspect(self.manifest, self.root)

    def test_test_authorization_state_rejected(self):
        self.manifest['test_inference_authorized'] = True
        with self.assertRaises(ValueError):
            report.inspect(self.manifest, self.root)


if __name__ == '__main__':
    unittest.main()
