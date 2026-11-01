"""Synthetic in-memory validation tests; no figures or scientific outputs."""
import csv
import io
import itertools
import json
import statistics
import unittest

import generate_focused_ieee as g


def encode(name, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=g.HEADERS[name])
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def fixture():
    seeds, arms, contrasts = [], [], []
    for surface, family, arm in itertools.product(g.SURFACES, g.FAMILIES, g.ARMS):
        values = [0.2 + 0.01 * g.ARMS.index(arm) + 0.001 * seed for seed in g.SEEDS]
        for seed, value in zip(g.SEEDS, values):
            seeds.append(dict(surface=surface, family=family, arm=arm, seed=seed,
                              average_precision=value, hit_at_3=value))
        arms.append(dict(surface=surface, family=family, arm=arm, ap_mean=statistics.mean(values),
                         ap_seed_sd=statistics.stdev(values), hit_mean=statistics.mean(values),
                         hit_seed_sd=statistics.stdev(values)))
    for surface, family, contrast, metric in itertools.product(g.SURFACES, g.FAMILIES, g.CONTRASTS, g.METRICS):
        left, right = contrast.split('-')
        point = 0.01 * (g.ARMS.index(left) - g.ARMS.index(right))
        low, high = point - 0.01, point + 0.01
        row = dict(surface=surface, family=family, contrast=contrast, metric=metric,
                   difference=point, ci_low=low, ci_high=high,
                   correct_per_100_races='', correct_per_100_low='', correct_per_100_high='')
        if metric == 'hit':
            row.update(correct_per_100_races=300 * point, correct_per_100_low=300 * low,
                       correct_per_100_high=300 * high)
        contrasts.append(row)
    payloads = {name: encode(name, rows) for name, rows in zip(g.FILES, (seeds, arms, contrasts))}
    coverage = {}
    for view, seed in itertools.product(('Gmixed', 'Gancestor'), g.SEEDS):
        surfaces = {}
        for surface in g.SURFACES:
            n = g.SURFACE_ENTRIES[surface]
            surfaces[surface] = {'entries': n, 'coverage': {'fallback': n},
                                 'parent_zero_counts': {'sire': 2, 'dam': 3, 'damsire': 4}}
        coverage[f'{view}/seed{seed}'] = {'surfaces': surfaces}
    report = {'status': 'complete', 'exploratory': True,
              'outputs': {name: g.digest(data) for name, data in payloads.items()},
              'coverage_and_parent_zero_counts': coverage}
    return report, payloads, (seeds, arms, contrasts)


def replace(report, payloads, name, rows):
    data = encode(name, rows)
    payloads[name] = data
    report['outputs'][name] = g.digest(data)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.report, self.payloads, self.rows = fixture()

    def test_complete_matrix_and_coverage(self):
        seed, arm, contrast, coverage = g.validate(self.report, self.payloads)
        self.assertEqual((len(seed), len(arm), len(contrast), len(coverage)), (72, 24, 36, 12))

    def test_hash_mismatch(self):
        self.payloads[g.FILES[0]] += b'\n'
        with self.assertRaises(ValueError):
            g.validate(self.report, self.payloads)

    def test_missing_and_duplicate_key(self):
        rows = self.rows[0]
        rows[-1] = dict(rows[0])
        replace(self.report, self.payloads, g.FILES[0], rows)
        with self.assertRaises(ValueError):
            g.validate(self.report, self.payloads)

    def test_extra_or_wrong_key(self):
        rows = self.rows[2]
        rows[0]['contrast'] = 'P-H'
        replace(self.report, self.payloads, g.FILES[2], rows)
        with self.assertRaises(ValueError):
            g.validate(self.report, self.payloads)

    def test_inconsistent_summary(self):
        rows = self.rows[1]
        rows[0]['ap_mean'] = 0.99
        replace(self.report, self.payloads, g.FILES[1], rows)
        with self.assertRaises(ValueError):
            g.validate(self.report, self.payloads)

    def test_nonfinite_and_invalid_interval(self):
        rows = self.rows[2]
        rows[0]['ci_low'] = 'nan'
        replace(self.report, self.payloads, g.FILES[2], rows)
        with self.assertRaises(ValueError):
            g.validate(self.report, self.payloads)

    def test_coverage_and_exploratory_gate(self):
        self.report['exploratory'] = False
        with self.assertRaises(ValueError):
            g.validate(self.report, self.payloads)
        self.report['exploratory'] = True
        self.report['coverage_and_parent_zero_counts']['Gancestor/seed2']['surfaces']['turf']['entries'] -= 1
        with self.assertRaises(ValueError):
            g.validate(self.report, self.payloads)


if __name__ == '__main__':
    unittest.main()
