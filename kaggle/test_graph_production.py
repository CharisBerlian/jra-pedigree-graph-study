"""Local checks for production orchestration; no ML libraries or fitting."""
import ast
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph_production as production
import review_production_outputs as review


class FakeArray:
    def __init__(self, values, shape=None):
        self.values = list(values)
        self.shape = shape or (len(self.values),)

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, index):
        if isinstance(index, (list, tuple)):
            return FakeArray([self.values[i] for i in index])
        return self.values[index]


class FakeRng:
    def choice(self, size, count, replace=False):
        return list(range(count))


class FakeNumpy:
    class random:
        @staticmethod
        def default_rng(seed):
            return FakeRng()

    @staticmethod
    def unique(values):
        return sorted(set(values))


class FakeKMeans:
    def __init__(self, n_clusters, n_init, max_iter, random_state):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit_predict(self, values):
        self.labels_ = FakeArray([i % self.n_clusters for i in range(len(values))])
        self.cluster_centers_ = FakeArray(
            [[0] * values.shape[1] for _ in range(self.n_clusters)],
            (self.n_clusters, values.shape[1]))
        self.inertia_ = float(self.n_clusters)
        self.n_iter_ = 2
        return self.labels_


def fake_silhouette(values, labels):
    # Deterministic stand-in: tests orchestration and fixed sampling only.
    return 1.0 / len(set(labels))


class ProductionChecks(unittest.TestCase):
    def test_selection_tie_prefers_smaller_k(self):
        diagnostics = {"8": {"silhouette_sample": 0.2},
                       "4": {"silhouette_sample": 0.2},
                       "16": {"silhouette_sample": None}}
        self.assertEqual(production.choose_k(diagnostics), 4)

    def test_cluster_diagnostics_use_mock_fit_only(self):
        values = FakeArray([[float(i), float(i + 1)] for i in range(15)], (15, 2))
        settings = dict(production.PRODUCTION_SETTINGS, silhouette_sample=10)
        diagnostics, models = production.cluster_training_vectors(
            values, [str(i) for i in range(len(values))], [4, 8], 0,
            settings, FakeKMeans, fake_silhouette, FakeNumpy)
        self.assertEqual(set(diagnostics), {"4", "8"})
        self.assertEqual(diagnostics["4"]["n_iter"], 2)
        self.assertEqual(sum(diagnostics["8"]["cluster_sizes"].values()), 15)
        self.assertEqual(len(models[4][1]), 15)

    def test_completed_run_refuses_config_mismatch_and_same_config(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            config = {"settings": {"seed": 0}}
            wanted = production.config_hash(config)
            (folder / "run_status.json").write_text(json.dumps({
                "status": "complete", "config_hash": wanted}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "different config"):
                production.begin_run(folder, {"settings": {"seed": 1}})
            with self.assertRaisesRegex(RuntimeError, "already complete"):
                production.begin_run(folder, config)

    def test_stale_run_is_recorded_before_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "run_status.json").write_text(json.dumps({
                "status": "running", "started_utc": "old"}), encoding="utf-8")
            value = production.begin_run(folder, {"settings": {"seed": 0}})
            stale = json.loads((folder / "stale_run.json").read_text(encoding="utf-8"))
            current = json.loads((folder / "run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(value, current["config_hash"])
            self.assertEqual(stale["status"], "running")
            self.assertEqual(current["previous_status"], "stale")

    def test_kaggle_guard_prevents_local_fitting(self):
        class Args:
            input_dir = Path("unused")
            output_dir = Path("unused")
            seed = 0
            clusters = None
        with self.assertRaisesRegex(RuntimeError, "in Kaggle"):
            production.run(Args())

    def test_selection_rejects_nonfinite_silhouette(self):
        with self.assertRaisesRegex(ValueError, "valid silhouette"):
            production.choose_k({"4": {"silhouette_sample": float("nan")}})

    def test_review_rejects_one_sided_config_hash_mismatch(self):
        saved = {"input_manifest_sha256": "i", "graph_sha256": "g",
                 "script_sha256": "s", "settings": {"seed": 0},
                 "candidate_clusters": [4]}
        digest = review.config_hash(saved)
        report = {"status": "complete", "config_hash": digest, **saved}
        status = {"status": "complete", "config_hash": digest}
        run_config = {"config_hash": digest, **saved}
        status["config_hash"] = "one-sided"
        with self.assertRaisesRegex(ValueError, "hashes differ"):
            review.validate_run_metadata(report, status, run_config)

    def test_review_selection_ignores_nonfinite_scores(self):
        self.assertEqual(review.choose_k({"4": {"silhouette_sample": float("nan")},
                                          "8": {"silhouette_sample": 0.2}}), 8)

    def test_outcome_manifest_is_rejected_before_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "manifest.json").write_text(json.dumps({
                "purpose": "training-only pedigree resource pilot",
                "outcome_columns_exported": True,
                "validation_test_roots_exported": False}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exports outcomes"):
                production.load_verified_inputs(folder)

    def test_immutable_bundle_loads_and_helper_hash_is_checked_first(self):
        bundle = Path(__file__).resolve().parent / "upload" / "jra-pedigree-pilot-v1"
        roots, nodes, edges, _ = production.load_verified_inputs(bundle)
        self.assertEqual((len(roots), len(nodes), len(edges)), (62270, 93474, 173412))
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "bundle"
            shutil.copytree(bundle, copy)
            helper = copy / "graph_pilot.py"
            helper.write_text(helper.read_text(encoding="utf-8") + "\n# corrupted\n",
                              encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                production.load_verified_inputs(copy)

    def test_notebook_embeds_exact_runner_source(self):
        notebook = json.loads((Path(__file__).resolve().parent /
                               "node2vec_production.ipynb").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                for entry in cell["source"][:-1]:
                    self.assertTrue(entry.endswith("\n"), "Kaggle source entry must be a complete line")
                ast.parse("\n".join(entry.rstrip("\n") for entry in cell["source"]))
        code = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                           if cell["cell_type"] == "code")
        tree = ast.parse(code)
        embedded = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "PRODUCTION_SOURCE"
                    for target in node.targets):
                embedded = ast.literal_eval(node.value)
                break
        self.assertEqual(embedded, (Path(__file__).resolve().parent /
                                    "graph_production.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
