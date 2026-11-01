"""Synthetic no-fit checks for validation SHAP sampling and attribution policy."""
import numpy as np

from run_shap_validation import (attribution_groups, explain_model,
                                 normalize_shap_values, validation_sample)


def _rows():
    rows = []
    for race_number, race_id in enumerate(("r2", "r1", "r3")):
        for horse_number in range(1, 4):
            rows.append({"race_id": race_id, "horse_id_netkeiba": f"{race_id}-h{horse_number}",
                         "horse_number": horse_number, "field_size": 3,
                         "surface": "turf", "split": "validation"})
    return rows


class _Explainer:
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs

    def shap_values(self, frame):
        return [np.asarray([[1.0, -2.0, 3.0], [2.0, -1.0, 1.0]])]


class _Shap:
    __version__ = "synthetic"
    TreeExplainer = _Explainer


def check():
    sampled, race_ids = validation_sample(_rows(), seed=20260913, min_entries=5)
    assert len(sampled) == 6 and len(race_ids) == 2
    assert all(sum(row["race_id"] == race for row in sampled) == 3 for race in race_ids)
    sampled_again, race_ids_again = validation_sample(_rows(), seed=20260913, min_entries=5)
    assert race_ids == race_ids_again and sampled == sampled_again
    columns = ("sire", "embedding_0", "embedding_1", "graph_cluster", "embedding_status")
    groups = attribution_groups("S6", columns)
    assert groups == {"graph_availability": (4,), "graph_embeddings": (1, 2),
                      "graph_cluster": (3,)}
    values = normalize_shap_values(np.zeros((2, 5, 2)), 2, 5)
    assert values.shape == (2, 5)
    summary, explainer = explain_model(object(), np.zeros((2, 3)),
                                       ("sire", "embedding_0", "graph_cluster"),
                                       {"pedigree": (0,), "graph_embeddings": (1,),
                                        "graph_cluster": (2,)}, _Shap)
    assert explainer.kwargs == {"feature_perturbation": "tree_path_dependent",
                                "model_output": "raw"}
    assert summary["feature_mean_abs"]["sire"] == 1.5
    assert summary["group_mean_abs_sum"]["graph_embeddings"] == 1.5
    print("PASS: complete-race sampling, native TreeExplainer policy, and grouped phi summaries")


if __name__ == "__main__":
    check()
