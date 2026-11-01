"""Checks for result-release figures and generated-artifact checksums."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_results_release as release


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        figure = root / "forest.svg"
        rows = [{"family": "xgboost", "surface": "turf", "contrast": "S6-S2",
                 "label": "xgboost / turf / S6-S2", "estimate": 0.02,
                 "ci95_lower": 0.01, "ci95_upper": 0.03}]
        release.forest_svg(figure, rows, "test forest", "label")
        release.tikz_forest(root / "forest.tex", rows, "test forest")
        assert "<svg" in figure.read_text(encoding="utf-8")
        assert "xgboost / turf / S6-S2" in (root / "forest.tex").read_text(encoding="utf-8")

        shap_path = root / "shap.tex"
        shap = [{"family": "catboost", "surface": "turf", "setting": "S6",
                 "seed": 0, "group_mean_abs_sum": {
                     "graph_embeddings": 0.2, "graph_cluster": 0.05,
                     "graph_availability": 0.0}}]
        release.tikz_shap(shap_path, shap)
        text = shap_path.read_text(encoding="utf-8")
        assert "Validation-only grouped SHAP" in text
        assert "availability" in text

        (root / "release_metadata.json").write_text("{}", encoding="utf-8")
        artifacts = release.artifact_inventory(root)
        assert {item["path"] for item in artifacts} == {
            "forest.svg", "forest.tex", "shap.tex"}
        assert all(len(item["sha256"]) == 64 and item["bytes"] > 0
                   for item in artifacts)
    print("PASS: results-release figures and artifact hashes")


if __name__ == "__main__":
    main()
