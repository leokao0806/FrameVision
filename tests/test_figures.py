import json
import math
import os

import pytest

from analysis.figures import (labels_to_plot, plot_ablation, plot_c1_heatmap, plot_c2, plot_main_results,
                              plot_per_label, plot_region_coverage)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "final_small.json")
C1 = {"labels": ["Economic", "Political", "Morality"],
      "conditional_matrix": [[0.5, 0.25, 0.0], [0.1, 0.8, 0.05], [math.nan, math.nan, math.nan]]}
C2 = {"splits": ["dev"], "base": "qwen_image_only", "by_split": {"dev": {"n_images": 20, "variants": {
    "qwen_image_only": {"overlap": {"mean_n_labels": 1.2}, "swap_vs_base": None},
    "qwen_2a_title": {"overlap": {"mean_n_labels": 1.3}, "swap_vs_base": {
        "added": 10, "added_correct": 1, "added_in_text": 7, "removed": 8, "removed_correct": 1}},
    "qwen_2a": {"overlap": {"mean_n_labels": 1.4}, "swap_vs_base": {
        "added": 7, "added_correct": 1, "added_in_text": 4, "removed": 3, "removed_correct": 1}}}}}}
COVERAGE = {"coco": {"n_images": 700, "all_detected": 420, "partial": 200, "all_fallback": 80, "mean_detected": 2.3},
            "gdino": {"n_images": 700, "all_detected": 500, "partial": 150, "all_fallback": 50, "mean_detected": 2.6}}


@pytest.fixture
def final():
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


def _written(out):
    png, pdf = out, os.path.splitext(out)[0] + ".pdf"
    return all(os.path.exists(p) and os.path.getsize(p) > 0 for p in (png, pdf))


def test_plot_main_results_skips_models_without_test(final, tmp_path):
    out = str(tmp_path / "main.png")
    plot_main_results(final, ["qwen_image_only", "e6_fusion_global_asl", "most_frequent", "not_a_model"], out)
    assert _written(out)


def test_plot_ablation_writes_png_and_pdf(final, tmp_path):
    out = str(tmp_path / "ablation.png")
    plot_ablation(final, out)
    assert _written(out)


def test_plot_ablation_without_trained_models_still_writes_a_figure(final, tmp_path):
    final["models"] = {k: v for k, v in final["models"].items() if not k.startswith("e")}
    out = str(tmp_path / "ablation.png")
    plot_ablation(final, out)
    assert _written(out)


def test_per_label_omits_labels_without_test_support(final):
    assert labels_to_plot(final, ["qwen_image_only", "most_frequent"]) == ["Economic", "Political", "None"]


def test_per_label_lists_none_once_when_labels_already_include_it(final):
    final["labels"] = final["labels"] + ["None"]
    assert labels_to_plot(final, ["qwen_image_only"]) == ["Economic", "Political", "None"]


def test_plot_per_label_writes_png_and_pdf(final, tmp_path):
    out = str(tmp_path / "per_label.png")
    plot_per_label(final, ["qwen_2a_fulltext", "qwen_image_only", "e2_vision_global"], out)
    assert _written(out)


def test_plot_c1_heatmap_handles_unseen_text_frames(tmp_path):
    out = str(tmp_path / "c1.png")
    plot_c1_heatmap(C1, out)
    assert _written(out)


def test_plot_region_coverage_writes_png_and_pdf(tmp_path):
    out = str(tmp_path / "coverage.png")
    plot_region_coverage(COVERAGE, out)
    assert _written(out)


def test_plot_c2_works_with_dev_only_analysis(final, tmp_path):
    out = str(tmp_path / "c2.png")
    plot_c2(C2, final, out)
    assert _written(out)
