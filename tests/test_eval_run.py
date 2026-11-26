import json
import os
import subprocess
import sys

import pytest

from evaluation.run_eval import (
    COMPLEXITY_ORDER,
    PRIMARY_COMPARISONS,
    TRAINED_MODELS,
    evaluate_all,
    select_best,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ["Economic", "Political", "Morality"]
GOLD = {
    "dev": "d1.jpg\tEconomic\nd2.jpg\n",
    "test": "t1.jpg\tPolitical\nt2.jpg\tMorality\tEconomic\nt3.jpg\n",
}
PRED = {
    "dev": "d1.jpg\tEconomic\nd2.jpg\tPolitical\n",
    "test": "t1.jpg\tPolitical\nt2.jpg\tMorality\nt3.jpg\n",
}


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _root(tmp_path, runs):
    """runs: {model: {seed: [splits]}}"""
    root = tmp_path / "pred"
    _write(root / "_gold" / "labels.txt", "\n".join(LABELS) + "\n")
    for split, text in GOLD.items():
        _write(root / "_gold" / f"{split}.tsv", text)
    for model, seeds in runs.items():
        for seed, splits in seeds.items():
            for split in splits:
                _write(root / model / f"seed{seed}" / f"{split}.tsv", PRED[split])
    scorer = _write(tmp_path / "fake_scorer.py", 'print("0.5\\t0.4")\n')
    return str(root), scorer


def test_constants_match_plan():
    assert TRAINED_MODELS[0] == "e1_text_only" and TRAINED_MODELS[-1] == "e8_fusion_gdino_asl"
    assert len(TRAINED_MODELS) == 8
    assert PRIMARY_COMPARISONS[0] == ("best", "most_frequent")
    assert COMPLEXITY_ORDER.index("most_frequent") < COMPLEXITY_ORDER.index("image_lr") \
        < COMPLEXITY_ORDER.index("qwen_image_only") < COMPLEXITY_ORDER.index("e2_vision_global") \
        < COMPLEXITY_ORDER.index("e5_fusion_coco_asl") < COMPLEXITY_ORDER.index("combo_qwen_best")


def test_select_best_ignores_models_without_dev():
    models = {"image_lr": {"dev": {"micro_mean": 0.3}}, "qwen_image_only": {"test": {"micro_mean": 0.9}}}
    assert select_best(models) == "image_lr"
    assert select_best({"qwen_image_only": {"test": {"micro_mean": 0.9}}}) is None


def test_tie_breaks_to_simpler_model():
    models = {"image_lr": {"dev": {"micro_mean": 0.4}}, "e5_fusion_coco_asl": {"dev": {"micro_mean": 0.4}}}
    assert select_best(models) == "image_lr"


def test_unlisted_model_loses_ties_to_listed_ones():
    models = {"e11_e5_fusion_coco_asl_no_batch1": {"dev": {"micro_mean": 0.4}},
              "combo_qwen_best": {"dev": {"micro_mean": 0.4}}}
    assert select_best(models) == "combo_qwen_best"


def test_select_best_respects_candidates():
    models = {"image_lr": {"dev": {"micro_mean": 0.9}}, "e2_vision_global": {"dev": {"micro_mean": 0.3}}}
    assert select_best(models, TRAINED_MODELS) == "e2_vision_global"


def test_missing_model_marks_comparison_missing(tmp_path):
    root, scorer = _root(tmp_path, {"e5_fusion_coco_asl": {13: ["dev", "test"]}})
    out = evaluate_all(root, scorer)
    comparison = {(c["spec_a"], c["spec_b"]): c for c in out["primary_comparisons"]}
    assert comparison[("e8_fusion_gdino_asl", "e5_fusion_coco_asl")]["status"] == "missing"
    assert comparison[("best", "most_frequent")]["status"] == "missing"


def test_model_with_only_dev_has_no_test_key(tmp_path):
    root, scorer = _root(tmp_path, {"e8_fusion_gdino_asl": {13: ["dev"]}, "e5_fusion_coco_asl": {13: ["dev", "test"]}})
    out = evaluate_all(root, scorer)
    assert "dev" in out["models"]["e8_fusion_gdino_asl"]
    assert "test" not in out["models"]["e8_fusion_gdino_asl"]
    comparison = {(c["spec_a"], c["spec_b"]): c for c in out["primary_comparisons"]}
    assert comparison[("e8_fusion_gdino_asl", "e5_fusion_coco_asl")]["status"] == "missing"


def test_best_equal_to_reference_is_same_model(tmp_path):
    # the fake scorer gives every model the same dev score, so the simplest model wins
    root, scorer = _root(tmp_path, {"most_frequent": {0: ["dev", "test"]}, "image_lr": {0: ["dev", "test"]}})
    out = evaluate_all(root, scorer)
    assert out["selection"]["best_model"] == "most_frequent"
    comparison = {(c["spec_a"], c["spec_b"]): c for c in out["primary_comparisons"]}
    assert comparison[("best", "most_frequent")]["status"] == "same_model"


def test_ok_comparison_has_diff_and_interval(tmp_path):
    root, scorer = _root(tmp_path, {"e6_fusion_global_asl": {13: ["dev", "test"]},
                                    "e2_vision_global": {13: ["dev", "test"]}})
    out = evaluate_all(root, scorer)
    comparison = {(c["spec_a"], c["spec_b"]): c for c in out["primary_comparisons"]}
    c = comparison[("e6_fusion_global_asl", "e2_vision_global")]
    assert c["status"] == "ok" and c["diff_mean"] == 0.0 and c["ci95"] == [0.0, 0.0]


def test_partial_seeds_are_reported(tmp_path):
    root, scorer = _root(tmp_path, {"e5_fusion_coco_asl": {13: ["dev", "test"], 42: ["dev", "test"]}})
    out = evaluate_all(root, scorer)
    model = out["models"]["e5_fusion_coco_asl"]
    assert model["seeds"] == [13, 42]
    assert model["test"]["micro_mean"] == 0.5 and model["test"]["micro_std"] == 0.0
    assert out["n_dev"] == 2 and out["n_test"] == 3
    assert out["labels"] == LABELS + ["None"]
    assert out["selection"]["best_trained_model"] == "e5_fusion_coco_asl"


def test_mismatched_images_stop_evaluation(tmp_path):
    root, scorer = _root(tmp_path, {"e5_fusion_coco_asl": {13: ["dev", "test"]}})
    _write(tmp_path / "pred" / "e5_fusion_coco_asl" / "seed13" / "test.tsv", "t1.jpg\nt2.jpg\n")
    with pytest.raises(ValueError, match=r"seed13[\\/]test\.tsv"):
        evaluate_all(root, scorer)


def test_reference_prior_is_copied(tmp_path):
    root, scorer = _root(tmp_path, {"most_frequent": {0: ["dev", "test"]}})
    _write(tmp_path / "pred" / "_reference" / "random_prior.json", json.dumps({"micro_f1": 0.12}))
    assert evaluate_all(root, scorer)["reference_only"]["random_prior"] == {"micro_f1": 0.12}


def test_cli_writes_json(tmp_path):
    root, scorer = _root(tmp_path, {"most_frequent": {0: ["dev", "test"]}})
    out_path = tmp_path / "results" / "subtask2_final.json"
    subprocess.run([sys.executable, "-m", "evaluation.run_eval", "--pred_root", root, "--scorer", scorer,
                    "--out", str(out_path)], check=True, cwd=REPO)
    data = json.loads(out_path.read_text(encoding="utf-8"))
    lo, hi = data["models"]["most_frequent"]["test"]["micro_ci95"]
    assert 0.0 <= lo <= hi <= 1.0
    assert data["selection"]["rule"] == "docs/preregistration.md"
