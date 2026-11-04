import os
import subprocess
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.losses.asymmetric_loss import AsymmetricLoss
from src.metrics import (
    NONE_LABEL,
    DEFAULT_CLASSES,
    compute_metrics,
    optimize_threshold,
    evaluate_predictions,
    export_predictions_tsv,
)


def test_asymmetric_loss_forward_and_backward():
    asl = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
    logits = torch.randn(8, 14, requires_grad=True)
    targets = (torch.rand(8, 14) > 0.8).float()
    
    loss = asl(logits, targets)
    assert loss.dim() == 0
    assert loss.item() > 0
    
    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
    assert not torch.isinf(logits.grad).any()


def test_asymmetric_loss_reductions():
    asl_mean = AsymmetricLoss(reduction="mean")
    asl_sum = AsymmetricLoss(reduction="sum")
    asl_none = AsymmetricLoss(reduction="none")
    
    logits = torch.randn(4, 14)
    targets = (torch.rand(4, 14) > 0.7).float()
    
    loss_mean = asl_mean(logits, targets)
    loss_sum = asl_sum(logits, targets)
    loss_none = asl_none(logits, targets)
    
    assert loss_mean.dim() == 0
    assert loss_sum.dim() == 0
    assert loss_none.shape == (4, 14)
    assert torch.isclose(loss_mean, loss_none.mean())
    assert torch.isclose(loss_sum, loss_none.sum())

    with pytest.raises(ValueError):
        AsymmetricLoss(reduction="invalid_reduction")


def test_asymmetric_loss_asymmetry():
    # When clip > 0 and probability is below clip, negative loss must be zero
    clip = 0.05
    asl = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=clip, reduction="none")
    
    # Very small logit corresponding to p < clip
    # e.g., sigmoid(-5.0) ~ 0.00669 < 0.05
    logits = torch.tensor([[-5.0]])
    target_neg = torch.tensor([[0.0]])
    
    loss_asl = asl(logits, target_neg)
    loss_bce = F.binary_cross_entropy_with_logits(logits, target_neg)
    
    # ASL shifts p by clip: max(p - 0.05, 0) = 0, so loss_asl is exactly 0
    assert torch.isclose(loss_asl, torch.tensor([[0.0]]), atol=1e-7)
    assert loss_bce.item() > 0.0
    
    # For a moderate probability p = 0.4 (logit ~ -0.405)
    # ASL negative loss should be heavily discounted compared to BCE due to (p-clip)^gamma_neg
    logit_mid = torch.tensor([[-0.4054651]])  # sigmoid(-0.4054651) ~ 0.4
    loss_mid_asl = asl(logit_mid, target_neg)
    loss_mid_bce = F.binary_cross_entropy_with_logits(logit_mid, target_neg)
    assert loss_mid_asl.item() < loss_mid_bce.item() * 0.1


def test_metrics_empty_prediction_none_mapping():
    # If all probabilities are below threshold, "None" is activated (1)
    # If targets have all zeros, "None" is also activated (1)
    # Both activate None -> perfect F1 = 1.0
    targets = np.zeros((3, 14), dtype=np.float32)
    probs = np.full((3, 14), 0.1, dtype=np.float32)
    
    metrics = compute_metrics(targets, probs, threshold=0.5)
    assert metrics["micro_f1"] == 1.0
    assert metrics["macro_f1"] == 1.0
    
    # If predictions trigger a class, but target had all zeros (i.e. target None = 1)
    probs_mismatch = np.zeros((3, 14), dtype=np.float32)
    probs_mismatch[:, 0] = 0.9  # class 0 predicted
    mismatch_metrics = compute_metrics(targets, probs_mismatch, threshold=0.5)
    assert mismatch_metrics["micro_f1"] == 0.0


def test_metrics_tensor_inputs():
    targets = torch.zeros((4, 14), dtype=torch.float32)
    targets[0, 1] = 1.0
    probs = torch.zeros((4, 14), dtype=torch.float32)
    probs[0, 1] = 0.8
    
    res = compute_metrics(targets, probs, threshold=0.5)
    assert isinstance(res["micro_f1"], float)
    assert isinstance(res["macro_f1"], float)
    assert res["micro_f1"] > 0.0


def test_optimize_threshold():
    # Construct synthetic targets and probs
    rng = np.random.RandomState(42)
    targets = (rng.rand(50, 14) > 0.8).astype(np.float32)
    # Signal: true positives have probs centered at 0.75, negatives at 0.2
    probs = np.where(targets == 1.0, rng.uniform(0.6, 0.9, size=targets.shape), rng.uniform(0.05, 0.35, size=targets.shape))
    
    best_th, best_micro, best_macro = optimize_threshold(targets, probs)
    
    assert 0.35 <= best_th <= 0.65
    assert best_micro > 0.8
    assert best_macro > 0.5


def test_export_predictions_tsv_and_evaluate(tmp_path):
    pred_file = str(tmp_path / "preds.tsv")
    gold_file = str(tmp_path / "gold.tsv")
    labels_file = str(tmp_path / "labels.txt")
    
    classes = ["Economic", "Morality", "Political"]
    with open(labels_file, "w") as f:
        f.write("\n".join(classes) + "\n")
        
    image_names = ["img_1.jpg", "img_2.jpg", "img_3.jpg"]
    probs = np.array([
        [0.8, 0.1, 0.2],  # img_1: Economic
        [0.9, 0.8, 0.1],  # img_2: Economic, Morality
        [0.1, 0.2, 0.3],  # img_3: None (all < 0.5)
    ])
    
    export_predictions_tsv(image_names, probs, threshold=0.5, class_names=classes, out_path=pred_file)
    
    # Verify TSV formatting
    with open(pred_file, "r") as f:
        lines = [line.rstrip("\n") for line in f]
    assert lines[0] == "img_1.jpg\tEconomic"
    # Order of frames corresponds to class order where prob >= threshold
    assert lines[1] == "img_2.jpg\tEconomic\tMorality"
    assert lines[2] == "img_3.jpg"
    
    # Write matching gold TSV
    with open(gold_file, "w") as f:
        f.write("img_1.jpg\tEconomic\n")
        f.write("img_2.jpg\tEconomic\tMorality\n")
        f.write("img_3.jpg\n")
        
    macro_f1, micro_f1 = evaluate_predictions(pred_file, gold_file, labels_file=labels_file)
    assert macro_f1 == 1.0
    assert micro_f1 == 1.0


def test_evaluate_predictions_matches_official_scorer():
    pred_file = "predictions_test_subtask2.txt"
    gold_file = "gold_test_subtask2.txt"
    labels_file = "data_official/labels.txt"
    
    # Official scorer command
    cmd = [
        "/Users/leokao/Documents/Python/task9/venv/bin/python",
        "starter-kit/subtask2a_2b/scorer-subtask-2.py",
        "-g", gold_file,
        "-p", pred_file,
        "-f", labels_file,
        "-o",
    ]
    res = subprocess.run(cmd, cwd="/Users/leokao/Documents/Python/task9", capture_output=True, text=True, check=True)
    official_micro, official_macro = [float(x) for x in res.stdout.strip().split("\t")]
    
    macro_f1, micro_f1 = evaluate_predictions(pred_file, gold_file, labels_file=labels_file)
    
    assert np.isclose(macro_f1, official_macro, atol=1e-5)
    assert np.isclose(micro_f1, official_micro, atol=1e-5)
