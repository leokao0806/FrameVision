"""
metrics.py
Evaluation metrics for Multimodal Framing Multi-label Classification.
Computes Macro-F1, Micro-F1, Hamming Loss, and precision/recall per framing class.
"""

import numpy as np


def compute_multilabel_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray, threshold: float = 0.5):
    """
    Args:
        y_true: Binary ground-truth matrix of shape (N, num_classes)
        y_pred_probs: Predicted probabilities of shape (N, num_classes)
        threshold: Decision threshold for positive classification (default: 0.5)
    Returns:
        Dictionary of computed metrics (Macro-F1, Micro-F1, Hamming Loss, etc.)
    """
    y_pred = (y_pred_probs >= threshold).astype(int)

    # True Positives, False Positives, False Negatives per class
    tp = np.sum((y_true == 1) & (y_pred == 1), axis=0)
    fp = np.sum((y_true == 0) & (y_pred == 1), axis=0)
    fn = np.sum((y_true == 1) & (y_pred == 0), axis=0)

    # Class-wise Precision, Recall, and F1
    precision = np.zeros(y_true.shape[1])
    recall = np.zeros(y_true.shape[1])
    f1_per_class = np.zeros(y_true.shape[1])

    for c in range(y_true.shape[1]):
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        precision[c] = p
        recall[c] = r
        f1_per_class[c] = f

    macro_f1 = np.mean(f1_per_class)

    # Micro F1
    total_tp = np.sum(tp)
    total_fp = np.sum(fp)
    total_fn = np.sum(fn)

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    # Hamming Loss: fraction of labels incorrectly predicted
    hamming_loss = np.mean(y_true != y_pred)

    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "hamming_loss": float(hamming_loss),
        "f1_per_class": f1_per_class.tolist(),
    }
