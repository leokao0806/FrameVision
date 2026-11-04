"""
Evaluation Metrics for Multi-Label Framing Analysis
CS 5134/6034 NLP Fall 2026 - Team FrameVision
Authors: Leo Kao, En-Shuo Zhang

Follows the official scorer: multi-label Micro F1 is the official measure, Macro F1 is
complementary, and an example with no framing counts as the label "None". Reported scores
come from the official scorer (src/official_scorer.py); this module must agree with it and
adds the diagnostics the scorer does not compute.
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
from src.data_loader import FRAME_TAXONOMY, NUM_CLASSES

NONE_LABEL = "None"
SCORED_LABELS = FRAME_TAXONOMY + [NONE_LABEL]


def add_none_column(y):
    """Appends a "None" column that is 1 exactly when a row has no framing."""
    y = np.asarray(y).astype(int)
    none_col = (y.sum(axis=1) == 0).astype(int).reshape(-1, 1)
    return np.hstack([y, none_col])


def compute_multilabel_metrics(y_true, y_pred):
    """
    Args:
        y_true, y_pred: binary arrays of shape (N, 14), without the "None" column.
    Returns:
        Micro F1 (official), Macro F1 over all 15 labels (as in the scorer, a label absent
        from both gold and predictions gets F1 = 1), Macro F1 over labels that occur in
        the gold data, Hamming loss, and a per-label breakdown.
    """
    t = add_none_column(y_true)
    p = add_none_column(y_pred)
    present = t.sum(axis=0) > 0

    per_label = {}
    for idx, name in enumerate(SCORED_LABELS):
        per_label[name] = {
            "f1": float(f1_score(t[:, idx], p[:, idx], zero_division=0)),
            "precision": float(precision_score(t[:, idx], p[:, idx], zero_division=0)),
            "recall": float(recall_score(t[:, idx], p[:, idx], zero_division=0)),
            "support": int(t[:, idx].sum()),
        }

    return {
        "micro_f1": float(f1_score(t, p, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(t, p, average="macro", zero_division=1)),
        "macro_f1_present": float(f1_score(t[:, present], p[:, present], average="macro", zero_division=0)),
        "micro_precision": float(precision_score(t, p, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(t, p, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(t, p)),
        "n_examples": int(len(t)),
        "per_label": per_label,
    }


def select_threshold_on_dev(dev_probs, dev_true):
    """Picks one global threshold on the development set to maximize Micro F1."""
    best_threshold, best_score = 0.5, -1.0
    for threshold in np.arange(0.05, 0.96, 0.05):
        score = compute_multilabel_metrics(dev_true, (dev_probs >= threshold).astype(int))["micro_f1"]
        if score > best_score:
            best_threshold, best_score = float(threshold), score
    return best_threshold, best_score


def _micro_counts_by_group(y_true, y_pred, groups):
    """True positive, false positive, and false negative counts per article."""
    t, p = add_none_column(y_true), add_none_column(y_pred)
    names = sorted(set(groups))
    index = {name: i for i, name in enumerate(names)}
    counts = np.zeros((len(names), 3))
    for row, group in enumerate(groups):
        counts[index[group]] += [(t[row] & p[row]).sum(), ((1 - t[row]) & p[row]).sum(), (t[row] & (1 - p[row])).sum()]
    return counts


def _micro_from_counts(counts):
    tp, fp, fn = counts.sum(axis=-2).T
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def bootstrap_micro_f1(y_true, y_pred, groups, n_boot=2000, seed=0):
    """95% percentile bootstrap interval for Micro F1, resampling whole articles."""
    counts = _micro_counts_by_group(y_true, y_pred, groups)
    idx = np.random.RandomState(seed).randint(0, len(counts), (n_boot, len(counts)))
    scores = _micro_from_counts(counts[idx])
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def bootstrap_micro_difference(y_true, y_pred_a, y_pred_b, groups, n_boot=2000, seed=0):
    """Paired 95% bootstrap interval for Micro F1 of a minus b, resampling whole articles."""
    counts_a = _micro_counts_by_group(y_true, y_pred_a, groups)
    counts_b = _micro_counts_by_group(y_true, y_pred_b, groups)
    idx = np.random.RandomState(seed).randint(0, len(counts_a), (n_boot, len(counts_a)))
    differences = _micro_from_counts(counts_a[idx]) - _micro_from_counts(counts_b[idx])
    return float(np.percentile(differences, 2.5)), float(np.percentile(differences, 97.5))


def compute_mismatch_matrix(y_true, y_pred):
    """For each missed gold frame, counts which frames were wrongly predicted instead."""
    mismatch = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for i in range(len(y_true)):
        missed = np.where((y_true[i] == 1) & (y_pred[i] == 0))[0]
        false_positives = np.where((y_true[i] == 0) & (y_pred[i] == 1))[0]
        for g in missed:
            for fp in false_positives:
                mismatch[g, fp] += 1
    return mismatch
