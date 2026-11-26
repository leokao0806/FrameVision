"""Article-level bootstrap for seed-averaged Micro F1.

Each resample draws whole articles with replacement. Every seed is scored on the same resample and the
seed scores are averaged, so the interval reflects article sampling; seed variation is reported as std.
Articles are ordered by name, as in the Intermediate Report, so a single seed gives the same interval.
"""
import os

import numpy as np


def group_of(image_name):
    return os.path.splitext(image_name)[0]


def _counts_by_group(y_true, y_pred, groups):
    t, p = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    names = sorted(set(groups))
    index = {name: i for i, name in enumerate(names)}
    counts = np.zeros((len(names), 3))
    for row, group in enumerate(groups):
        counts[index[group]] += [(t[row] & p[row]).sum(), ((1 - t[row]) & p[row]).sum(), (t[row] & (1 - p[row])).sum()]
    return counts


def _micro_from_counts(counts):
    tp, fp, fn = counts.sum(axis=-2).T
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def _resampled_mean(y_true, preds_by_seed, groups, idx):
    per_seed = [_micro_from_counts(_counts_by_group(y_true, p, groups)[idx]) for p in preds_by_seed]
    return np.mean(per_seed, axis=0)


def _indices(groups, n_boot, seed):
    n_groups = len(set(groups))
    return np.random.RandomState(seed).randint(0, n_groups, (n_boot, n_groups))


def _interval(scores):
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def bootstrap_ci(y_true, preds_by_seed, groups, n_boot=2000, seed=0):
    idx = _indices(groups, n_boot, seed)
    return _interval(_resampled_mean(y_true, preds_by_seed, groups, idx))


def paired_ci(y_true, preds_a, preds_b, groups, n_boot=2000, seed=0):
    idx = _indices(groups, n_boot, seed)
    diff = _resampled_mean(y_true, preds_a, groups, idx) - _resampled_mean(y_true, preds_b, groups, idx)
    return _interval(diff)
