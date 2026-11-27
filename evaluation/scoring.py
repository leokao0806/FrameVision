"""Score prediction files with the unmodified official Subtask 2 scorer.

Matrices have one column per frame plus a final None column. The official scorer's numbers are the ones
reported; ``micro_f1`` is only a cross-check, and disagreements are counted as ``mismatches``.
The scorer is given canonical copies (rows in gold order, no BOM, empty row for None), so every form
that ``read_frames_tsv`` accepts is scored exactly as it was parsed.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

from .prediction_io import NONE_LABEL, check_same_images, read_frames_tsv, read_labels, to_matrix, write_frames_tsv

MISMATCH_TOLERANCE = 1e-6


def run_official_scorer(scorer_path, gold_tsv, pred_tsv, labels_path, source=None):
    source = source or pred_tsv
    result = subprocess.run(
        [sys.executable, scorer_path, "-g", gold_tsv, "-p", pred_tsv, "-f", labels_path, "-o"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"official scorer failed (exit {result.returncode}) for {source}: {result.stderr.strip()}")
    lines = result.stdout.strip().splitlines()
    if not lines:
        raise RuntimeError(f"official scorer printed no score for {source}: {result.stderr.strip()}")
    micro, macro = lines[-1].split("\t")[:2]
    return float(micro), float(macro)


def _counts(y_true, y_pred):
    t, p = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    tp = (t & p).sum(axis=0)
    fp = ((1 - t) & p).sum(axis=0)
    fn = (t & (1 - p)).sum(axis=0)
    return tp, fp, fn


def micro_f1(y_true, y_pred):
    tp, fp, fn = (c.sum() for c in _counts(y_true, y_pred))
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def hamming(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) != np.asarray(y_pred)))


def per_label_f1(y_true, y_pred, names):
    tp, fp, fn = _counts(y_true, y_pred)
    out = {}
    for j, name in enumerate(names):
        denom = 2 * tp[j] + fp[j] + fn[j]
        out[name] = None if denom == 0 else float(2 * tp[j] / denom)
    return out


def _average_per_label(per_seed):
    out = {}
    for name in per_seed[0]:
        values = [d[name] for d in per_seed if d[name] is not None]
        out[name] = float(np.mean(values)) if values else None
    return out


def score_split(pred_paths, gold_path, labels_path, scorer_path):
    labels = read_labels(labels_path)
    names = labels + [NONE_LABEL]
    gold_rows = read_frames_tsv(gold_path, labels)
    image_names = list(gold_rows)
    y_true = to_matrix(gold_rows, image_names, labels)

    out = {"micro": [], "macro": [], "hamming": [], "matrices": [], "mismatches": 0}
    per_seed_f1 = []
    with tempfile.TemporaryDirectory() as tmp:
        canonical_labels = os.path.join(tmp, "labels.txt")
        with open(canonical_labels, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(labels) + "\n")
        canonical_gold = os.path.join(tmp, "gold.tsv")
        write_frames_tsv(canonical_gold, gold_rows)
        for k, path in enumerate(pred_paths):
            rows = read_frames_tsv(path, labels)
            check_same_images(rows, gold_rows, path)
            y_pred = to_matrix(rows, image_names, labels)
            canonical_pred = os.path.join(tmp, f"pred{k}.tsv")
            write_frames_tsv(canonical_pred, {name: rows[name] for name in image_names})
            micro, macro = run_official_scorer(scorer_path, canonical_gold, canonical_pred, canonical_labels, source=path)
            if abs(micro - micro_f1(y_true, y_pred)) > MISMATCH_TOLERANCE:
                out["mismatches"] += 1
            out["micro"].append(micro)
            out["macro"].append(macro)
            out["hamming"].append(hamming(y_true, y_pred))
            out["matrices"].append(y_pred)
            per_seed_f1.append(per_label_f1(y_true, y_pred, names))

    out["per_label_f1"] = _average_per_label(per_seed_f1) if per_seed_f1 else {}
    out["support"] = {name: int(n) for name, n in zip(names, y_true.sum(axis=0))}
    out["y_true"] = y_true
    out["image_names"] = image_names
    return out
