"""E9: combine Qwen image-only 0/1 predictions with a trained model's probabilities.

score = alpha * qwen + (1 - alpha) * probs; a frame is predicted when score >= threshold, and an image with no
frame is None. alpha and the threshold are chosen on dev only (docs/preregistration.md), separately for every
seed of the source model.

Usage:
    python -m ensemble.combine --pred_root predictions/subtask2                       # E9: source = best trained model
    python -m ensemble.combine --pred_root predictions/subtask2 --source image_lr --name combo_qwen_image_lr
"""
import argparse
import json
import os
import re

import numpy as np

from evaluation.prediction_io import read_frames_tsv, read_labels, to_matrix, write_frames_tsv
from evaluation.scoring import micro_f1

ALPHAS = (0.25, 0.5, 0.75)
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)  # rounded, so dev fitting and test use the same values
QWEN = "qwen_image_only"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FINAL = os.path.join(REPO, "results", "subtask2_final.json")


def load_probs(path, labels):
    """Returns (image_names, probs) with the columns reordered to follow labels."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    classes = list(data["classes"])
    if sorted(classes) != sorted(labels):
        raise ValueError(f"{path}: classes {classes} do not match labels.txt {labels}")
    probs = np.asarray(data["probs"], dtype=float)
    order = [classes.index(label) for label in labels]
    return list(data["image_names"]), probs[:, order]


def combine(qwen, probs, alpha):
    return alpha * np.asarray(qwen, dtype=float) + (1 - alpha) * np.asarray(probs, dtype=float)


def _with_none(y14):
    y14 = np.asarray(y14).astype(int)
    return np.hstack([y14, (y14.sum(axis=1) == 0).astype(int).reshape(-1, 1)])


def fit_on_dev(qwen_dev, probs_dev, y_dev):
    """(alpha, threshold, dev_micro) with the highest dev Micro F1; ties keep the smaller alpha, then threshold."""
    y = _with_none(y_dev)
    best = (None, None, -1.0)
    for alpha in ALPHAS:
        scores = combine(qwen_dev, probs_dev, alpha)
        for threshold in THRESHOLDS:
            micro = micro_f1(y, _with_none(scores >= threshold))
            if micro > best[2]:
                best = (alpha, float(threshold), micro)
    return best


def _aligned_probs(path, labels, names):
    prob_names, probs = load_probs(path, labels)
    if set(prob_names) != set(names):
        raise ValueError(f"{path}: images differ from gold")
    index = {n: i for i, n in enumerate(prob_names)}
    return probs[[index[n] for n in names]]


def _rows(scores, threshold, names, labels):
    return {name: [labels[j] for j in np.flatnonzero(row >= threshold)] for name, row in zip(names, scores)}


def run(pred_root, source, name):
    labels = read_labels(os.path.join(pred_root, "_gold", "labels.txt"))
    gold = {s: read_frames_tsv(os.path.join(pred_root, "_gold", f"{s}.tsv"), labels) for s in ("dev", "test")}
    names = {s: list(gold[s]) for s in gold}
    qwen = {s: to_matrix(read_frames_tsv(os.path.join(pred_root, QWEN, "seed0", f"{s}.tsv"), labels), names[s], labels)[:, :-1]
            for s in gold}
    y_dev = to_matrix(gold["dev"], names["dev"], labels)[:, :-1]
    seed_dirs = sorted(d for d in os.listdir(os.path.join(pred_root, source)) if re.fullmatch(r"seed\d+", d))
    selection = {"source": source, "qwen": QWEN, "alphas": list(ALPHAS), "seeds": {}}
    for seed_dir in seed_dirs:
        probs = {s: _aligned_probs(os.path.join(pred_root, source, seed_dir, f"{s}.probs.json"), labels, names[s])
                 for s in gold}
        alpha, threshold, dev_micro = fit_on_dev(qwen["dev"], probs["dev"], y_dev)
        for s in gold:
            scores = combine(qwen[s], probs[s], alpha)
            write_frames_tsv(os.path.join(pred_root, name, seed_dir, f"{s}.tsv"), _rows(scores, threshold, names[s], labels))
        selection["seeds"][seed_dir[len("seed"):]] = {"alpha": alpha, "threshold": threshold, "dev_micro": dev_micro}
    with open(os.path.join(pred_root, name, "selection.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(selection, f, indent=2)
        f.write("\n")
    return selection


def main(argv=None):
    parser = argparse.ArgumentParser(description="E9 Qwen + trained-model combination.")
    parser.add_argument("--pred_root", default=os.path.join(REPO, "predictions", "subtask2"))
    parser.add_argument("--source", help="model whose {dev,test}.probs.json are combined; default: best trained model")
    parser.add_argument("--name", default="combo_qwen_best")
    parser.add_argument("--final", default=DEFAULT_FINAL, help="results JSON that names the best trained model")
    args = parser.parse_args(argv)
    source = args.source
    if source is None:
        with open(args.final, encoding="utf-8") as f:
            source = json.load(f)["selection"].get("best_trained_model")
        if not source:
            raise SystemExit(f"{args.final} has no selection.best_trained_model yet (E1–E8 have not been evaluated)")
    selection = run(args.pred_root, source, args.name)
    print(f"Wrote {args.name} from {source}: " + ", ".join(
        f"seed{k} alpha={v['alpha']} thr={v['threshold']} dev={v['dev_micro']:.3f}" for k, v in selection["seeds"].items()))


if __name__ == "__main__":
    main()
