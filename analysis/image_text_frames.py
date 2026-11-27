"""C1: how an article's image frames relate to its text frames (union of its paragraph frames).

Only gold labels are used. Before results/LOCKED exists only train and dev are read; --include_test
requires the lock.

Usage:
    python -m analysis.image_text_frames --pred_root predictions/subtask2 [--include_test] \
        [--out results/c1_image_text.json]
"""
import argparse
import json
import math
import os

import numpy as np

from evaluation.prediction_io import read_frames_tsv, read_labels

from . import lock

RELATIONS = ("aligned", "complementary", "divergent", "image_none")
DEFAULT_OUT = os.path.join(lock.REPO, "results", "c1_image_text.json")


def frame_relation(image_frames, text_frames):
    image_frames, text_frames = set(image_frames), set(text_frames)
    if not image_frames:
        return "image_none"
    if image_frames <= text_frames:
        return "aligned"
    if image_frames & text_frames:
        return "complementary"
    return "divergent"


def conditional_matrix(pairs, labels):
    """Row i, column j = P(image has frame j | text has frame i); NaN rows for text frames never seen."""
    index = {label: k for k, label in enumerate(labels)}
    joint = np.zeros((len(labels), len(labels)))
    text_counts = np.zeros(len(labels))
    for image_frames, text_frames in pairs:
        for t in text_frames:
            text_counts[index[t]] += 1
            for i in image_frames:
                joint[index[t], index[i]] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(text_counts[:, None] > 0, joint / text_counts[:, None], np.nan)


def _language(image_name):
    return image_name.split("_", 1)[0]


def _summary(pairs):
    relations = {r: 0 for r in RELATIONS}
    for image_frames, text_frames in pairs:
        relations[frame_relation(image_frames, text_frames)] += 1
    tp = sum(len(i & t) for i, t in pairs)
    fp = sum(len(t - i) for i, t in pairs)
    fn = sum(len(i - t) for i, t in pairs)
    n = len(pairs)
    return {
        "n_articles": n,
        "relations": relations,
        "relation_share": {r: c / n for r, c in relations.items()} if n else {},
        "mean_text_frames_per_article": float(np.mean([len(t) for _, t in pairs])) if n else None,
        "mean_image_frames_per_image": float(np.mean([len(i) for i, _ in pairs])) if n else None,
        "copy_text_frames": {"precision": tp / (tp + fp) if tp + fp else None,
                             "recall": tp / (tp + fn) if tp + fn else None},
    }


def _load_split(pred_root, split, labels):
    gold = os.path.join(pred_root, "_gold")
    image = read_frames_tsv(os.path.join(gold, f"{split}.tsv"), labels)
    text = read_frames_tsv(os.path.join(gold, f"{split}_text_frames.tsv"), labels)
    if set(image) != set(text):
        raise ValueError(f"{split}: image and text-frame files list different images")
    return [(name, set(image[name]), set(text[name])) for name in image]


def analyse(pred_root, splits):
    labels = read_labels(os.path.join(pred_root, "_gold", "labels.txt"))
    rows = {split: _load_split(pred_root, split, labels) for split in splits}
    everything = [r for split in splits for r in rows[split]]
    pairs = [(i, t) for _, i, t in everything]
    out = {"splits": list(splits), "labels": labels, **_summary(pairs)}
    out["text_frame_articles"] = {label: sum(label in t for _, t in pairs) for label in labels}
    out["image_frame_images"] = {label: sum(label in i for i, _ in pairs) for label in labels}
    matrix = conditional_matrix(pairs, labels)
    out["conditional_matrix"] = [[None if math.isnan(x) else float(x) for x in row] for row in matrix]
    out["conditional_matrix_axes"] = "rows = text frame, columns = image frame, value = P(image frame | text frame)"
    languages = sorted({_language(name) for name, _, _ in everything})
    out["by_language"] = {lang: _summary([(i, t) for name, i, t in everything if _language(name) == lang])
                          for lang in languages}
    out["by_split"] = {split: _summary([(i, t) for _, i, t in rows[split]]) for split in splits}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="C1 image-vs-text frame relations.")
    parser.add_argument("--pred_root", default=os.path.join(lock.REPO, "predictions", "subtask2"))
    parser.add_argument("--include_test", action="store_true", help="also read test (requires results/LOCKED)")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    splits = ["train", "dev"]
    if args.include_test:
        lock.require_lock()
        splits.append("test")
    result = analyse(args.pred_root, splits)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"Wrote {args.out}: splits={splits} n_articles={result['n_articles']} relations={result['relations']}")


if __name__ == "__main__":
    main()
