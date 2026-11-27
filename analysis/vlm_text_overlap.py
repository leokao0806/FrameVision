"""C2: why adding article text changes Qwen2.5-VL's image predictions.

For each Qwen variant: how many labels it predicts, how much of its prediction overlaps the article's gold
text frames, and which labels it adds or removes relative to image-only (and how many of those are correct
or come from the text frames). Also reports how much of each article the 1,500-character prompt shows.
Before results/LOCKED exists only dev is read; --include_test requires the lock.

Usage:
    python -m analysis.vlm_text_overlap --pred_root predictions/subtask2 [--include_test] \
        [--out results/c2_vlm_text.json]
"""
import argparse
import importlib.util
import json
import os

import numpy as np

from evaluation.prediction_io import read_frames_tsv, read_labels

from . import lock

BASE = "qwen_image_only"
VARIANTS = ["qwen_image_only", "qwen_image_only_v2", "qwen_2a_title", "qwen_2a", "qwen_2a_fulltext", "qwen_2b"]
MAX_TEXT_CHARS = 1500  # baselines/src/models/vlm_pipeline.py, used by qwen_2a and qwen_2b
DEFAULT_OUT = os.path.join(lock.REPO, "results", "c2_vlm_text.json")


def overlap_stats(pred_rows, text_rows):
    names = list(pred_rows)
    n_pred = sum(len(pred_rows[n]) for n in names)
    in_text = sum(len(set(pred_rows[n]) & set(text_rows[n])) for n in names)
    jaccards = []
    for n in names:
        p, t = set(pred_rows[n]), set(text_rows[n])
        jaccards.append(1.0 if not p and not t else len(p & t) / len(p | t))
    return {
        "mean_n_labels": n_pred / len(names),
        "frac_pred_in_text": in_text / n_pred if n_pred else None,
        "jaccard_mean": float(np.mean(jaccards)),
    }


def swap_stats(base_rows, variant_rows, gold_rows, text_rows):
    out = {"added": 0, "added_correct": 0, "added_in_text": 0, "removed": 0, "removed_correct": 0}
    for name in base_rows:
        base, variant = set(base_rows[name]), set(variant_rows[name])
        gold, text = set(gold_rows[name]), set(text_rows[name])
        added, removed = variant - base, base - variant
        out["added"] += len(added)
        out["added_correct"] += len(added & gold)
        out["added_in_text"] += len(added & text)
        out["removed"] += len(removed)
        out["removed_correct"] += len(removed & gold)
    return out


def text_coverage(texts, max_chars):
    lengths = np.array([len(t) for t in texts])
    return {
        "median_chars": float(np.median(lengths)),
        "frac_truncated": float(np.mean(lengths > max_chars)),
        "median_share_seen": float(np.median(np.minimum(max_chars / lengths, 1.0))),
    }


def load_article_texts():
    """Article texts exactly as the VLM saw them, from baselines/src/data_loader.py.

    Loaded from its file path because the repository root has its own, unrelated ``src`` package.
    """
    path = os.path.join(lock.REPO, "baselines", "src", "data_loader.py")
    spec = importlib.util.spec_from_file_location("baselines_data_loader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, documents = module.load_official_data()
    return {d["image_name"]: d["text"] for d in documents if d["lang"] == "EN" and d["source"] == "second_chunk"}


def _pred_path(pred_root, model, split):
    return os.path.join(pred_root, model, "seed0", f"{split}.tsv")


def analyse(pred_root, splits, article_texts):
    labels = read_labels(os.path.join(pred_root, "_gold", "labels.txt"))
    models = [m for m in VARIANTS if all(os.path.isfile(_pred_path(pred_root, m, s)) for s in splits)]
    out = {"splits": list(splits), "base": BASE, "models": models,
           "missing": [m for m in VARIANTS if m not in models], "by_split": {}}
    for split in splits:
        gold = read_frames_tsv(os.path.join(pred_root, "_gold", f"{split}.tsv"), labels)
        text = read_frames_tsv(os.path.join(pred_root, "_gold", f"{split}_text_frames.tsv"), labels)
        preds = {m: read_frames_tsv(_pred_path(pred_root, m, split), labels) for m in models}
        base = preds.get(BASE)
        out["by_split"][split] = {
            "n_images": len(gold),
            "gold": overlap_stats(gold, text),
            "variants": {m: {"overlap": overlap_stats(preds[m], text),
                             "swap_vs_base": None if m == BASE or base is None else swap_stats(base, preds[m], gold, text)}
                         for m in models},
            f"text_coverage_{MAX_TEXT_CHARS}": text_coverage([article_texts[n] for n in gold], MAX_TEXT_CHARS),
        }
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="C2 VLM text overlap.")
    parser.add_argument("--pred_root", default=os.path.join(lock.REPO, "predictions", "subtask2"))
    parser.add_argument("--include_test", action="store_true", help="also read test (requires results/LOCKED)")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    splits = ["dev"]
    if args.include_test:
        lock.require_lock()
        splits.append("test")
    result = analyse(args.pred_root, splits, load_article_texts())
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"Wrote {args.out}: splits={splits} models={result['models']}")


if __name__ == "__main__":
    main()
