"""Evaluate every prediction folder and write results/subtask2_final.json.

Selection follows docs/preregistration.md: dev only, seed-averaged Micro F1, ties go to the simpler model.

Usage:
    python -m evaluation.run_eval --pred_root predictions/subtask2 --scorer <scorer-subtask-2.py> \
        --out results/subtask2_final.json
"""
import argparse
import json
import os

import numpy as np

from .bootstrap import bootstrap_ci, group_of, paired_ci
from .prediction_io import NONE_LABEL, discover_runs, read_labels
from .scoring import score_split

TRAINED_MODELS = [
    "e1_text_only",
    "e2_vision_global",
    "e3_vision_region_coco",
    "e4_fusion_coco_bce",
    "e5_fusion_coco_asl",
    "e6_fusion_global_asl",
    "e7_vision_region_gdino",
    "e8_fusion_gdino_asl",
]

# Simplest first (docs/preregistration.md): trivial < logistic regression < zero-shot VLM
# < trained single-modality < trained fusion < combination.
COMPLEXITY_ORDER = [
    "most_frequent", "copy_text_frames_2b",
    "image_lr", "text_lr", "late_fusion_2a", "late_fusion_2b",
    "qwen_image_only", "qwen_image_only_v2", "qwen_2a_title", "qwen_2a", "qwen_2a_fulltext", "qwen_2b",
    "e1_text_only", "e2_vision_global", "e3_vision_region_coco", "e7_vision_region_gdino",
    "e6_fusion_global_asl", "e4_fusion_coco_bce", "e5_fusion_coco_asl", "e8_fusion_gdino_asl",
    "combo_qwen_best",
]

PRIMARY_COMPARISONS = [
    ("best", "most_frequent"),
    ("best", "qwen_image_only"),
    ("e8_fusion_gdino_asl", "e5_fusion_coco_asl"),
    ("e6_fusion_global_asl", "e2_vision_global"),
]

RULE_FILE = "docs/preregistration.md"
TIE_TOLERANCE = 1e-12


def _complexity_rank(name):
    if name in COMPLEXITY_ORDER:
        return (COMPLEXITY_ORDER.index(name), "")
    return (len(COMPLEXITY_ORDER), name)


def select_best(models, candidates=None):
    names = [m for m in (candidates if candidates is not None else models) if "dev" in models.get(m, {})]
    if not names:
        return None
    top = max(models[m]["dev"]["micro_mean"] for m in names)
    tied = [m for m in names if top - models[m]["dev"]["micro_mean"] <= TIE_TOLERANCE]
    return min(tied, key=_complexity_rank)


def _dev_summary(scores):
    return {
        "micro_mean": float(np.mean(scores["micro"])),
        "micro_std": float(np.std(scores["micro"])),
        "macro_mean": float(np.mean(scores["macro"])),
        "hamming_mean": float(np.mean(scores["hamming"])),
    }


def _test_summary(scores):
    groups = [group_of(name) for name in scores["image_names"]]
    lo, hi = bootstrap_ci(scores["y_true"], scores["matrices"], groups)
    return {
        "micro_mean": float(np.mean(scores["micro"])),
        "micro_std": float(np.std(scores["micro"])),
        "micro_ci95": [lo, hi],
        "macro_mean": float(np.mean(scores["macro"])),
        "macro_std": float(np.std(scores["macro"])),
        "hamming_mean": float(np.mean(scores["hamming"])),
        "per_label_f1": scores["per_label_f1"],
        "support": scores["support"],
    }


def _compare(spec_a, spec_b, best, models, test_scores):
    a = best if spec_a == "best" else spec_a
    b = best if spec_b == "best" else spec_b
    entry = {"spec_a": spec_a, "spec_b": spec_b, "a": a, "b": b, "diff_mean": None, "ci95": None}
    if a is None or b is None or a not in test_scores or b not in test_scores:
        entry["status"] = "missing"
    elif a == b:
        entry["status"] = "same_model"
    else:
        sa, sb = test_scores[a], test_scores[b]
        groups = [group_of(name) for name in sa["image_names"]]
        lo, hi = paired_ci(sa["y_true"], sa["matrices"], sb["matrices"], groups)
        entry.update(status="ok", ci95=[lo, hi],
                     diff_mean=models[a]["test"]["micro_mean"] - models[b]["test"]["micro_mean"])
    return entry


def evaluate_all(pred_root, scorer_path):
    gold_dir = os.path.join(pred_root, "_gold")
    labels_path = os.path.join(gold_dir, "labels.txt")
    gold = {split: os.path.join(gold_dir, f"{split}.tsv") for split in ("dev", "test")}
    labels = read_labels(labels_path)

    models, test_scores, sizes = {}, {}, {}
    for model, seeds in discover_runs(pred_root).items():
        entry = {"seeds": sorted(seeds), "scorer_mismatches": 0}
        for split in ("dev", "test"):
            paths = [seeds[s][split] for s in sorted(seeds) if split in seeds[s]]
            if not paths:
                continue
            scores = score_split(paths, gold[split], labels_path, scorer_path)
            sizes[split] = len(scores["image_names"])
            entry["scorer_mismatches"] += scores["mismatches"]
            if split == "dev":
                entry["dev"] = _dev_summary(scores)
            else:
                entry["test"] = _test_summary(scores)
                test_scores[model] = scores
        models[model] = entry

    best = select_best(models)
    out = {
        "labels": labels + [NONE_LABEL],
        "n_dev": sizes.get("dev"),
        "n_test": sizes.get("test"),
        "models": models,
        "reference_only": {},
        "selection": {"best_model": best, "best_trained_model": select_best(models, TRAINED_MODELS),
                      "rule": RULE_FILE},
        "primary_comparisons": [_compare(a, b, best, models, test_scores) for a, b in PRIMARY_COMPARISONS],
    }
    prior_path = os.path.join(pred_root, "_reference", "random_prior.json")
    if os.path.isfile(prior_path):
        with open(prior_path, encoding="utf-8") as f:
            out["reference_only"]["random_prior"] = json.load(f)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pred_root", default="predictions/subtask2")
    parser.add_argument("--scorer", required=True)
    parser.add_argument("--out", default="results/subtask2_final.json")
    args = parser.parse_args(argv)
    result = evaluate_all(args.pred_root, args.scorer)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"Wrote {args.out}: {len(result['models'])} models, best = {result['selection']['best_model']}")


if __name__ == "__main__":
    main()
