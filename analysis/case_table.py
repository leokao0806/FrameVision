"""C4: one row per dev (and, after the lock, test) image with gold frames and each model's prediction.

Used to pick about 8 cases for the qualitative error analysis. A model with several seeds is shown with
seed 42 when it exists, otherwise its smallest seed. The name `best` resolves to selection.best_model in
results/subtask2_final.json. Before results/LOCKED exists only dev is read; --include_test requires the lock.

Usage:
    python -m analysis.case_table --pred_root predictions/subtask2 --models best qwen_image_only e2_vision_global \
        [--include_test] [--out results/case_table.md]
"""
import argparse
import json
import os

from evaluation.prediction_io import discover_runs, read_frames_tsv, read_labels

from . import lock

DEFAULT_OUT = os.path.join(lock.REPO, "results", "case_table.md")
DEFAULT_FINAL = os.path.join(lock.REPO, "results", "subtask2_final.json")
PREFERRED_SEED = 42


def _fmt(frames):
    return ", ".join(sorted(frames)) or "None"


def _pick_seed(seeds):
    return PREFERRED_SEED if PREFERRED_SEED in seeds else min(seeds)


def build_case_table(pred_root, models, splits, labels_shown=None):
    """models: real model folder names; labels_shown: optional display names in the same order."""
    labels = read_labels(os.path.join(pred_root, "_gold", "labels.txt"))
    runs = discover_runs(pred_root)
    shown = labels_shown or models
    lines = ["# C4 case table", "", f"Splits: {', '.join(splits)}. ✓ = prediction equals the gold frames exactly.", ""]
    columns = []
    for model, label in zip(models, shown):
        seeds = {s: paths for s, paths in runs.get(model, {}).items() if all(sp in paths for sp in splits)}
        if not seeds:
            lines.append(f"- {model}: no predictions for {', '.join(splits)} (skipped)")
            continue
        seed = _pick_seed(seeds)
        columns.append((label, {sp: read_frames_tsv(seeds[seed][sp], labels) for sp in splits}))
        if len(seeds) > 1:
            lines.append(f"- {model}: seed {seed}")
    lines += ["", "| split | image | gold | " + " | ".join(label for label, _ in columns) + " |",
              "|" + "---|" * (3 + len(columns))]
    for split in splits:
        gold = read_frames_tsv(os.path.join(pred_root, "_gold", f"{split}.tsv"), labels)
        for name, frames in gold.items():
            cells = []
            for _, preds in columns:
                pred = preds[split][name]
                cells.append(f"{_fmt(pred)} {'✓' if set(pred) == set(frames) else '✗'}")
            lines.append(f"| {split} | {name} | {_fmt(frames)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="C4 case table.")
    parser.add_argument("--pred_root", default=os.path.join(lock.REPO, "predictions", "subtask2"))
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--final", default=DEFAULT_FINAL, help="used to resolve the name `best`")
    parser.add_argument("--include_test", action="store_true", help="also show test (requires results/LOCKED)")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    splits = ["dev"]
    if args.include_test:
        lock.require_lock()
        splits.append("test")
    models, shown = [], []
    for name in args.models:
        if name == "best":
            with open(args.final, encoding="utf-8") as f:
                name = json.load(f)["selection"]["best_model"]
            models.append(name)
            shown.append(f"{name} (best)")
        else:
            models.append(name)
            shown.append(name)
    table = build_case_table(args.pred_root, models, splits, shown)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(table)
    print(f"Wrote {args.out}: splits={splits} models={models}")


if __name__ == "__main__":
    main()
