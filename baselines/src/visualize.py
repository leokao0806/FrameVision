"""
Report Figures from the Saved Experiment Results
CS 5134/6034 NLP Fall 2026 - Team FrameVision
Authors: Leo Kao, En-Shuo Zhang
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
EMPHASIS_MUTED = "#c3c2b7"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": EMPHASIS_MUTED,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
})


def _style_axis(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)


def plot_subtask2_comparison(rows, n_images, out_path):
    """rows: (model name, Micro F1, is_trivial_baseline), drawn in the given order."""
    names = [r[0] for r in rows]
    scores = [r[1] for r in rows]
    colors = [EMPHASIS_MUTED if r[2] else SERIES_1 for r in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(6.5, 2.9), dpi=300)
    ax.barh(y, scores, height=0.62, color=colors)
    for yi, score in zip(y, scores):
        ax.text(score + 0.012, yi, f"{score:.3f}", va="center", color=INK, fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"Micro F1 on {n_images} held-out EN images (gray = trivial baseline)", fontsize=8)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_per_label(first, second, support, names, xlabel, out_path):
    """Per-label F1 of two models, for labels that occur in the gold test data."""
    labels = sorted([l for l, s in support.items() if s > 0], key=lambda l: -support[l])
    y = np.arange(len(labels))
    height = 0.36

    fig, ax = plt.subplots(figsize=(6.5, 3.1), dpi=300)
    ax.barh(y - height / 2 - 0.01, [first[l] for l in labels], height=height, color=SERIES_1, label=names[0])
    ax.barh(y + height / 2 + 0.01, [second[l] for l in labels], height=height, color=SERIES_2, label=names[1])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{l.replace('_', ' ')} (n={support[l]})" for l in labels], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _f1_by_label(per_label):
    return {label: values["f1"] for label, values in per_label.items()}


def main():
    results_dir = os.path.join(project_root, "results")
    plots_dir = os.path.join(project_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    with open(os.path.join(results_dir, "subtask2_metrics.json"), encoding="utf-8") as f:
        subtask2 = json.load(f)
    with open(os.path.join(results_dir, "vlm_test_metrics.json"), encoding="utf-8") as f:
        vlm = json.load(f)["variants"]
    with open(os.path.join(results_dir, "subtask1_metrics.json"), encoding="utf-8") as f:
        subtask1 = json.load(f)
    supervised = subtask2["models"]
    n_images = subtask2["n_test_images"]

    rows = [
        ("Random (training prior)", supervised["Random (training prior)"]["micro_f1"], True),
        ("Most frequent label", supervised["Most frequent label"]["micro_f1"], True),
        ("Copy text frames (2b)", supervised["Copy text frames (2b)"]["micro_f1"], True),
        ("Image LR (SigLIP)", supervised["Image LR (SigLIP)"]["micro_f1"], False),
        ("Text LR (DeBERTa-v3)", supervised["Text LR (DeBERTa-v3)"]["micro_f1"], False),
        ("Late fusion", supervised["Late fusion (image + text)"]["micro_f1"], False),
        ("Late fusion + text frames (2b)", supervised["Late fusion + text frames (2b)"]["micro_f1"], False),
        ("Qwen2.5-VL, image only", vlm["image_only"]["micro_f1"], False),
        ("Qwen2.5-VL, image + text (2a)", vlm["subtask_2a"]["micro_f1"], False),
        ("Qwen2.5-VL, + text frames (2b)", vlm["subtask_2b"]["micro_f1"], False),
    ]
    plot_subtask2_comparison(rows, n_images, os.path.join(plots_dir, "figure1_subtask2_micro_f1.png"))

    image_lr = supervised["Image LR (SigLIP)"]["per_label"]
    plot_per_label(_f1_by_label(image_lr), _f1_by_label(vlm["image_only"]["per_label"]),
                   {label: values["support"] for label, values in image_lr.items()},
                   ("Image LR (SigLIP)", "Qwen2.5-VL, image only"),
                   f"F1 on the course test images ({n_images} images)",
                   os.path.join(plots_dir, "figure3_subtask2_per_label_f1.png"))

    # Subtask 1 results are written only after every seed has finished.
    if "selected_on_dev" not in subtask1:
        print("Subtask 1 results are not final yet; skipping its figure.")
        return
    selected = subtask1["selected_on_dev"]
    n_seeds = len(subtask1["models"][selected].get("runs", []))
    selected_name = f"{selected}, mean of {n_seeds} seeds" if n_seeds else selected
    plot_per_label(_f1_by_label(subtask1["tfidf_per_label"]), subtask1["selected_per_label_mean_f1"],
                   subtask1["selected_support"], ("TF-IDF + LR", selected_name),
                   f"F1 on the course test articles ({subtask1['split']['test']} paragraphs)",
                   os.path.join(plots_dir, "figure2_subtask1_per_label_f1.png"))


if __name__ == "__main__":
    main()
