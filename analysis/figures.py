"""Report figures from the result JSONs (plan Task 12). PNG and PDF for each figure, in results/figures/.

Usage (from the repo root): python -m analysis.figures
Colors are the validated light-mode slots of baselines/src/visualize.py: blue, orange, then violet for a third
series (blue/orange/violet passes the dataviz validator's adjacent CVD, normal-vision and contrast checks).
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from report.build_tables import DISPLAY  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "results", "figures")
SERIES = ["#2a78d6", "#eb6834", "#6250d6"]
EMPHASIS_MUTED = "#c3c2b7"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ABLATION = ["e2_vision_global", "e3_vision_region_coco", "e5_fusion_coco_asl", "e6_fusion_global_asl",
            "e7_vision_region_gdino", "e8_fusion_gdino_asl"]
TEXT_LENGTH = [("qwen_image_only", "Image only"), ("qwen_2a_title", "Title"), ("qwen_2a", "Title +\n1,500 chars"),
               ("qwen_2a_fulltext", "Full text"), ("qwen_2b", "+ gold\nframes")]
SHORT = {"Legality_Constitutionality_and_jurisprudence": "Legality", "Policy_prescription_and_evaluation": "Policy",
         "External_regulation_and_reputation": "External reputation", "Capacity_and_resources": "Capacity",
         "Fairness_and_equality": "Fairness", "Crime_and_punishment": "Crime", "Security_and_defense": "Security",
         "Health_and_safety": "Health", "Quality_of_life": "Quality of life", "Cultural_identity": "Cultural identity",
         "Public_opinion": "Public opinion"}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": EMPHASIS_MUTED,
    "axes.labelcolor": INK_SECONDARY, "xtick.color": INK_SECONDARY, "ytick.color": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "legend.frameon": False,
})


def _style(ax, grid_axis="x"):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)


def _save(fig, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf")
    plt.close(fig)


def _short(label):
    return SHORT.get(label, label)


def _label(name, final):
    exploratory = name in final["selection"].get("exploratory_not_selectable", [])
    return DISPLAY.get(name, name) + (" (exploratory)" if exploratory else "")


def plot_main_results(final, models, out_path):
    """Test Micro F1 with its 95% CI; the model selected on dev is the only colored one."""
    rows = [m for m in models if "test" in final["models"].get(m, {})][::-1]
    best = final["selection"].get("best_model")
    fig, ax = plt.subplots(figsize=(6.2, 0.32 * len(rows) + 0.8))
    for y, name in enumerate(rows):
        test = final["models"][name]["test"]
        color = SERIES[0] if name == best else INK_SECONDARY
        ax.hlines(y, *test["micro_ci95"], color=SERIES[0] if name == best else EMPHASIS_MUTED, linewidth=2)
        ax.plot(test["micro_mean"], y, "o", ms=7, color=color, mec=SURFACE, mew=2)
        if name == best:
            ax.annotate(f"{test['micro_mean']:.3f} (selected on dev)", (test["micro_ci95"][1], y),
                        xytext=(6, 0), textcoords="offset points", va="center", color=INK_SECONDARY)
    ax.set_yticks(range(len(rows)), [_label(n, final) for n in rows])
    ax.set_xlabel("Test Micro F1 (95% article-level bootstrap CI)")
    ax.set_xlim(0, 1)
    _style(ax)
    _save(fig, out_path)


def plot_ablation(final, out_path):
    """Dev and test Micro F1 (mean ± std over seeds) for the trained ablations."""
    rows = [m for m in ABLATION if m in final["models"]][::-1]
    fig, ax = plt.subplots(figsize=(6.2, 0.4 * max(len(rows), 2) + 0.9))
    if not rows:
        ax.text(0.5, 0.5, "No trained-model results yet.", ha="center", va="center", color=INK_SECONDARY)
        ax.axis("off")
    for k, (split, offset) in enumerate((("dev", 0.15), ("test", -0.15))):
        points = [(y + offset, final["models"][n][split]) for y, n in enumerate(rows) if split in final["models"][n]]
        if points:
            ax.errorbar([p["micro_mean"] for _, p in points], [y for y, _ in points],
                        xerr=[p["micro_std"] for _, p in points], fmt="o", ms=7, color=SERIES[k], mec=SURFACE,
                        mew=2, elinewidth=2, capsize=0, label=split)
    if rows:
        ax.set_yticks(range(len(rows)), [DISPLAY.get(n, n) for n in rows])
        ax.set_xlabel("Micro F1 (mean ± std over 3 seeds)")
        ax.legend(loc="lower right")
        _style(ax)
    _save(fig, out_path)


def labels_to_plot(final, models):
    """Labels (14 frames, then None) with at least one gold test image."""
    support = next(final["models"][m]["test"]["support"] for m in models if "test" in final["models"].get(m, {}))
    labels = final["labels"] + ([] if "None" in final["labels"] else ["None"])
    return [label for label in labels if support.get(label, 0) > 0]


def plot_per_label(final, models, out_path):
    models = [m for m in models if "test" in final["models"].get(m, {})][:3]
    labels = labels_to_plot(final, models)[::-1]
    support = final["models"][models[0]]["test"]["support"]
    fig, ax = plt.subplots(figsize=(6.2, 0.3 * len(labels) * len(models) / 2 + 1))
    height = 0.8 / len(models)
    for k, name in enumerate(models):
        f1 = [final["models"][name]["test"]["per_label_f1"].get(label) or 0.0 for label in labels]
        ys = np.arange(len(labels)) + (len(models) - 1 - k - (len(models) - 1) / 2) * height
        ax.barh(ys, f1, height=height, color=SERIES[k], edgecolor=SURFACE, linewidth=2, label=DISPLAY.get(name, name))
    ax.set_yticks(range(len(labels)), [f"{_short(label)} (n={support[label]})" for label in labels])
    ax.set_xlabel("Test F1 per label")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right")
    _style(ax)
    _save(fig, out_path)


def plot_c1_heatmap(c1, out_path):
    matrix = np.array(c1["conditional_matrix"], dtype=float)
    labels = [_short(label) for label in c1["labels"]]
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP).with_extremes(bad=GRID)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Image frame")
    ax.set_ylabel("Text frame in the article")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, shrink=0.8, label="P(image frame | text frame)")
    _save(fig, out_path)


def plot_region_coverage(cov, out_path):
    """Share of images whose regions are all detected, partly detected or all fixed-crop fallbacks."""
    parts = [("all_detected", "all regions detected", SERIES[0]), ("partial", "some regions detected", SERIES[1]),
             ("all_fallback", "fixed crops only", EMPHASIS_MUTED)]
    detectors = list(cov)[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 0.5 * len(detectors) + 1))
    for y, name in enumerate(detectors):
        left = 0.0
        for key, label, color in parts:
            share = cov[name][key] / cov[name]["n_images"]
            ax.barh(y, share, left=left, color=color, edgecolor=SURFACE, linewidth=2, label=label if y == 0 else None)
            left += share
    ax.set_yticks(range(len(detectors)), detectors)
    ax.set_xlabel("Share of images")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3)
    _style(ax)
    _save(fig, out_path)


def plot_c2(c2, final, out_path):
    """Left: Qwen Micro F1 as it reads more text. Right: labels added and removed relative to image only."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
    present = [(n, label) for n, label in TEXT_LENGTH if n in final["models"]]
    for k, split in enumerate(("dev", "test")):
        xs = [i for i, (n, _) in enumerate(present) if split in final["models"][n]]
        means = [final["models"][present[i][0]][split]["micro_mean"] for i in xs]
        if split == "test":
            cis = [final["models"][present[i][0]]["test"]["micro_ci95"] for i in xs]
            left.vlines(np.array(xs) + 0.08, [c[0] for c in cis], [c[1] for c in cis], color=SERIES[1], linewidth=2)
        left.plot(np.array(xs) + (0.08 if split == "test" else -0.08), means, "o", ms=7, color=SERIES[k],
                  mec=SURFACE, mew=2, label=split)
    left.set_xticks(range(len(present)), [label for _, label in present])
    left.set_ylabel("Micro F1")
    left.set_ylim(0, 1)
    left.legend(loc="upper right")
    _style(left, "y")
    split = "test" if "test" in c2["by_split"] else c2["splits"][-1]
    variants = c2["by_split"][split]["variants"]
    shown = [(n, label) for n, label in TEXT_LENGTH[1:] if (variants.get(n) or {}).get("swap_vs_base")]
    series = [("added", "added"), ("added_in_text", "added: text frame"), ("added_correct", "added: correct"),
              ("removed", "removed")]
    width = 0.8 / len(series)
    for k, (key, label) in enumerate(series):
        values = [variants[n]["swap_vs_base"][key] for n, _ in shown]
        color = SERIES[k] if k < len(SERIES) else EMPHASIS_MUTED
        right.bar(np.arange(len(shown)) + (k - 1.5) * width, values, width=width, color=color, edgecolor=SURFACE,
                  linewidth=2, label=label)
    right.set_xticks(range(len(shown)), [label for _, label in shown])
    right.set_ylabel(f"Labels vs image only ({split}, {c2['by_split'][split]['n_images']} images)")
    right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=8)
    _style(right, "y")
    _save(fig, out_path)


def main():
    def load(name):
        path = os.path.join(REPO, "results", name)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    final = load("subtask2_final.json")
    models = [n for n in DISPLAY if n in final["models"]] + sorted(set(final["models"]) - set(DISPLAY))
    best = final["selection"].get("best_model")
    plot_main_results(final, models, os.path.join(OUT_DIR, "main_results.png"))
    plot_ablation(final, os.path.join(OUT_DIR, "ablation.png"))
    per_label_models = [m for m in (best, "qwen_image_only", "e2_vision_global") if m]
    plot_per_label(final, list(dict.fromkeys(per_label_models)), os.path.join(OUT_DIR, "per_label.png"))
    for source, plot, figure in (("c1_image_text.json", plot_c1_heatmap, "c1_heatmap.png"),
                                 ("region_coverage.json", plot_region_coverage, "region_coverage.png")):
        data = load(source)
        if data is not None:
            plot(data, os.path.join(OUT_DIR, figure))
    c2 = load("c2_vlm_text.json")
    if c2 is not None:
        plot_c2(c2, final, os.path.join(OUT_DIR, "c2.png"))
    print(f"Figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
