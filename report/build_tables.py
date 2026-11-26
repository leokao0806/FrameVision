"""Fill the table markers of report/final_report.md from results/subtask2_final.json.

Usage (from the repo root): python report/build_tables.py
Writes report/final_report.built.md. Markers: <!-- table:main --> and <!-- table:ablation -->.
"""

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from evaluation.run_eval import COMPLEXITY_ORDER  # noqa: E402

MD_PATH = os.path.join(REPO, "report", "final_report.md")
BUILT_PATH = os.path.join(REPO, "report", "final_report.built.md")
FINAL_PATH = os.path.join(REPO, "results", "subtask2_final.json")

DISPLAY = {
    "most_frequent": "Most frequent label (Political)",
    "copy_text_frames_2b": "Copy gold text frames (2b)",
    "image_lr": "Image LR (SigLIP)",
    "text_lr": "Text LR (DeBERTa-v3)",
    "late_fusion_2a": "Late fusion, image + text (2a)",
    "late_fusion_2b": "Late fusion + gold frames (2b)",
    "qwen_image_only": "Qwen2.5-VL, image only",
    "qwen_image_only_v2": "Qwen2.5-VL, image only, None option (E10)",
    "qwen_2a_title": "Qwen2.5-VL, title (E12a)",
    "qwen_2a": "Qwen2.5-VL, title + 1,500 characters (2a)",
    "qwen_2a_fulltext": "Qwen2.5-VL, title + full text (E12b)",
    "qwen_2b": "Qwen2.5-VL, + gold frames (2b)",
    "e1_text_only": "E1: text only",
    "e2_vision_global": "E2: image only (global)",
    "e3_vision_region_coco": "E3: image + COCO regions",
    "e4_fusion_coco_bce": "E4: fusion + COCO regions, BCE",
    "e5_fusion_coco_asl": "E5: fusion + COCO regions, ASL",
    "e6_fusion_global_asl": "E6: fusion, no regions, ASL",
    "e7_vision_region_gdino": "E7: image + Grounding DINO regions",
    "e8_fusion_gdino_asl": "E8: fusion + Grounding DINO regions, ASL",
    "combo_qwen_best": "E9: Qwen + best trained model",
    "combo_qwen_image_lr": "Qwen + Image LR",
}


def _is_trained(name):
    return name[:1] == "e" and name[1:2].isdigit()


def _order(names):
    rank = {name: i for i, name in enumerate(COMPLEXITY_ORDER)}
    return sorted(names, key=lambda n: (rank.get(n, len(rank)), n))


def _mean(summary, multi):
    text = f"{summary['micro_mean']:.3f}"
    return f"{text} ± {summary['micro_std']:.3f}" if multi else text


def _test_cell(summary, multi):
    low, high = summary["micro_ci95"]
    return f"{_mean(summary, multi)} [{low:.3f}, {high:.3f}]"


def _name(name, final):
    label = DISPLAY.get(name, name)
    if name in final["selection"].get("exploratory_not_selectable", []):
        label += " (exploratory)"
    return f"**{label}**" if name == final["selection"].get("best_model") else label


def main_table(final):
    """Every model with test results except the trained ablations; the best trained model joins them."""
    best_trained = final["selection"].get("best_trained_model")
    names = [n for n, m in final["models"].items()
             if "test" in m and (not _is_trained(n) or n == best_trained)]
    rows = ["| Model | Dev Micro | Test Micro [95% CI] | Test Macro | Test Hamming |", "|---|---|---|---|---|"]
    for name in _order(names):
        model = final["models"][name]
        multi = len(model["seeds"]) > 1
        rows.append(f"| {_name(name, final)} | {_mean(model['dev'], multi)} | {_test_cell(model['test'], multi)} | "
                    f"{model['test']['macro_mean']:.3f} | {model['test']['hamming_mean']:.3f} |")
    return "\n".join(rows)


def ablation_table(final):
    """Trained models E1–E8 (and any E11 run): mean ± std over seeds."""
    names = _order([n for n in final["models"] if _is_trained(n)])
    if not names:
        return "*No trained-model results yet.*"
    rows = ["| Model | Dev Micro (mean ± std) | Test Micro (mean ± std) [95% CI] |", "|---|---|---|"]
    for name in names:
        model = final["models"][name]
        test = _test_cell(model["test"], True) if "test" in model else "—"
        rows.append(f"| {_name(name, final)} | {_mean(model['dev'], True)} | {test} |")
    return "\n".join(rows)


def build(md_text, final):
    return (md_text.replace("<!-- table:main -->", main_table(final))
            .replace("<!-- table:ablation -->", ablation_table(final)))


def main():
    with open(MD_PATH, encoding="utf-8") as f:
        md_text = f.read()
    with open(FINAL_PATH, encoding="utf-8") as f:
        final = json.load(f)
    with open(BUILT_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(build(md_text, final))
    print(f"Wrote {BUILT_PATH}")


if __name__ == "__main__":
    main()
