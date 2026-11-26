"""
Export the Intermediate Report baselines as official-format prediction files for evaluation/.
CS 5134/6034 NLP Fall 2026 - Team FrameVision

Usage (from baselines/): python scripts/export_predictions.py [out_root]

Reads results/subtask2_metrics.json (run `scripts/run_experiments.py subtask2` first, so dev predictions
exist), results/vlm_{dev,test}_metrics.json and results/course_split.json, and writes
<out_root>/<model>/seed0/{dev,test}.tsv plus <out_root>/_gold and <out_root>/_reference.
Each line is `image_name<TAB>frame1<TAB>frame2...`; an image with no frame is just `image_name`.
"""

import json
import os
import shutil
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_loader import FRAME_TAXONOMY, load_official_data  # noqa: E402
from src.official_scorer import vector_to_frames  # noqa: E402

RESULTS_DIR = os.path.join(project_root, "results")
DEFAULT_OUT = os.path.join(os.path.dirname(project_root), "predictions", "subtask2")

MODEL_NAMES = {
    "Most frequent label": "most_frequent",
    "Copy text frames (2b)": "copy_text_frames_2b",
    "Image LR (SigLIP)": "image_lr",
    "Text LR (DeBERTa-v3)": "text_lr",
    "Late fusion (image + text)": "late_fusion_2a",
    "Late fusion + text frames (2b)": "late_fusion_2b",
}
VLM_NAMES = {"image_only": "qwen_image_only", "subtask_2a": "qwen_2a", "subtask_2b": "qwen_2b",
             "image_only_v2": "qwen_image_only_v2"}
RANDOM_NAME = "Random (training prior)"


def write_tsv(path, rows):
    """rows: list of (image_name, frames)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for name, frames in rows:
            f.write("\t".join([name] + list(frames)) + "\n")


def _rows(doc_ids, matrix, by_id):
    if len(doc_ids) != len(matrix):
        raise ValueError(f"{len(doc_ids)} documents but {len(matrix)} prediction rows")
    return [(by_id[doc_id]["image_name"], vector_to_frames(row)) for doc_id, row in zip(doc_ids, matrix)]


def _write_model(out_root, model, split, rows):
    write_tsv(os.path.join(out_root, model, "seed0", f"{split}.tsv"), rows)


def export_gold(documents, split, out_root):
    """Gold image frames and the union of paragraph (text) frames for train/dev/test."""
    gold = os.path.join(out_root, "_gold")
    os.makedirs(gold, exist_ok=True)
    with open(os.path.join(gold, "labels.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(FRAME_TAXONOMY) + "\n")
    held_out = set(split["dev"]) | set(split["test"])
    parts = {
        "dev": [d for d in documents if d["doc_id"] in set(split["dev"])],
        "test": [d for d in documents if d["doc_id"] in set(split["test"])],
        # Only articles with paragraph labels, so image and text frames can be compared.
        "train": [d for d in documents if d["doc_id"] not in held_out and d["has_text_frames"]],
    }
    for name, docs in parts.items():
        write_tsv(os.path.join(gold, f"{name}.tsv"), [(d["image_name"], d["frames"]) for d in docs])
        write_tsv(os.path.join(gold, f"{name}_text_frames.tsv"),
                  [(d["image_name"], vector_to_frames(d["text_frame_labels"])) for d in docs])
    return parts


def export_baselines(st2, by_id, out_root):
    if "dev_predictions" not in st2:
        raise KeyError("subtask2_metrics.json has no dev predictions; rerun `scripts/run_experiments.py subtask2`")
    written = []
    for name, model in MODEL_NAMES.items():
        if name not in st2["predictions"]:
            continue
        _write_model(out_root, model, "test", _rows(st2["doc_order"], st2["predictions"][name], by_id))
        _write_model(out_root, model, "dev", _rows(st2["dev_doc_order"], st2["dev_predictions"][name], by_id))
        written.append(model)
    return written


def export_vlm(vlm_by_split, by_id, out_root):
    """vlm_by_split: {"dev": vlm_dev_metrics, "test": vlm_test_metrics}."""
    written = []
    for variant, model in VLM_NAMES.items():
        for split, data in vlm_by_split.items():
            if variant in data["variants"]:
                _write_model(out_root, model, split, _rows(data["doc_order"], data["variants"][variant]["predictions"], by_id))
                if model not in written:
                    written.append(model)
    return written


def export_random_prior(st2, out_root):
    """The random baseline is an expectation over draws, not one prediction file, so it is reference only."""
    path = os.path.join(out_root, "_reference", "random_prior.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(st2["models"][RANDOM_NAME], f, indent=2)
        f.write("\n")


def _load(name):
    with open(os.path.join(RESULTS_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def main(out_root=DEFAULT_OUT):
    _, documents = load_official_data()
    by_id = {d["doc_id"]: d for d in documents}
    st2 = _load("subtask2_metrics.json")
    for model in list(MODEL_NAMES.values()) + list(VLM_NAMES.values()):
        shutil.rmtree(os.path.join(out_root, model, "seed0"), ignore_errors=True)
    parts = export_gold(documents, _load("course_split.json"), out_root)
    models = export_baselines(st2, by_id, out_root)
    models += export_vlm({"dev": _load("vlm_dev_metrics.json"), "test": _load("vlm_test_metrics.json")}, by_id, out_root)
    export_random_prior(st2, out_root)
    print(f"Wrote {len(models)} models to {out_root}: {', '.join(models)}")
    print(f"Gold: train={len(parts['train'])} dev={len(parts['dev'])} test={len(parts['test'])}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
