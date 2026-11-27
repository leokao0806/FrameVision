"""Read and write official Subtask 2 prediction files.

Format: one line per image, ``image_name<TAB>frame1<TAB>frame2...``. A line with only the image name
(or with the literal ``None``) means the image has no frame, which the official scorer counts as the
``None`` class.
"""
import os
import re

import numpy as np

NONE_LABEL = "None"
_SEED_DIR = re.compile(r"^seed(\d+)$")
_SPLITS = ("dev", "test")


def read_labels(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_frames_tsv(path, labels):
    allowed = set(labels)
    rows = {}
    with open(path, encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip()
            if not line:
                continue
            name, *frames = line.split("\t")
            frames = [fr for fr in frames if fr and fr != NONE_LABEL]
            unknown = [fr for fr in frames if fr not in allowed]
            if unknown:
                raise ValueError(f"{path}:{line_no}: unknown frame(s) {unknown} for {name}")
            if name in rows:
                raise ValueError(f"{path}:{line_no}: duplicate image {name}")
            rows[name] = frames
    return rows


def write_frames_tsv(path, rows):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for name, frames in rows.items():
            f.write("\t".join([name] + list(frames)) + "\n")


def to_matrix(rows, image_names, labels):
    index = {label: j for j, label in enumerate(labels)}
    matrix = np.zeros((len(image_names), len(labels) + 1), dtype=int)
    for i, name in enumerate(image_names):
        frames = rows[name]
        if frames:
            for fr in frames:
                matrix[i, index[fr]] = 1
        else:
            matrix[i, -1] = 1
    return matrix


def check_same_images(rows, gold_rows, source):
    missing = sorted(set(gold_rows) - set(rows))
    extra = sorted(set(rows) - set(gold_rows))
    if missing or extra:
        raise ValueError(f"{source}: images differ from gold; missing={missing} extra={extra}")


def discover_runs(pred_root):
    runs = {}
    for model in sorted(os.listdir(pred_root)):
        model_dir = os.path.join(pred_root, model)
        if model.startswith("_") or not os.path.isdir(model_dir):
            continue
        for seed_dir in sorted(os.listdir(model_dir)):
            match = _SEED_DIR.match(seed_dir)
            if not match:
                continue
            for split in _SPLITS:
                path = os.path.join(model_dir, seed_dir, f"{split}.tsv")
                if os.path.isfile(path):
                    runs.setdefault(model, {}).setdefault(int(match.group(1)), {})[split] = path
    return runs
