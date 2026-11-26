"""
Scoring through the Unmodified Official Scorers
CS 5134/6034 NLP Fall 2026 - Team FrameVision
Authors: Leo Kao, En-Shuo Zhang

Writes gold and prediction files in the official format and runs the starter-kit scorer.
"""

import os
import subprocess
import sys

from src.data_loader import FRAME_TAXONOMY, project_root

KIT = os.path.join(project_root, "data", "starter-kit", "starter-kit")
SCORER_1 = os.path.join(KIT, "subtask1", "scorer-subtask-1.py")
SCORER_2 = os.path.join(KIT, "subtask2a_2b", "scorer-subtask-2.py")
WORK_DIR = os.path.join(project_root, "results", "official_scoring")


def vector_to_frames(row):
    return [FRAME_TAXONOMY[i] for i, value in enumerate(row) if value]


def _write_lines(path, lines):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def _run_scorer(scorer, gold_path, pred_path):
    labels_path = os.path.join(WORK_DIR, "labels.txt")
    _write_lines(labels_path, FRAME_TAXONOMY)
    result = subprocess.run([sys.executable, scorer, "-g", gold_path, "-p", pred_path, "-f", labels_path, "-o"],
                            capture_output=True, text=True, check=True)
    fields = result.stdout.strip().splitlines()[-1].split("\t")
    return float(fields[0]), float(fields[1])


def score_subtask1(paragraphs, predictions, tag):
    """Returns the official (micro_f1, macro_f1) for one row of predictions per paragraph."""
    os.makedirs(WORK_DIR, exist_ok=True)
    keys = [[p["doc_name"], str(p["start"]), str(p["end"])] for p in paragraphs]
    gold_path, pred_path = os.path.join(WORK_DIR, "gold_st1.tsv"), os.path.join(WORK_DIR, f"st1_{tag}.tsv")
    _write_lines(gold_path, ["\t".join(k + p["frames"]) for k, p in zip(keys, paragraphs)])
    _write_lines(pred_path, ["\t".join(k + vector_to_frames(row)) for k, row in zip(keys, predictions)])
    return _run_scorer(SCORER_1, gold_path, pred_path)


def score_subtask2(documents, predictions, tag):
    """Returns the official (micro_f1, macro_f1) for one row of predictions per image."""
    os.makedirs(WORK_DIR, exist_ok=True)
    gold_path, pred_path = os.path.join(WORK_DIR, "gold_st2.tsv"), os.path.join(WORK_DIR, f"st2_{tag}.tsv")
    _write_lines(gold_path, ["\t".join([d["image_name"]] + d["frames"]) for d in documents])
    _write_lines(pred_path, ["\t".join([d["image_name"]] + vector_to_frames(row)) for d, row in zip(documents, predictions)])
    return _run_scorer(SCORER_2, gold_path, pred_path)
