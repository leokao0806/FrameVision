"""
SemEval-2027 Task 9 Official Data Loader
CS 5134/6034 NLP Fall 2026 - Team FrameVision
Authors: Leo Kao, En-Shuo Zhang

Parses the official training releases and the trial release:
- Second chunk (EN, BG, PT): paragraph frames, image frames, and "None" examples. It
  already contains the trial articles of those languages, so only trial RU is added.
- First chunk, Subtask 1: SemEval-2023 articles in nine languages, with no "None" examples.
- First chunk, Subtask 2: 400 English article-image pairs, with no gold text frames.

Splits are always made by article, never by paragraph, because all paragraphs of an
article share a topic, a vocabulary, and one image.
"""

import os
import re
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL_ROOT = os.path.join(project_root, "data", "data-second-batch", "data")
SECOND_CHUNK = os.path.join(OFFICIAL_ROOT, "training-second-batch-oct")
FIRST_CHUNK_ST1 = os.path.join(OFFICIAL_ROOT, "initial-training-data", "subtask1")
FIRST_CHUNK_ST2 = os.path.join(OFFICIAL_ROOT, "initial-training-data", "subtask2")
TRIAL_ROOT = os.path.join(project_root, "data", "trialdataSemEval2027Task9")
SECOND_CHUNK_LANGUAGES = ["EN", "BG", "PT"]
# The second chunk writes "no framing" in these ways as well as with an empty field.
NO_FRAMING_MARKS = {"", "-", "No framing"}


def load_label_names(data_root=TRIAL_ROOT):
    with open(os.path.join(data_root, "labels.txt"), encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


FRAME_TAXONOMY = load_label_names()
FRAME_TO_ID = {frame: idx for idx, frame in enumerate(FRAME_TAXONOMY)}
NUM_CLASSES = len(FRAME_TAXONOMY)


def labels_to_vector(frames):
    vec = np.zeros(NUM_CLASSES, dtype=np.int64)
    for frame in frames:
        vec[FRAME_TO_ID[frame]] = 1
    return vec


def clean_frames(cells):
    return sorted({cell.strip() for cell in cells} - NO_FRAMING_MARKS)


def _read_tsv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [line.rstrip("\r\n").split("\t") for line in f if line.strip()]


def _read_article(path):
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        return f.read()


def _load_paragraphs(folder, lang, source):
    """Reads Subtask 1 annotations; a repeated span keeps its last row, as the scorer does."""
    raw_dir = os.path.join(folder, "raw-documents")
    listed = {tuple(row[:3]): row[-1] for row in _read_tsv(os.path.join(folder, "subtask-1-annotations-with-paragraphs.txt"))}
    rows = {tuple(row[:3]): row for row in _read_tsv(os.path.join(folder, "subtask-1-annotations.txt"))}
    paragraphs = []
    for key, row in rows.items():
        doc_name, start, end = row[0], int(row[1]), int(row[2])
        frames = clean_frames(row[3:])
        text = listed.get(key) or _read_article(os.path.join(raw_dir, doc_name))[start:end]
        paragraphs.append({
            "id": f"{doc_name}:{start}-{end}",
            "doc_name": doc_name,
            "doc_id": os.path.splitext(doc_name)[0],
            "lang": lang,
            "source": source,
            "start": start,
            "end": end,
            "text": text.strip(),
            "frames": frames,
            "labels": labels_to_vector(frames),
        })
    return paragraphs


def _load_annotated_folder(folder, lang, source):
    """Returns (paragraphs, documents) for a folder with both Subtask 1 and Subtask 2 labels."""
    raw_dir = os.path.join(folder, "raw-documents")
    paragraphs = _load_paragraphs(folder, lang, source)
    by_doc = {}
    for paragraph in paragraphs:
        by_doc.setdefault(paragraph["doc_id"], []).append(paragraph)

    documents = []
    for row in sorted(_read_tsv(os.path.join(folder, "subtask-2-annotations.txt"))):
        image_name = row[0]
        doc_id = os.path.splitext(image_name)[0]
        frames = clean_frames(row[1:])
        text_frames = sorted({f for p in by_doc[doc_id] for f in p["frames"]})
        documents.append({
            "doc_id": doc_id,
            "lang": lang,
            "source": source,
            "text": _read_article(os.path.join(raw_dir, f"{doc_id}.txt")),
            "image_path": os.path.join(raw_dir, image_name),
            "image_name": image_name,
            "frames": frames,
            "labels": labels_to_vector(frames),
            "has_text_frames": True,
            "text_frames": text_frames,
            "text_frame_labels": labels_to_vector(text_frames),
        })
    return paragraphs, documents


def _load_first_chunk_paragraphs():
    paragraphs = []
    for part in ("A", "B"):
        base = os.path.join(FIRST_CHUNK_ST1, f"SEMEVAL23_FRAMING_INITIAL_TRAIN-{part}")
        for lang in sorted(os.listdir(base)):
            if os.path.isdir(os.path.join(base, lang)):
                paragraphs += _load_paragraphs(os.path.join(base, lang), lang, "semeval23")
    return paragraphs


def _load_first_chunk_images():
    documents = []
    image_dir = os.path.join(FIRST_CHUNK_ST2, "images")
    for row in sorted(_read_tsv(os.path.join(FIRST_CHUNK_ST2, "subtask-2-annotations.txt"))):
        image_name = row[0]
        number = re.fullmatch(r"EN_doc_(\d+)_image\.\w+", image_name).group(1)
        frames = clean_frames(row[1:])
        documents.append({
            "doc_id": f"EN_M_TRAIN_doc_{number}",
            "lang": "EN",
            "source": "first_chunk",
            "text": _read_article(os.path.join(image_dir, f"EN_M_TRAIN_doc_{number}.txt")),
            "image_path": os.path.join(image_dir, image_name),
            "image_name": image_name,
            "frames": frames,
            "labels": labels_to_vector(frames),
            "has_text_frames": False,
            "text_frames": [],
            "text_frame_labels": labels_to_vector([]),
        })
    return documents


def load_official_data():
    """Returns all paragraphs and documents; each record says its language and source."""
    paragraphs, documents = [], []
    for lang in SECOND_CHUNK_LANGUAGES:
        lang_paragraphs, lang_documents = _load_annotated_folder(os.path.join(SECOND_CHUNK, lang), lang, "second_chunk")
        paragraphs += lang_paragraphs
        documents += lang_documents
    trial_paragraphs, trial_documents = _load_annotated_folder(os.path.join(TRIAL_ROOT, "RU"), "RU", "trial")
    paragraphs += trial_paragraphs + _load_first_chunk_paragraphs()
    documents += trial_documents + _load_first_chunk_images()
    return paragraphs, documents


def split_article_paragraphs(text):
    return [part.strip() for part in re.split(r"\r?\n\s*\r?\n", text) if part.strip()]


def stack_labels(records):
    return np.vstack([r["labels"] for r in records])
