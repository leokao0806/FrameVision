"""
metrics.py
Official 15-class evaluation harness for SemEval-2027 Task 9 Subtask 2.
Implements:
- 15-class evaluation with explicit "None" frame handling
- Micro-F1 and Macro-F1 computation (zero_division=1)
- Optimal decision threshold calibration via grid search
- Official TSV scorer evaluation logic and submission export
"""

import os
from typing import Dict, Iterable, List, Optional, Tuple, Union, Any
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

NONE_LABEL = "None"

DEFAULT_CLASSES: List[str] = [
    "Economic",
    "Capacity_and_resources",
    "Morality",
    "Fairness_and_equality",
    "Legality_Constitutionality_and_jurisprudence",
    "Policy_prescription_and_evaluation",
    "Crime_and_punishment",
    "Security_and_defense",
    "Health_and_safety",
    "Quality_of_life",
    "Cultural_identity",
    "Public_opinion",
    "Political",
    "External_regulation_and_reputation",
]


def _to_numpy(arr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Helper to convert torch.Tensor or array-like to np.ndarray."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def compute_metrics(
    targets: Union[np.ndarray, torch.Tensor],
    probs: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Evaluates multi-label framing predictions using the official 15-class scheme.

    An explicit 'None' class is activated whenever no frame exceeds threshold in
    predictions, or when all frame labels are 0 in gold targets.

    Args:
        targets: Ground-truth binary indicators of shape (N, C), where C is typically 14.
        probs: Predicted probabilities of shape (N, C).
        threshold: Decision threshold for positive frame classification.
        class_names: Optional list of class names (defaults to DEFAULT_CLASSES).

    Returns:
        Dictionary containing 'micro_f1' and 'macro_f1' (zero_division=1).
    """
    targets_np = _to_numpy(targets)
    probs_np = _to_numpy(probs)

    if len(targets_np) == 0:
        return {"micro_f1": 0.0, "macro_f1": 0.0}

    num_classes = len(class_names) if class_names is not None else len(DEFAULT_CLASSES)

    # Binary predictions for the frame classes
    pred_bin = (probs_np >= threshold).astype(int)
    targets_bin = (targets_np >= 0.5).astype(int)

    # If already 15 columns, use directly; otherwise append None class column
    if pred_bin.shape[1] == num_classes:
        pred_none = (pred_bin.sum(axis=1) == 0).astype(int)
        pred_15 = np.column_stack([pred_bin, pred_none])
    else:
        pred_15 = pred_bin

    if targets_bin.shape[1] == num_classes:
        target_none = (targets_bin.sum(axis=1) == 0).astype(int)
        targets_15 = np.column_stack([targets_bin, target_none])
    else:
        targets_15 = targets_bin

    micro_f1 = float(f1_score(targets_15, pred_15, average="micro", zero_division=1))
    macro_f1 = float(f1_score(targets_15, pred_15, average="macro", zero_division=1))

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }


def optimize_threshold(
    targets: Union[np.ndarray, torch.Tensor],
    probs: Union[np.ndarray, torch.Tensor],
    thresholds: Optional[Iterable[float]] = None,
) -> Tuple[float, float, float]:
    """
    Grid search over decision thresholds to maximize validation Micro-F1.

    Args:
        targets: Ground-truth binary matrix of shape (N, C).
        probs: Predicted probability matrix of shape (N, C).
        thresholds: Iterable of candidate thresholds (default: 0.10 to 0.88 with step 0.02).

    Returns:
        Tuple of (best_threshold, best_micro_f1, corresponding_macro_f1).
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.90, 0.02)

    best_th = 0.5
    best_micro = -1.0
    best_macro = -1.0

    for th in thresholds:
        th_val = float(th)
        m = compute_metrics(targets, probs, threshold=th_val)
        if m["micro_f1"] > best_micro:
            best_micro = m["micro_f1"]
            best_macro = m["macro_f1"]
            best_th = th_val

    return best_th, best_micro, best_macro


def _read_tsv_labels(file_path: str) -> List[Dict[str, Any]]:
    """Reads TSV predictions or annotations into list of dicts."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if not parts[0]:
                raise ValueError(f"Line missing file name: {line}")
            file_name = parts[0].strip()
            frames = [p.strip() for p in parts[1:] if p.strip()]
            records.append({"file_name": file_name, "frames": frames})
    return records


def _dedupe_keeping_last(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicates rows keeping the last occurrence per image file name."""
    deduped = {}
    for entry in records:
        deduped[entry["file_name"]] = entry
    return list(deduped.values())


def evaluate_predictions(
    pred_tsv: str,
    gold_tsv: str,
    labels_file: str = "data_official/labels.txt",
) -> Tuple[float, float]:
    """
    Evaluates predicted TSV against gold TSV matching official SemEval scorer logic.

    Args:
        pred_tsv: Path to prediction TSV.
        gold_tsv: Path to gold annotations TSV.
        labels_file: Path to labels.txt containing class names.

    Returns:
        Tuple of (macro_f1, micro_f1).
    """
    # Load class list
    if os.path.exists(labels_file):
        with open(labels_file, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
    else:
        # Check relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate = os.path.join(project_root, labels_file)
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                classes = [line.strip() for line in f if line.strip()]
        else:
            classes = list(DEFAULT_CLASSES)

    pred_records = _dedupe_keeping_last(_read_tsv_labels(pred_tsv))
    gold_records = _dedupe_keeping_last(_read_tsv_labels(gold_tsv))

    pred_by_id = {entry["file_name"]: entry["frames"] for entry in pred_records}

    pred_values = []
    gold_values = []
    for entry in gold_records:
        gold_values.append(entry["frames"] if entry["frames"] else [NONE_LABEL])
        predicted_frames = pred_by_id.get(entry["file_name"], [])
        pred_values.append(predicted_frames if predicted_frames else [NONE_LABEL])

    mlb = MultiLabelBinarizer()
    mlb.fit([classes + [NONE_LABEL]])
    gold_mat = mlb.transform(gold_values)
    pred_mat = mlb.transform(pred_values)

    macro_f1 = float(f1_score(gold_mat, pred_mat, average="macro", zero_division=1))
    micro_f1 = float(f1_score(gold_mat, pred_mat, average="micro", zero_division=1))

    return macro_f1, micro_f1


def export_predictions_tsv(
    image_names: List[str],
    probs: Union[np.ndarray, torch.Tensor],
    threshold: float,
    class_names: List[str],
    out_path: str,
) -> None:
    """
    Exports probability predictions into official submission TSV format.
    Format: file_name \\t frame1 \\t frame2 ... (or just file_name if no frame >= threshold)

    Args:
        image_names: List of image file names.
        probs: Array or tensor of shape (N, C) containing predicted probabilities.
        threshold: Classification threshold.
        class_names: Names corresponding to columns of probs.
        out_path: Destination path for output TSV.
    """
    probs_np = _to_numpy(probs)
    if len(image_names) != len(probs_np):
        raise ValueError(
            f"Length mismatch: {len(image_names)} image names vs {len(probs_np)} probability rows."
        )

    parent_dir = os.path.dirname(os.path.abspath(out_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for img_name, prob_vec in zip(image_names, probs_np):
            active_classes = [
                cls_name
                for cls_name, p in zip(class_names, prob_vec)
                if p >= threshold
            ]
            if active_classes:
                line = f"{img_name}\t" + "\t".join(active_classes)
            else:
                line = f"{img_name}"
            f.write(f"{line}\n")
