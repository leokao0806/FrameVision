#!/usr/bin/env python3
"""
04_run_ablations.py
Automated Ablation Benchmark Suite & Markdown Report Generator
for SemEval-2027 Task 9 Subtask 2.

Runs 7 ablation experiments on the unified Grounding-DINO + SigLIP pipeline, each answering
one question, across multiple seeds (mean +/- std):
  - Exp 1: Text-Only (SigLIP text + ASL)
  - Exp 2: Vision-Only Global (SigLIP global + ASL)
  - Exp 3: Vision-Only Region (SigLIP global + Grounding-DINO regions + ASL)
  - Exp 4: Global + Text Cross-Attention, no regions (ASL)
  - Exp 5: Late Fusion (global image + text probability average, ASL)
  - Exp 6: Full Cross-Modal (Grounding-DINO regions + text, BCE)
  - Exp 7: Full Cross-Modal (Grounding-DINO regions + text, ASL)

Per-seed predictions are exported as predictions/subtask2/<slug>/seed<k>/{dev,test}.{tsv,probs.json}
and a Markdown comparison table (with Qwen2.5-VL reference rows) is written to reports/.
"""

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import f1_score, hamming_loss
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Register self in sys.modules
sys.modules["scripts.04_run_ablations"] = sys.modules[__name__]

from src.cached_dataset import CachedFramingDataset, collate_cached_batch
from src.losses.asymmetric_loss import AsymmetricLoss
from src.metrics import (
    DEFAULT_CLASSES,
    evaluate_predictions,
    export_predictions_tsv,
    optimize_threshold,
)
from src.models.baselines import (
    LateFusionModel,
    TextOnlyModel,
    VisionGlobalModel,
    VisionRegionModel,
)
from src.models.fusion_model import CrossModalFramingModel

INTERMEDIATE_REPORT_BASELINES: List[Dict[str, Any]] = [
    {
        "name": "Image LR (SigLIP)",
        "micro_f1": 0.353,
        "macro_f1": 0.281,
        "hamming_loss": None,
        "notes": "Intermediate Report baseline",
    },
    {
        "name": "Late fusion (Image + Text)",
        "micro_f1": 0.367,
        "macro_f1": 0.350,
        "hamming_loss": None,
        "notes": "Intermediate Report baseline",
    },
    {
        "name": "Qwen2.5-VL (Image only)",
        "micro_f1": 0.417,
        "macro_f1": 0.205,
        "hamming_loss": None,
        "notes": "Intermediate Report zero-shot",
    },
    {
        "name": "Qwen2.5-VL (Image + Text)",
        "micro_f1": 0.333,
        "macro_f1": 0.229,
        "hamming_loss": None,
        "notes": "Intermediate Report zero-shot",
    },
]


def _resolve_path(path: str) -> str:
    """Resolve path against current working directory or project root."""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return path
    candidate = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(candidate):
        return candidate
    return path


def _fmt_metric(val: Any, precision: int = 3) -> str:
    """Formats float metric or returns '-' for None/missing values."""
    if val is None or val == "" or val == "-" or str(val).upper() == "N/A":
        return "-"
    try:
        f_val = float(val)
        return f"{f_val:.{precision}f}"
    except (ValueError, TypeError):
        return str(val)


def format_ablation_markdown_table(
    results: List[Dict[str, Any]],
    baselines: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Returns GitHub Flavored Markdown table with columns:
      | Model / Setting | Micro F1 (Primary) | Macro F1 | Hamming Loss | Notes / Status |

    Displays teammate's Intermediate Report Table 1 baselines as reference:
      - Image LR (SigLIP): Micro 0.353, Macro 0.281
      - Late fusion (Image + Text): Micro 0.367, Macro 0.350
      - Qwen2.5-VL (Image only): Micro 0.417, Macro 0.205
      - Qwen2.5-VL (Image + Text): Micro 0.333, Macro 0.229

    Args:
        results: List of result dictionaries for ablation experiments.
        baselines: Optional list of baseline dictionaries (defaults to Intermediate Report baselines).

    Returns:
        Formatted markdown table string.
    """
    ref_baselines = baselines if baselines is not None else INTERMEDIATE_REPORT_BASELINES

    lines = [
        "| Model / Setting | Micro F1 (Primary) | Macro F1 | Hamming Loss | Notes / Status |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    # Add reference baselines
    for b in ref_baselines:
        name = b.get("name") or b.get("model") or "Baseline"
        micro = _fmt_metric(b.get("micro_f1") if "micro_f1" in b else b.get("test_micro", b.get("micro")))
        macro = _fmt_metric(b.get("macro_f1") if "macro_f1" in b else b.get("test_macro", b.get("macro")))
        hamming = _fmt_metric(b.get("hamming_loss") if "hamming_loss" in b else b.get("hamming"))
        notes = b.get("notes") or b.get("status") or "-"
        lines.append(f"| {name} | {micro} | {macro} | {hamming} | {notes} |")

    # Add ablation experiment rows
    for r in results:
        name = r.get("name") or r.get("model_type") or r.get("setting") or "Experiment"
        micro = _fmt_metric(r.get("micro_f1") if "micro_f1" in r else r.get("test_micro", r.get("micro")))
        macro = _fmt_metric(r.get("macro_f1") if "macro_f1" in r else r.get("test_macro", r.get("macro")))
        if r.get("micro_f1_std") is not None:
            micro = f"{micro} ± {_fmt_metric(r['micro_f1_std'])}"
        if r.get("macro_f1_std") is not None:
            macro = f"{macro} ± {_fmt_metric(r['macro_f1_std'])}"
        hamming = _fmt_metric(r.get("hamming_loss") if "hamming_loss" in r else r.get("hamming"))
        notes = r.get("notes") or r.get("status") or "Completed"
        lines.append(f"| {name} | {micro} | {macro} | {hamming} | {notes} |")

    return "\n".join(lines)


EXPERIMENTS: List[Dict[str, str]] = [
    {"model_type": "text_only", "loss_type": "asl", "slug": "sig_e1_text_only",
     "name": "Exp 1: Text-Only (SigLIP text + ASL)"},
    {"model_type": "vision_global", "loss_type": "asl", "slug": "sig_e2_vision_global",
     "name": "Exp 2: Vision-Only Global (SigLIP + ASL)"},
    {"model_type": "vision_region", "loss_type": "asl", "slug": "sig_e3_vision_region_gdino",
     "name": "Exp 3: Vision-Only Global + Grounding-DINO Regions (ASL)"},
    {"model_type": "fusion_global", "loss_type": "asl", "slug": "sig_e4_fusion_global_asl",
     "name": "Exp 4: Global + Text Cross-Attention, no regions (ASL)"},
    {"model_type": "late_fusion", "loss_type": "asl", "slug": "sig_e5_late_fusion_asl",
     "name": "Exp 5: Late Fusion (global image + text, ASL)"},
    {"model_type": "fusion_bce", "loss_type": "bce", "slug": "sig_e6_fusion_gdino_bce",
     "name": "Exp 6: Full Cross-Modal, Grounding-DINO (BCE)"},
    {"model_type": "fusion_asl", "loss_type": "asl", "slug": "sig_e7_fusion_gdino_asl",
     "name": "Exp 7: Full Cross-Modal, Grounding-DINO (ASL)"},
]
DEFAULT_SEEDS: List[int] = [13, 42, 2026]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _write_probs_json(path: str, image_names: List[str], classes: List[str], probs: Any, threshold: float) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "image_names": list(image_names),
                "classes": list(classes),
                "probs": np.asarray(probs, dtype=float).tolist(),
                "threshold": float(threshold),
            },
            f,
        )


def aggregate_seed_results(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Averages per-seed results; micro_f1/macro_f1/hamming_loss hold the means, *_std the sample std."""
    def _mean_std(key: str):
        vals = np.array([float(r[key]) for r in per_seed], dtype=float)
        return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

    out = dict(per_seed[0])
    for key, prefix in (("micro_f1", "micro_f1"), ("macro_f1", "macro_f1"), ("hamming_loss", "hamming_loss"),
                        ("dev_micro", "dev_micro")):
        m, sd = _mean_std(key)
        out[key] = m
        out[f"{prefix}_mean"] = m
        out[f"{prefix}_std"] = sd
    out["hamming"] = out["hamming_loss"]
    out["test_micro"] = out["micro_f1"]
    out["test_macro"] = out["macro_f1"]
    out["seeds"] = [r.get("seed") for r in per_seed]
    out["notes"] = f"mean ± std over {len(per_seed)} seeds"
    return out


def train_unimodal_or_ablation(
    model_type: str,
    loss_type: str = "asl",
    feature_dir: str = "cache/features",
    split_file: str = "course_split.json",
    data_root: str = "data_official",
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    embed_dim: int = 512,
    num_heads: int = 8,
    dropout: float = 0.1,
    checkpoint_dir: str = "checkpoints/ablations",
    device: Optional[Any] = None,
    seed: int = 42,
    predictions_root: Optional[str] = None,
    exp_slug: Optional[str] = None,
    exp_name: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Trains and evaluates a specific unimodal baseline or cross-modal ablation variant.

    Args:
        model_type: One of 'text_only', 'vision_global', 'vision_region', 'fusion_global', 'late_fusion', 'fusion_bce', 'fusion_asl'.
        seed: Random seed (model init, shuffling).
        predictions_root: If set, exports predictions/<exp_slug>/seed<seed>/{dev,test}.{tsv,probs.json} there.
        loss_type: 'asl' or 'bce'. (For 'fusion_bce', automatically defaults to 'bce').
        feature_dir: Directory containing cached .pt features.
        split_file: Path to course_split.json.
        data_root: Path to official data root.
        epochs: Number of training epochs.
        batch_size: Batch size for training and evaluation.
        lr: Learning rate for AdamW.
        weight_decay: Weight decay.
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads (where applicable).
        dropout: Dropout rate.
        checkpoint_dir: Directory to save model checkpoint and predictions.
        device: Target execution device.

    Returns:
        Result dictionary containing micro_f1, macro_f1, hamming_loss, and metadata.
    """
    # Normalize model_type and loss_type
    mt = model_type.lower().strip()
    if mt == "fusion_bce" and loss_type == "asl":
        effective_loss = "bce"
    else:
        effective_loss = loss_type.lower().strip()

    # Determine human-friendly name
    display_names = {e["model_type"]: e["name"] for e in EXPERIMENTS}
    if exp_name is None:
        exp_name = display_names.get(mt, f"Ablation ({mt})")
    _set_seed(seed)

    # 1. Resolve Accelerator Device
    if device is None:
        if torch.cuda.is_available():
            resolved_device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            resolved_device = torch.device("mps")
        else:
            resolved_device = torch.device("cpu")
    elif isinstance(device, str):
        resolved_device = torch.device(device)
    else:
        resolved_device = device

    # 2. Setup Checkpoint Directory
    checkpoint_dir = _resolve_path(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"best_{mt}_seed{seed}.pt")
    predictions_tsv_path = os.path.join(checkpoint_dir, f"predictions_{mt}_seed{seed}.tsv")

    # 3. Load Cached Datasets & Loaders
    train_dataset = CachedFramingDataset(
        feature_dir=feature_dir,
        split="train",
        split_file=split_file,
        data_root=data_root,
    )
    dev_dataset = CachedFramingDataset(
        feature_dir=feature_dir,
        split="dev",
        split_file=split_file,
        data_root=data_root,
    )
    test_dataset = CachedFramingDataset(
        feature_dir=feature_dir,
        split="test",
        split_file=split_file,
        data_root=data_root,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_cached_batch,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_cached_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_cached_batch,
    )

    num_classes = train_dataset.num_classes
    class_names = train_dataset.classes

    # 4. Instantiate Model Architecture
    if mt == "text_only":
        model = TextOnlyModel(
            text_dim=768,
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout,
        ).to(resolved_device)
    elif mt == "vision_global":
        model = VisionGlobalModel(
            vision_dim=768,
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout,
        ).to(resolved_device)
    elif mt == "vision_region":
        model = VisionRegionModel(
            vision_dim=768,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_classes=num_classes,
            dropout=dropout,
        ).to(resolved_device)
    elif mt in ("fusion_bce", "fusion_asl", "fusion_global", "cross_modal"):
        model = CrossModalFramingModel(
            text_dim=768,
            vision_dim=768,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_classes=num_classes,
            dropout=dropout,
        ).to(resolved_device)
    elif mt == "late_fusion":
        model = LateFusionModel(
            text_dim=768,
            vision_dim=768,
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout,
        ).to(resolved_device)
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'")

    # 5. Loss Function
    if effective_loss == "asl":
        criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
    elif effective_loss == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported loss_type '{effective_loss}'. Expected 'asl' or 'bce'.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=1e-6,
    )

    def _forward(batch_dict: Dict[str, Any]) -> torch.Tensor:
        v_tokens = batch_dict["visual_tokens"].to(resolved_device)
        t_tokens = batch_dict["text_tokens"].to(resolved_device)
        t_mask = batch_dict["text_mask"].to(resolved_device)

        if mt == "text_only":
            return model(t_tokens, text_mask=t_mask)
        elif mt in ("vision_global", "vision_region"):
            return model(v_tokens)
        elif mt == "fusion_global":
            logits, _ = model(v_tokens[:, :1, :], t_tokens, text_mask=t_mask)
            return logits
        elif mt in ("fusion_bce", "fusion_asl", "cross_modal"):
            logits, _ = model(v_tokens, t_tokens, text_mask=t_mask)
            return logits
        elif mt == "late_fusion":
            probs = model(v_tokens, t_tokens, text_mask=t_mask)
            return torch.logit(probs.clamp(1e-6, 1.0 - 1e-6))
        else:
            raise ValueError(f"Unknown model_type {mt}")

    # 6. Epoch Training & Dev Checkpointing
    best_dev_micro = -1.0
    best_dev_macro = 0.0
    best_threshold = 0.5
    best_epoch = 1

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            targets = batch["target"].to(resolved_device)
            optimizer.zero_grad()
            logits = _forward(batch)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Dev Evaluation
        model.eval()
        dev_targets_list = []
        dev_probs_list = []

        with torch.no_grad():
            for batch in dev_loader:
                logits = _forward(batch)
                probs = torch.sigmoid(logits).cpu()
                dev_targets_list.append(batch["target"])
                dev_probs_list.append(probs)

        if dev_targets_list:
            dev_targets = torch.cat(dev_targets_list, dim=0)
            dev_probs = torch.cat(dev_probs_list, dim=0)
            tau_star, dev_micro, dev_macro = optimize_threshold(dev_targets, dev_probs)
        else:
            tau_star, dev_micro, dev_macro = 0.5, 0.0, 0.0

        if dev_micro > best_dev_micro:
            best_dev_micro = dev_micro
            best_dev_macro = dev_macro
            best_threshold = tau_star
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_threshold": float(best_threshold),
                    "dev_micro": float(best_dev_micro),
                    "dev_macro": float(best_dev_macro),
                    "model_type": mt,
                    "classes": class_names,
                },
                checkpoint_path,
            )

    # 7. Post-Training Test Evaluation
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        calibrated_threshold = ckpt.get("best_threshold", best_threshold)
    else:
        calibrated_threshold = best_threshold

    model.eval()

    def _collect(loader):
        names, plist = [], []
        with torch.no_grad():
            for batch in loader:
                plist.append(torch.sigmoid(_forward(batch)).cpu())
                names.extend(batch["image_name"])
        return names, (torch.cat(plist, dim=0) if plist else torch.zeros(0, num_classes))

    pred_dir = None
    if predictions_root is not None and exp_slug is not None:
        pred_dir = os.path.join(_resolve_path(predictions_root), exp_slug, f"seed{seed}")
        dev_names, dev_all_probs = _collect(dev_loader)
        export_predictions_tsv(dev_names, dev_all_probs, calibrated_threshold, class_names,
                               os.path.join(pred_dir, "dev.tsv"))
        _write_probs_json(os.path.join(pred_dir, "dev.probs.json"), dev_names, class_names,
                          dev_all_probs.numpy(), calibrated_threshold)

    test_image_names = []
    test_probs_list = []
    test_targets_list = []

    with torch.no_grad():
        for batch in test_loader:
            logits = _forward(batch)
            probs = torch.sigmoid(logits).cpu()
            test_probs_list.append(probs)
            test_targets_list.append(batch["target"])
            test_image_names.extend(batch["image_name"])

    if test_probs_list and len(test_image_names) > 0:
        all_test_probs = torch.cat(test_probs_list, dim=0).numpy()
        all_test_targets = torch.cat(test_targets_list, dim=0).numpy()

        export_predictions_tsv(
            image_names=test_image_names,
            probs=torch.from_numpy(all_test_probs),
            threshold=calibrated_threshold,
            class_names=class_names,
            out_path=predictions_tsv_path,
        )
        if pred_dir is not None:
            export_predictions_tsv(test_image_names, torch.from_numpy(all_test_probs), calibrated_threshold,
                                   class_names, os.path.join(pred_dir, "test.tsv"))
            _write_probs_json(os.path.join(pred_dir, "test.probs.json"), test_image_names, class_names,
                              all_test_probs, calibrated_threshold)

        pred_bin = (all_test_probs >= calibrated_threshold).astype(int)
        target_bin = (all_test_targets >= 0.5).astype(int)

        pred_none = (pred_bin.sum(axis=1) == 0).astype(int)
        target_none = (target_bin.sum(axis=1) == 0).astype(int)

        pred_15 = np.column_stack([pred_bin, pred_none])
        targets_15 = np.column_stack([target_bin, target_none])

        test_micro = float(f1_score(targets_15, pred_15, average="micro", zero_division=1))
        test_macro = float(f1_score(targets_15, pred_15, average="macro", zero_division=1))
        test_hamming = float(hamming_loss(targets_15, pred_15))
    else:
        test_micro = 0.0
        test_macro = 0.0
        test_hamming = 0.0

    return {
        "model_type": mt,
        "seed": seed,
        "name": exp_name,
        "loss_type": effective_loss,
        "best_epoch": best_epoch,
        "best_threshold": float(best_threshold),
        "dev_micro": float(best_dev_micro),
        "dev_macro": float(best_dev_macro),
        "micro_f1": float(test_micro),
        "macro_f1": float(test_macro),
        "test_micro": float(test_micro),
        "test_macro": float(test_macro),
        "hamming_loss": float(test_hamming),
        "hamming": float(test_hamming),
        "checkpoint_path": checkpoint_path,
        "predictions_tsv": predictions_tsv_path,
        "notes": "Completed",
        "status": "PASS",
    }


def run_ablation_suite(
    feature_dir: str = "cache/features_gdino_siglip",
    split_file: str = "course_split.json",
    data_root: str = "data_official",
    report_dir: str = "reports",
    checkpoint_dir: str = "checkpoints/ablations_gdino_siglip",
    predictions_root: str = "predictions/subtask2",
    report_name: str = "ablation_results_gdino_siglip.md",
    seeds: Optional[List[int]] = None,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-4,
    embed_dim: int = 512,
    num_heads: int = 8,
    device: Optional[Any] = None,
    dry_run: bool = False,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Runs the 7 ablation experiments over `seeds` (default [13, 42, 2026]) and writes a
    Markdown report with mean +/- std plus per-seed prediction exports.

    Returns:
        One aggregated result dict per experiment (micro_f1_mean/_std, macro_f1_mean/_std, ...).
    """
    seeds = list(seeds) if seeds else list(DEFAULT_SEEDS)
    report_dir = _resolve_path(report_dir)
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, report_name)

    results: List[Dict[str, Any]] = []
    if dry_run:
        print("[*] Running ablation suite in dry_run mode...")
        for i, exp in enumerate(EXPERIMENTS):
            base = 0.30 + 0.02 * i
            per_seed = [
                {
                    **exp, "name": exp["name"], "seed": sd, "best_epoch": 1, "best_threshold": 0.5,
                    "dev_micro": base - 0.01, "dev_macro": base - 0.05,
                    "micro_f1": base + 0.002 * j, "macro_f1": base - 0.04 + 0.002 * j,
                    "hamming_loss": 0.12 - 0.004 * i, "status": "PASS",
                }
                for j, sd in enumerate(seeds)
            ]
            agg = aggregate_seed_results(per_seed)
            agg["notes"] = "dry run; " + agg["notes"]
            results.append(agg)
    else:
        for exp in EXPERIMENTS:
            per_seed = []
            for sd in seeds:
                print(f"\n=======================================================")
                print(f"[*] {exp['name']} | seed {sd}")
                print(f"=======================================================")
                per_seed.append(
                    train_unimodal_or_ablation(
                        model_type=exp["model_type"],
                        loss_type=exp["loss_type"],
                        feature_dir=feature_dir,
                        split_file=split_file,
                        data_root=data_root,
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=lr,
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        checkpoint_dir=os.path.join(checkpoint_dir, exp["slug"]),
                        device=device,
                        seed=sd,
                        predictions_root=predictions_root,
                        exp_slug=exp["slug"],
                        exp_name=exp["name"],
                        **kwargs,
                    )
                )
            results.append(aggregate_seed_results(per_seed))

    md_table = format_ablation_markdown_table(results)
    report_content = f"""# SemEval-2027 Task 9 Subtask 2 - Grounding-DINO + SigLIP Ablations

Seven experiments, each answering one question; scores are mean ± std over seeds {seeds}.
Qwen2.5-VL rows are En-Shuo's zero-shot baselines for reference.

{md_table}
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n[✓] Ablation Suite Completed! Benchmark report written to:\n    {report_path}\n")
    print(md_table)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="SemEval-2027 Task 9 Subtask 2 Ablation Runner & Report Generator"
    )
    parser.add_argument("--feature_dir", type=str, default="cache/features_gdino_siglip", help="Path to cached .pt features")
    parser.add_argument("--split_file", type=str, default="course_split.json", help="Path to course_split.json")
    parser.add_argument("--data_root", type=str, default="data_official", help="Path to official data root")
    parser.add_argument("--report_dir", type=str, default="reports", help="Directory to save ablation_results.md")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/ablations_gdino_siglip", help="Directory for checkpoints")
    parser.add_argument("--predictions_root", type=str, default="predictions/subtask2", help="Root for per-seed prediction exports")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Random seeds (default: 13 42 2026)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs per ablation")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--embed_dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda, mps)")
    parser.add_argument("--dry_run", action="store_true", help="Generate mock report without full training")
    args = parser.parse_args()

    run_ablation_suite(
        feature_dir=args.feature_dir,
        split_file=args.split_file,
        data_root=args.data_root,
        report_dir=args.report_dir,
        checkpoint_dir=args.checkpoint_dir,
        predictions_root=args.predictions_root,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        device=args.device,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
