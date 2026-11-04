#!/usr/bin/env python3
"""
03_train.py
Modular Training, Threshold Calibration, and Dev Checkpoint Pipeline
for SemEval-2027 Task 9 Subtask 2.

Features:
  - Trains CrossModalFramingModel on pre-extracted cached features
  - Supports Asymmetric Loss (ASL) and Binary Cross-Entropy (BCE)
  - AdamW optimizer with CosineAnnealingLR
  - Per-epoch Dev validation with optimal threshold calibration (tau*)
  - Best model checkpointing when Dev Micro-F1 improves
  - Post-training Test evaluation and official submission TSV generation
"""

import argparse
import os
import sys
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cached_dataset import CachedFramingDataset, collate_cached_batch
from src.losses.asymmetric_loss import AsymmetricLoss
from src.metrics import (
    evaluate_predictions,
    export_predictions_tsv,
    optimize_threshold,
)
from src.models.fusion_model import CrossModalFramingModel


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


def train_pipeline(
    feature_dir: str = "cache/features",
    split_file: str = "course_split.json",
    data_root: str = "data_official",
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    embed_dim: int = 512,
    num_heads: int = 8,
    loss_type: str = "asl",
    checkpoint_dir: str = "checkpoints",
    device: Optional[Any] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Executes modular training, calibration, and checkpoint management pipeline.

    Args:
        feature_dir: Directory containing cached .pt feature tensors.
        split_file: Path to course_split.json.
        data_root: Path to official data root.
        epochs: Number of training epochs.
        batch_size: Batch size for training and evaluation.
        lr: Learning rate for AdamW.
        weight_decay: Weight decay for AdamW.
        embed_dim: Latent projection dimension for cross-modal attention.
        num_heads: Number of attention heads.
        loss_type: 'asl' (default) or 'bce'.
        checkpoint_dir: Directory to save best checkpoint and predictions TSV.
        device: 'cuda', 'mps', 'cpu', or None for automatic accelerator detection.

    Returns:
        Dictionary with training and evaluation summary metrics:
          {"best_epoch": int, "dev_micro": float, "dev_macro": float, "best_threshold": float,
           "test_micro": float, "test_macro": float, "checkpoint_path": str, "predictions_tsv": str}
    """
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

    print(f"[*] Training pipeline running on device: {resolved_device}")

    # 2. Setup Directories
    checkpoint_dir = _resolve_path(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    predictions_tsv_path = os.path.join(checkpoint_dir, "predictions_test_subtask2.tsv")

    # 3. Load Datasets & DataLoaders
    print("[*] Loading cached datasets...")
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

    print(
        f"    Train samples: {len(train_dataset)} | "
        f"Dev samples: {len(dev_dataset)} | "
        f"Test samples: {len(test_dataset)}"
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

    # 4. Initialize Architecture
    model = CrossModalFramingModel(
        text_dim=768,
        vision_dim=768,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_classes=num_classes,
        dropout=0.1,
    ).to(resolved_device)

    # 5. Loss Function
    loss_type_lower = loss_type.lower()
    if loss_type_lower == "asl":
        criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
    elif loss_type_lower == "bce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(
            f"Unsupported loss_type '{loss_type}'. Expected 'asl' or 'bce'."
        )

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

    # 6. Epoch Training Loop & Dev Checkpointing
    best_dev_micro = -1.0
    best_dev_macro = 0.0
    best_threshold = 0.5
    best_epoch = 1

    print(f"[*] Starting training for {epochs} epochs (loss: {loss_type_lower})...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            v_tokens = batch["visual_tokens"].to(resolved_device)
            t_tokens = batch["text_tokens"].to(resolved_device)
            t_mask = batch["text_mask"].to(resolved_device)
            targets = batch["target"].to(resolved_device)

            optimizer.zero_grad()
            logits, _ = model(v_tokens, t_tokens, text_mask=t_mask)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_train_loss = total_train_loss / max(1, num_batches)

        # Dev Evaluation & Threshold Calibration
        model.eval()
        dev_targets_list = []
        dev_probs_list = []

        with torch.no_grad():
            for batch in dev_loader:
                v_tokens = batch["visual_tokens"].to(resolved_device)
                t_tokens = batch["text_tokens"].to(resolved_device)
                t_mask = batch["text_mask"].to(resolved_device)
                targets = batch["target"]

                logits, _ = model(v_tokens, t_tokens, text_mask=t_mask)
                probs = torch.sigmoid(logits).cpu()

                dev_targets_list.append(targets)
                dev_probs_list.append(probs)

        if dev_targets_list:
            dev_targets = torch.cat(dev_targets_list, dim=0)
            dev_probs = torch.cat(dev_probs_list, dim=0)
            tau_star, dev_micro, dev_macro = optimize_threshold(dev_targets, dev_probs)
        else:
            tau_star, dev_micro, dev_macro = 0.5, 0.0, 0.0

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Dev Micro-F1: {dev_micro:.4f} | "
            f"Dev Macro-F1: {dev_macro:.4f} | "
            f"tau*: {tau_star:.2f}"
        )

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
                    "classes": class_names,
                    "embed_dim": embed_dim,
                    "num_heads": num_heads,
                    "num_classes": num_classes,
                    "loss_type": loss_type,
                },
                checkpoint_path,
            )
            print(
                f"  --> [★ Best Model Saved] Dev Micro-F1 improved to {best_dev_micro:.4f} "
                f"(tau*={best_threshold:.2f}) -> {checkpoint_path}"
            )

    # 7. Post-Training Test Evaluation
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        calibrated_threshold = ckpt.get("best_threshold", best_threshold)
    else:
        calibrated_threshold = best_threshold

    model.eval()
    test_image_names = []
    test_probs_list = []
    test_labels_list = []

    with torch.no_grad():
        for batch in test_loader:
            v_tokens = batch["visual_tokens"].to(resolved_device)
            t_tokens = batch["text_tokens"].to(resolved_device)
            t_mask = batch["text_mask"].to(resolved_device)

            logits, _ = model(v_tokens, t_tokens, text_mask=t_mask)
            probs = torch.sigmoid(logits).cpu()

            test_probs_list.append(probs)
            test_image_names.extend(batch["image_name"])
            test_labels_list.extend(batch["labels"])

    if test_probs_list and len(test_image_names) > 0:
        all_test_probs = torch.cat(test_probs_list, dim=0)

        # Export official format predictions TSV
        export_predictions_tsv(
            image_names=test_image_names,
            probs=all_test_probs,
            threshold=calibrated_threshold,
            class_names=class_names,
            out_path=predictions_tsv_path,
        )

        # Generate gold test TSV for evaluation
        gold_tsv_path = os.path.join(checkpoint_dir, "gold_test_subtask2.tsv")
        with open(gold_tsv_path, "w", encoding="utf-8") as f:
            for img_name, lbls in zip(test_image_names, test_labels_list):
                if lbls:
                    f.write(f"{img_name}\t" + "\t".join(lbls) + "\n")
                else:
                    f.write(f"{img_name}\n")

        labels_file = getattr(
            test_dataset.base_dataset,
            "labels_file",
            os.path.join(data_root, "labels.txt"),
        )
        test_macro, test_micro = evaluate_predictions(
            pred_tsv=predictions_tsv_path,
            gold_tsv=gold_tsv_path,
            labels_file=labels_file,
        )
    else:
        test_micro = 0.0
        test_macro = 0.0

    print(
        f"\n[✓] Post-Training Test Evaluation:\n"
        f"    Best Epoch: {best_epoch}\n"
        f"    Dev Micro-F1: {best_dev_micro:.4f} | Dev Macro-F1: {best_dev_macro:.4f}\n"
        f"    Calibrated Threshold (tau*): {calibrated_threshold:.2f}\n"
        f"    Test Micro-F1: {test_micro:.4f} | Test Macro-F1: {test_macro:.4f}\n"
        f"    Checkpoint Path: {checkpoint_path}\n"
        f"    Predictions TSV: {predictions_tsv_path}\n"
        f"[✓] Training pipeline finished successfully."
    )

    return {
        "best_epoch": int(best_epoch),
        "dev_micro": float(best_dev_micro),
        "dev_macro": float(best_dev_macro),
        "best_threshold": float(calibrated_threshold),
        "test_micro": float(test_micro),
        "test_macro": float(test_macro),
        "checkpoint_path": checkpoint_path,
        "predictions_tsv": predictions_tsv_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="SemEval-2027 Task 9 Subtask 2 Modular Training Pipeline"
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="cache/features",
        help="Path to pre-extracted feature tensors (default: cache/features)",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default="course_split.json",
        help="Path to partition split json (default: course_split.json)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data_official",
        help="Path to data_official directory (default: data_official)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="AdamW learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="AdamW weight decay (default: 1e-2)",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=512,
        help="Latent projection dimension (default: 512)",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Number of attention heads (default: 8)",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="asl",
        choices=["asl", "bce"],
        help="Loss function: 'asl' or 'bce' (default: asl)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints and predictions (default: checkpoints)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Accelerator device ('cuda', 'mps', 'cpu', or None for auto)",
    )

    args = parser.parse_args()

    train_pipeline(
        feature_dir=args.feature_dir,
        split_file=args.split_file,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        loss_type=args.loss_type,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
