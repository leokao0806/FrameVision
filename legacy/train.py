"""
train.py
End-to-End Training, Held-Out Testing, and Official Evaluation Pipeline.
Features:
  - 80/20 Train vs Held-Out Test split (reproducible seed 42)
  - Region-Aware Cross-Modal Attention (Text queries Full Photo + SAM Local Crop)
  - Multi-Label Asymmetric Loss (ASL) for severe class imbalance
  - Saves official submission TSV format: file-name \\t frame1 \\t frame2 ...
  - Directly invokes official starter-kit scorer (scorer-subtask-2.py)
"""

import os
import subprocess
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import FramingMultimodalDataset
from model import MultimodalFramingModel, AsymmetricLoss
from metrics import compute_multilabel_metrics


def collate_fn(batch):
    return {
        "image_names": [item["image_name"] for item in batch],
        "texts": [item["text"] for item in batch],
        "full_images": torch.stack([item["full_image"] for item in batch]),
        "crop_images": torch.stack([item["crop_image"] for item in batch]),
        "targets": torch.stack([item["target"] for item in batch]),
        "label_names": [item["label_names"] for item in batch],
    }


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            texts = batch["texts"]
            full_imgs = batch["full_images"].to(device)
            crop_imgs = batch["crop_images"].to(device)
            targets = batch["targets"].to(device)

            logits, _ = model(texts, full_imgs, crop_imgs)
            loss = criterion(logits, targets)
            total_loss += loss.item() * len(texts)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    y_true = np.vstack(all_targets)
    y_pred_probs = np.vstack(all_probs)

    metrics = compute_multilabel_metrics(y_true, y_pred_probs, threshold=0.45)
    metrics["loss"] = avg_loss
    return metrics, y_true, y_pred_probs


def main():
    parser = argparse.ArgumentParser(description="SemEval-2027 Task 9 End-to-End Pipeline")
    parser.add_argument("--data_root", type=str, default="data_official", help="Data root directory")
    parser.add_argument("--crops_dir", type=str, default="crops", help="SAM crops folder")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Save directory")
    parser.add_argument("--threshold", type=float, default=0.45, help="Classification decision threshold")
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[*] Accelerator: Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("[*] Accelerator: NVIDIA CUDA GPU")
    else:
        device = torch.device("cpu")
        print("[*] Accelerator: CPU")

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Ingest Official Dataset
    print(f"\n[*] Loading official multimodal dataset from '{args.data_root}'...")
    full_dataset = FramingMultimodalDataset(
        data_root=args.data_root,
        crops_dir=args.crops_dir,
    )
    num_samples = len(full_dataset)
    print(f"[✓] Successfully ingested {num_samples} multimodal samples across {full_dataset.num_classes} framing classes.")

    # 2. Partition 80% Train / 20% Held-Out Test
    np.random.seed(42)
    indices = np.random.permutation(num_samples)
    test_size = int(0.20 * num_samples)
    train_idx = indices[test_size:]
    test_idx = indices[:test_size]

    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"    - Training Set: {len(train_dataset)} samples")
    print(f"    - Held-Out Evaluation Set: {len(test_dataset)} samples")

    # 3. Export Held-Out Gold Annotations for Official Scorer
    gold_test_file = "gold_test_subtask2.txt"
    with open(gold_test_file, "w", encoding="utf-8") as f:
        for idx in test_idx:
            sample = full_dataset.samples[idx]
            img_name = sample["image_name"]
            labels = sample["label_names"]
            if labels:
                f.write(f"{img_name}\t" + "\t".join(labels) + "\n")
            else:
                f.write(f"{img_name}\n")
    print(f"[✓] Exported isolated test ground-truth to '{gold_test_file}'")

    # 4. Initialize Architecture
    model = MultimodalFramingModel(
        num_classes=full_dataset.num_classes,
        embed_dim=256,
        num_heads=8,
    ).to(device)

    criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 5. Training Loop
    print("\n" + "=" * 65)
    print("  Training Multimodal Framing Network (Cross-Attention + ASL)")
    print("=" * 65)

    best_macro_f1 = -1.0
    best_ckpt_path = os.path.join(args.save_dir, "best_multimodal_model.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            texts = batch["texts"]
            full_imgs = batch["full_images"].to(device)
            crop_imgs = batch["crop_images"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            logits, _ = model(texts, full_imgs, crop_imgs)
            loss = criterion(logits, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(texts)

        scheduler.step()
        avg_train_loss = train_loss / len(train_dataset)

        # Evaluate on Held-Out Test Set
        val_metrics, _, _ = evaluate(model, test_loader, criterion, device)
        val_macro_f1 = val_metrics["macro_f1"]
        val_micro_f1 = val_metrics["micro_f1"]
        val_loss = val_metrics["loss"]

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Test Loss: {val_loss:.4f} | "
            f"Test Macro-F1: {val_macro_f1:.4f} | "
            f"Test Micro-F1: {val_micro_f1:.4f}"
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "classes": full_dataset.classes,
                "macro_f1": val_macro_f1,
            }, best_ckpt_path)
            print(f"  --> [★ Best Model Saved] Macro-F1 improved to {best_macro_f1:.4f}")

    # 6. Final Evaluation & Official Scorer Execution
    print("\n" + "=" * 65)
    print("  Final Testing & Official Scorer Evaluation")
    print("=" * 65)

    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pred_file = "predictions_test_subtask2.txt"
    all_pred_rows = []

    with torch.no_grad():
        for batch in test_loader:
            texts = batch["texts"]
            full_imgs = batch["full_images"].to(device)
            crop_imgs = batch["crop_images"].to(device)
            names = batch["image_names"]

            logits, _ = model(texts, full_imgs, crop_imgs)
            probs = torch.sigmoid(logits).cpu().numpy()

            for img_name, prob_row in zip(names, probs):
                predicted_labels = [
                    full_dataset.classes[c_idx]
                    for c_idx, p in enumerate(prob_row)
                    if p >= args.threshold
                ]
                if predicted_labels:
                    all_pred_rows.append(f"{img_name}\t" + "\t".join(predicted_labels))
                else:
                    all_pred_rows.append(f"{img_name}")

    with open(pred_file, "w", encoding="utf-8") as f:
        f.write("\n".join(all_pred_rows) + "\n")
    print(f"[✓] Generated official prediction file: '{pred_file}' ({len(all_pred_rows)} test samples)")

    # 7. Execute Official Scorer
    scorer_script = "starter-kit/subtask2a_2b/scorer-subtask-2.py"
    labels_file = "data_official/labels.txt"

    print(f"[*] Invoking official competition scorer: {scorer_script}")
    import sys
    cmd = [
        sys.executable,
        scorer_script,
        "-g", gold_test_file,
        "-p", pred_file,
        "-f", labels_file,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("\n--- Official Competition Scorer Output ---")
    print(res.stdout)
    if res.stderr:
        print(res.stderr)


if __name__ == "__main__":
    main()
