#!/usr/bin/env python3
"""
05_visualize_attention.py
Cross-modal attention heatmap and interpretability visualizer CLI script
for SemEval-2027 Task 9 Subtask 2.
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional
import numpy as np
import torch
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.explain import extract_attention_for_sample, visualize_sample_attention
from src.models import CrossModalFramingModel
from src.dataset import FramingDataset


def resolve_path(path: str) -> str:
    """Resolves path against current working directory or project root."""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return path
    candidate = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(candidate):
        return candidate
    return path


def find_image_file(image_name: str, data_root: str) -> Optional[str]:
    """Search for the image file across known data directories."""
    candidate_dirs = [
        os.path.join(data_root, "initial-training-data", "subtask2", "images"),
        os.path.join(data_root, "training-second-batch-oct", "EN", "raw-documents"),
    ]
    for d in candidate_dirs:
        resolved_d = resolve_path(d)
        if os.path.exists(resolved_d):
            full_path = os.path.join(resolved_d, image_name)
            if os.path.exists(full_path):
                return full_path
    return None


def run_dry_run(output_dir: str):
    """Executes a self-contained dry-run generating a synthetic attention figure."""
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "dry_run_attention.png")

    img = Image.new("RGB", (400, 400), color=(140, 150, 160))
    boxes = [
        [30, 40, 160, 200],
        [180, 50, 350, 220],
        [80, 230, 300, 370],
    ]
    attn_scores = np.array([0.35, 0.28, 0.22, 0.15])
    text_tokens = [
        "citizens",
        "gathered",
        "to",
        "protest",
        "police",
        "reforms",
        "outside",
        "city",
        "hall",
    ]
    token_scores = np.array([0.45, 0.20, 0.10, 0.85, 0.80, 0.55, 0.30, 0.75, 0.60])
    top_indices = [3, 4, 7, 8, 5]

    visualize_sample_attention(
        image=img,
        boxes=boxes,
        attn_scores=attn_scores,
        text_tokens=text_tokens,
        top_text_indices=top_indices,
        text_token_scores=token_scores,
        title="[Dry Run] Cross-Modal Attention Visualization",
        output_path=out_file,
    )

    print(f"[✓] Dry run completed successfully. Generated: {out_file}")


def main(args: Optional[argparse.Namespace] = None):
    if args is None:
        parser = argparse.ArgumentParser(
            description="Visualize Cross-Modal Attention Heatmaps and Keywords"
        )
        parser.add_argument(
            "--checkpoint",
            type=str,
            default="checkpoints/best_model.pt",
            help="Path to trained model checkpoint (default: checkpoints/best_model.pt)",
        )
        parser.add_argument(
            "--feature_dir",
            type=str,
            default="cache/features",
            help="Path to directory containing cached features (default: cache/features)",
        )
        parser.add_argument(
            "--crop_dir",
            type=str,
            default="cache/crops",
            help="Path to directory containing entity crops and metadata.json (default: cache/crops)",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default="reports/figures",
            help="Output directory to save attention figures (default: reports/figures)",
        )
        parser.add_argument(
            "--data_root",
            type=str,
            default="data_official",
            help="Path to official data directory (default: data_official)",
        )
        parser.add_argument(
            "--split",
            type=str,
            default="test",
            help="Split to visualize samples from (default: test)",
        )
        parser.add_argument(
            "--split_file",
            type=str,
            default="course_split.json",
            help="Split partitioning JSON file (default: course_split.json)",
        )
        parser.add_argument(
            "--num_samples",
            type=int,
            default=5,
            help="Number of samples to visualize (default: 5)",
        )
        parser.add_argument(
            "--dry_run",
            action="store_true",
            help="Run in dry-run mode using synthetic data",
        )
        parser.add_argument(
            "--device",
            type=str,
            default=None,
            help="Inference device ('mps', 'cuda', 'cpu', or None for auto)",
        )

        args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.dry_run:
        run_dry_run(output_dir)
        return

    # Select execution device
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Locate checkpoint
    ckpt_path = resolve_path(args.checkpoint)
    if not os.path.exists(ckpt_path):
        # Fallback to best_multimodal_model.pt
        alt_ckpt = resolve_path("checkpoints/best_multimodal_model.pt")
        if os.path.exists(alt_ckpt):
            ckpt_path = alt_ckpt
            print(f"[*] Using fallback checkpoint: {ckpt_path}")
        else:
            print(f"[!] Warning: Checkpoint not found at {args.checkpoint}. Initializing untrained model.")
            ckpt_path = None

    # Load model architecture and weights
    embed_dim = 512
    num_heads = 8
    num_classes = 14
    model = CrossModalFramingModel(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_classes=num_classes,
    )

    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
                state_dict = ckpt_data["model_state_dict"]
                embed_dim = ckpt_data.get("embed_dim", embed_dim)
                num_heads = ckpt_data.get("num_heads", num_heads)
                num_classes = ckpt_data.get("num_classes", num_classes)
                model = CrossModalFramingModel(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_classes=num_classes,
                )
                model.load_state_dict(state_dict)
            elif isinstance(ckpt_data, dict):
                model.load_state_dict(ckpt_data)
            print(f"[✓] Loaded model weights from: {ckpt_path}")
        except Exception as e:
            print(f"[!] Warning: Could not load state_dict directly ({e}). Initializing standard model.")
            model = CrossModalFramingModel(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_classes=num_classes,
            )

    model.to(device)
    model.eval()

    # Load crop metadata
    crop_dir = resolve_path(args.crop_dir)
    metadata_file = os.path.join(crop_dir, "metadata.json")
    metadata: Dict[str, dict] = {}
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            print(f"[✓] Loaded crop metadata for {len(metadata)} images from {metadata_file}")
        except Exception as e:
            print(f"[!] Warning loading metadata.json: {e}")

    # Discover candidate feature files
    feature_dir = resolve_path(args.feature_dir)
    feature_files = sorted(glob.glob(os.path.join(feature_dir, "*.pt")))

    if not feature_files:
        print(f"[!] No feature files found in {feature_dir}. Running dry-run instead.")
        run_dry_run(output_dir)
        return

    # Try mapping to split samples if available
    sample_text_map: Dict[str, str] = {}
    try:
        ds = FramingDataset(
            split=args.split,
            split_file=args.split_file,
            data_root=args.data_root,
        )
        for s in ds:
            sample_text_map[s["id"]] = s.get("text", "")
    except Exception:
        pass

    if sample_text_map:
        feature_files = [
            p for p in feature_files
            if os.path.splitext(os.path.basename(p))[0] in sample_text_map
        ]

    processed = 0
    for feat_path in feature_files:
        if processed >= args.num_samples:
            break

        try:
            data = torch.load(feat_path, map_location="cpu", weights_only=False)
            doc_id = data.get("id", os.path.splitext(os.path.basename(feat_path))[0])
            image_name = data.get("image_name", f"{doc_id}.jpg")
            visual_tokens = data["visual_tokens"]
            text_tokens = data["text_tokens"]
            text_mask = data.get("text_mask", None)

            # 1. Run attention extraction
            vis_attn_scores, text_token_scores = extract_attention_for_sample(
                model=model,
                visual_tokens=visual_tokens,
                text_tokens=text_tokens,
                text_mask=text_mask,
            )

            # 2. Locate source image
            img_path = find_image_file(image_name, args.data_root)
            if img_path and os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
            else:
                img = Image.new("RGB", (500, 500), color=(180, 180, 180))

            # 3. Locate bounding boxes
            base_img = os.path.splitext(image_name)[0]
            boxes = metadata.get(doc_id, {}).get("boxes")
            if not boxes:
                boxes = metadata.get(base_img, {}).get("boxes")
            if not boxes:
                # Fallback boxes
                w, h = img.size
                num_crops = max(1, len(vis_attn_scores) - 1)
                boxes = []
                for b_i in range(num_crops):
                    step_w = w // (num_crops + 1)
                    boxes.append([b_i * step_w, 20, (b_i + 1) * step_w, h // 2])

            # 4. Extract words
            text = sample_text_map.get(doc_id, "")
            if text:
                words = text.split()
            else:
                words = [f"word_{i}" for i in range(len(text_token_scores))]

            if len(words) < len(text_token_scores):
                words = words + [f"w_{i}" for i in range(len(words), len(text_token_scores))]
            else:
                words = words[: len(text_token_scores)]

            # 5. Top text indices
            top_indices = (
                np.argsort(text_token_scores)[::-1][: min(10, len(text_token_scores))].tolist()
            )

            # 6. Render and save visualization
            out_file = os.path.join(output_dir, f"{doc_id}_attention.png")
            visualize_sample_attention(
                image=img,
                boxes=boxes,
                attn_scores=vis_attn_scores,
                text_tokens=words,
                top_text_indices=top_indices,
                text_token_scores=text_token_scores,
                title=f"Cross-Modal Attention: {doc_id}",
                output_path=out_file,
            )
            print(f"[{processed+1}/{args.num_samples}] Generated figure: {out_file}")
            processed += 1

        except Exception as e:
            print(f"[!] Error processing {feat_path}: {e}", file=sys.stderr)

    print(f"[✓] Attention visualization complete. Generated {processed} figures in {output_dir}")


if __name__ == "__main__":
    main()
