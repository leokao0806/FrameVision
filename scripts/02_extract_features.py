#!/usr/bin/env python3
"""
02_extract_features.py
Batch feature extraction CLI script for SemEval-2027 Task 9 Subtask 2.
Extracts multimodal pre-features (SigLIP visual tokens and SigLIP text tokens)
and caches them to disk as PyTorch tensor files (.pt).
"""

import argparse
import os
import sys
from typing import Dict, List, Set
from PIL import Image
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import FramingDataset
from src.entity_extractor import extract_salient_regions
from src.feature_extractor import FeatureExtractor


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


def discover_all_samples(data_root: str) -> List[Dict]:
    """
    Discovers all unique samples across train, dev, and test splits
    from FramingDataset.
    """
    seen_ids: Set[str] = set()
    all_samples: List[Dict] = []

    for split in ["train", "dev", "test"]:
        try:
            ds = FramingDataset(split=split, data_root=data_root)
            for sample in ds:
                if sample["id"] not in seen_ids:
                    seen_ids.add(sample["id"])
                    all_samples.append(sample)
        except Exception as e:
            print(f"[!] Warning loading split '{split}': {e}", file=sys.stderr)

    return all_samples


def main():
    parser = argparse.ArgumentParser(
        description="Extract multimodal features (unified SigLIP vision + text) and cache to disk."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data_official",
        help="Path to data_official directory (default: data_official)",
    )
    parser.add_argument(
        "--crop_dir",
        type=str,
        default="cache/crops",
        help="Path to directory containing pre-extracted crops (default: cache/crops)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cache/features",
        help="Path to directory to save feature tensors (default: cache/features)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of salient crops per image (default: 3)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Limit number of samples to process (-1 for all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for feature extraction ('mps', 'cuda', 'cpu', or None for auto)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already extracted feature files",
    )

    args = parser.parse_args()

    data_root = resolve_path(args.data_root)
    crop_dir = resolve_path(args.crop_dir)

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[*] Discovering dataset samples from: {data_root}")
    all_samples = discover_all_samples(data_root=data_root)
    print(f"[*] Discovered {len(all_samples)} total unique samples across splits.")

    if args.limit > 0:
        all_samples = all_samples[: args.limit]
        print(f"[*] Processing limited to {len(all_samples)} samples.")

    print(f"[*] Initializing FeatureExtractor (device={args.device})...")
    extractor = FeatureExtractor(device=args.device)

    processed_count = 0
    skipped_count = 0

    for idx, sample in enumerate(all_samples):
        doc_id = sample["id"]
        out_path = os.path.join(output_dir, f"{doc_id}.pt")

        if os.path.exists(out_path) and not args.overwrite:
            skipped_count += 1
            continue

        try:
            # 1. Load full image
            full_img_path = sample["image_path"]
            if not os.path.exists(full_img_path):
                print(f"[!] Image not found: {full_img_path}", file=sys.stderr)
                continue

            with Image.open(full_img_path) as img:
                full_image = img.convert("RGB")

            # 2. Retrieve crops from crop_dir or extract on the fly
            crop_filenames = [
                os.path.join(crop_dir, f"{doc_id}_crop_{k}.jpg")
                for k in range(args.top_k)
            ]
            all_crops_exist = all(os.path.exists(cf) for cf in crop_filenames)

            crop_images: List[Image.Image] = []
            if all_crops_exist:
                for cf in crop_filenames:
                    with Image.open(cf) as c_img:
                        crop_images.append(c_img.convert("RGB"))
            else:
                # Extract on the fly
                crops, _ = extract_salient_regions(
                    full_image, top_k=args.top_k, device=args.device
                )
                crop_images = [c.convert("RGB") for c in crops]

            # 3. Compute multimodal features
            text = sample.get("text", "")
            feat_dict = extractor.encode_sample(
                text=text,
                full_image=full_image,
                crop_images=crop_images,
            )

            # 4. Assemble payload
            target = sample.get("target")
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target, dtype=torch.float32)

            payload = {
                "id": doc_id,
                "image_name": sample["image_name"],
                "visual_tokens": feat_dict["visual_tokens"].detach().cpu(),  # (1+K, 768) float32
                "text_tokens": feat_dict["text_tokens"].detach().cpu(),      # (64, 768) float32
                "text_mask": feat_dict["text_mask"].detach().cpu(),          # (64,) int64
                "target": target.detach().cpu(),                              # (14,) float32
                "labels": sample["labels"],                                  # List[str]
            }

            torch.save(payload, out_path)
            processed_count += 1

        except Exception as e:
            print(f"[!] Error extracting features for {doc_id}: {e}", file=sys.stderr)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(all_samples):
            print(
                f"    Progress: {idx + 1}/{len(all_samples)} "
                f"(Extracted: {processed_count}, Skipped: {skipped_count})"
            )

    print(
        f"[✓] Feature extraction finished. Processed: {processed_count}, "
        f"Skipped: {skipped_count}."
    )
    print(f"[✓] Saved to: {output_dir}")


if __name__ == "__main__":
    main()
