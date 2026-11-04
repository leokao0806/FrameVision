#!/usr/bin/env python3
"""
01_extract_regions.py
Batch salient region extraction CLI script for official SemEval-2027 Task 9 dataset.
Extracts Top-K semantic entity crops and saves them to cache/crops/ with metadata.json.
"""

import argparse
import json
import os
import sys
from typing import Dict, List
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.entity_extractor import SalientEntityExtractor


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


def discover_images(data_root: str) -> List[str]:
    """
    Discovers all images across:
      1. data_official/initial-training-data/subtask2/images
      2. data_official/training-second-batch-oct/EN/raw-documents
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    dirs_to_scan = [
        os.path.join(data_root, "initial-training-data", "subtask2", "images"),
        os.path.join(data_root, "training-second-batch-oct", "EN", "raw-documents"),
    ]

    image_paths: List[str] = []
    for d in dirs_to_scan:
        resolved_d = resolve_path(d)
        if not os.path.exists(resolved_d):
            print(f"[!] Warning: scan directory not found: {resolved_d}")
            continue

        entries = sorted(os.listdir(resolved_d))
        for fname in entries:
            ext = os.path.splitext(fname)[1].lower()
            if ext in valid_exts:
                image_paths.append(os.path.join(resolved_d, fname))

    return image_paths


def main():
    parser = argparse.ArgumentParser(
        description="Extract Top-K salient semantic entity crops and save metadata."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data_official",
        help="Path to data_official directory (default: data_official)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cache/crops",
        help="Path to output directory for crops (default: cache/crops)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of crops per image (default: 3)",
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.3,
        help="Confidence threshold for entity detections (default: 0.3)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Limit number of images to process (-1 for all)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already extracted crops and metadata",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run detection on ('mps', 'cuda', 'cpu', or None for auto)",
    )

    args = parser.parse_args()

    data_root = resolve_path(args.data_root)
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    metadata_path = os.path.join(output_dir, "metadata.json")
    metadata: Dict[str, dict] = {}
    if os.path.exists(metadata_path) and not args.overwrite:
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"[!] Warning reading existing metadata: {e}. Starting fresh.")
            metadata = {}

    all_images = discover_images(data_root)
    print(f"[*] Discovered {len(all_images)} total candidate images across partitions.")

    if args.limit > 0:
        all_images = all_images[: args.limit]
        print(f"[*] Processing limited to {len(all_images)} images.")

    print(f"[*] Initializing SalientEntityExtractor (device={args.device})...")
    extractor = SalientEntityExtractor(device=args.device)

    processed_count = 0
    skipped_count = 0

    for idx, img_path in enumerate(all_images):
        fname = os.path.basename(img_path)
        base_name = os.path.splitext(fname)[0]

        crop_filenames = [f"{base_name}_crop_{k}.jpg" for k in range(args.top_k)]
        all_crops_exist = all(
            os.path.exists(os.path.join(output_dir, cf)) for cf in crop_filenames
        )

        if not args.overwrite and all_crops_exist and base_name in metadata:
            skipped_count += 1
            continue

        try:
            with Image.open(img_path) as img:
                crops, boxes = extractor.extract_regions(
                    img, top_k=args.top_k, conf_thresh=args.conf_thresh
                )

            # Save crops
            for k, (crop, cf) in enumerate(zip(crops, crop_filenames)):
                crop_path = os.path.join(output_dir, cf)
                crop.save(crop_path, quality=95)

            # Record metadata
            metadata[base_name] = {
                "boxes": boxes,
                "crops": crop_filenames,
            }
            processed_count += 1

        except Exception as e:
            print(f"[!] Error processing {img_path}: {e}", file=sys.stderr)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(all_images):
            print(
                f"    Progress: {idx + 1}/{len(all_images)} "
                f"(New: {processed_count}, Skipped: {skipped_count})"
            )

    # Save metadata JSON
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"[✓] Extraction finished. Processed: {processed_count}, "
        f"Skipped: {skipped_count}. Total in metadata: {len(metadata)}"
    )
    print(f"[✓] Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
