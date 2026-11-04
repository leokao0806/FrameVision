"""
sam_extractor.py
Batch salient region extraction for official SemEval-2027 Task 9 dataset.
Extracts high-information foreground entity crops (e.g., protest signs, faces, gear)
and caches them in the crops/ folder for fast multimodal training.
"""

import os
import glob
from PIL import Image, ImageFilter, ImageStat


def extract_salient_crop(img: Image.Image) -> Image.Image:
    """
    Fast, robust edge/gradient density saliency extraction:
    Locates the most informative focal region in the news photograph.
    """
    w, h = img.size
    if w < 64 or h < 64:
        return img.copy()

    # Convert to grayscale and find high-contrast edges/details
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)

    # 3x3 grid search for highest detail density
    grid_w, grid_h = w // 3, h // 3
    best_score = -1
    best_box = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)

    for i in range(3):
        for j in range(3):
            box = (i * grid_w, j * grid_h, (i + 1) * grid_w, (j + 1) * grid_h)
            cell = edges.crop(box)
            stat = ImageStat.Stat(cell)
            score = stat.mean[0]
            if score > best_score:
                best_score = score
                # Expand box slightly to capture full entity context
                x1 = max(0, int(box[0] - 0.25 * grid_w))
                y1 = max(0, int(box[1] - 0.25 * grid_h))
                x2 = min(w, int(box[2] + 0.25 * grid_w))
                y2 = min(h, int(box[3] + 0.25 * grid_h))
                best_box = (x1, y1, x2, y2)

    return img.crop(best_box)


def process_directories(data_root: str = "data_official", output_dir: str = "crops"):
    os.makedirs(output_dir, exist_ok=True)

    dirs_to_scan = [
        os.path.join(data_root, "initial-training-data", "subtask2", "images"),
        os.path.join(data_root, "training-second-batch-oct", "EN", "raw-documents"),
    ]

    all_images = []
    for d in dirs_to_scan:
        if os.path.exists(d):
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                all_images.extend(glob.glob(os.path.join(d, f"*{ext}")))

    print(f"[*] Found {len(all_images)} total official news photos to process.")
    print(f"[*] Extracting salient entity crops to {output_dir}/...")

    processed = 0
    skipped = 0
    for idx, img_path in enumerate(all_images):
        fname = os.path.basename(img_path)
        out_path = os.path.join(output_dir, fname)

        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            with Image.open(img_path) as img:
                crop = extract_salient_crop(img.convert("RGB"))
                crop.save(out_path)
                processed += 1
        except Exception as e:
            print(f"    [!] Error processing {fname}: {e}")

        if (idx + 1) % 50 == 0 or (idx + 1) == len(all_images):
            print(f"    Progress: {idx + 1}/{len(all_images)} (New: {processed}, Cached: {skipped})")

    print(f"[✓] Completed! Total crops available in {output_dir}: {len(os.listdir(output_dir))}")


if __name__ == "__main__":
    process_directories()
