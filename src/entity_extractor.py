"""
entity_extractor.py
Salient semantic entity extractor with deterministic crop fallback for SemEval-2027 Task 9.
Uses Faster R-CNN MobileNetV3-Large 320 FPN to detect salient foreground entities
and extracts Top-K localized region crops with coordinate bounding boxes.
"""

import os
from typing import List, Optional, Sequence, Tuple
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
)


def _clamp_box(box: Sequence[float], w: int, h: int) -> List[int]:
    """Clamps bounding box to [0, 0, w, h] ensuring x2 > x1 and y2 > y1."""
    x1 = max(0, min(w - 1, int(round(box[0]))))
    y1 = max(0, min(h - 1, int(round(box[1]))))
    x2 = max(x1 + 1, min(w, int(round(box[2]))))
    y2 = max(y1 + 1, min(h, int(round(box[3]))))
    return [x1, y1, x2, y2]


class SalientEntityExtractor:
    """
    Lightweight object detection backbone for salient foreground entity extraction.
    Loads fasterrcnn_mobilenet_v3_large_320_fpn with automatic hardware acceleration
    (MPS on Mac, CUDA on NVIDIA GPU, CPU fallback).
    """

    def __init__(
        self,
        device: Optional[str] = None,
        weights: Optional[FasterRCNN_MobileNet_V3_Large_320_FPN_Weights] = None,
    ):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        if weights is None:
            weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT

        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
        self.model.to(self.device)
        self.model.eval()

    def extract_regions(
        self,
        img: Image.Image,
        top_k: int = 3,
        conf_thresh: float = 0.3,
    ) -> Tuple[List[Image.Image], List[List[int]]]:
        """
        Runs object detection on `img`, selects confident bounding boxes, and returns
        Top-K crops along with integer bounding box coordinates [x1, y1, x2, y2].

        If fewer than top_k detections meet conf_thresh, fills remaining slots with
        deterministic multi-scale focal crops:
          Fallback 1: Center focal crop [0.15*w, 0.15*h, 0.85*w, 0.85*h]
          Fallback 2: Upper focus / headline region [0.05*w, 0.05*h, 0.95*w, 0.60*h]
          Fallback 3: Golden-ratio center-left crop [0.05*w, 0.20*h, 0.70*w, 0.80*h]
        """
        if top_k <= 0:
            return [], []

        img_rgb = img.convert("RGB")
        w, h = img_rgb.size
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid image dimensions: {w}x{h}")

        img_tensor = TF.to_tensor(img_rgb).to(self.device)

        with torch.no_grad():
            preds = self.model([img_tensor])[0]

        boxes_t = preds["boxes"].detach().cpu()
        scores_t = preds["scores"].detach().cpu()

        # Collect detections above confidence threshold
        candidates = []
        for i in range(len(scores_t)):
            score = float(scores_t[i])
            if score >= conf_thresh:
                raw_box = boxes_t[i].tolist()
                clamped = _clamp_box(raw_box, w, h)
                area = (clamped[2] - clamped[0]) * (clamped[3] - clamped[1])
                candidates.append({
                    "box": clamped,
                    "score": score,
                    "area": area,
                })

        # Sort candidate detections by confidence score descending, then area descending
        candidates.sort(key=lambda c: (c["score"], c["area"]), reverse=True)

        selected_boxes: List[List[int]] = []
        for c in candidates:
            # Avoid duplicate identical boxes
            if c["box"] not in selected_boxes:
                selected_boxes.append(c["box"])
            if len(selected_boxes) == top_k:
                break

        # Fallback multi-scale focal crops if < top_k detections
        fallback_ratios = [
            (0.15, 0.15, 0.85, 0.85),  # 1: Center focal crop
            (0.05, 0.05, 0.95, 0.60),  # 2: Upper focus / headline region
            (0.05, 0.20, 0.70, 0.80),  # 3: Golden-ratio center-left crop
            (0.30, 0.20, 0.95, 0.80),  # 4: Golden-ratio center-right crop
            (0.10, 0.50, 0.90, 0.95),  # 5: Lower focal crop
            (0.00, 0.00, 1.00, 1.00),  # 6: Full context crop
        ]

        fb_idx = 0
        while len(selected_boxes) < top_k:
            rx1, ry1, rx2, ry2 = fallback_ratios[fb_idx % len(fallback_ratios)]
            fb_box = _clamp_box([rx1 * w, ry1 * h, rx2 * w, ry2 * h], w, h)
            selected_boxes.append(fb_box)
            fb_idx += 1

        # Generate crops
        crops = [img_rgb.crop((b[0], b[1], b[2], b[3])) for b in selected_boxes]
        return crops, selected_boxes


_SHARED_EXTRACTOR: Optional[SalientEntityExtractor] = None


def get_shared_extractor(device: Optional[str] = None) -> SalientEntityExtractor:
    """Returns or creates a shared singleton SalientEntityExtractor instance."""
    global _SHARED_EXTRACTOR
    if _SHARED_EXTRACTOR is None:
        _SHARED_EXTRACTOR = SalientEntityExtractor(device=device)
    return _SHARED_EXTRACTOR


def extract_salient_regions(
    img: Image.Image,
    top_k: int = 3,
    conf_thresh: float = 0.3,
    device: Optional[str] = None,
) -> Tuple[List[Image.Image], List[List[int]]]:
    """
    Convenience wrapper function calling the shared singleton SalientEntityExtractor.
    """
    extractor = get_shared_extractor(device=device)
    return extractor.extract_regions(img, top_k=top_k, conf_thresh=conf_thresh)
