"""
entity_extractor.py
Open-vocabulary entity extractor with deterministic crop fallback for SemEval-2027 Task 9.
Uses Grounding-DINO with a frame-neutral physical prompt to detect entities and extracts
Top-K localized region crops with coordinate bounding boxes.
"""

import warnings
from typing import Dict, List, Optional, Sequence, Tuple
from PIL import Image
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

DEFAULT_MODEL_NAME = "IDEA-Research/grounding-dino-tiny"
# Frame-neutral physical prompt: no framing-category words, so the detector only decomposes
# the photo into physical entities and never encodes a human prior about the answer labels.
DEFAULT_PROMPT = "person. crowd. face. sign. text. building. vehicle. object."


def _clamp_box(box: Sequence[float], w: int, h: int) -> List[int]:
    """Clamps bounding box to [0, 0, w, h] ensuring x2 > x1 and y2 > y1."""
    x1 = max(0, min(w - 1, int(round(box[0]))))
    y1 = max(0, min(h - 1, int(round(box[1]))))
    x2 = max(x1 + 1, min(w, int(round(box[2]))))
    y2 = max(y1 + 1, min(h, int(round(box[3]))))
    return [x1, y1, x2, y2]


def _box_iou(a: Sequence[int], b: Sequence[int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def select_distinct_boxes(
    candidates: List[Dict], top_k: int, iou_thresh: float = 0.7
) -> List[List[int]]:
    """Sorts candidates by (score, area) desc and keeps up to top_k boxes with pairwise IoU <= iou_thresh."""
    if top_k <= 0:
        return []
    ordered = sorted(candidates, key=lambda c: (c["score"], c["area"]), reverse=True)
    selected: List[List[int]] = []
    for c in ordered:
        if all(_box_iou(c["box"], s) <= iou_thresh for s in selected):
            selected.append(c["box"])
        if len(selected) == top_k:
            break
    return selected


class SalientEntityExtractor:
    """
    Open-vocabulary entity extractor built on Grounding-DINO with a frame-neutral prompt.
    Runs on MPS/CUDA when available and falls back to CPU once if the accelerator fails.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        prompt: str = DEFAULT_PROMPT,
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

        self.prompt = prompt
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _detect(self, img_rgb: Image.Image, conf_thresh: float) -> List[Dict]:
        w, h = img_rgb.size
        inputs = self.processor(images=img_rgb, text=self.prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"], threshold=conf_thresh,
                text_threshold=conf_thresh, target_sizes=[(h, w)],
            )[0]
        except TypeError:  # older transformers name
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"], box_threshold=conf_thresh,
                text_threshold=conf_thresh, target_sizes=[(h, w)],
            )[0]
        candidates = []
        for box, score in zip(results["boxes"].detach().cpu().tolist(), results["scores"].detach().cpu().tolist()):
            clamped = _clamp_box(box, w, h)
            candidates.append({
                "box": clamped,
                "score": float(score),
                "area": (clamped[2] - clamped[0]) * (clamped[3] - clamped[1]),
            })
        return candidates

    def extract_regions(
        self,
        img: Image.Image,
        top_k: int = 3,
        conf_thresh: float = 0.25,
    ) -> Tuple[List[Image.Image], List[List[int]]]:
        """
        Detects entities matching the frame-neutral prompt and returns Top-K distinct crops with
        integer boxes [x1, y1, x2, y2]. Missing slots are filled with deterministic focal crops.
        """
        if top_k <= 0:
            return [], []

        img_rgb = img.convert("RGB")
        w, h = img_rgb.size
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid image dimensions: {w}x{h}")

        try:
            candidates = self._detect(img_rgb, conf_thresh)
        except RuntimeError as e:
            if self.device.type == "cpu":
                raise
            warnings.warn(f"Grounding-DINO failed on {self.device} ({e}); falling back to CPU.")
            self.device = torch.device("cpu")
            self.model.to(self.device)
            candidates = self._detect(img_rgb, conf_thresh)

        selected_boxes = select_distinct_boxes(candidates, top_k=top_k, iou_thresh=0.7)

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
    conf_thresh: float = 0.25,
    device: Optional[str] = None,
) -> Tuple[List[Image.Image], List[List[int]]]:
    """
    Convenience wrapper function calling the shared singleton SalientEntityExtractor.
    """
    if top_k <= 0:
        return [], []
    extractor = get_shared_extractor(device=device)
    return extractor.extract_regions(img, top_k=top_k, conf_thresh=conf_thresh)
