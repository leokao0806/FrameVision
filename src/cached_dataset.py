"""
cached_dataset.py
PyTorch Dataset adapter and batch collator for pre-extracted multimodal features
(SigLIP visual tokens and DeBERTa-v3 text tokens) cached as .pt tensors.
"""

import os
from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import Dataset

from src.dataset import FramingDataset


def _resolve_path(path: str) -> str:
    """Resolve path against current working directory and project root."""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(project_root, path)
    if os.path.exists(candidate):
        return candidate
    return path


class CachedFramingDataset(Dataset):
    """
    Dataset adapter that reads pre-extracted multimodal features from disk.
    Ingests sample metadata via FramingDataset and pairs them with cached .pt tensor files.
    """

    def __init__(
        self,
        feature_dir: str = "cache/features",
        split: str = "train",
        split_file: str = "course_split.json",
        data_root: str = "data_official",
        filter_missing: bool = False,
    ):
        super().__init__()
        self.feature_dir = _resolve_path(feature_dir)
        self.base_dataset = FramingDataset(
            split=split,
            split_file=split_file,
            data_root=data_root,
        )
        self.split = self.base_dataset.split
        self.classes = self.base_dataset.classes
        self.num_classes = self.base_dataset.num_classes
        self.class_to_idx = self.base_dataset.class_to_idx

        if filter_missing:
            self.samples = [
                s for s in self.base_dataset.samples
                if os.path.exists(os.path.join(self.feature_dir, f"{s['id']}.pt"))
            ]
        else:
            self.samples = self.base_dataset.samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        sample_id = sample["id"]
        pt_path = os.path.join(self.feature_dir, f"{sample_id}.pt")

        if not os.path.exists(pt_path):
            raise FileNotFoundError(
                f"Cached feature file not found for sample '{sample_id}' at: {pt_path}"
            )

        data = torch.load(pt_path, map_location="cpu", weights_only=False)

        image_name = sample.get("image_name") or data.get("image_name", f"{sample_id}.jpg")
        visual_tokens = data["visual_tokens"]
        text_tokens = data["text_tokens"]
        text_mask = data["text_mask"]

        target = sample.get("target")
        if target is None:
            target = data.get("target")
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.float32)
        else:
            target = target.float()

        labels = sample.get("labels")
        if labels is None:
            labels = data.get("labels", [])

        return {
            "id": sample_id,
            "image_name": image_name,
            "visual_tokens": visual_tokens,
            "text_tokens": text_tokens,
            "text_mask": text_mask,
            "target": target,
            "labels": labels,
        }


def collate_cached_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for DataLoader of CachedFramingDataset.
    Stacks visual tokens (B, 1+K, 768), text tokens (B, Lt, 768),
    text masks (B, Lt), and targets (B, 14).
    """
    visual_tokens = torch.stack([item["visual_tokens"] for item in batch], dim=0)
    text_tokens = torch.stack([item["text_tokens"] for item in batch], dim=0)
    text_mask = torch.stack([item["text_mask"] for item in batch], dim=0)
    targets = torch.stack([item["target"] for item in batch], dim=0)

    ids = [item["id"] for item in batch]
    image_names = [item["image_name"] for item in batch]
    labels = [item["labels"] for item in batch]

    return {
        "id": ids,
        "image_name": image_names,
        "visual_tokens": visual_tokens,
        "text_tokens": text_tokens,
        "text_mask": text_mask,
        "target": targets,
        "labels": labels,
        "ids": ids,
        "image_names": image_names,
        "targets": targets,
    }
