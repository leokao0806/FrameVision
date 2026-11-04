"""
dataset.py
PyTorch Dataset adapter for SemEval-2027 Task 9 Subtask 2 (Multimodal Framing in Images).
Ingests official course partition (course_split.json) and pairs text and image documents
from data_official/.
"""

import json
import os
from typing import Dict, List, Optional, Any
import torch
from torch.utils.data import Dataset


class FramingDataset(Dataset):
    """
    Dataset adapter for SemEval-2027 Task 9 Subtask 2.
    Consumes course_split.json and official data directories:
      - Batch 1 (400 samples): data_official/initial-training-data/subtask2/ (all in 'train')
      - Batch 2 (97 samples): data_official/training-second-batch-oct/EN/ (split into train/dev/test)
    """

    def __init__(
        self,
        split: str,
        data_root: str = "data_official",
        split_file: str = "course_split.json",
        labels_file: Optional[str] = None,
        max_text_chars: Optional[int] = None,
    ):
        super().__init__()
        self.split_raw = split
        split_lower = split.lower()
        if split_lower == "val":
            split_lower = "dev"
        if split_lower not in ("train", "dev", "test"):
            raise ValueError(
                f"Invalid split '{split}'. Expected 'train', 'dev', or 'test'."
            )
        self.split = split_lower
        self.max_text_chars = max_text_chars

        # Resolve paths
        self.data_root = self._resolve_path(data_root)
        self.split_file = self._resolve_path(split_file)

        # 1. Load labels
        if labels_file is None:
            labels_file = os.path.join(self.data_root, "labels.txt")
        self.labels_file = self._resolve_path(labels_file)

        if not os.path.exists(self.labels_file):
            raise FileNotFoundError(f"Labels file not found at: {self.labels_file}")

        with open(self.labels_file, "r", encoding="utf-8") as f:
            self.classes = [line.strip() for line in f if line.strip()]
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # 2. Load course split
        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"Split file not found at: {self.split_file}")

        with open(self.split_file, "r", encoding="utf-8") as f:
            self.course_split = json.load(f)

        if self.split not in self.course_split:
            raise KeyError(
                f"Split '{self.split}' not found in split file: {self.split_file}"
            )
        self.allowed_ids = set(self.course_split[self.split])

        # 3. Load samples
        self.samples: List[Dict[str, Any]] = []
        if self.split == "train":
            self._load_batch1()
        self._load_batch2()

    def _resolve_path(self, path: str) -> str:
        """Resolve path against current directory and project root."""
        if os.path.isabs(path) and os.path.exists(path):
            return path
        if os.path.exists(path):
            return path
        # Try relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate = os.path.join(project_root, path)
        if os.path.exists(candidate):
            return candidate
        return path

    def _load_batch1(self):
        """Loads data_official/initial-training-data/subtask2 (400 samples, all train)."""
        b1_dir = os.path.join(self.data_root, "initial-training-data", "subtask2")
        annot_file = os.path.join(b1_dir, "subtask-2-annotations.txt")
        img_dir = os.path.join(b1_dir, "images")

        if not os.path.exists(annot_file):
            return

        with open(annot_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                img_name = parts[0].strip()
                label_names = [p.strip() for p in parts[1:] if p.strip()]

                base_name = os.path.splitext(img_name)[0]
                # Text file: EN_doc_{N}_image.jpg -> EN_M_TRAIN_doc_{N}.txt
                name_parts = img_name.split("_")
                doc_num = name_parts[2]
                txt_name = f"EN_M_TRAIN_doc_{doc_num}.txt"

                img_path = os.path.join(img_dir, img_name)
                txt_path = os.path.join(img_dir, txt_name)

                target = self._labels_to_target(label_names)

                self.samples.append({
                    "id": base_name,
                    "image_name": img_name,
                    "image_path": img_path,
                    "text_path": txt_path,
                    "labels": label_names,
                    "target": target,
                    "source": "batch1",
                })

    def _load_batch2(self):
        """Loads data_official/training-second-batch-oct/EN (97 samples filtered by split)."""
        b2_dir = os.path.join(self.data_root, "training-second-batch-oct", "EN")
        annot_file = os.path.join(b2_dir, "subtask-2-annotations.txt")
        raw_dir = os.path.join(b2_dir, "raw-documents")

        if not os.path.exists(annot_file):
            return

        with open(annot_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                img_name = parts[0].strip()
                label_names = [p.strip() for p in parts[1:] if p.strip()]

                base_name = os.path.splitext(img_name)[0]

                # Filter by course split
                if base_name not in self.allowed_ids:
                    continue

                txt_name = f"{base_name}.txt"
                img_path = os.path.join(raw_dir, img_name)
                txt_path = os.path.join(raw_dir, txt_name)

                target = self._labels_to_target(label_names)

                self.samples.append({
                    "id": base_name,
                    "image_name": img_name,
                    "image_path": img_path,
                    "text_path": txt_path,
                    "labels": label_names,
                    "target": target,
                    "source": "batch2",
                })

    def _labels_to_target(self, label_names: List[str]) -> torch.Tensor:
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for lbl in label_names:
            if lbl in self.class_to_idx:
                target[self.class_to_idx[lbl]] = 1.0
            else:
                for c_name, c_idx in self.class_to_idx.items():
                    if c_name.lower() == lbl.lower():
                        target[c_idx] = 1.0
                        break
        return target

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        text_content = ""
        if os.path.exists(sample["text_path"]):
            with open(sample["text_path"], "r", encoding="utf-8", errors="ignore") as f:
                text_content = f.read().strip()

        if self.max_text_chars and len(text_content) > self.max_text_chars:
            text_content = text_content[:self.max_text_chars]

        return {
            "id": sample["id"],
            "image_name": sample["image_name"],
            "image_path": sample["image_path"],
            "text": text_content,
            "labels": sample["labels"],
            "target": sample["target"].clone(),
        }
