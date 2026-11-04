"""
dataset.py
PyTorch Dataset for SemEval-2027 Task 9 Subtask 2 (Multimodal Framing in Images).
Supports both official training data batches:
  - Batch 1: data_official/initial-training-data/subtask2 (400 samples)
  - Batch 2: data_official/training-second-batch-oct/EN (97 samples)
"""

import os
from typing import List, Optional, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class FramingMultimodalDataset(Dataset):
    """
    Dataset for paired News Text, Full Image, and Local Salient Crop (from SAM/Saliency).
    """

    def __init__(
        self,
        data_root: str = "data_official",
        crops_dir: str = "crops",
        labels_file: Optional[str] = None,
        include_batch1: bool = True,
        include_batch2: bool = True,
        image_transform=None,
        crop_transform=None,
        max_text_chars: int = 2000,
    ):
        self.data_root = data_root
        self.crops_dir = crops_dir
        self.max_text_chars = max_text_chars

        # 1. Load 14 framing labels
        if labels_file is None:
            labels_file = os.path.join(data_root, "labels.txt")
        if not os.path.exists(labels_file):
            labels_file = "data_official/labels.txt"

        with open(labels_file, "r", encoding="utf-8") as f:
            self.classes = [line.strip() for line in f if line.strip()]
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # 2. Image Transforms
        if image_transform is not None:
            self.image_transform = image_transform
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

        if crop_transform is not None:
            self.crop_transform = crop_transform
        else:
            self.crop_transform = self.image_transform

        # 3. Load Samples
        self.samples = []
        if include_batch1:
            self._load_batch1()
        if include_batch2:
            self._load_batch2()

    def _load_batch1(self):
        """Loads data_official/initial-training-data/subtask2 (400 samples)"""
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

                # Text file: EN_doc_{N}_image.jpg -> EN_M_TRAIN_doc_{N}.txt
                name_parts = img_name.split("_")
                doc_num = name_parts[2]
                txt_name = f"EN_M_TRAIN_doc_{doc_num}.txt"

                img_path = os.path.join(img_dir, img_name)
                txt_path = os.path.join(img_dir, txt_name)

                binary_target = self._labels_to_multihot(label_names)

                self.samples.append({
                    "id": img_name,
                    "image_name": img_name,
                    "image_path": img_path,
                    "text_path": txt_path,
                    "label_names": label_names,
                    "target": binary_target,
                    "source": "batch1",
                })

    def _load_batch2(self):
        """Loads data_official/training-second-batch-oct/EN (97 samples)"""
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
                txt_name = f"{base_name}.txt"

                img_path = os.path.join(raw_dir, img_name)
                txt_path = os.path.join(raw_dir, txt_name)

                binary_target = self._labels_to_multihot(label_names)

                self.samples.append({
                    "id": img_name,
                    "image_name": img_name,
                    "image_path": img_path,
                    "text_path": txt_path,
                    "label_names": label_names,
                    "target": binary_target,
                    "source": "batch2",
                })

    def _labels_to_multihot(self, label_names: List[str]) -> List[int]:
        target = [0] * self.num_classes
        for lbl in label_names:
            if lbl in self.class_to_idx:
                target[self.class_to_idx[lbl]] = 1
            else:
                for c_name, c_idx in self.class_to_idx.items():
                    if c_name.lower() == lbl.lower():
                        target[c_idx] = 1
                        break
        return target

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        # 1. Read Text
        text_content = ""
        if os.path.exists(sample["text_path"]):
            with open(sample["text_path"], "r", encoding="utf-8", errors="ignore") as f:
                text_content = f.read().strip()
        if self.max_text_chars and len(text_content) > self.max_text_chars:
            text_content = text_content[:self.max_text_chars]

        # 2. Read Full Image
        if os.path.exists(sample["image_path"]):
            try:
                raw_image = Image.open(sample["image_path"]).convert("RGB")
            except Exception:
                raw_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        else:
            raw_image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        # 3. Read Salient Crop
        crop_path = os.path.join(self.crops_dir, sample["image_name"])
        if os.path.exists(crop_path):
            try:
                raw_crop = Image.open(crop_path).convert("RGB")
            except Exception:
                raw_crop = raw_image.copy()
        else:
            # Fallback center crop
            w, h = raw_image.size
            crop_box = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
            raw_crop = raw_image.crop(crop_box)

        full_img_tensor = self.image_transform(raw_image)
        crop_img_tensor = self.crop_transform(raw_crop)
        target_tensor = torch.tensor(sample["target"], dtype=torch.float32)

        return {
            "image_name": sample["image_name"],
            "text": text_content,
            "full_image": full_img_tensor,
            "crop_image": crop_img_tensor,
            "target": target_tensor,
            "label_names": sample["label_names"],
        }
