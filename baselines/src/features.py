"""
Frozen Pretrained Feature Extraction (SigLIP images, DeBERTa-v3 text)
CS 5134/6034 NLP Fall 2026 - Team FrameVision
Authors: Leo Kao, En-Shuo Zhang

Features come from fixed pretrained weights, so nothing here is learned from the labels.
Results are cached under results/cache so the experiments can be rerun quickly.
"""

import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from src.data_loader import project_root

SIGLIP_NAME = "google/siglip-base-patch16-224"
DEBERTA_NAME = "microsoft/deberta-v3-large"
CACHE_DIR = os.path.join(project_root, "results", "cache")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _cached(name, compute):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}.npy")
    if os.path.exists(path):
        return np.load(path)
    values = compute()
    np.save(path, values)
    return values


def extract_image_features(documents):
    """Returns L2-normalized SigLIP image embeddings keyed by doc_id."""
    def compute():
        processor = AutoImageProcessor.from_pretrained(SIGLIP_NAME)
        model = AutoModel.from_pretrained(SIGLIP_NAME, dtype=torch.float32).to(DEVICE).eval()
        rows = []
        with torch.no_grad():
            for doc in documents:
                image = Image.open(doc["image_path"]).convert("RGB")
                pixels = processor(images=image, return_tensors="pt")["pixel_values"].to(DEVICE)
                output = model.vision_model(pixel_values=pixels).pooler_output
                rows.append(torch.nn.functional.normalize(output, dim=-1).squeeze(0).cpu().numpy())
        del model
        torch.cuda.empty_cache()
        return np.vstack(rows)

    values = _cached("siglip_official_images", compute)
    return {doc["doc_id"]: values[i] for i, doc in enumerate(documents)}


def encode_texts(texts, cache_name, batch_size=8, max_length=256):
    """Mean-pooled DeBERTa-v3 embeddings for a list of texts."""
    def compute():
        tokenizer = AutoTokenizer.from_pretrained(DEBERTA_NAME)
        model = AutoModel.from_pretrained(DEBERTA_NAME, dtype=torch.float32).to(DEVICE).eval()
        rows = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                                max_length=max_length, return_tensors="pt").to(DEVICE)
                hidden = model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
                rows.append(torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy())
        del model
        torch.cuda.empty_cache()
        return np.vstack(rows)

    return _cached(cache_name, compute)
