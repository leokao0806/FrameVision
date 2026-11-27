"""
tests/test_feature_extractor.py
Unit tests for FeatureExtractor (unified SigLIP vision + text pre-feature extraction).
"""

import pytest
import torch
from PIL import Image

from src.feature_extractor import FeatureExtractor


@pytest.fixture(scope="module")
def extractor():
    """Module-level fixture to load FeatureExtractor once for tests."""
    return FeatureExtractor()


def test_feature_extractor_shapes_and_types(extractor):
    """
    Checks output shapes (4, 768) and (64, 768), verifies dtype == torch.float32,
    and attention mask shape (64,).
    """
    # 1 full image and 3 crops = 4 images total
    full_image = Image.new("RGB", (224, 224), color=(255, 0, 0))
    crops = [
        Image.new("RGB", (100, 100), color=(0, 255, 0)),
        Image.new("RGB", (120, 80), color=(0, 0, 255)),
        Image.new("RGB", (64, 64), color=(128, 128, 128)),
    ]
    text = "A multimodal framing sample containing an image and text caption."

    out = extractor.encode_sample(text=text, full_image=full_image, crop_images=crops)

    assert "visual_tokens" in out
    assert "text_tokens" in out
    assert "text_mask" in out

    visual_tokens = out["visual_tokens"]
    text_tokens = out["text_tokens"]
    text_mask = out["text_mask"]

    assert visual_tokens.shape == (4, 768)
    assert visual_tokens.dtype == torch.float32

    assert text_tokens.shape == (64, 768)
    assert text_tokens.dtype == torch.float32

    assert text_mask.shape == (64,)
    assert text_mask.dtype == torch.int64
    assert not torch.isnan(text_tokens).any()
    assert text_mask.sum() >= 1


def test_feature_extractor_empty_text(extractor):
    """
    Handles empty text string and None gracefully without crashing.
    """
    full_image = Image.new("RGB", (200, 200), color=(50, 50, 50))
    crops = [
        Image.new("RGB", (50, 50), color=(100, 100, 100)),
        Image.new("RGB", (50, 50), color=(150, 150, 150)),
        Image.new("RGB", (50, 50), color=(200, 200, 200)),
    ]

    # Test empty string
    out_empty = extractor.encode_sample(text="", full_image=full_image, crop_images=crops)
    assert out_empty["text_tokens"].shape == (64, 768)
    assert out_empty["text_tokens"].dtype == torch.float32
    assert out_empty["text_mask"].shape == (64,)
    assert out_empty["text_mask"].sum() >= 1  # never all-zero: avoids 0/0 in masked pooling
    assert not torch.isnan(out_empty["text_tokens"]).any()
    assert out_empty["visual_tokens"].shape == (4, 768)

    # Test None as text
    out_none = extractor.encode_sample(text=None, full_image=full_image, crop_images=crops)
    assert out_none["text_tokens"].shape == (64, 768)
    assert out_none["text_tokens"].dtype == torch.float32
    assert out_none["text_mask"].shape == (64,)
