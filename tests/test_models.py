"""
tests/test_models.py
Unit tests for CrossModalFramingModel and unimodal ablation baselines.
"""

import pytest
import torch
import torch.nn as nn

from src.models.fusion_model import CrossModalFramingModel
from src.models.baselines import (
    TextOnlyModel,
    VisionGlobalModel,
    VisionRegionModel,
    LateFusionModel,
)


def test_cross_modal_fusion_forward():
    """
    Checks output shapes (B, 14) for logits and (B, Lt, 1+K) for attention weights.
    Checks backward pass and gradient flow through both vision and text projections.
    """
    B, Lt, K_plus_1, D = 2, 256, 4, 768
    num_classes = 14
    embed_dim = 512
    num_heads = 8

    model = CrossModalFramingModel(
        text_dim=D,
        vision_dim=D,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_classes=num_classes,
        dropout=0.1,
    )

    v_tokens = torch.randn(B, K_plus_1, D)
    t_tokens = torch.randn(B, Lt, D)
    t_mask = torch.ones(B, Lt, dtype=torch.bool)
    t_mask[:, 100:] = False  # padding tokens

    logits, attn_weights = model(v_tokens, t_tokens, t_mask)

    assert logits.shape == (B, num_classes)
    assert attn_weights.shape == (B, Lt, K_plus_1)

    # Check backward pass & gradient flow
    loss = logits.sum()
    loss.backward()

    assert model.proj_text.weight.grad is not None
    assert model.proj_vision.weight.grad is not None
    assert not torch.isnan(model.proj_text.weight.grad).any()
    assert not torch.isnan(model.proj_vision.weight.grad).any()


def test_cross_modal_fusion_masking():
    """
    Checks that padding tokens do not distort pooled embeddings or logits.
    """
    B, Lt, K_plus_1, D = 2, 64, 4, 768
    valid_len = 20

    model = CrossModalFramingModel(
        text_dim=D,
        vision_dim=D,
        embed_dim=256,
        num_heads=4,
        num_classes=14,
        dropout=0.0,
    )
    model.eval()

    v_tokens = torch.randn(B, K_plus_1, D)

    # Base valid tokens
    base_t_tokens = torch.randn(B, valid_len, D)

    # Sample 1: padding filled with zeros
    t_tokens1 = torch.zeros(B, Lt, D)
    t_tokens1[:, :valid_len, :] = base_t_tokens

    # Sample 2: padding filled with random noise
    t_tokens2 = torch.randn(B, Lt, D) * 10.0
    t_tokens2[:, :valid_len, :] = base_t_tokens

    # Masks (valid up to valid_len)
    mask_bool = torch.zeros(B, Lt, dtype=torch.bool)
    mask_bool[:, :valid_len] = True

    mask_int = torch.zeros(B, Lt, dtype=torch.int64)
    mask_int[:, :valid_len] = 1

    with torch.no_grad():
        logits1, _ = model(v_tokens, t_tokens1, mask_bool)
        logits2, _ = model(v_tokens, t_tokens2, mask_bool)
        logits3, _ = model(v_tokens, t_tokens1, mask_int)

    # Masking should completely isolate the valid tokens
    assert torch.allclose(logits1, logits2, atol=1e-5), "Padding tokens distorted pooled logits!"
    assert torch.allclose(logits1, logits3, atol=1e-5), "Bool and Int masks produced different logits!"


def test_unimodal_baselines():
    """
    Verifies forward passes and output shapes for TextOnlyModel, VisionGlobalModel,
    VisionRegionModel, and LateFusionModel.
    """
    B, Lt, K_plus_1, D = 2, 64, 4, 768
    num_classes = 14
    embed_dim = 256

    # 1. TextOnlyModel
    text_model = TextOnlyModel(
        text_dim=D,
        embed_dim=embed_dim,
        num_classes=num_classes,
        dropout=0.1,
    )
    t_tokens = torch.randn(B, Lt, D)
    t_mask = torch.ones(B, Lt, dtype=torch.int64)
    t_mask[:, 30:] = 0

    t_logits = text_model(t_tokens, text_mask=t_mask)
    assert t_logits.shape == (B, num_classes)

    loss_t = t_logits.sum()
    loss_t.backward()
    assert text_model.proj_text.weight.grad is not None

    # 2. VisionGlobalModel
    vision_global_model = VisionGlobalModel(
        vision_dim=D,
        embed_dim=embed_dim,
        num_classes=num_classes,
        dropout=0.1,
    )
    v_tokens = torch.randn(B, K_plus_1, D)
    vg_logits = vision_global_model(v_tokens)
    assert vg_logits.shape == (B, num_classes)

    # Also test with 2D tensor (B, D)
    vg_2d_logits = vision_global_model(v_tokens[:, 0, :])
    assert vg_2d_logits.shape == (B, num_classes)

    loss_vg = vg_logits.sum()
    loss_vg.backward()
    assert vision_global_model.proj_vision.weight.grad is not None

    # 3. VisionRegionModel
    vision_region_model = VisionRegionModel(
        vision_dim=D,
        embed_dim=embed_dim,
        num_heads=4,
        num_classes=num_classes,
        dropout=0.1,
    )
    vr_logits = vision_region_model(v_tokens)
    assert vr_logits.shape == (B, num_classes)

    loss_vr = vr_logits.sum()
    loss_vr.backward()
    assert vision_region_model.proj_vision.weight.grad is not None

    # 4. LateFusionModel
    late_fusion = LateFusionModel(
        text_dim=D,
        vision_dim=D,
        embed_dim=embed_dim,
        num_classes=num_classes,
        dropout=0.1,
    )
    lf_probs = late_fusion(v_tokens, t_tokens, text_mask=t_mask)
    assert lf_probs.shape == (B, num_classes)
    assert (lf_probs >= 0.0).all() and (lf_probs <= 1.0).all()
