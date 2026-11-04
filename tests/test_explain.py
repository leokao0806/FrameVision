"""
tests/test_explain.py
Unit and integration tests for cross-modal attention interpretability and visualization.
"""

import os
import subprocess
import sys
import numpy as np
import pytest
import torch
from PIL import Image

from src.models import CrossModalFramingModel
from src.explain import (
    extract_attention_for_sample,
    visualize_sample_attention,
)


def test_visualize_attention_generates_figure(tmp_path):
    """
    Test that visualize_sample_attention produces a non-empty image file
    with overlaid bounding boxes and attended text tokens.
    """
    img = Image.new("RGB", (300, 300), color=(150, 150, 150))
    boxes = [[20, 20, 100, 100], [120, 120, 200, 200], [50, 150, 150, 250]]
    attn_weights = torch.softmax(torch.randn(4), dim=-1).numpy()  # 1 global + 3 crops
    out_file = tmp_path / "attn_test.png"

    fig = visualize_sample_attention(
        image=img,
        boxes=boxes,
        attn_scores=attn_weights,
        text_tokens=["city", "council", "protest", "police"],
        top_text_indices=[2, 3],
        title="Test Attention Visualization",
        output_path=str(out_file),
    )

    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 0


def test_visualize_attention_without_text(tmp_path):
    """
    Test that visualize_sample_attention works cleanly without optional text tokens.
    """
    img = Image.new("RGB", (200, 200), color=(100, 100, 100))
    boxes = [[10, 10, 80, 80], [100, 100, 180, 180]]
    attn_weights = np.array([0.5, 0.3, 0.2])  # 1 global + 2 crops
    out_file = tmp_path / "attn_no_text.png"

    fig = visualize_sample_attention(
        image=img,
        boxes=boxes,
        attn_scores=attn_weights,
        output_path=str(out_file),
    )

    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 0


def test_extract_attention_for_sample():
    """
    Test extract_attention_for_sample produces correct shapes and normalized scores.
    """
    model = CrossModalFramingModel(
        text_dim=64,
        vision_dim=64,
        embed_dim=32,
        num_heads=4,
        num_classes=14,
    )
    model.eval()

    # 1 global + 3 entity crops (1+K = 4)
    visual_tokens = torch.randn(4, 64)
    # Sequence length 10
    text_tokens = torch.randn(10, 64)
    # 7 active tokens, 3 padded tokens
    text_mask = torch.tensor([1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=torch.long)

    vis_attn_scores, text_token_scores = extract_attention_for_sample(
        model=model,
        visual_tokens=visual_tokens,
        text_tokens=text_tokens,
        text_mask=text_mask,
    )

    # vis_attn_scores should have shape (4,) for 1 global + 3 crops
    assert isinstance(vis_attn_scores, np.ndarray)
    assert vis_attn_scores.shape == (4,)
    # Attention across all visual tokens sums to 1.0
    assert np.isclose(vis_attn_scores.sum(), 1.0, atol=1e-5)
    assert np.all(vis_attn_scores >= 0.0)

    # text_token_scores should have shape (7,) corresponding to the 7 active tokens
    assert isinstance(text_token_scores, np.ndarray)
    assert text_token_scores.shape == (7,)
    assert np.all(text_token_scores >= 0.0)
    assert np.all(text_token_scores <= 1.0 + 1e-5)


def test_visualize_script_cli(tmp_path):
    """
    Test scripts/05_visualize_attention.py CLI runs with --dry_run.
    """
    python_exe = sys.executable
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts",
        "05_visualize_attention.py",
    )

    cmd = [
        python_exe,
        script_path,
        "--dry_run",
        "--output_dir",
        str(tmp_path),
    ]

    res = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert res.returncode == 0, f"Script failed with stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
    # Verify at least one figure was produced in tmp_path
    figures = [f for f in os.listdir(tmp_path) if f.endswith(".png")]
    assert len(figures) > 0, f"Expected output figures in {tmp_path}, got: {os.listdir(tmp_path)}"
