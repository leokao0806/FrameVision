"""
explain.py
Cross-modal attention interpretability and visualization pipeline
for SemEval-2027 Task 9 Subtask 2.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def extract_attention_for_sample(
    model: nn.Module,
    visual_tokens: torch.Tensor,
    text_tokens: torch.Tensor,
    text_mask: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts cross-modal attention scores for a single multimodal sample.

    Args:
        model: Trained CrossModalFramingModel or nn.Module that outputs (logits, attn_weights).
        visual_tokens: Visual token tensor of shape (1+K, vision_dim) or (1, 1+K, vision_dim).
        text_tokens: Text sequence token tensor of shape (Lt, text_dim) or (1, Lt, text_dim).
        text_mask: Optional mask tensor of shape (Lt,) or (1, Lt) indicating active (1) vs padding (0).

    Returns:
        Tuple of:
          - vis_attn_scores: np.ndarray of shape (1+K,) containing average attention weights
            allocated to the global image (idx 0) and each of the K entity crops. Sums to 1.0.
          - text_token_scores: np.ndarray of shape (Lt_active,) representing how strongly
            each active text token attended to visual entity crops (sum over crop tokens 1..K).
    """
    model.eval()

    # Determine model device & dtype
    params = list(model.parameters())
    if params:
        device = params[0].device
        dtype = params[0].dtype
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    # Ensure batch dimension
    if visual_tokens.dim() == 2:
        v_in = visual_tokens.unsqueeze(0)
    else:
        v_in = visual_tokens

    if text_tokens.dim() == 2:
        t_in = text_tokens.unsqueeze(0)
    else:
        t_in = text_tokens

    v_in = v_in.to(device=device, dtype=dtype)
    t_in = t_in.to(device=device, dtype=dtype)

    m_in: Optional[torch.Tensor] = None
    if text_mask is not None:
        if text_mask.dim() == 1:
            m_in = text_mask.unsqueeze(0)
        else:
            m_in = text_mask
        m_in = m_in.to(device=device)

    with torch.no_grad():
        out = model(visual_tokens=v_in, text_tokens=t_in, text_mask=m_in)
        if isinstance(out, tuple):
            _, attn_weights = out
        else:
            raise ValueError(f"Expected model output tuple (logits, attn_weights), got {type(out)}")

    # attn_weights shape: (B, Lt, 1+K)
    attn = attn_weights.squeeze(0)  # (Lt, 1+K)
    lt, num_vis = attn.shape

    # Identify active tokens
    if m_in is not None:
        mask_1d = m_in.squeeze(0) > 0
        active_indices = torch.nonzero(mask_1d, as_tuple=True)[0]
        if len(active_indices) == 0:
            active_indices = torch.arange(lt, device=device)
    else:
        active_indices = torch.arange(lt, device=device)

    # Active attention slice: (Lt_active, 1+K)
    active_attn = attn[active_indices]

    # vis_attn_scores: Mean attention from active text queries across visual keys
    vis_attn = active_attn.mean(dim=0)
    vis_attn_scores = vis_attn.detach().cpu().numpy().astype(np.float64)

    # Normalize across visual tokens
    v_sum = vis_attn_scores.sum()
    if v_sum > 0:
        vis_attn_scores = vis_attn_scores / v_sum

    # text_token_scores: how strongly each active text query attends to visual entity crops (1..K)
    if num_vis > 1:
        crop_attn = active_attn[:, 1:].sum(dim=-1)
        text_token_scores = crop_attn.detach().cpu().numpy().astype(np.float64)
    else:
        text_token_scores = np.zeros(len(active_indices), dtype=np.float64)

    return vis_attn_scores, text_token_scores


def visualize_sample_attention(
    image: Image.Image,
    boxes: List[List[int]],
    attn_scores: np.ndarray,
    text_tokens: Optional[List[str]] = None,
    top_text_indices: Optional[List[int]] = None,
    text_token_scores: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Renders and saves a visual attention figure displaying:
      - The source image overlaid with entity bounding boxes weighted and labeled by attention scores.
      - Annotation for the global image attention weight.
      - An optional side-by-side bar chart of top attended text tokens/keywords.

    Args:
        image: PIL Image of the document / scene.
        boxes: List of bounding boxes [x1, y1, x2, y2] for entity crops.
        attn_scores: Array of shape (1+K,) for [global_score, crop_1_score, ..., crop_K_score].
        text_tokens: Optional list of text tokens or words from the text description.
        top_text_indices: Optional list of indices indicating top attended tokens.
        text_token_scores: Optional array of attention scores for active text tokens.
        title: Optional figure title.
        output_path: Optional file path to save the generated figure.

    Returns:
        matplotlib.pyplot.Figure instance.
    """
    has_text = text_tokens is not None and len(text_tokens) > 0

    if has_text:
        fig, (ax_img, ax_text) = plt.subplots(
            1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.4, 1.0]}
        )
    else:
        fig, ax_img = plt.subplots(1, 1, figsize=(8, 7))
        ax_text = None

    # 1. Plot source image
    ax_img.imshow(image)
    ax_img.axis("off")

    global_score = float(attn_scores[0]) if len(attn_scores) > 0 else 0.0
    crop_scores = attn_scores[1:] if len(attn_scores) > 1 else np.array([])

    img_title = f"Visual Region Cross-Attention\nGlobal Image Weight: {global_score:.3f} ({global_score*100:.1f}%)"
    ax_img.set_title(img_title, fontsize=12, pad=10)

    # Palette of colors for entity crops
    colors = ["#FF3366", "#33CC66", "#3399FF", "#FF9933", "#9933FF", "#00E5FF"]

    # Overlay bounding boxes
    num_boxes_to_draw = min(len(boxes), len(crop_scores))
    for i in range(num_boxes_to_draw):
        box = boxes[i]
        score = float(crop_scores[i])
        color = colors[i % len(colors)]

        x1, y1, x2, y2 = box
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        # Draw transparent filled box
        rect_fill = patches.Rectangle(
            (x1, y1),
            w,
            h,
            linewidth=0,
            edgecolor="none",
            facecolor=color,
            alpha=0.20,
        )
        ax_img.add_patch(rect_fill)

        # Draw solid bounding rectangle
        rect_edge = patches.Rectangle(
            (x1, y1),
            w,
            h,
            linewidth=2.5,
            edgecolor=color,
            facecolor="none",
            alpha=0.90,
        )
        ax_img.add_patch(rect_edge)

        # Label above or inside box
        label_text = f"Crop {i+1}: {score:.3f} ({score*100:.1f}%)"
        ax_img.text(
            x1 + 4,
            max(15, y1 - 6),
            label_text,
            fontsize=10,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor=color,
                edgecolor="none",
                alpha=0.85,
            ),
        )

    # 2. Plot text tokens panel if available
    if has_text and ax_text is not None:
        assert text_tokens is not None

        # Determine which tokens and scores to display
        tokens_to_plot: List[str] = []
        scores_to_plot: List[float] = []

        if top_text_indices is not None and len(top_text_indices) > 0:
            for idx in top_text_indices:
                if idx < len(text_tokens):
                    tokens_to_plot.append(str(text_tokens[idx]))
                    if text_token_scores is not None and idx < len(text_token_scores):
                        scores_to_plot.append(float(text_token_scores[idx]))
                    else:
                        scores_to_plot.append(1.0)
        elif text_token_scores is not None and len(text_token_scores) > 0:
            # Sort top 10 by score
            k_top = min(10, len(text_token_scores), len(text_tokens))
            sorted_idx = np.argsort(text_token_scores)[::-1][:k_top]
            for idx in sorted_idx:
                tokens_to_plot.append(str(text_tokens[idx]))
                scores_to_plot.append(float(text_token_scores[idx]))
        else:
            # Fallback to first few tokens
            k_top = min(10, len(text_tokens))
            tokens_to_plot = [str(text_tokens[i]) for i in range(k_top)]
            scores_to_plot = [1.0 / (i + 1) for i in range(k_top)]

        # Reverse for horizontal bar chart (highest score on top)
        tokens_rev = tokens_to_plot[::-1]
        scores_rev = scores_to_plot[::-1]
        y_pos = np.arange(len(tokens_rev))

        # Color intensity mapped to score
        max_s = max(max(scores_rev) if scores_rev else 1.0, 1e-6)
        norm_scores = [s / max_s for s in scores_rev]
        bar_colors = [plt.cm.viridis(0.3 + 0.6 * ns) for ns in norm_scores]

        bars = ax_text.barh(y_pos, scores_rev, color=bar_colors, edgecolor="none", height=0.6)
        ax_text.set_yticks(y_pos)
        ax_text.set_yticklabels(tokens_rev, fontsize=11, fontweight="normal")
        ax_text.set_xlabel("Cross-Attention to Entity Crops", fontsize=11)
        ax_text.set_title("Attended Text Keywords", fontsize=12, pad=10)
        ax_text.grid(True, axis="x", linestyle="--", alpha=0.5)

        # Display numeric values on bars
        for bar in bars:
            val = bar.get_width()
            ax_text.text(
                val + max_s * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                ha="left",
                fontsize=9,
                color="#333333",
            )
        ax_text.set_xlim(0, max_s * 1.25)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    plt.tight_layout()

    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

    return fig
