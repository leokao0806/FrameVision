"""
baselines.py
Unimodal and late-fusion baseline models for SemEval-2027 Task 9 Subtask 2 ablation studies.
Includes:
  - TextOnlyModel (DeBERTa-v3 tokens alone)
  - VisionGlobalModel (SigLIP global image token alone)
  - VisionRegionModel (SigLIP global + regional crops with self-attention)
  - LateFusionModel (Averaging predicted sigmoid probabilities from text and vision models)
"""

from typing import Optional
import torch
import torch.nn as nn


class TextOnlyModel(nn.Module):
    """
    Unimodal text baseline: projects text tokens, applies FFN and masked pooling,
    and maps to classification logits.
    """

    def __init__(
        self,
        text_dim: int = 768,
        embed_dim: int = 512,
        num_classes: int = 14,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.proj_text = nn.Linear(text_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for TextOnlyModel.

        Args:
            text_tokens: (B, Lt, text_dim) or (Lt, text_dim)
            text_mask: Optional (B, Lt) or (Lt,) boolean or integer mask

        Returns:
            logits: (B, num_classes)
        """
        if text_tokens.dim() == 2:
            text_tokens = text_tokens.unsqueeze(0)
        if text_mask is not None and text_mask.dim() == 1:
            text_mask = text_mask.unsqueeze(0)

        x = self.norm1(self.proj_text(text_tokens))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        if text_mask is not None:
            mask = text_mask.unsqueeze(-1).to(dtype=x.dtype)
            sum_pooled = (x * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-9)
            pooled = sum_pooled / denom
        else:
            pooled = x.mean(dim=1)

        return self.classifier(pooled)


class VisionGlobalModel(nn.Module):
    """
    Unimodal vision baseline using only the global image token (index 0 of visual_tokens).
    """

    def __init__(
        self,
        vision_dim: int = 768,
        embed_dim: int = 512,
        num_classes: int = 14,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.proj_vision = nn.Linear(vision_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for VisionGlobalModel.

        Args:
            visual_tokens: (B, 1+K, vision_dim) or (B, vision_dim) or (1+K, vision_dim)

        Returns:
            logits: (B, num_classes)
        """
        if visual_tokens.dim() == 3:
            global_token = visual_tokens[:, 0, :]
        elif visual_tokens.dim() == 2:
            if visual_tokens.shape[-1] == self.vision_dim:
                # Could be (B, vision_dim) or (1+K, vision_dim)
                # If first dim matches 1+K where K>0, check if user intended single sample
                global_token = visual_tokens
            else:
                raise ValueError(f"Unexpected shape for visual_tokens: {visual_tokens.shape}")
        elif visual_tokens.dim() == 1:
            global_token = visual_tokens.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected shape for visual_tokens: {visual_tokens.shape}")

        # If 2D and shape[1] == vision_dim, it's (B, vision_dim)
        x = self.norm(self.proj_vision(global_token))
        return self.classifier(x)


class VisionRegionModel(nn.Module):
    """
    Unimodal vision baseline using global + regional crop tokens with self-attention.
    """

    def __init__(
        self,
        vision_dim: int = 768,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_classes: int = 14,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_classes = num_classes

        self.proj_vision = nn.Linear(vision_dim, embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for VisionRegionModel.

        Args:
            visual_tokens: (B, 1+K, vision_dim) or (1+K, vision_dim)

        Returns:
            logits: (B, num_classes)
        """
        if visual_tokens.dim() == 2:
            visual_tokens = visual_tokens.unsqueeze(0)

        v_proj = self.proj_vision(visual_tokens)  # (B, 1+K, embed_dim)

        attn_out, _ = self.self_attn(
            query=v_proj,
            key=v_proj,
            value=v_proj,
            need_weights=False,
        )

        x = self.norm1(v_proj + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        # Pool across all tokens (global + regions)
        pooled = x.mean(dim=1)
        return self.classifier(pooled)


class LateFusionModel(nn.Module):
    """
    Late fusion baseline: averages predicted sigmoid probabilities:
      0.5 * sigma(text_logits) + 0.5 * sigma(vision_logits)
    """

    def __init__(
        self,
        text_dim: int = 768,
        vision_dim: int = 768,
        embed_dim: int = 512,
        num_classes: int = 14,
        dropout: float = 0.1,
        text_model: Optional[TextOnlyModel] = None,
        vision_model: Optional[VisionGlobalModel] = None,
    ):
        super().__init__()
        self.text_model = text_model if text_model is not None else TextOnlyModel(
            text_dim=text_dim,
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.vision_model = vision_model if vision_model is not None else VisionGlobalModel(
            vision_dim=vision_dim,
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for LateFusionModel.

        Returns:
            Ensemble predicted probabilities: (B, num_classes) in [0, 1].
        """
        text_logits = self.text_model(text_tokens, text_mask=text_mask)
        vision_logits = self.vision_model(visual_tokens)

        text_probs = torch.sigmoid(text_logits)
        vision_probs = torch.sigmoid(vision_logits)

        return 0.5 * text_probs + 0.5 * vision_probs
