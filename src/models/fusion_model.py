"""
fusion_model.py
Region-Aware Cross-Modal Attention Model for SemEval-2027 Task 9 Subtask 2.
Projects DeBERTa-v3 text sequence tokens and SigLIP region/global visual tokens
into a shared latent space and attends text queries to visual region keys/values.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class CrossModalFramingModel(nn.Module):
    """
    Region-Aware Cross-Modal Attention Model.
    
    Architecture:
      - Text Projection: text_dim -> embed_dim
      - Vision Projection: vision_dim -> embed_dim
      - Multi-Head Cross-Attention:
          Queries: Text tokens (B, Lt, embed_dim)
          Keys & Values: Visual tokens (B, 1+K, embed_dim)
          -> Attended representation + attention weights (B, Lt, 1+K)
      - Residual + LayerNorm
      - Feed-Forward Network: embed_dim -> 2*embed_dim -> embed_dim with GELU & Dropout
      - Residual + LayerNorm
      - Masked Mean-Pooling across active text queries using text_mask
      - Classification Head: MLP mapping to num_classes logits
    """

    def __init__(
        self,
        text_dim: int = 768,
        vision_dim: int = 768,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_classes: int = 14,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_classes = num_classes

        # Projections
        self.proj_text = nn.Linear(text_dim, embed_dim)
        self.proj_vision = nn.Linear(vision_dim, embed_dim)

        # Cross-Modal Multi-Head Attention (Queries = Text, Keys/Values = Vision)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Position-wise Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for CrossModalFramingModel.

        Args:
            visual_tokens: (B, 1+K, vision_dim) or (1+K, vision_dim)
            text_tokens: (B, Lt, text_dim) or (Lt, text_dim)
            text_mask: Optional (B, Lt) or (Lt,) boolean or integer mask (1 = valid, 0 = pad)

        Returns:
            Tuple of:
              - logits: (B, num_classes)
              - attn_weights: (B, Lt, 1+K)
        """
        # Ensure batch dimension
        if visual_tokens.dim() == 2:
            visual_tokens = visual_tokens.unsqueeze(0)
        if text_tokens.dim() == 2:
            text_tokens = text_tokens.unsqueeze(0)
        if text_mask is not None and text_mask.dim() == 1:
            text_mask = text_mask.unsqueeze(0)

        # Project modalities into shared embed_dim
        t_proj = self.proj_text(text_tokens)      # (B, Lt, embed_dim)
        v_proj = self.proj_vision(visual_tokens)  # (B, 1+K, embed_dim)

        # Cross-attention: Query = Text, Key/Value = Vision
        attn_out, attn_weights = self.cross_attn(
            query=t_proj,
            key=v_proj,
            value=v_proj,
            need_weights=True,
            average_attn_weights=True,
        )

        # Residual connection 1 + LayerNorm
        x = self.norm1(t_proj + self.dropout1(attn_out))

        # Feed-forward network + Residual connection 2 + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        # Masked pooling over text sequence length (dim=1)
        if text_mask is not None:
            mask = text_mask.unsqueeze(-1).to(dtype=x.dtype)
            sum_pooled = (x * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-9)
            pooled = sum_pooled / denom
        else:
            pooled = x.mean(dim=1)

        # Multi-label classification logits
        logits = self.classifier(pooled)

        return logits, attn_weights
