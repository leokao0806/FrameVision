"""
Region-Aware Cross-Modal Attention Architecture (SAM + Cross-Attention)
CS 5134/6034 NLP Fall 2026 - Team FrameVision
Authors: Leo Kao (Lead Multimodal Architect), En-Shuo Zhang
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.data_loader import NUM_CLASSES

class RegionAwareCrossModalAttention(nn.Module):
    """
    Prototype of the architecture proposed in NLP_proposal.pdf; only smoke-tested so far.
    It has not been evaluated: SAM / Grounding-DINO regions are not integrated yet, and
    the trial data offers too few images to train it. Planned design: decompose images
    into salient semantic regions and cross-attend them with text representations.

    Architecture components:
    1. Region & Text feature projections into a shared latent space.
    2. Multi-Head Cross-Modal Attention (Text queries attending to Image Region keys/values).
    3. Gated Residual Cross-Modal Fusion.
    4. Multi-Label Classification Head optimized via Multi-Label Asymmetric Loss (ASL).
    """
    def __init__(self, text_dim=768, vision_dim=768, embed_dim=256, num_heads=4, dropout=0.2, num_classes=NUM_CLASSES):
        super(RegionAwareCrossModalAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.region_proj = nn.Linear(vision_dim, embed_dim)
        self.global_v_proj = nn.Linear(vision_dim, embed_dim)
        
        # Multi-Head Cross-Attention (Q: text, K/V: image regions)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_attn = nn.LayerNorm(embed_dim)
        
        # Gating Mechanism to balance text vs visual evidence
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
        # Classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, batch):
        """
        Args:
            batch['text_feat']: (B, text_dim)
            batch['region_feat']: (B, num_regions, vision_dim)
            batch['vision_feat']: (B, vision_dim)
        Returns:
            logits: (B, num_classes)
        """
        text_f = batch["text_feat"]      # (B, text_dim)
        region_f = batch["region_feat"]  # (B, num_regions, vision_dim)
        global_v = batch["vision_feat"]  # (B, vision_dim)
        
        # Project to common embedding dimension
        q_text = self.text_proj(text_f).unsqueeze(1)    # (B, 1, embed_dim)
        kv_regions = self.region_proj(region_f)        # (B, num_regions, embed_dim)
        v_global = self.global_v_proj(global_v)        # (B, embed_dim)
        
        # Text attends to visual entity regions
        attn_out, attn_weights = self.cross_attn(query=q_text, key=kv_regions, value=kv_regions)
        attn_out = self.norm_attn(attn_out.squeeze(1) + q_text.squeeze(1))  # (B, embed_dim)
        
        # Dynamic modal gating
        gate = self.gate_net(torch.cat([attn_out, v_global], dim=-1))
        fused_visual = gate * attn_out + (1.0 - gate) * v_global
        
        # Final joint multimodal representation
        joint_repr = torch.cat([q_text.squeeze(1), fused_visual], dim=-1) # (B, embed_dim * 2)
        logits = self.classifier(joint_repr)
        
        return logits
