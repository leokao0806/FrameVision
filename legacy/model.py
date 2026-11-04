"""
model.py
Multimodal Framing Architecture with Region-Aware Cross-Attention & Asymmetric Loss.
Designed for SemEval-2027 Task 9 Subtask 2 (Framing in Images).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for Multi-Label Classification under severe class imbalance.
    Paper: 'Asymmetric Loss For Multi-Label Classification' (ICCV 2021)
    Dynamically downweights easy negative examples using asymmetric focusing parameters and margin.
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (batch_size, num_classes) - raw unnormalized logits
        targets: (batch_size, num_classes) - binary 0/1 labels
        """
        probs = torch.sigmoid(logits)

        # Positive loss
        pos_loss = targets * torch.log(probs.clamp(min=self.eps)) * ((1.0 - probs) ** self.gamma_pos)

        # Negative loss with probability shifting / margin clipping
        if self.clip > 0:
            probs_neg = (probs - self.clip).clamp(min=0.0)
        else:
            probs_neg = probs
        neg_loss = (1.0 - targets) * torch.log((1.0 - probs_neg).clamp(min=self.eps)) * (probs_neg ** self.gamma_neg)

        loss = - (pos_loss + neg_loss)
        return loss.sum(dim=-1).mean()


class CrossModalAttentionBlock(nn.Module):
    """
    Cross-Attention mechanism where Text actively queries Global & Local Visual cues.
    Query: Text Token Features
    Key/Value: Concatenated Visual Features (Global Image + SAM Local Crop)
    """

    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, text_feats: torch.Tensor, visual_feats: torch.Tensor) -> torch.Tensor:
        """
        text_feats: (B, L_t, D)
        visual_feats: (B, L_v, D) - concatenated global + local visual tokens
        """
        # Cross-Attention: Text queries Visual elements
        attn_out, attn_weights = self.cross_attn(query=text_feats, key=visual_feats, value=visual_feats)
        x = self.norm1(text_feats + attn_out)

        # Feed-forward residual
        out = self.norm2(x + self.ffn(x))
        return out, attn_weights


class MultimodalFramingModel(nn.Module):
    """
    Full End-to-End Multimodal Framing Network.
    - Encodes raw news text into contextual token embeddings.
    - Encodes full image and SAM-cropped local entity into visual tokens.
    - Fuses via Cross-Attention and projects to 14 framing classes.
    """

    def __init__(
        self,
        num_classes: int = 14,
        embed_dim: int = 512,
        num_heads: int = 8,
        use_pretrained_backbone: bool = False,
        backbone_name: str = "google/siglip-base-patch16-224",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_pretrained_backbone = use_pretrained_backbone

        if use_pretrained_backbone:
            # Load pre-trained HuggingFace vision and text models if available
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
            self.backbone = AutoModel.from_pretrained(backbone_name)
            hidden_size = self.backbone.config.hidden_size if hasattr(self.backbone.config, 'hidden_size') else 768
            self.proj_text = nn.Linear(hidden_size, embed_dim)
            self.proj_vis = nn.Linear(hidden_size, embed_dim)
        else:
            # Lightweight modular encoders for fast local execution on MacBook (MPS/CPU)
            # 1. Text Encoder (Word embedding + 2-layer Bidirectional GRU)
            self.vocab_size = 30000
            self.word_emb = nn.Embedding(self.vocab_size, 256, padding_idx=0)
            self.text_encoder = nn.GRU(256, embed_dim // 2, num_layers=2, batch_first=True, bidirectional=True)

            # 2. Vision Encoder (Lightweight CNN feature extractor for full image and crop)
            self.vis_encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),

                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),

                nn.Conv2d(256, embed_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
            )

        # Cross-Modal Attention Fusion
        self.fusion = CrossModalAttentionBlock(embed_dim=embed_dim, num_heads=num_heads)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def encode_text_simple(self, text_list, device):
        """Simple hash tokenizer for lightweight local execution"""
        max_len = 128
        batch_tokens = []
        for text in text_list:
            words = text.lower().split()[:max_len]
            tokens = [(abs(hash(w)) % (self.vocab_size - 1)) + 1 for w in words]
            if len(tokens) < max_len:
                tokens += [0] * (max_len - len(tokens))
            batch_tokens.append(tokens)
        token_tensor = torch.tensor(batch_tokens, dtype=torch.long, device=device)
        emb = self.word_emb(token_tensor)
        out, _ = self.text_encoder(emb)
        return out  # (B, L_t, embed_dim)

    def forward(self, text_list, full_images, crop_images):
        """
        text_list: list of strings (B,)
        full_images: (B, 3, 224, 224)
        crop_images: (B, 3, 224, 224) - SAM salient entity crops
        """
        device = full_images.device

        # 1. Text representations: (B, L_t, D)
        text_feats = self.encode_text_simple(text_list, device)

        # 2. Visual representations
        # Extract features for both full scene and localized SAM crop
        feat_global = self.vis_encoder(full_images)  # (B, D, H, W)
        feat_crop = self.vis_encoder(crop_images)    # (B, D, H, W)

        B, D, H, W = feat_global.shape
        # Flatten spatial dimensions into token sequence: (B, H*W, D)
        tok_global = feat_global.flatten(2).permute(0, 2, 1)
        tok_crop = feat_crop.flatten(2).permute(0, 2, 1)

        # Concatenate global scene + local entity tokens: (B, 2*H*W, D)
        visual_tokens = torch.cat([tok_global, tok_crop], dim=1)

        # 3. Cross-Modal Fusion: Text queries Visual cues
        fused_tokens, attn_weights = self.fusion(text_feats, visual_tokens)

        # Pool across sequence dimension (Mean pooling over text queries)
        pooled = fused_tokens.mean(dim=1)  # (B, D)

        # 4. Multi-Label Classification Logits
        logits = self.classifier(pooled)   # (B, 14)

        return logits, attn_weights
