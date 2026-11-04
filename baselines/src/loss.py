"""
Multi-Label Loss Functions for Severe Class Imbalance
CS 5134/6034 NLP Fall 2026 - Team FrameVision
Authors: Leo Kao, En-Shuo Zhang

Implements:
1. Multi-Label Asymmetric Loss (ASL) (Ridnik et al., ICCV 2021)
2. Class-Weighted Binary Cross-Entropy Loss with Logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for multi-label classification.
    Dynamically downweights easy negatives while preserving focus on hard positives.
    Critical for severe class imbalance in the 14-frame taxonomy.
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        """
        logits: (N, C) unnormalized model outputs
        targets: (N, C) binary labels {0, 1}
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        # Asymmetric Clipping for negatives
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Basic CE calculation
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1.0 - targets)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            loss = loss * one_sided_w

        return -loss.mean()


class WeightedBCELoss(nn.Module):
    """
    Binary Cross Entropy with positive weight balancing.
    """
    def __init__(self, pos_weights=None):
        super(WeightedBCELoss, self).__init__()
        if pos_weights is not None:
            self.pos_weight = torch.tensor(pos_weights, dtype=torch.float32)
        else:
            self.pos_weight = None

    def forward(self, logits, targets):
        if self.pos_weight is not None and self.pos_weight.device != logits.device:
            self.pos_weight = self.pos_weight.to(logits.device)
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )
