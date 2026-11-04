"""
asymmetric_loss.py
Asymmetric Loss (ASL) for multi-label classification.
Addresses extreme class imbalance by decoupling positive and negative focusing
and applying asymmetric probability margin clipping on negative labels.
Reference: Ridnik et al., "Asymmetric Loss For Multi-Label Classification" (ICCV 2021).
"""

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) with probability margin clipping.

    Args:
        gamma_neg: Focusing parameter for negative samples (gamma-). Higher values
                   suppress easy negatives more aggressively.
        gamma_pos: Focusing parameter for positive samples (gamma+). Typically <= gamma_neg.
        clip: Margin clipping parameter (m) for negatives. Probabilities below clip
              are zeroed out, discarding very easy negatives completely.
        eps: Small epsilon for numerical stability inside logarithm.
        reduction: Reduction method: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"Unsupported reduction '{reduction}'. Expected 'mean', 'sum', or 'none'."
            )
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            logits: Predicted logits of shape (N, C) or arbitrary shape.
            targets: Binary ground truth targets of matching shape.

        Returns:
            Reduced loss tensor or element-wise loss according to self.reduction.
        """
        if targets.dtype != logits.dtype:
            targets = targets.to(dtype=logits.dtype)

        # Probabilities
        p = torch.sigmoid(logits)

        # Margin clipping for negatives
        p_m = torch.clamp(p - self.clip, min=0.0)

        # Positive and negative loss components
        # L+ = y * (1 - p)^gamma_pos * log(p.clamp(min=eps))
        pos_loss = (
            targets
            * torch.pow(1.0 - p, self.gamma_pos)
            * torch.log(torch.clamp(p, min=self.eps))
        )

        # L- = (1 - y) * (p_m)^gamma_neg * log((1 - p_m).clamp(min=eps))
        neg_loss = (
            (1.0 - targets)
            * torch.pow(p_m, self.gamma_neg)
            * torch.log(torch.clamp(1.0 - p_m, min=self.eps))
        )

        loss = -(pos_loss + neg_loss)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")
