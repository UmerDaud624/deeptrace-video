"""
src/training/losses.py
-----------------------
Loss functions for the DeepTrace video deepfake detection pipeline.

Focal Loss is used instead of standard Binary Cross Entropy because:
    - The real/fake class imbalance in DFDC (20699 real vs 3802 fake)
      causes BCE to bias toward predicting the majority class
    - Focal loss down-weights easy examples (confident correct predictions)
      and focuses training on hard examples (ambiguous or misclassified)
    - This is particularly important for deepfake detection where
      high-quality fakes are the hard examples that matter most

Focal Loss formula:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where:
        p_t   = sigmoid(logit) if label=1 else 1 - sigmoid(logit)
        alpha = class balancing factor (default 0.25 from params.yaml)
        gamma = focusing parameter (default 2.0 from params.yaml)

    At gamma=0 this reduces to weighted BCE.
    At gamma=2 hard examples receive 4x more weight than easy ones.

Reference:
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs.config import cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Binary Focal Loss for single-output deepfake detection.

    Expects raw logits as input (before sigmoid).
    Labels should be float tensors of 0.0 or 1.0.

    Args:
        alpha: Weighting factor for the rare (positive/fake) class.
               Set between 0 and 1. Default 0.25 from params.yaml.
        gamma: Focusing parameter. Higher values focus more on hard
               examples. Default 2.0 from params.yaml.
        reduction: "mean" averages loss over batch.
                   "sum" sums over batch.
                   "none" returns per-sample loss tensor.

    Example:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        loss = criterion(logits, labels)
    """

    def __init__(
        self,
        alpha     : float = None,
        gamma     : float = None,
        reduction : str   = "mean",
    ) -> None:
        super().__init__()

        train_cfg = cfg.training
        self.alpha     = alpha if alpha is not None else train_cfg.focal_alpha
        self.gamma     = gamma if gamma is not None else train_cfg.focal_gamma
        self.reduction = reduction

        assert reduction in ("mean", "sum", "none"), (
            f"reduction must be 'mean', 'sum', or 'none', got: {reduction}"
        )

        logger.info(
            "FocalLoss -- alpha: %.3f | gamma: %.1f | reduction: %s",
            self.alpha, self.gamma, self.reduction,
        )

    def forward(
        self,
        logits : torch.Tensor,
        labels : torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Raw model output of shape (batch, 1) or (batch,).
                    Must be before sigmoid.
            labels: Ground truth of shape (batch, 1) or (batch,).
                    Float tensor with values 0.0 or 1.0.

        Returns:
            Scalar loss tensor if reduction is "mean" or "sum".
            Per-sample tensor of shape (batch,) if reduction is "none".
        """
        # Flatten to (batch,)
        logits = logits.view(-1)
        labels = labels.view(-1).float()

        # Numerically stable BCE per sample
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )

        # p_t = probability of the true class
        probs = torch.sigmoid(logits)
        p_t   = probs * labels + (1.0 - probs) * (1.0 - labels)

        # alpha_t = alpha for positive class, (1-alpha) for negative class
        alpha_t = self.alpha * labels + (1.0 - self.alpha) * (1.0 - labels)

        # Focal weight
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma

        focal_loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

    def __repr__(self) -> str:
        return (
            f"FocalLoss(alpha={self.alpha}, gamma={self.gamma}, "
            f"reduction={self.reduction})"
        )


# ---------------------------------------------------------------------------
# Loss Factory
# ---------------------------------------------------------------------------

def build_loss(
    alpha     : float = None,
    gamma     : float = None,
    reduction : str   = "mean",
) -> FocalLoss:
    """
    Build the focal loss criterion.

    Args:
        alpha:     Class weighting factor. Defaults to params.yaml value.
        gamma:     Focusing exponent. Defaults to params.yaml value.
        reduction: Batch reduction strategy.

    Returns:
        FocalLoss instance.
    """
    return FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)