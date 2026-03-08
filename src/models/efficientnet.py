"""
src/models/efficientnet.py
---------------------------
EfficientNet-B4 based spatial deepfake detector.

Architecture:
    1. EfficientNet-B4 backbone pretrained on ImageNet (via timm)
    2. Generalized Mean (GeM) pooling replaces default average pooling
    3. Dropout for regularization
    4. Single linear output for binary classification

Why GeM pooling over standard average pooling:
    Average pooling treats all spatial locations equally.
    GeM pooling raises each activation to power p before averaging,
    which emphasizes the most discriminative spatial regions --
    particularly useful for detecting localized deepfake artifacts
    such as blending boundaries and texture inconsistencies that
    occupy a small portion of the face image.

Input:  (batch, 3, 224, 224) normalized RGB face crop
Output: (batch, 1) raw logit -- apply sigmoid for probability
"""

from __future__ import annotations

import logging
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs.config import cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GeM Pooling
# ---------------------------------------------------------------------------

class GeMPooling(nn.Module):
    """
    Generalized Mean Pooling.

    Computes pooled feature maps as:
        GeM(x) = ( mean(x^p) ) ^ (1/p)

    At p=1 this is standard average pooling.
    At p -> inf this approaches max pooling.
    Optimal p is typically learned or set to 3.0.

    Args:
        p:   Initial pooling exponent. Trainable if learn_p is True.
        eps: Small value to avoid numerical instability with negative
             activations before exponentiation.
        learn_p: If True, p is a trainable parameter. If False, fixed.

    Reference:
        Radenovic et al., "Fine-tuning CNN Image Retrieval with No
        Human Annotation", TPAMI 2019.
    """

    def __init__(
        self,
        p       : float = 3.0,
        eps     : float = 1e-6,
        learn_p : bool  = True,
    ) -> None:
        super().__init__()
        if learn_p:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = torch.tensor(p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map of shape (batch, channels, height, width).

        Returns:
            Pooled tensor of shape (batch, channels, 1, 1).
        """
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            kernel_size=(x.size(-2), x.size(-1)),
        ).pow(1.0 / self.p)


# ---------------------------------------------------------------------------
# EfficientNet-B4 Detector
# ---------------------------------------------------------------------------

class EfficientNetDetector(nn.Module):
    """
    EfficientNet-B4 spatial deepfake detector with GeM pooling.

    The classifier head is replaced with:
        GeM pooling -> Flatten -> Dropout -> Linear(num_features, 1)

    All backbone weights are unfrozen for end-to-end fine-tuning.
    The backbone is initialized from ImageNet pretrained weights via timm.

    Args:
        model_name:  timm model string. Default from params.yaml.
        pretrained:  Load ImageNet weights if True.
        num_classes: Output dimension. Always 1 for binary detection.
        dropout:     Dropout probability before classifier head.
        gem_p:       Initial GeM pooling exponent.

    Example:
        model = EfficientNetDetector()
        logits = model(batch_tensor)   # shape (batch, 1)
        probs  = torch.sigmoid(logits) # shape (batch, 1)
    """

    def __init__(
        self,
        model_name  : str   = None,
        pretrained  : bool  = None,
        num_classes : int   = None,
        dropout     : float = None,
        gem_p       : float = None,
    ) -> None:
        super().__init__()

        # Fall back to params.yaml values if not explicitly passed
        eff_cfg     = cfg.efficientnet
        model_name  = model_name  or eff_cfg.model_name
        pretrained  = pretrained  if pretrained  is not None else eff_cfg.pretrained
        num_classes = num_classes or eff_cfg.num_classes
        dropout     = dropout     if dropout     is not None else eff_cfg.dropout
        gem_p       = gem_p       if gem_p       is not None else eff_cfg.gem_p

        # Load backbone without classification head
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,        # removes classifier head
            global_pool="",       # removes default pooling
        )

        num_features = self.backbone.num_features

        # Custom head
        self.pool       = GeMPooling(p=gem_p, learn_p=True)
        self.flatten    = nn.Flatten()
        self.dropout    = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(num_features, num_classes)

        # Weight initialization for classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        logger.info(
            "EfficientNetDetector -- backbone: %s | features: %d | "
            "pretrained: %s | dropout: %.2f | gem_p: %.1f",
            model_name, num_features, pretrained, dropout, gem_p,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, H, W).

        Returns:
            Raw logits of shape (batch, 1).
            Apply torch.sigmoid() to get probabilities.
        """
        features = self.backbone(x)       # (batch, num_features, h, w)
        pooled   = self.pool(features)    # (batch, num_features, 1, 1)
        flat     = self.flatten(pooled)   # (batch, num_features)
        dropped  = self.dropout(flat)     # (batch, num_features)
        logits   = self.classifier(dropped)  # (batch, 1)
        return logits

    def get_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature vector before the classifier head.

        Used by the ensemble model to get embeddings from both
        streams before final fusion.

        Args:
            x: Input tensor of shape (batch, 3, H, W).

        Returns:
            Feature vector of shape (batch, num_features).
        """
        features = self.backbone(x)
        pooled   = self.pool(features)
        flat     = self.flatten(pooled)
        return flat


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

def build_efficientnet(
    checkpoint_path : str = None,
) -> EfficientNetDetector:
    """
    Build EfficientNetDetector and optionally load a checkpoint.

    Args:
        checkpoint_path: Path to a saved .pt checkpoint file.
                         If None, returns freshly initialized model.

    Returns:
        EfficientNetDetector instance.

    Raises:
        FileNotFoundError: If checkpoint_path provided but does not exist.
    """
    model = EfficientNetDetector()

    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )
        state = torch.load(checkpoint_path, map_location="cpu")

        # Handle both raw state dict and wrapped checkpoint dicts
        if "model_state_dict" in state:
            state = state["model_state_dict"]

        model.load_state_dict(state)
        logger.info("Loaded EfficientNet checkpoint: %s", checkpoint_path)

    return model