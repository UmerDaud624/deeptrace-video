"""
src/models/tall_swin.py
------------------------
Swin Transformer based temporal deepfake detector using TALL grids.

Architecture:
    1. Swin Transformer backbone pretrained on ImageNet (via timm)
    2. Standard average pooling (Swin already has efficient attention pooling)
    3. Dropout for regularization
    4. Single linear output for binary classification

Why Swin Transformer for TALL grids:
    The TALL grid tiles multiple video frames into a single image arranged
    in a spatial grid. Swin Transformer processes images through shifted
    window attention, where each window naturally corresponds to one tile
    in the TALL grid. This means the model attends within individual frames
    (local artifact detection) and across frame boundaries (temporal
    inconsistency detection) without any architectural modification.
    This is the exact setup validated in:
        Xu et al., "TALL: Thumbnail Layout for Deepfake Video Detection",
        ICCV 2023.

Input:  (batch, 3, 224, 224) TALL grid -- 4x4 tiled face crops
Output: (batch, 1) raw logit -- apply sigmoid for probability
"""

from __future__ import annotations

import logging
from pathlib import Path

import timm
import torch
import torch.nn as nn

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs.config import cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TALL Swin Detector
# ---------------------------------------------------------------------------

class TALLSwinDetector(nn.Module):
    """
    Swin Transformer temporal deepfake detector operating on TALL grids.

    The classifier head is replaced with:
        AdaptiveAvgPool -> Flatten -> Dropout -> Linear(num_features, 1)

    All backbone weights are unfrozen for end-to-end fine-tuning.
    The backbone is initialized from ImageNet pretrained weights via timm.

    Args:
        model_name:  timm model string. Default from params.yaml.
        pretrained:  Load ImageNet weights if True.
        num_classes: Output dimension. Always 1 for binary detection.
        dropout:     Dropout probability before classifier head.

    Example:
        model = TALLSwinDetector()
        logits = model(batch_tensor)   # shape (batch, 1)
        probs  = torch.sigmoid(logits) # shape (batch, 1)
    """

    def __init__(
        self,
        model_name  : str   = None,
        pretrained  : bool  = None,
        num_classes : int   = None,
        dropout     : float = None,
    ) -> None:
        super().__init__()

        swin_cfg    = cfg.tall_swin
        model_name  = model_name  or swin_cfg.model_name
        pretrained  = pretrained  if pretrained  is not None else swin_cfg.pretrained
        num_classes = num_classes or swin_cfg.num_classes
        dropout     = dropout     if dropout     is not None else swin_cfg.dropout

        # Load Swin backbone without classification head
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,      # removes classifier head
            global_pool="",     # removes default pooling, we handle it
        )

        # Swin outputs (batch, H, W, C) or (batch, seq_len, C) depending
        # on version. Use timm's num_features to get correct channel count.
        num_features = self.backbone.num_features

        self.pool       = nn.AdaptiveAvgPool1d(1)
        self.dropout    = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(num_features, num_classes)

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        logger.info(
            "TALLSwinDetector -- backbone: %s | features: %d | "
            "pretrained: %s | dropout: %.2f",
            model_name, num_features, pretrained, dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, H, W).
               Should be a TALL grid with tiled face crops.

        Returns:
            Raw logits of shape (batch, 1).
            Apply torch.sigmoid() to get probabilities.
        """
        features = self.backbone(x)   # (batch, seq_len, C) or (batch, H, W, C)

        # Normalize to (batch, C, seq_len) for AdaptiveAvgPool1d
        if features.dim() == 4:
            # (batch, H, W, C) -> (batch, C, H*W)
            b, h, w, c = features.shape
            features = features.reshape(b, h * w, c)

        # (batch, seq_len, C) -> (batch, C, seq_len)
        features = features.permute(0, 2, 1)

        pooled  = self.pool(features).squeeze(-1)   # (batch, C)
        dropped = self.dropout(pooled)
        logits  = self.classifier(dropped)          # (batch, 1)
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

        if features.dim() == 4:
            b, h, w, c = features.shape
            features = features.reshape(b, h * w, c)

        features = features.permute(0, 2, 1)
        pooled   = self.pool(features).squeeze(-1)
        return pooled


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

def build_tall_swin(
    checkpoint_path : str = None,
) -> TALLSwinDetector:
    """
    Build TALLSwinDetector and optionally load a checkpoint.

    Args:
        checkpoint_path: Path to a saved .pt checkpoint file.
                         If None, returns freshly initialized model.

    Returns:
        TALLSwinDetector instance.

    Raises:
        FileNotFoundError: If checkpoint_path provided but does not exist.
    """
    model = TALLSwinDetector()

    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )
        state = torch.load(checkpoint_path, map_location="cpu")

        if "model_state_dict" in state:
            state = state["model_state_dict"]

        model.load_state_dict(state)
        logger.info("Loaded TALL-Swin checkpoint: %s", checkpoint_path)

    return model