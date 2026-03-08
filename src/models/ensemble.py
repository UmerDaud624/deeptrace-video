"""
src/models/ensemble.py
-----------------------
Learned weighted fusion ensemble of EfficientNet-B4 and TALL-Swin-T.

Two fusion strategies are implemented:

    1. LogitFusionEnsemble (default, used during training):
        Takes raw logits from both models and combines them as:
            final_logit = w_eff * logit_eff + w_swin * logit_swin
        where w_eff and w_swin are learnable scalar parameters
        initialized from params.yaml (0.6 and 0.4 respectively).
        Both models are trained end-to-end jointly with the ensemble.

    2. FeatureFusionEnsemble (alternative, higher capacity):
        Concatenates feature vectors from both backbones:
            features = [feat_eff (1792) | feat_swin (768)] = 2560-dim
        Passes through a small MLP to produce final logit.
        Use this if LogitFusion underfits.

LogitFusion is the default because:
    - Fewer parameters, less risk of overfitting on FYP dataset size
    - Weights are directly interpretable (how much each stream contributes)
    - Directly matches the ensemble strategy in cited papers

Both models share the same input batch -- EfficientNet receives face crops
and Swin-T receives TALL grids. The trainer feeds both simultaneously.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs.config import cfg
from src.models.efficientnet import EfficientNetDetector
from src.models.tall_swin import TALLSwinDetector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logit Fusion Ensemble (default)
# ---------------------------------------------------------------------------

class LogitFusionEnsemble(nn.Module):
    """
    Weighted logit fusion of EfficientNet-B4 and TALL-Swin-T.

    Both submodels produce a (batch, 1) logit independently.
    A learnable weight for each stream is applied before summing.
    Weights are constrained to be positive via softplus.

    Args:
        efficientnet: Pre-built EfficientNetDetector instance.
        tall_swin:    Pre-built TALLSwinDetector instance.
        init_w_eff:   Initial weight for EfficientNet stream.
        init_w_swin:  Initial weight for Swin-T stream.

    Example:
        eff  = EfficientNetDetector(pretrained=False)
        swin = TALLSwinDetector(pretrained=False)
        model = LogitFusionEnsemble(eff, swin)

        face_batch = torch.randn(2, 3, 224, 224)
        tall_batch = torch.randn(2, 3, 224, 224)
        logits = model(face_batch, tall_batch)   # (2, 1)
    """

    def __init__(
        self,
        efficientnet : EfficientNetDetector,
        tall_swin    : TALLSwinDetector,
        init_w_eff   : float = None,
        init_w_swin  : float = None,
    ) -> None:
        super().__init__()

        ens_cfg     = cfg.ensemble
        init_w_eff  = init_w_eff  if init_w_eff  is not None else ens_cfg.init_weight_efficientnet
        init_w_swin = init_w_swin if init_w_swin is not None else ens_cfg.init_weight_swin

        self.efficientnet = efficientnet
        self.tall_swin    = tall_swin

        # Learnable fusion weights -- stored as raw values, softplus applied
        # in forward to keep them strictly positive
        self.w_eff  = nn.Parameter(torch.tensor(init_w_eff,  dtype=torch.float32))
        self.w_swin = nn.Parameter(torch.tensor(init_w_swin, dtype=torch.float32))

        logger.info(
            "LogitFusionEnsemble -- init weights: eff=%.2f, swin=%.2f",
            init_w_eff, init_w_swin,
        )

    def forward(
        self,
        face_input : torch.Tensor,
        tall_input : torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through both streams and weighted logit fusion.

        Args:
            face_input: Face crop batch (batch, 3, 224, 224) for EfficientNet.
            tall_input: TALL grid batch (batch, 3, 224, 224) for Swin-T.

        Returns:
            Fused logits of shape (batch, 1).
            Apply torch.sigmoid() to get final probability.
        """
        logit_eff  = self.efficientnet(face_input)   # (batch, 1)
        logit_swin = self.tall_swin(tall_input)       # (batch, 1)

        # Softplus ensures weights stay positive during training
        w_eff  = nn.functional.softplus(self.w_eff)
        w_swin = nn.functional.softplus(self.w_swin)

        # Normalize so weights sum to 1 for stable training
        w_sum  = w_eff + w_swin
        w_eff  = w_eff  / w_sum
        w_swin = w_swin / w_sum

        fused = w_eff * logit_eff + w_swin * logit_swin
        return fused

    def get_stream_probabilities(
        self,
        face_input : torch.Tensor,
        tall_input : torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Get individual stream probabilities alongside fused output.

        Used during evaluation to analyze each stream's contribution
        and identify which stream is more reliable per sample.

        Args:
            face_input: Face crop batch for EfficientNet.
            tall_input: TALL grid batch for Swin-T.

        Returns:
            Dict with keys:
                fused       -- final ensemble probability (batch, 1)
                efficientnet -- EfficientNet probability (batch, 1)
                swin         -- Swin-T probability (batch, 1)
                weight_eff   -- normalized EfficientNet weight (scalar)
                weight_swin  -- normalized Swin-T weight (scalar)
        """
        with torch.no_grad():
            logit_eff  = self.efficientnet(face_input)
            logit_swin = self.tall_swin(tall_input)

            w_eff  = nn.functional.softplus(self.w_eff)
            w_swin = nn.functional.softplus(self.w_swin)
            w_sum  = w_eff + w_swin
            w_eff  = w_eff  / w_sum
            w_swin = w_swin / w_sum

            fused = w_eff * logit_eff + w_swin * logit_swin

        return {
            "fused"        : torch.sigmoid(fused),
            "efficientnet" : torch.sigmoid(logit_eff),
            "swin"         : torch.sigmoid(logit_swin),
            "weight_eff"   : w_eff.item(),
            "weight_swin"  : w_swin.item(),
        }


# ---------------------------------------------------------------------------
# Feature Fusion Ensemble (alternative, higher capacity)
# ---------------------------------------------------------------------------

class FeatureFusionEnsemble(nn.Module):
    """
    Feature concatenation ensemble of EfficientNet-B4 and TALL-Swin-T.

    Extracts feature vectors from both backbones, concatenates them,
    and passes through a small MLP for final classification.

    Use this instead of LogitFusionEnsemble if logit fusion underfits,
    as this allows the classifier to learn cross-stream feature interactions.

    Args:
        efficientnet:  Pre-built EfficientNetDetector instance.
        tall_swin:     Pre-built TALLSwinDetector instance.
        hidden_dim:    Hidden layer size in the fusion MLP.
        dropout:       Dropout in the fusion MLP.
    """

    def __init__(
        self,
        efficientnet : EfficientNetDetector,
        tall_swin    : TALLSwinDetector,
        hidden_dim   : int   = 512,
        dropout      : float = 0.4,
    ) -> None:
        super().__init__()

        self.efficientnet = efficientnet
        self.tall_swin    = tall_swin

        eff_features  = efficientnet.backbone.num_features   # 1792
        swin_features = tall_swin.backbone.num_features      # 768
        combined_dim  = eff_features + swin_features         # 2560

        self.fusion_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Weight initialization
        for layer in self.fusion_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        logger.info(
            "FeatureFusionEnsemble -- combined_dim: %d -> hidden: %d -> 1",
            combined_dim, hidden_dim,
        )

    def forward(
        self,
        face_input : torch.Tensor,
        tall_input : torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            face_input: Face crop batch (batch, 3, 224, 224).
            tall_input: TALL grid batch (batch, 3, 224, 224).

        Returns:
            Logits of shape (batch, 1).
        """
        feat_eff  = self.efficientnet.get_feature_vector(face_input)  # (batch, 1792)
        feat_swin = self.tall_swin.get_feature_vector(tall_input)      # (batch, 768)

        combined = torch.cat([feat_eff, feat_swin], dim=1)            # (batch, 2560)
        logits   = self.fusion_head(combined)                          # (batch, 1)
        return logits


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

def build_ensemble(
    fusion_type          : str  = "logit",
    pretrained           : bool = None,
    eff_checkpoint_path  : str  = None,
    swin_checkpoint_path : str  = None,
) -> LogitFusionEnsemble | FeatureFusionEnsemble:
    """
    Build the full ensemble model.

    Args:
        fusion_type:          "logit" for LogitFusionEnsemble (default),
                              "feature" for FeatureFusionEnsemble.
        pretrained:           Whether to load ImageNet weights for backbones.
                              Defaults to cfg.efficientnet.pretrained.
        eff_checkpoint_path:  Optional path to EfficientNet checkpoint.
        swin_checkpoint_path: Optional path to TALL-Swin checkpoint.

    Returns:
        Fully assembled ensemble model ready for training or inference.

    Raises:
        ValueError: If fusion_type is not "logit" or "feature".
    """
    if pretrained is None:
        pretrained = cfg.efficientnet.pretrained

    efficientnet = EfficientNetDetector(pretrained=pretrained)
    tall_swin    = TALLSwinDetector(pretrained=pretrained)

    if eff_checkpoint_path is not None:
        path = Path(eff_checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(eff_checkpoint_path)
        state = torch.load(eff_checkpoint_path, map_location="cpu")
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        efficientnet.load_state_dict(state)
        logger.info("Loaded EfficientNet checkpoint: %s", eff_checkpoint_path)

    if swin_checkpoint_path is not None:
        path = Path(swin_checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(swin_checkpoint_path)
        state = torch.load(swin_checkpoint_path, map_location="cpu")
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        tall_swin.load_state_dict(state)
        logger.info("Loaded TALL-Swin checkpoint: %s", swin_checkpoint_path)

    if fusion_type == "logit":
        model = LogitFusionEnsemble(efficientnet, tall_swin)
    elif fusion_type == "feature":
        model = FeatureFusionEnsemble(efficientnet, tall_swin)
    else:
        raise ValueError(
            f"fusion_type must be 'logit' or 'feature', got: {fusion_type}"
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Ensemble built -- fusion: %s | total params: %s | trainable: %s",
        fusion_type,
        f"{total_params:,}",
        f"{trainable:,}",
    )

    return model