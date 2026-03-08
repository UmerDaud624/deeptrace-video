"""
configs/config.py
-----------------
Single source of truth for all hyperparameters, paths, and settings.
All other modules import from here. Never hardcode values elsewhere.

Parameters are loaded from params.yaml at the project root.
Changing params.yaml is the only place you need to update hyperparameters.

Environment detection:
    - If /content/drive exists --> running on Colab, use Drive paths
    - Otherwise               --> running locally, use relative project paths
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import yaml


# ---------------------------------------------------------------------------
# Load params.yaml
# ---------------------------------------------------------------------------

def _load_params() -> dict:
    """
    Load params.yaml from project root.

    Searches from configs/ directory up one level to find params.yaml.

    Returns:
        Dict of all parameters.

    Raises:
        FileNotFoundError: If params.yaml not found at project root.
    """
    config_dir = Path(__file__).resolve().parent
    params_path = config_dir.parent / "params.yaml"

    if not params_path.exists():
        raise FileNotFoundError(
            f"params.yaml not found at {params_path}. "
            "Make sure it exists in the project root."
        )

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    return params


# Loaded once at import time
_P = _load_params()


# ---------------------------------------------------------------------------
# Environment Detection
# ---------------------------------------------------------------------------

def _on_colab() -> bool:
    """Detect if running on Google Colab."""
    return Path("/content/drive").exists()


# ---------------------------------------------------------------------------
# Path Resolution
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if _on_colab():
    DRIVE_ROOT        = Path("/content/drive/MyDrive/DeepTrace")
    DATASET_ROOT      = DRIVE_ROOT / "datasets"
    CHECKPOINT_ROOT   = DRIVE_ROOT / "video" / "checkpoints"
    MLFLOW_ROOT       = DRIVE_ROOT / "video" / "mlruns"
    PREPROCESSED_ROOT = DRIVE_ROOT / "video" / "preprocessed"
    MANIFEST_ROOT     = DRIVE_ROOT / "video" / "manifests"
else:
    DATASET_ROOT      = PROJECT_ROOT / "data" / "datasets"
    CHECKPOINT_ROOT   = PROJECT_ROOT / "data" / "models"
    MLFLOW_ROOT       = PROJECT_ROOT / "data" / "mlruns"
    PREPROCESSED_ROOT = PROJECT_ROOT / "data" / "preprocessed"
    MANIFEST_ROOT     = PROJECT_ROOT / "data" / "manifests"


# ---------------------------------------------------------------------------
# Dataset Paths
# ---------------------------------------------------------------------------

@dataclass
class DatasetPaths:
    """Paths to each dataset. Resolves correctly on both Colab and local."""

    ff_plus_plus : Path = field(
        default_factory=lambda: DATASET_ROOT / "FaceForensics"
    )
    celeb_df     : Path = field(
        default_factory=lambda: DATASET_ROOT / "CelebDF_v2"
    )
    dfdc         : Path = field(
        default_factory=lambda: DATASET_ROOT / "DFDC"
    )

    @property
    def ff_original(self) -> Path:
        return self.ff_plus_plus / "original_sequences/youtube/c23/videos"

    @property
    def ff_deepfakes(self) -> Path:
        return self.ff_plus_plus / "manipulated_sequences/Deepfakes/c23/videos"

    @property
    def ff_face2face(self) -> Path:
        return self.ff_plus_plus / "manipulated_sequences/Face2Face/c23/videos"

    @property
    def ff_faceswap(self) -> Path:
        return self.ff_plus_plus / "manipulated_sequences/FaceSwap/c23/videos"

    @property
    def ff_neural_textures(self) -> Path:
        return (
            self.ff_plus_plus
            / "manipulated_sequences/NeuralTextures/c23/videos"
        )


# ---------------------------------------------------------------------------
# Preprocessing Config
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingConfig:
    """Controls face extraction and frame sampling. Values from params.yaml."""

    frames_per_video : int   = field(
        default_factory=lambda: _P["preprocessing"]["frames_per_video"]
    )
    tall_grid_size   : int   = field(
        default_factory=lambda: _P["preprocessing"]["tall_grid_size"]
    )
    tall_grid_rows   : int   = field(
        default_factory=lambda: _P["preprocessing"]["tall_grid_rows"]
    )
    tall_grid_cols   : int   = field(
        default_factory=lambda: _P["preprocessing"]["tall_grid_cols"]
    )
    face_size        : int   = field(
        default_factory=lambda: _P["preprocessing"]["face_size"]
    )
    tall_face_size   : int   = field(
        default_factory=lambda: _P["preprocessing"]["tall_face_size"]
    )
    face_confidence  : float = field(
        default_factory=lambda: _P["preprocessing"]["face_confidence"]
    )
    face_margin      : float = field(
        default_factory=lambda: _P["preprocessing"]["face_margin"]
    )
    jpg_quality      : int   = field(
        default_factory=lambda: _P["preprocessing"]["jpg_quality"]
    )


# ---------------------------------------------------------------------------
# Split Config
# ---------------------------------------------------------------------------

@dataclass
class SplitConfig:
    """Train/val/test split following FF++ official protocol."""

    train_size  : float = field(
        default_factory=lambda: _P["split"]["train_size"]
    )
    val_size    : float = field(
        default_factory=lambda: _P["split"]["val_size"]
    )
    test_size   : float = field(
        default_factory=lambda: _P["split"]["test_size"]
    )
    random_seed : int   = field(
        default_factory=lambda: _P["split"]["random_seed"]
    )


# ---------------------------------------------------------------------------
# EfficientNet-B4 Config
# ---------------------------------------------------------------------------

@dataclass
class EfficientNetConfig:
    """Spatial deepfake detector based on EfficientNet-B4 with GeM pooling."""

    model_name  : str   = field(
        default_factory=lambda: _P["efficientnet"]["model_name"]
    )
    pretrained  : bool  = field(
        default_factory=lambda: _P["efficientnet"]["pretrained"]
    )
    num_classes : int   = field(
        default_factory=lambda: _P["efficientnet"]["num_classes"]
    )
    dropout     : float = field(
        default_factory=lambda: _P["efficientnet"]["dropout"]
    )
    input_size  : int   = field(
        default_factory=lambda: _P["efficientnet"]["input_size"]
    )
    gem_p       : float = field(
        default_factory=lambda: _P["efficientnet"]["gem_p"]
    )


# ---------------------------------------------------------------------------
# TALL + Swin-T Config
# ---------------------------------------------------------------------------

@dataclass
class TALLSwinConfig:
    """Temporal detector using TALL thumbnail grid + Swin Transformer."""

    model_name  : str   = field(
        default_factory=lambda: _P["tall_swin"]["model_name"]
    )
    pretrained  : bool  = field(
        default_factory=lambda: _P["tall_swin"]["pretrained"]
    )
    num_classes : int   = field(
        default_factory=lambda: _P["tall_swin"]["num_classes"]
    )
    dropout     : float = field(
        default_factory=lambda: _P["tall_swin"]["dropout"]
    )
    input_size  : int   = field(
        default_factory=lambda: _P["tall_swin"]["input_size"]
    )


# ---------------------------------------------------------------------------
# Ensemble Config
# ---------------------------------------------------------------------------

@dataclass
class EnsembleConfig:
    """Learned weighted fusion of EfficientNet and Swin-T."""

    init_weight_efficientnet : float = field(
        default_factory=lambda: _P["ensemble"]["init_weight_efficientnet"]
    )
    init_weight_swin         : float = field(
        default_factory=lambda: _P["ensemble"]["init_weight_swin"]
    )
    num_classes              : int   = field(
        default_factory=lambda: _P["ensemble"]["num_classes"]
    )


# ---------------------------------------------------------------------------
# Training Config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Training loop hyperparameters."""

    seed            : int   = field(
        default_factory=lambda: _P["training"]["seed"]
    )
    num_epochs      : int   = field(
        default_factory=lambda: _P["training"]["num_epochs"]
    )
    batch_size      : int   = field(
        default_factory=lambda: _P["training"]["batch_size"]
    )
    num_workers     : int   = field(
        default_factory=lambda: _P["training"]["num_workers"]
    )
    pin_memory      : bool  = field(
        default_factory=lambda: _P["training"]["pin_memory"]
    )
    mixed_precision : bool  = field(
        default_factory=lambda: _P["training"]["mixed_precision"]
    )
    optimizer       : str   = field(
        default_factory=lambda: _P["training"]["optimizer"]
    )
    learning_rate   : float = field(
        default_factory=lambda: _P["training"]["learning_rate"]
    )
    weight_decay    : float = field(
        default_factory=lambda: _P["training"]["weight_decay"]
    )
    scheduler       : str   = field(
        default_factory=lambda: _P["training"]["scheduler"]
    )
    warmup_epochs   : int   = field(
        default_factory=lambda: _P["training"]["warmup_epochs"]
    )
    min_lr          : float = field(
        default_factory=lambda: _P["training"]["min_lr"]
    )
    focal_alpha     : float = field(
        default_factory=lambda: _P["training"]["focal_alpha"]
    )
    focal_gamma     : float = field(
        default_factory=lambda: _P["training"]["focal_gamma"]
    )
    patience        : int   = field(
        default_factory=lambda: _P["training"]["patience"]
    )
    monitor         : str   = field(
        default_factory=lambda: _P["training"]["monitor"]
    )
    save_top_k      : int   = field(
        default_factory=lambda: _P["training"]["save_top_k"]
    )
    grad_clip       : float = field(
        default_factory=lambda: _P["training"]["grad_clip"]
    )


# ---------------------------------------------------------------------------
# Evaluation Config
# ---------------------------------------------------------------------------

@dataclass
class EvaluationConfig:
    """Evaluation metrics and thresholds."""

    primary_metric     : str   = field(
        default_factory=lambda: _P["evaluation"]["primary_metric"]
    )
    decision_threshold : float = field(
        default_factory=lambda: _P["evaluation"]["decision_threshold"]
    )
    cross_dataset_test : str   = field(
        default_factory=lambda: _P["evaluation"]["cross_dataset_test"]
    )


# ---------------------------------------------------------------------------
# MLflow Config
# ---------------------------------------------------------------------------

@dataclass
class MLflowConfig:
    """MLflow experiment tracking."""

    tracking_uri    : str  = str(MLFLOW_ROOT)
    experiment_name : str  = "DeepTrace-Video"
    run_tags        : dict = field(default_factory=lambda: {
        "project"   : "DeepTrace",
        "modality"  : "video",
        "framework" : "pytorch",
    })


# ---------------------------------------------------------------------------
# Master Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """
    Master config object. Import cfg from this module everywhere.

    Example:
        from configs.config import cfg
        lr = cfg.training.learning_rate
        ff_path = cfg.paths.ff_original
    """

    paths         : DatasetPaths        = field(default_factory=DatasetPaths)
    preprocessing : PreprocessingConfig = field(default_factory=PreprocessingConfig)
    split         : SplitConfig         = field(default_factory=SplitConfig)
    efficientnet  : EfficientNetConfig  = field(default_factory=EfficientNetConfig)
    tall_swin     : TALLSwinConfig      = field(default_factory=TALLSwinConfig)
    ensemble      : EnsembleConfig      = field(default_factory=EnsembleConfig)
    training      : TrainingConfig      = field(default_factory=TrainingConfig)
    evaluation    : EvaluationConfig    = field(default_factory=EvaluationConfig)
    mlflow        : MLflowConfig        = field(default_factory=MLflowConfig)

    def validate(self) -> None:
        """Validate config consistency. Call before any training run."""
        assert (
            self.preprocessing.tall_grid_rows
            * self.preprocessing.tall_grid_cols
            == self.preprocessing.tall_grid_size
        ), "tall_grid_rows * tall_grid_cols must equal tall_grid_size"

        assert abs(
            self.split.train_size
            + self.split.val_size
            + self.split.test_size
            - 1.0
        ) < 1e-6, "Split sizes must sum to 1.0"

        assert self.training.batch_size > 0, "Batch size must be positive"
        assert 0.0 < self.training.learning_rate < 1.0, "LR out of range"

    def make_dirs(self) -> None:
        """Create all output directories."""
        for d in [
            PREPROCESSED_ROOT,
            MANIFEST_ROOT,
            CHECKPOINT_ROOT,
            MLFLOW_ROOT,
            PROJECT_ROOT / "data" / "reports",
            PROJECT_ROOT / "data" / "metrics",
        ]:
            d.mkdir(parents=True, exist_ok=True)


# Module-level singleton -- import this everywhere
cfg = Config()