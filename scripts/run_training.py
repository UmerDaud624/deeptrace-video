"""
scripts/run_training.py
------------------------
Launches training of the DeepTrace dual-stream ensemble.

Run this on Google Colab after preprocessing is complete.
Loads train/val manifests, builds DataLoaders, trains ensemble,
saves checkpoints to Drive, logs everything to MLflow.

Usage:
    python scripts/run_training.py
    python scripts/run_training.py --fusion logit
    python scripts/run_training.py --fusion feature
    python scripts/run_training.py --run_name experiment_v2
    python scripts/run_training.py --resume checkpoints/ensemble_latest.pt

Expected runtime on Colab T4 (16GB):
    ~3-5 hours for 30 epochs depending on dataset size
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch

from configs.config import cfg, MANIFEST_ROOT, CHECKPOINT_ROOT
from src.data.dataset import build_dataloaders
from src.data.manifest import load_manifest, get_manifest_stats
from src.models.ensemble import build_ensemble
from src.training.trainer import Trainer, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepTrace ensemble training"
    )
    parser.add_argument(
        "--fusion",
        type=str,
        choices=["logit", "feature"],
        default="logit",
        help="Ensemble fusion strategy. Default: logit.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="MLflow run name. Auto-generated if not provided.",
    )
    parser.add_argument(
        "--manifest_dir",
        type=str,
        default=str(MANIFEST_ROOT),
        help="Directory containing train.csv and val.csv.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use ImageNet pretrained weights. Default: True.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("DeepTrace Training Pipeline")
    logger.info("=" * 60)
    logger.info("Fusion type  : %s", args.fusion)
    logger.info("Run name     : %s", args.run_name or "auto")
    logger.info("Manifest dir : %s", args.manifest_dir)
    logger.info("Resume from  : %s", args.resume or "None")
    logger.info("Device       : %s",
                "cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info("GPU          : %s", torch.cuda.get_device_name(0))
        logger.info("VRAM         : %.1f GB",
                    torch.cuda.get_device_properties(0).total_memory / 1e9)
    logger.info("=" * 60)

    cfg.validate()
    set_seed(cfg.training.seed)

    # --- Load Manifests ---
    manifest_dir = Path(args.manifest_dir)
    train_csv    = manifest_dir / "train.csv"
    val_csv      = manifest_dir / "val.csv"

    if not train_csv.exists():
        logger.error(
            "train.csv not found at %s. "
            "Run scripts/run_preprocessing.py first.", train_csv
        )
        sys.exit(1)

    if not val_csv.exists():
        logger.error(
            "val.csv not found at %s. "
            "Run scripts/run_preprocessing.py first.", val_csv
        )
        sys.exit(1)

    train_df = load_manifest(str(train_csv))
    val_df   = load_manifest(str(val_csv))

    logger.info("Train set stats: %s", get_manifest_stats(train_df))
    logger.info("Val set stats  : %s", get_manifest_stats(val_df))

    # Dummy test_df for build_dataloaders (not used during training)
    test_df = val_df.copy()

    # --- Build DataLoaders ---
    logger.info("Building DataLoaders...")
    loaders = build_dataloaders(train_df, val_df, test_df)

    face_loaders = {
        "train" : loaders["face"]["train"],
        "val"   : loaders["face"]["val"],
    }
    tall_loaders = {
        "train" : loaders["tall"]["train"],
        "val"   : loaders["tall"]["val"],
    }

    logger.info(
        "Train batches: %d | Val batches: %d",
        len(face_loaders["train"]),
        len(face_loaders["val"]),
    )

    # --- Build Model ---
    logger.info("Building ensemble model (fusion=%s)...", args.fusion)
    model = build_ensemble(
        fusion_type=args.fusion,
        pretrained=args.pretrained,
    )

    # Resume from checkpoint if specified
    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            logger.error("Resume checkpoint not found: %s", args.resume)
            sys.exit(1)

        checkpoint = torch.load(args.resume, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state)
        start_epoch = checkpoint.get("epoch", 0)
        logger.info(
            "Resumed from checkpoint: %s (epoch %d)",
            args.resume, start_epoch
        )

    # --- Train ---
    run_name = args.run_name or f"deeptrace_{args.fusion}_{int(time.time())}"

    trainer = Trainer(
        model=model,
        face_loaders=face_loaders,
        tall_loaders=tall_loaders,
        run_name=run_name,
    )

    logger.info("Starting training run: %s", run_name)
    best_metrics = trainer.train()

    logger.info("=" * 60)
    logger.info("Training complete. Best metrics:")
    for k, v in best_metrics.items():
        logger.info("  %s: %s", k, v)
    logger.info("Best checkpoint: %s", trainer.ckpt_manager.best_checkpoint())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()