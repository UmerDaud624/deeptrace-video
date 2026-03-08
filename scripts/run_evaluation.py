"""
scripts/run_evaluation.py
--------------------------
Runs cross-dataset evaluation of a trained ensemble on the Celeb-DF v2
test set and optionally on the FF++ internal test split.

This is the final evaluation step that produces the numbers reported
in the FYP presentation and report.

Usage:
    python scripts/run_evaluation.py --checkpoint path/to/checkpoint.pt
    python scripts/run_evaluation.py --checkpoint path/to/checkpoint.pt --splits celeb ff
    python scripts/run_evaluation.py --checkpoint path/to/checkpoint.pt --save_predictions

Output:
    - Metrics printed to console
    - Metrics logged to MLflow under the same experiment
    - Optional: predictions CSV saved for further analysis
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import pandas as pd

from configs.config import cfg, MANIFEST_ROOT
from src.data.dataset import FaceDataset, TALLDataset
from src.data.manifest import load_manifest
from src.evaluation.metrics import Evaluator, compute_metrics
from src.models.ensemble import build_ensemble
from torch.utils.data import DataLoader

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
        description="DeepTrace cross-dataset evaluation"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained ensemble checkpoint (.pt file).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["celeb", "ff", "val"],
        default=["celeb"],
        help="Which test splits to evaluate on. Default: celeb (cross-dataset).",
    )
    parser.add_argument(
        "--manifest_dir",
        type=str,
        default=str(MANIFEST_ROOT),
        help="Directory containing manifest CSV files.",
    )
    parser.add_argument(
        "--fusion",
        type=str,
        choices=["logit", "feature"],
        default="logit",
        help="Fusion type matching the trained checkpoint.",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save per-sample predictions to a CSV file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/reports",
        help="Directory to save prediction CSVs.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------

def evaluate_split(
    model       : torch.nn.Module,
    manifest_df : pd.DataFrame,
    split_name  : str,
    device      : torch.device,
) -> dict[str, float]:
    """
    Evaluate model on one manifest split.

    Args:
        model:       Trained ensemble.
        manifest_df: Manifest DataFrame for this split.
        split_name:  Name for logging.
        device:      Compute device.

    Returns:
        Dict of evaluation metrics.
    """
    train_cfg = cfg.training

    face_ds = FaceDataset(
        manifest_df,
        split="test",
        input_size=cfg.efficientnet.input_size,
    )
    tall_ds = TALLDataset(
        manifest_df,
        split="test",
        input_size=cfg.tall_swin.input_size,
    )

    eval_kwargs = {
        "batch_size"  : train_cfg.batch_size,
        "shuffle"     : False,
        "num_workers" : train_cfg.num_workers,
        "pin_memory"  : train_cfg.pin_memory,
        "drop_last"   : False,
    }

    face_loader = DataLoader(face_ds, **eval_kwargs)
    tall_loader = DataLoader(tall_ds, **eval_kwargs)

    evaluator = Evaluator(
        model=model,
        face_loader=face_loader,
        tall_loader=tall_loader,
        device=device,
    )

    metrics = evaluator.evaluate(
        split_name=split_name,
        log_to_mlflow=True,
    )
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("DeepTrace Evaluation Pipeline")
    logger.info("=" * 60)
    logger.info("Checkpoint  : %s", args.checkpoint)
    logger.info("Splits      : %s", args.splits)
    logger.info("Fusion type : %s", args.fusion)
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # --- Load Model ---
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        sys.exit(1)

    model = build_ensemble(fusion_type=args.fusion, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    saved_epoch = checkpoint.get("epoch", "unknown")
    saved_score = checkpoint.get("score", "unknown")
    logger.info(
        "Loaded checkpoint: epoch=%s | val_auc=%s",
        saved_epoch, saved_score
    )

    # Log ensemble weights
    import torch.nn.functional as F
    w_eff  = F.softplus(model.w_eff).item()
    w_swin = F.softplus(model.w_swin).item()
    w_sum  = w_eff + w_swin
    logger.info(
        "Ensemble weights: EfficientNet=%.3f | Swin=%.3f",
        w_eff / w_sum, w_swin / w_sum,
    )

    # --- Load Manifests ---
    manifest_dir  = Path(args.manifest_dir)
    split_map = {
        "celeb" : manifest_dir / "test.csv",
        "ff"    : manifest_dir / "test.csv",
        "val"   : manifest_dir / "val.csv",
    }

    all_results: dict[str, dict] = {}

    for split in args.splits:
        csv_path = split_map[split]
        if not csv_path.exists():
            logger.error("Manifest not found: %s", csv_path)
            continue

        df = load_manifest(str(csv_path))

        # Filter Celeb-DF only for cross-dataset eval
        if split == "celeb":
            df = df[df["dataset"] == "celeb_df"].reset_index(drop=True)
            split_name = "celeb_df_crossdataset"
        elif split == "ff":
            df = df[df["dataset"] == "ff_plus_plus"].reset_index(drop=True)
            split_name = "ff_internal_test"
        else:
            split_name = "val"

        if len(df) == 0:
            logger.warning(
                "No samples found for split '%s' after filtering.", split
            )
            continue

        logger.info(
            "Evaluating on %s: %d samples", split_name, len(df)
        )
        metrics = evaluate_split(model, df, split_name, device)
        all_results[split_name] = metrics

        if args.save_predictions:
            _save_predictions(
                model=model,
                manifest_df=df,
                split_name=split_name,
                output_dir=Path(args.output_dir),
                device=device,
            )

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)
    for split_name, metrics in all_results.items():
        logger.info("[%s]", split_name)
        logger.info(
            "  AUC-ROC  : %.4f", metrics["auc_roc"]
        )
        logger.info(
            "  EER      : %.4f (%.2f%%)",
            metrics["eer"], metrics["eer"] * 100
        )
        logger.info(
            "  AP       : %.4f", metrics["ap"]
        )
        logger.info(
            "  Accuracy : %.4f", metrics["accuracy"]
        )
        logger.info(
            "  F1       : %.4f", metrics["f1"]
        )
        logger.info(
            "  TP=%d FP=%d TN=%d FN=%d",
            metrics["tp"], metrics["fp"],
            metrics["tn"], metrics["fn"],
        )
    logger.info("=" * 60)


def _save_predictions(
    model       : torch.nn.Module,
    manifest_df : pd.DataFrame,
    split_name  : str,
    output_dir  : Path,
    device      : torch.device,
) -> None:
    """
    Save per-sample predictions to a CSV file for detailed analysis.

    Args:
        model:       Trained ensemble.
        manifest_df: Manifest DataFrame.
        split_name:  Name used for output filename.
        output_dir:  Directory to save CSV.
        device:      Compute device.
    """
    from src.data.dataset import FaceDataset, TALLDataset

    face_ds = FaceDataset(manifest_df, split="test",
                          input_size=cfg.efficientnet.input_size)
    tall_ds = TALLDataset(manifest_df, split="test",
                          input_size=cfg.tall_swin.input_size)

    eval_kwargs = {
        "batch_size"  : cfg.training.batch_size,
        "shuffle"     : False,
        "num_workers" : cfg.training.num_workers,
        "pin_memory"  : cfg.training.pin_memory,
        "drop_last"   : False,
    }

    face_loader = DataLoader(face_ds, **eval_kwargs)
    tall_loader = DataLoader(tall_ds, **eval_kwargs)

    evaluator = Evaluator(
        model=model,
        face_loader=face_loader,
        tall_loader=tall_loader,
        device=device,
    )

    results = evaluator.get_per_sample_predictions()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"predictions_{split_name}.csv"

    pred_df = manifest_df[["tall_path", "label", "dataset", "subset"]].copy()
    pred_df["prob_fake"] = results["probs"]
    pred_df["pred"]      = results["preds"]
    pred_df["correct"]   = (pred_df["pred"] == pred_df["label"]).astype(int)

    pred_df.to_csv(out_path, index=False)
    logger.info("Predictions saved to: %s", out_path)


if __name__ == "__main__":
    main()