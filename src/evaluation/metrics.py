"""
src/evaluation/metrics.py
--------------------------
Evaluation metrics for the DeepTrace deepfake detection pipeline.

Metrics computed:
    AUC-ROC  -- primary metric, threshold-independent, standard in literature
    EER      -- Equal Error Rate, standard in audio/video deepfake papers
    AP       -- Average Precision (area under precision-recall curve)
    Accuracy -- at decision_threshold from params.yaml
    Precision, Recall, F1 -- at decision_threshold
    Confusion matrix breakdown

All metrics are computed from raw probabilities (after sigmoid),
not from logits. The evaluator also logs everything to MLflow.

Why EER in addition to AUC:
    EER is the operating point where FAR == FRR (false accept == false reject).
    It is the standard metric in the ASVspoof challenge and used in all
    deepfake detection papers cited in the DeepTrace literature review.
    Lower EER is better. A random classifier has EER of 50%.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
import mlflow
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs.config import cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EER Computation
# ---------------------------------------------------------------------------

def compute_eer(labels: np.ndarray, probs: np.ndarray) -> float:
    """
    Compute Equal Error Rate from binary labels and predicted probabilities.

    EER is the threshold at which False Accept Rate equals False Reject Rate.
    Computed by finding the crossing point on the ROC curve.

    Args:
        labels: Ground truth binary labels (0 or 1). Shape (N,).
        probs:  Predicted probabilities for the positive class. Shape (N,).

    Returns:
        EER as a float between 0 and 1. Lower is better.
        Returns 0.5 if computation fails (degenerate case).
    """
    try:
        fpr, tpr, _ = roc_curve(labels, probs, pos_label=1)
        fnr         = 1.0 - tpr
        # Find index where FPR and FNR are closest
        idx = np.nanargmin(np.abs(fnr - fpr))
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
        return eer
    except Exception as exc:
        logger.warning("EER computation failed: %s", exc)
        return 0.5


# ---------------------------------------------------------------------------
# Full Metrics Computation
# ---------------------------------------------------------------------------

def compute_metrics(
    labels    : np.ndarray,
    probs     : np.ndarray,
    threshold : float = None,
) -> dict[str, float]:
    """
    Compute all evaluation metrics from labels and predicted probabilities.

    Args:
        labels:    Ground truth binary labels. Shape (N,).
        probs:     Predicted probabilities for fake class. Shape (N,).
        threshold: Decision threshold for binary prediction.
                   Defaults to cfg.evaluation.decision_threshold.

    Returns:
        Dict with keys: auc_roc, eer, ap, accuracy, precision,
                        recall, f1, tp, fp, tn, fn.
    """
    if threshold is None:
        threshold = cfg.evaluation.decision_threshold

    preds = (probs >= threshold).astype(int)

    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = 0.5

    try:
        ap = float(average_precision_score(labels, probs))
    except ValueError:
        ap = 0.0

    eer = compute_eer(labels, probs)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metrics = {
        "auc_roc"   : round(auc, 6),
        "eer"       : round(eer, 6),
        "ap"        : round(ap,  6),
        "accuracy"  : round(float(accuracy_score(labels, preds)), 6),
        "precision" : round(float(precision_score(labels, preds, zero_division=0)), 6),
        "recall"    : round(float(recall_score(labels, preds, zero_division=0)), 6),
        "f1"        : round(float(f1_score(labels, preds, zero_division=0)), 6),
        "tp"        : int(tp),
        "fp"        : int(fp),
        "tn"        : int(tn),
        "fn"        : int(fn),
    }
    return metrics


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Runs inference on a test set and computes all metrics.

    Designed for cross-dataset evaluation on Celeb-DF v2.
    Logs results to MLflow if an active run exists.

    Args:
        model:       Trained LogitFusionEnsemble.
        face_loader: DataLoader for FaceDataset test split.
        tall_loader: DataLoader for TALLDataset test split.
        device:      Compute device. Auto-detects if None.

    Example:
        evaluator = Evaluator(model, face_loader, tall_loader)
        metrics   = evaluator.evaluate(split_name="celeb_df_test")
    """

    def __init__(
        self,
        model       : nn.Module,
        face_loader : DataLoader,
        tall_loader : DataLoader,
        device      : Optional[torch.device] = None,
    ) -> None:
        self.model       = model
        self.face_loader = face_loader
        self.tall_loader = tall_loader
        self.device      = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def evaluate(
        self,
        split_name     : str  = "test",
        log_to_mlflow  : bool = True,
    ) -> dict[str, float]:
        """
        Run full evaluation on the provided DataLoaders.

        Args:
            split_name:    Name prefix for MLflow metric keys.
            log_to_mlflow: Log metrics to active MLflow run if True.

        Returns:
            Dict of all computed metrics.
        """
        self.model.eval()
        all_labels : list[int]   = []
        all_probs  : list[float] = []

        with torch.no_grad():
            for (face_imgs, labels), (tall_imgs, _) in zip(
                tqdm(self.face_loader, desc=f"Evaluating [{split_name}]",
                     leave=True),
                self.tall_loader,
            ):
                face_imgs = face_imgs.to(self.device, non_blocking=True)
                tall_imgs = tall_imgs.to(self.device, non_blocking=True)

                logits = self.model(face_imgs, tall_imgs)
                probs  = torch.sigmoid(logits).cpu().numpy().flatten()
                labs   = labels.numpy().flatten().astype(int)

                all_probs.extend(probs.tolist())
                all_labels.extend(labs.tolist())

        labels_arr = np.array(all_labels)
        probs_arr  = np.array(all_probs)

        metrics = compute_metrics(labels_arr, probs_arr)

        logger.info(
            "[%s] AUC=%.4f | EER=%.4f | AP=%.4f | "
            "Acc=%.4f | P=%.4f | R=%.4f | F1=%.4f",
            split_name,
            metrics["auc_roc"],
            metrics["eer"],
            metrics["ap"],
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )
        logger.info(
            "[%s] Confusion -- TP=%d FP=%d TN=%d FN=%d",
            split_name,
            metrics["tp"], metrics["fp"],
            metrics["tn"], metrics["fn"],
        )

        if log_to_mlflow:
            try:
                mlflow_metrics = {
                    f"{split_name}_{k}": v
                    for k, v in metrics.items()
                    if isinstance(v, float)
                }
                mlflow.log_metrics(mlflow_metrics)
            except Exception:
                logger.debug("MLflow logging skipped (no active run)")

        return metrics

    def get_per_sample_predictions(self) -> dict[str, np.ndarray]:
        """
        Return per-sample probabilities and labels for detailed analysis.

        Useful for plotting ROC curves, PR curves, and score histograms.

        Returns:
            Dict with keys "probs", "labels", "preds".
        """
        self.model.eval()
        all_labels : list[int]   = []
        all_probs  : list[float] = []

        with torch.no_grad():
            for (face_imgs, labels), (tall_imgs, _) in zip(
                self.face_loader, self.tall_loader
            ):
                face_imgs = face_imgs.to(self.device, non_blocking=True)
                tall_imgs = tall_imgs.to(self.device, non_blocking=True)

                logits = self.model(face_imgs, tall_imgs)
                probs  = torch.sigmoid(logits).cpu().numpy().flatten()
                labs   = labels.numpy().flatten().astype(int)

                all_probs.extend(probs.tolist())
                all_labels.extend(labs.tolist())

        probs_arr  = np.array(all_probs)
        labels_arr = np.array(all_labels)

        return {
            "probs"  : probs_arr,
            "labels" : labels_arr,
            "preds"  : (probs_arr >= cfg.evaluation.decision_threshold).astype(int),
        }