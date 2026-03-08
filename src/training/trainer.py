"""
src/training/trainer.py
------------------------
Training loop for the DeepTrace dual-stream ensemble.

Handles:
    - End-to-end training of LogitFusionEnsemble (both streams jointly)
    - Mixed precision training via torch.cuda.amp
    - Cosine annealing LR scheduler with linear warmup
    - Early stopping monitored on validation AUC
    - Checkpoint saving (top-k by val_auc + latest)
    - MLflow experiment tracking (params, metrics, artifacts)
    - Gradient clipping for training stability
    - Reproducible seeding

Training loop processes both streams in lockstep:
    for face_batch, tall_batch in zip(face_loader, tall_loader):
        logits = ensemble(face_batch, tall_batch)
        loss   = focal_loss(logits, labels)
        ...

Both DataLoaders share the same WeightedRandomSampler so the
(face_crop, tall_grid) pairs always correspond to the same video.
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import mlflow
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs.config import cfg, CHECKPOINT_ROOT
from src.models.ensemble import LogitFusionEnsemble
from src.training.losses import FocalLoss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Warmup + Cosine LR Scheduler
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """
    Linear warmup followed by cosine annealing.

    During warmup epochs (0 to warmup_epochs), LR increases linearly
    from 0 to base_lr. After warmup, cosine annealing takes over until
    min_lr is reached at total_epochs.

    Args:
        optimizer:     PyTorch optimizer.
        warmup_epochs: Number of linear warmup epochs.
        total_epochs:  Total training epochs.
        base_lr:       Peak learning rate after warmup.
        min_lr:        Minimum learning rate at end of cosine schedule.
    """

    def __init__(
        self,
        optimizer     : torch.optim.Optimizer,
        warmup_epochs : int,
        total_epochs  : int,
        base_lr       : float,
        min_lr        : float,
    ) -> None:
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.base_lr       = base_lr
        self.min_lr        = min_lr
        self.cosine        = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=min_lr,
        )
        self.current_epoch = 0

    def step(self) -> float:
        """
        Advance the scheduler by one epoch.

        Returns:
            Current learning rate after the step.
        """
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        else:
            self.cosine.step()
            lr = self.optimizer.param_groups[0]["lr"]

        self.current_epoch += 1
        return lr

    def get_lr(self) -> float:
        """Return current learning rate."""
        return self.optimizer.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Early stopping based on validation metric improvement.

    Args:
        patience:  Number of epochs to wait without improvement.
        min_delta: Minimum change to count as improvement.
        mode:      "max" for metrics like AUC, "min" for loss.
    """

    def __init__(
        self,
        patience  : int   = 7,
        min_delta : float = 1e-4,
        mode      : str   = "max",
    ) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.counter    = 0
        self.best_score : Optional[float] = None
        self.stop       = False

    def __call__(self, score: float) -> bool:
        """
        Update state with new score.

        Args:
            score: Current epoch metric value.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
                return True

        return False


# ---------------------------------------------------------------------------
# Checkpoint Manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Saves top-k checkpoints by validation AUC plus the latest checkpoint.

    Args:
        save_dir:   Directory to save checkpoint files.
        top_k:      Number of best checkpoints to keep.
        model_name: Prefix for checkpoint filenames.
    """

    def __init__(
        self,
        save_dir   : str,
        top_k      : int = 3,
        model_name : str = "ensemble",
    ) -> None:
        self.save_dir   = Path(save_dir)
        self.top_k      = top_k
        self.model_name = model_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[tuple[float, str]] = []   # (score, path)

    def save(
        self,
        model     : nn.Module,
        optimizer : torch.optim.Optimizer,
        epoch     : int,
        score     : float,
        metrics   : dict,
    ) -> str:
        """
        Save checkpoint and prune if more than top_k saved.

        Args:
            model:     Model to save.
            optimizer: Optimizer state to save.
            epoch:     Current epoch number.
            score:     Validation metric used for ranking.
            metrics:   Dict of all metrics to store in checkpoint.

        Returns:
            Path to saved checkpoint file.
        """
        filename = (
            self.save_dir
            / f"{self.model_name}_epoch{epoch:03d}_auc{score:.4f}.pt"
        )

        torch.save({
            "epoch"            : epoch,
            "model_state_dict" : model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "score"            : score,
            "metrics"          : metrics,
        }, filename)

        # Always save latest
        latest = self.save_dir / f"{self.model_name}_latest.pt"
        torch.save({
            "epoch"            : epoch,
            "model_state_dict" : model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "score"            : score,
            "metrics"          : metrics,
        }, latest)

        self.records.append((score, str(filename)))
        self.records.sort(key=lambda x: x[0], reverse=True)

        # Remove checkpoints beyond top_k
        while len(self.records) > self.top_k:
            _, old_path = self.records.pop()
            if Path(old_path).exists():
                Path(old_path).unlink()
                logger.info("Removed old checkpoint: %s", old_path)

        logger.info(
            "Saved checkpoint: %s (score=%.4f)", filename.name, score
        )
        return str(filename)

    def best_checkpoint(self) -> Optional[str]:
        """Return path to the best checkpoint saved so far."""
        if not self.records:
            return None
        return self.records[0][1]


# ---------------------------------------------------------------------------
# One Epoch Train / Validate
# ---------------------------------------------------------------------------

def _run_epoch(
    model       : LogitFusionEnsemble,
    face_loader : DataLoader,
    tall_loader : DataLoader,
    criterion   : FocalLoss,
    optimizer   : Optional[torch.optim.Optimizer],
    scaler      : Optional[GradScaler],
    device      : torch.device,
    grad_clip   : float,
    is_train    : bool,
) -> dict[str, float]:
    """
    Run one full epoch of training or validation.

    Args:
        model:       Ensemble model.
        face_loader: DataLoader for FaceDataset.
        tall_loader: DataLoader for TALLDataset.
        criterion:   Focal loss instance.
        optimizer:   Optimizer (None during validation).
        scaler:      AMP GradScaler (None during validation).
        device:      Compute device.
        grad_clip:   Max gradient norm for clipping.
        is_train:    True for training mode, False for eval mode.

    Returns:
        Dict with keys: loss, auc, accuracy, precision, recall, f1.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    model.train() if is_train else model.eval()

    total_loss  = 0.0
    all_labels  : list[int]   = []
    all_probs   : list[float] = []
    num_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for (face_imgs, labels), (tall_imgs, _) in zip(
            tqdm(face_loader, desc="train" if is_train else "val",
                 leave=False),
            tall_loader,
        ):
            face_imgs = face_imgs.to(device, non_blocking=True)
            tall_imgs = tall_imgs.to(device, non_blocking=True)
            labels    = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad()
                with autocast(enabled=scaler is not None):
                    logits = model(face_imgs, tall_imgs) # forward pass
                    loss   = criterion(logits, labels) # calculate the loss

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
            else:
                with autocast(enabled=False):
                    logits = model(face_imgs, tall_imgs)
                    loss   = criterion(logits, labels)

            total_loss  += loss.item()
            num_batches += 1

            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            labs  = labels.detach().cpu().numpy().flatten().astype(int)
            all_probs.extend(probs.tolist())
            all_labels.extend(labs.tolist())

    avg_loss = total_loss / max(num_batches, 1)
    preds    = (np.array(all_probs) >= 0.5).astype(int)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5   # only one class present in batch

    return {
        "loss"      : round(avg_loss, 6),
        "auc"       : round(float(auc), 6),
        "accuracy"  : round(float(accuracy_score(all_labels, preds)), 6),
        "precision" : round(float(precision_score(all_labels, preds, zero_division=0)), 6),
        "recall"    : round(float(recall_score(all_labels, preds, zero_division=0)), 6),
        "f1"        : round(float(f1_score(all_labels, preds, zero_division=0)), 6),
    }


# ---------------------------------------------------------------------------
# Main Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Full training orchestrator for the DeepTrace ensemble.

    Manages the complete training lifecycle:
        - Optimizer and scheduler setup
        - MLflow run initialization
        - Per-epoch train/val loop
        - Checkpoint saving
        - Early stopping
        - Final metric logging

    Args:
        model:       LogitFusionEnsemble instance.
        face_loaders: Dict with keys "train", "val" for FaceDataset.
        tall_loaders: Dict with keys "train", "val" for TALLDataset.
        run_name:    MLflow run name. Auto-generated if None.

    Example:
        trainer = Trainer(model, face_loaders, tall_loaders)
        trainer.train()
    """

    def __init__(
        self,
        model        : LogitFusionEnsemble,
        face_loaders : dict[str, DataLoader],
        tall_loaders : dict[str, DataLoader],
        run_name     : Optional[str] = None,
    ) -> None:
        self.model        = model
        self.face_loaders = face_loaders
        self.tall_loaders = tall_loaders
        self.run_name     = run_name or f"run_{int(time.time())}"

        train_cfg   = cfg.training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        logger.info("Training on device: %s", self.device)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )

        # Scheduler
        self.scheduler = WarmupCosineScheduler(
            optimizer=self.optimizer,
            warmup_epochs=train_cfg.warmup_epochs,
            total_epochs=train_cfg.num_epochs,
            base_lr=train_cfg.learning_rate,
            min_lr=train_cfg.min_lr,
        )

        # Loss
        self.criterion = FocalLoss()

        # AMP scaler (only on CUDA)
        self.scaler = (
            GradScaler()
            if train_cfg.mixed_precision and self.device.type == "cuda"
            else None
        )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=train_cfg.patience,
            mode="max",
        )

        # Checkpoints
        self.ckpt_manager = CheckpointManager(
            save_dir=str(CHECKPOINT_ROOT),
            top_k=train_cfg.save_top_k,
        )

        self.num_epochs = train_cfg.num_epochs
        self.grad_clip  = train_cfg.grad_clip

    def train(self) -> dict[str, float]:
        """
        Run the full training loop.

        Returns:
            Dict of best validation metrics achieved during training.
        """
        set_seed(cfg.training.seed)

        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)

        best_metrics: dict[str, float] = {}

        with mlflow.start_run(run_name=self.run_name, tags=cfg.mlflow.run_tags):

            # Log all params from params.yaml
            self._log_params_to_mlflow()

            logger.info("Starting training: %s", self.run_name)

            for epoch in range(1, self.num_epochs + 1):
                epoch_start = time.time()

                # Train
                train_metrics = _run_epoch(
                    model=self.model,
                    face_loader=self.face_loaders["train"],
                    tall_loader=self.tall_loaders["train"],
                    criterion=self.criterion,
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                    device=self.device,
                    grad_clip=self.grad_clip,
                    is_train=True,
                )

                # Validate
                val_metrics = _run_epoch(
                    model=self.model,
                    face_loader=self.face_loaders["val"],
                    tall_loader=self.tall_loaders["val"],
                    criterion=self.criterion,
                    optimizer=None,
                    scaler=None,
                    device=self.device,
                    grad_clip=self.grad_clip,
                    is_train=False,
                )

                # LR step
                current_lr = self.scheduler.step()

                # Log ensemble weights
                w_eff  = nn.functional.softplus(self.model.w_eff).item()
                w_swin = nn.functional.softplus(self.model.w_swin).item()
                w_sum  = w_eff + w_swin

                epoch_time = time.time() - epoch_start
                val_auc    = val_metrics["auc"]

                logger.info(
                    "Epoch %03d/%03d | "
                    "train_loss=%.4f train_auc=%.4f | "
                    "val_loss=%.4f val_auc=%.4f | "
                    "lr=%.2e | w_eff=%.3f w_swin=%.3f | "
                    "time=%.1fs",
                    epoch, self.num_epochs,
                    train_metrics["loss"], train_metrics["auc"],
                    val_metrics["loss"],   val_metrics["auc"],
                    current_lr,
                    w_eff / w_sum, w_swin / w_sum,
                    epoch_time,
                )

                # MLflow logging
                mlflow_metrics = {}
                for k, v in train_metrics.items():
                    mlflow_metrics[f"train_{k}"] = v
                for k, v in val_metrics.items():
                    mlflow_metrics[f"val_{k}"] = v
                mlflow_metrics["lr"]            = current_lr
                mlflow_metrics["weight_eff"]    = w_eff / w_sum
                mlflow_metrics["weight_swin"]   = w_swin / w_sum
                mlflow_metrics["epoch_time_s"]  = epoch_time
                mlflow.log_metrics(mlflow_metrics, step=epoch)

                # Save checkpoint
                self.ckpt_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    score=val_auc,
                    metrics={**train_metrics, **{f"val_{k}": v
                             for k, v in val_metrics.items()}},
                )

                # Track best
                if not best_metrics or val_auc > best_metrics.get("val_auc", 0):
                    best_metrics = {
                        f"val_{k}": v for k, v in val_metrics.items()
                    }
                    best_metrics["epoch"] = epoch

                # Early stopping
                if self.early_stopping(val_auc):
                    logger.info(
                        "Early stopping at epoch %d. "
                        "Best val_auc: %.4f",
                        epoch, self.early_stopping.best_score,
                    )
                    break

            # Log best metrics to MLflow
            mlflow.log_metrics(
                {f"best_{k}": v for k, v in best_metrics.items()},
                step=best_metrics.get("epoch", 0),
            )

            # Log best checkpoint as artifact
            best_ckpt = self.ckpt_manager.best_checkpoint()
            if best_ckpt and Path(best_ckpt).exists():
                mlflow.log_artifact(best_ckpt, artifact_path="checkpoints")

            logger.info("Training complete. Best metrics: %s", best_metrics)

        return best_metrics

    def _log_params_to_mlflow(self) -> None:
        """Log all params.yaml values to MLflow as flat key-value pairs."""
        import yaml
        params_path = (
            Path(__file__).resolve().parent.parent.parent / "params.yaml"
        )
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)

        flat: dict[str, str] = {}
        for section, values in params.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    flat[f"{section}.{k}"] = str(v)
            else:
                flat[section] = str(values)

        mlflow.log_params(flat)