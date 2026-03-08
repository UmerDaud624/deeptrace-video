"""
src/data/dataset.py
--------------------
PyTorch Dataset classes for the DeepTrace video deepfake detection pipeline.

Two datasets are provided:
    FaceDataset   -- loads individual face crops for EfficientNet-B4
    TALLDataset   -- loads pre-built TALL grid images for Swin Transformer

Both read from manifest CSVs produced by manifest.py.
Augmentation is applied only during training.

Augmentation strategy:
    Training:
        - Horizontal flip (p=0.5)
        - ColorJitter with moderate strength (brightness, contrast, hue)
        - JPEG compression simulation to match real-world codec artifacts
        - Random rotation (+/- 10 degrees)
        - Normalization with ImageNet statistics
    Val/Test:
        - Resize only
        - Normalization with ImageNet statistics
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs.config import cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Augmentation Pipelines
# ---------------------------------------------------------------------------

def _get_face_transforms(split: str, input_size: int) -> Callable:
    """
    Build torchvision transform pipeline for face crops.

    Args:
        split:      One of "train", "val", "test".
        input_size: Target image size in pixels (square).

    Returns:
        Composed torchvision transform.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split == "train":
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05,
            ),
            # Simulate JPEG compression artifacts seen in real-world videos
            transforms.RandomApply([
                transforms.Lambda(lambda img: _jpeg_compress(img, quality=60))
            ], p=0.3),
            transforms.ToTensor(),
            normalize,
        ])

    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize,
    ])


def _get_tall_transforms(split: str, input_size: int) -> Callable:
    """
    Build torchvision transform pipeline for TALL grid images.

    TALL grids are not flipped horizontally -- the spatial layout of
    tiles carries temporal ordering information that should be preserved.

    Args:
        split:      One of "train", "val", "test".
        input_size: Target image size in pixels (square).

    Returns:
        Composed torchvision transform.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split == "train":
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
            ),
            transforms.ToTensor(),
            normalize,
        ])

    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize,
    ])


def _jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    """
    Simulate JPEG compression by encoding and decoding in memory.

    Used as a training augmentation to improve robustness against
    codec artifacts present in real-world social media video.

    Args:
        img:     PIL Image in RGB mode.
        quality: JPEG quality factor (1-95). Lower = more compression.

    Returns:
        PIL Image after JPEG round-trip.
    """
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ---------------------------------------------------------------------------
# Face Dataset (EfficientNet-B4 input)
# ---------------------------------------------------------------------------

class FaceDataset(Dataset):
    """
    Loads individual face crop JPGs for EfficientNet-B4.

    Each row in the manifest corresponds to one video. This dataset
    samples one random face crop per video per epoch during training,
    and uses the middle crop during val/test for determinism.

    Args:
        manifest_df: DataFrame from load_manifest().
        split:       One of "train", "val", "test".
        input_size:  Image size expected by the model.

    Example:
        df = load_manifest("data/manifests/train.csv")
        ds = FaceDataset(df, split="train", input_size=224)
        img, label = ds[0]
    """

    def __init__(
        self,
        manifest_df : pd.DataFrame,
        split       : str,
        input_size  : int = 224,
    ) -> None:
        self.df         = manifest_df.reset_index(drop=True)
        self.split      = split
        self.transform  = _get_face_transforms(split, input_size)

        # Validate required columns
        for col in ("face_dir", "label"):
            if col not in self.df.columns:
                raise ValueError(f"Manifest missing required column: {col}")

        logger.info(
            "FaceDataset [%s]: %d samples", split, len(self.df)
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row       = self.df.iloc[idx]
        face_dir  = Path(row["face_dir"])
        label     = int(row["label"])

        crops = sorted(face_dir.glob("*.jpg"))
        if len(crops) == 0:
            # Return a black image if face_dir is somehow empty
            # This should not happen if preprocessing ran correctly
            logger.warning("Empty face_dir at idx %d: %s", idx, face_dir)
            img = Image.fromarray(
                np.zeros(
                    (cfg.efficientnet.input_size, cfg.efficientnet.input_size, 3),
                    dtype=np.uint8,
                )
            )
        else:
            if self.split == "train":
                crop_path = random.choice(crops)
            else:
                crop_path = crops[len(crops) // 2]

            img = Image.open(crop_path).convert("RGB")

        tensor = self.transform(img)
        return tensor, torch.tensor(label, dtype=torch.float32)

    def get_labels(self) -> list[int]:
        """Return all labels as a list. Used for WeightedRandomSampler."""
        return self.df["label"].tolist()


# ---------------------------------------------------------------------------
# TALL Dataset (Swin Transformer input)
# ---------------------------------------------------------------------------

class TALLDataset(Dataset):
    """
    Loads pre-built TALL grid JPGs for Swin Transformer.

    Each row in the manifest corresponds to one TALL grid image
    that encodes N frames of a video in a spatial tile layout.

    Args:
        manifest_df: DataFrame from load_manifest().
        split:       One of "train", "val", "test".
        input_size:  Image size expected by the model.

    Example:
        df = load_manifest("data/manifests/train.csv")
        ds = TALLDataset(df, split="train", input_size=224)
        img, label = ds[0]
    """

    def __init__(
        self,
        manifest_df : pd.DataFrame,
        split       : str,
        input_size  : int = 224,
    ) -> None:
        self.df        = manifest_df.reset_index(drop=True)
        self.split     = split
        self.transform = _get_tall_transforms(split, input_size)

        for col in ("tall_path", "label"):
            if col not in self.df.columns:
                raise ValueError(f"Manifest missing required column: {col}")

        logger.info(
            "TALLDataset [%s]: %d samples", split, len(self.df)
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row       = self.df.iloc[idx]
        tall_path = Path(row["tall_path"])
        label     = int(row["label"])

        if not tall_path.exists():
            logger.warning("TALL grid missing at idx %d: %s", idx, tall_path)
            img = Image.fromarray(
                np.zeros(
                    (cfg.tall_swin.input_size, cfg.tall_swin.input_size, 3),
                    dtype=np.uint8,
                )
            )
        else:
            img = Image.open(tall_path).convert("RGB")

        tensor = self.transform(img)
        return tensor, torch.tensor(label, dtype=torch.float32)

    def get_labels(self) -> list[int]:
        """Return all labels as a list. Used for WeightedRandomSampler."""
        return self.df["label"].tolist()


# ---------------------------------------------------------------------------
# Weighted Sampler (handles class imbalance)
# ---------------------------------------------------------------------------

def build_weighted_sampler(labels: list[int]) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that up-samples the minority class.

    Used for the training DataLoader to counteract the real/fake
    class imbalance present especially in DFDC.

    Args:
        labels: List of integer labels (0 or 1) from dataset.get_labels().

    Returns:
        WeightedRandomSampler configured for one full epoch.
    """
    label_array  = np.array(labels)
    class_counts = np.bincount(label_array)
    class_weights = 1.0 / class_counts.astype(np.float32)

    sample_weights = class_weights[label_array]
    sample_weights = torch.from_numpy(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    logger.info(
        "WeightedRandomSampler -- class counts: %s | class weights: %s",
        class_counts.tolist(),
        [round(w, 4) for w in class_weights.tolist()],
    )
    return sampler


# ---------------------------------------------------------------------------
# DataLoader Factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    train_df : pd.DataFrame,
    val_df   : pd.DataFrame,
    test_df  : pd.DataFrame,
) -> dict[str, dict[str, DataLoader]]:
    """
    Build all DataLoaders for both model streams.

    Returns a nested dict:
        loaders["face"]["train"]  --> FaceDataset train DataLoader
        loaders["face"]["val"]    --> FaceDataset val DataLoader
        loaders["face"]["test"]   --> FaceDataset test DataLoader
        loaders["tall"]["train"]  --> TALLDataset train DataLoader
        loaders["tall"]["val"]    --> TALLDataset val DataLoader
        loaders["tall"]["test"]   --> TALLDataset test DataLoader

    Training DataLoaders use WeightedRandomSampler.
    Val/Test DataLoaders use sequential sampling (no shuffle).

    Args:
        train_df: Training manifest DataFrame.
        val_df:   Validation manifest DataFrame.
        test_df:  Test manifest DataFrame.

    Returns:
        Nested dict of DataLoaders.
    """
    train_cfg = cfg.training
    eff_size  = cfg.efficientnet.input_size
    tall_size = cfg.tall_swin.input_size

    # Build datasets
    face_train = FaceDataset(train_df, split="train", input_size=eff_size)
    face_val   = FaceDataset(val_df,   split="val",   input_size=eff_size)
    face_test  = FaceDataset(test_df,  split="test",  input_size=eff_size)

    tall_train = TALLDataset(train_df, split="train", input_size=tall_size)
    tall_val   = TALLDataset(val_df,   split="val",   input_size=tall_size)
    tall_test  = TALLDataset(test_df,  split="test",  input_size=tall_size)

    # Weighted sampler for training (shared between both streams
    # since they share the same manifest row ordering)
    sampler = build_weighted_sampler(face_train.get_labels())

    common_train_kwargs = {
        "batch_size"  : train_cfg.batch_size,
        "sampler"     : sampler,
        "num_workers" : train_cfg.num_workers,
        "pin_memory"  : train_cfg.pin_memory,
        "drop_last"   : True,
    }
    common_eval_kwargs = {
        "batch_size"  : train_cfg.batch_size,
        "shuffle"     : False,
        "num_workers" : train_cfg.num_workers,
        "pin_memory"  : train_cfg.pin_memory,
        "drop_last"   : False,
    }

    loaders = {
        "face": {
            "train" : DataLoader(face_train, **common_train_kwargs),
            "val"   : DataLoader(face_val,   **common_eval_kwargs),
            "test"  : DataLoader(face_test,  **common_eval_kwargs),
        },
        "tall": {
            "train" : DataLoader(tall_train, **common_train_kwargs),
            "val"   : DataLoader(tall_val,   **common_eval_kwargs),
            "test"  : DataLoader(tall_test,  **common_eval_kwargs),
        },
    }

    logger.info(
        "DataLoaders built -- face train batches: %d | tall train batches: %d",
        len(loaders["face"]["train"]),
        len(loaders["tall"]["train"]),
    )
    return loaders