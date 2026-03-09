"""
src/data/manifest.py
---------------------
Generates and saves train/val/test CSV manifests from preprocessed records.

Split strategy:
    Train  --> FF++ train split + all DFDC
    Val    --> FF++ val split only
    Test   --> Celeb-DF v2 entire dataset (cross-dataset, unseen domain)

This matches the evaluation protocol used in the TALL and Guardian-AI
papers cited in the DeepTrace literature review.

CSV columns:
    face_dir    --> directory containing face crop JPGs (EfficientNet input)
    tall_path   --> TALL grid JPG path (Swin-T input)
    label       --> 0 = real, 1 = fake
    dataset     --> source dataset name
    subset      --> manipulation type within dataset
    split       --> train / val / test
    num_faces   --> number of face crops extracted
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs.config import cfg, MANIFEST_ROOT
from src.data.preprocessing import VideoRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def records_to_dataframe(records: list[VideoRecord]) -> pd.DataFrame:
    """
    Convert a list of VideoRecord objects to a pandas DataFrame.

    Filters out records where tall_path or face_dir do not exist
    on disk -- ensures DataLoader never gets a broken path.

    Args:
        records: Output from any preprocessor's run() method.

    Returns:
        DataFrame with manifest schema columns.
    """
    rows = []
    skipped = 0

    for rec in records:
        # Both outputs must exist and face_dir must be non-empty
        if not Path(rec.tall_path).exists():
            skipped += 1
            continue
        if not Path(rec.face_dir).exists():
            skipped += 1
            continue
        if len(list(Path(rec.face_dir).glob("*.jpg"))) == 0:
            skipped += 1
            continue

        rows.append({
            "face_dir"  : rec.face_dir,
            "tall_path" : rec.tall_path,
            "label"     : rec.label,
            "dataset"   : rec.dataset,
            "subset"    : rec.subset,
            "split"     : rec.split,
            "num_faces" : rec.num_faces,
        })

    if skipped > 0:
        logger.warning(
            "Skipped %d records with missing outputs on disk", skipped
        )

    df = pd.DataFrame(rows)
    logger.info("DataFrame: %d valid records", len(df))
    return df


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_ff_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign train/val/test splits to FF++ records.

    Uses stratified splitting on label to maintain real/fake
    ratio across all three splits.

    Args:
        df: FF++ DataFrame from records_to_dataframe().

    Returns:
        Same DataFrame with split column filled.
    """
    split_cfg = cfg.split
    df = df.copy()

    test_and_val_size = split_cfg.val_size + split_cfg.test_size

    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        df.index,
        test_size=test_and_val_size,
        stratify=df.loc[df.index, "label"],
        random_state=split_cfg.random_seed,
    )

    # Second split: val vs test from temp
    val_fraction = split_cfg.val_size / test_and_val_size
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_fraction),
        stratify=df.loc[temp_idx, "label"],
        random_state=split_cfg.random_seed,
    )

    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx,   "split"] = "val"
    df.loc[test_idx,  "split"] = "test"

    logger.info(
        "FF++ splits -- train: %d, val: %d, test: %d",
        len(train_idx), len(val_idx), len(test_idx),
    )
    return df


# ---------------------------------------------------------------------------
# Manifest Builder
# ---------------------------------------------------------------------------

def build_manifests(
    ff_records     : list[VideoRecord],
    celeb_records  : list[VideoRecord],
    dfdc_records   : list[VideoRecord],
) -> dict[str, pd.DataFrame]:
    """
    Build train/val/test manifests from all three datasets.

    Train: FF++ train split + all DFDC
    Val:   FF++ val split
    Test:  Celeb-DF v2 (full, cross-dataset evaluation)

    Args:
        ff_records:    From FFPlusPlusPreprocessor.run()
        celeb_records: From CelebDFPreprocessor.run()
        dfdc_records:  From DFDCPreprocessor.run()

    Returns:
        Dict with keys "train", "val", "test".
        Each value is a DataFrame ready for the DataLoader.

    Raises:
        ValueError: If FF++ or Celeb-DF produced no valid records.
    """
    split_cfg = cfg.split

    # --- FF++ ---
    ff_df = records_to_dataframe(ff_records)
    if len(ff_df) == 0:
        raise ValueError(
            "FF++ produced no valid records. "
            "Check preprocessing completed successfully."
        )

    ff_df = split_ff_dataframe(ff_df)
    ff_train = ff_df[ff_df["split"] == "train"].copy()
    ff_val   = ff_df[ff_df["split"] == "val"].copy()

    # --- DFDC (all to train) ---
    dfdc_df = records_to_dataframe(dfdc_records)
    dfdc_df["split"] = "train"
    logger.info("DFDC train samples: %d", len(dfdc_df))

    # --- Celeb-DF (official test list, cross-dataset) ---
    celeb_df = records_to_dataframe(celeb_records)
    if len(celeb_df) == 0:
        raise ValueError(
            "Celeb-DF produced no valid records. "
            "Check preprocessing completed successfully."
        )

    # Filter to official test split using List_of_testing_videos.txt.
    # This ensures AUC scores are directly comparable to published benchmarks
    # (TALL paper, Celeb-DF v2 paper). The file lists relative video paths,
    # one per line, prefixed by a label int (0=real, 1=fake).
    # Example lines:
    #   0 YouTube-real/00001.mp4
    #   1 Celeb-synthesis/id0_id1_0001.mp4
    test_list_path = cfg.paths.celeb_df_test_list
    if test_list_path.exists():
        official_stems: set[str] = set()
        with open(test_list_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Each line: "<label> <relative/path/video.mp4>"
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    # Extract the video stem without extension
                    official_stems.add(Path(parts[1]).stem)

        # Match against source_path stems in our records
        # source_path is the original mp4, so stem matches the test list
        pre_filter = len(celeb_df)
        celeb_df["_stem"] = celeb_df["face_dir"].apply(
            lambda p: Path(p).name  # face_dir leaf == video stem
        )
        celeb_df = celeb_df[
            celeb_df["_stem"].isin(official_stems)
        ].drop(columns=["_stem"]).copy()

        logger.info(
            "Celeb-DF official test filter: %d -> %d records (%d stems in list)",
            pre_filter, len(celeb_df), len(official_stems),
        )
        if len(celeb_df) == 0:
            logger.warning(
                "Official test filter removed ALL records. "
                "Falling back to full Celeb-DF dataset as test set."
            )
            celeb_df = records_to_dataframe(celeb_records)
    else:
        logger.warning(
            "List_of_testing_videos.txt not found at %s. "
            "Using full Celeb-DF as test set.",
            test_list_path,
        )

    celeb_df["split"] = "test"
    logger.info("Celeb-DF test samples: %d", len(celeb_df))

    # --- Combine ---
    train_df = pd.concat(
        [ff_train, dfdc_df], ignore_index=True
    ).sample(
        frac=1.0, random_state=split_cfg.random_seed
    ).reset_index(drop=True)

    val_df  = ff_val.reset_index(drop=True)
    test_df = celeb_df.reset_index(drop=True)

    # Log class distribution for each split
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        real = (df["label"] == 0).sum()
        fake = (df["label"] == 1).sum()
        logger.info(
            "%s -- total: %d | real: %d | fake: %d | imbalance ratio: %.2f",
            name, len(df), real, fake,
            fake / max(real, 1),
        )

    return {"train": train_df, "val": val_df, "test": test_df}


# ---------------------------------------------------------------------------
# Save and Load
# ---------------------------------------------------------------------------

def save_manifests(
    manifests  : dict[str, pd.DataFrame],
    output_dir : Path = MANIFEST_ROOT,
) -> dict[str, str]:
    """
    Save each manifest split as a CSV file.

    Args:
        manifests:  Dict from build_manifests().
        output_dir: Directory to write CSVs into.

    Returns:
        Dict mapping split name to saved CSV path string.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    for split_name, df in manifests.items():
        out_path = output_dir / f"{split_name}.csv"
        df.to_csv(out_path, index=False)
        saved[split_name] = str(out_path)
        logger.info(
            "Saved %s manifest: %s (%d rows)", split_name, out_path, len(df)
        )

    return saved


def load_manifest(csv_path: str) -> pd.DataFrame:
    """
    Load a manifest CSV and validate required columns exist.

    Args:
        csv_path: Path to manifest CSV file.

    Returns:
        Validated DataFrame.

    Raises:
        FileNotFoundError: If CSV does not exist.
        ValueError: If required columns are missing.
    """
    required_cols = {
        "face_dir", "tall_path", "label", "dataset", "subset", "split"
    }

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")

    logger.info(
        "Loaded manifest: %s (%d rows)", csv_path, len(df)
    )
    return df


def get_manifest_stats(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics for a manifest DataFrame.

    Useful for logging before training starts.

    Args:
        df: Manifest DataFrame from load_manifest().

    Returns:
        Dict with count, label distribution, and dataset breakdown.
    """
    stats = {
        "total"           : len(df),
        "real"            : int((df["label"] == 0).sum()),
        "fake"            : int((df["label"] == 1).sum()),
        "datasets"        : df["dataset"].value_counts().to_dict(),
        "subsets"         : df["subset"].value_counts().to_dict(),
        "imbalance_ratio" : round(
            (df["label"] == 1).sum() / max((df["label"] == 0).sum(), 1), 3
        ),
    }
    return stats