"""
scripts/run_preprocessing.py
------------------------------
Runs the full preprocessing pipeline for all three datasets and
generates train/val/test manifest CSV files.

Run this on Google Colab after mounting Drive.
All paths resolve automatically via config.py environment detection.

Usage:
    python scripts/run_preprocessing.py
    python scripts/run_preprocessing.py --datasets ff celeb dfdc
    python scripts/run_preprocessing.py --datasets ff --skip_manifest

Steps:
    1. FF++ preprocessing  -- frame sampling + face detection + TALL grid
    2. Celeb-DF preprocessing -- same pipeline
    3. DFDC preprocessing  -- load pre-extracted crops + TALL grid
    4. Build train/val/test manifest CSVs
    5. Print dataset statistics

Expected runtime on Colab T4:
    FF++     -- ~2-3 hours (5000 videos x 20 frames x RetinaFace)
    Celeb-DF -- ~1-2 hours (1000 videos)
    DFDC     -- ~15 minutes (pre-extracted, no face detection)
    Total    -- ~4-5 hours

Resume support:
    If interrupted, re-running will skip already-processed videos.
    Face crops and TALL grids are only created if they do not exist.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.config import cfg, MANIFEST_ROOT, PREPROCESSED_ROOT
from src.data.manifest import (
    build_manifests,
    get_manifest_stats,
    save_manifests,
)
from src.data.preprocessing import (
    CelebDFPreprocessor,
    DFDCPreprocessor,
    FFPlusPlusPreprocessor,
    VideoRecord,
)

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
        description="DeepTrace preprocessing pipeline"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["ff", "celeb", "dfdc"],
        default=["ff", "celeb", "dfdc"],
        help="Which datasets to preprocess. Default: all three.",
    )
    parser.add_argument(
        "--skip_manifest",
        action="store_true",
        help="Skip manifest generation (useful if only re-running one dataset).",
    )
    parser.add_argument(
        "--manifest_dir",
        type=str,
        default=str(MANIFEST_ROOT),
        help="Directory to save manifest CSVs.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("DeepTrace Preprocessing Pipeline")
    logger.info("=" * 60)
    logger.info("Datasets to process: %s", args.datasets)
    logger.info("Preprocessed output: %s", PREPROCESSED_ROOT)
    logger.info("Manifest output    : %s", args.manifest_dir)
    logger.info("=" * 60)

    cfg.validate()
    cfg.make_dirs()

    ff_records    : list[VideoRecord] = []
    celeb_records : list[VideoRecord] = []
    dfdc_records  : list[VideoRecord] = []

    total_start = time.time()

    # --- FF++ ---
    if "ff" in args.datasets:
        logger.info("Starting FF++ preprocessing...")
        start = time.time()
        ff_records = FFPlusPlusPreprocessor().run()
        elapsed = time.time() - start
        logger.info(
            "FF++ done: %d records in %.1f minutes",
            len(ff_records), elapsed / 60,
        )
    else:
        logger.info("Skipping FF++ (not in --datasets)")

    # --- Celeb-DF ---
    if "celeb" in args.datasets:
        logger.info("Starting Celeb-DF preprocessing...")
        start = time.time()
        celeb_records = CelebDFPreprocessor().run()
        elapsed = time.time() - start
        logger.info(
            "Celeb-DF done: %d records in %.1f minutes",
            len(celeb_records), elapsed / 60,
        )
    else:
        logger.info("Skipping Celeb-DF (not in --datasets)")

    # --- DFDC ---
    if "dfdc" in args.datasets:
        logger.info("Starting DFDC preprocessing...")
        start = time.time()
        dfdc_records = DFDCPreprocessor().run()
        elapsed = time.time() - start
        logger.info(
            "DFDC done: %d records in %.1f minutes",
            len(dfdc_records), elapsed / 60,
        )
    else:
        logger.info("Skipping DFDC (not in --datasets)")

    total_elapsed = time.time() - total_start
    logger.info(
        "Total preprocessing time: %.1f minutes", total_elapsed / 60
    )

    # --- Manifest Generation ---
    if args.skip_manifest:
        logger.info("Skipping manifest generation (--skip_manifest set)")
        return

    if not ff_records:
        logger.error(
            "FF++ records are empty. Cannot build manifests without FF++. "
            "Run with --datasets ff first."
        )
        sys.exit(1)

    if not celeb_records:
        logger.error(
            "Celeb-DF records are empty. Cannot build test manifest. "
            "Run with --datasets celeb first."
        )
        sys.exit(1)

    logger.info("Building train/val/test manifests...")
    manifests = build_manifests(ff_records, celeb_records, dfdc_records)
    saved     = save_manifests(manifests, output_dir=Path(args.manifest_dir))

    logger.info("=" * 60)
    logger.info("Manifest files saved:")
    for split, path in saved.items():
        logger.info("  %s --> %s", split, path)

    logger.info("=" * 60)
    logger.info("Dataset statistics:")
    for split, df in manifests.items():
        stats = get_manifest_stats(df)
        logger.info(
            "  [%s] total=%d | real=%d | fake=%d | "
            "imbalance_ratio=%.2f",
            split,
            stats["total"],
            stats["real"],
            stats["fake"],
            stats["imbalance_ratio"],
        )
        logger.info("  [%s] datasets: %s", split, stats["datasets"])

    logger.info("=" * 60)
    logger.info("Preprocessing pipeline complete.")


if __name__ == "__main__":
    main()