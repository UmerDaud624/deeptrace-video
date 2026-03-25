"""
scripts/predict_video.py
-------------------------
Run deepfake inference on a single raw video file.

Performs the full pipeline internally:
    1. Sample frames from video
    2. Detect and crop faces using RetinaFace
    3. Build TALL thumbnail grid
    4. Run ensemble model (EfficientNet-B4 + Swin-T)
    5. Print confidence score

Usage:
    python scripts/predict_video.py --video path/to/video.mp4
    python scripts/predict_video.py --video path/to/video.mp4 --checkpoint path/to/ckpt.pt
    python scripts/predict_video.py --video path/to/video.mp4 --frames 32

Output:
    REAL   (confidence: 12.34%)   <-- low fake probability
    FAKE   (confidence: 87.65%)   <-- high fake probability
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import cfg, CHECKPOINT_ROOT
from src.data.preprocessing import (
    sample_frames,
    detect_face,
    build_tall_grid,
)
from src.models.ensemble import build_ensemble

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transforms (must match training exactly)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

FACE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

TALL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(
    video_path  : str,
    checkpoint  : str,
    num_frames  : int,
    device      : str,
    threshold   : float = 0.5,
) -> dict:
    """
    Run full deepfake inference pipeline on one video file.

    Args:
        video_path: Path to input video file.
        checkpoint: Path to trained model checkpoint (.pt).
        num_frames: Number of frames to sample from the video.
        device:     'cuda' or 'cpu'.
        threshold:  Decision threshold for FAKE verdict. Default: 0.5.

    Returns:
        Dict with keys:
            verdict     -- 'FAKE' or 'REAL'
            prob_fake   -- float in [0, 1]
            prob_real   -- float in [0, 1]
            faces_found -- int, number of valid face crops extracted
            frames_used -- int, number of frames sampled
            threshold   -- float, threshold used for verdict

    Raises:
        FileNotFoundError: If video or checkpoint not found.
        RuntimeError: If no faces detected in video.
    """
    video_path = str(video_path)
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    # --- Load model ---
    logger.info("Loading model from: %s", checkpoint)
    model = build_ensemble(fusion_type="logit", pretrained=False)
    ckpt  = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    model.to(device)

    val_auc = ckpt.get("score", "unknown")
    epoch   = ckpt.get("epoch", "unknown")
    logger.info("Checkpoint: epoch=%s | val_auc=%s", epoch, val_auc)

    # Log learned ensemble weights
    w_eff  = F.softplus(model.w_eff).item()
    w_swin = F.softplus(model.w_swin).item()
    w_sum  = w_eff + w_swin
    logger.info(
        "Ensemble weights: EfficientNet=%.3f | Swin-T=%.3f",
        w_eff / w_sum, w_swin / w_sum,
    )

    # --- Extract frames ---
    logger.info("Sampling %d frames from: %s", num_frames, video_path)
    frames = sample_frames(video_path, num_frames)
    if not frames:
        raise RuntimeError(f"Could not extract any frames from: {video_path}")
    logger.info("Extracted %d frames", len(frames))

    # --- Detect faces ---
    preproc = cfg.preprocessing
    crops: list[np.ndarray] = []

    for i, frame in enumerate(frames):
        result = detect_face(
            frame_bgr   = frame,
            frame_idx   = i,
            target_size = preproc.face_size,
            min_conf    = preproc.face_confidence,
            margin      = preproc.face_margin,
        )
        if result.success and result.crop is not None:
            crops.append(result.crop)

    logger.info(
        "Face detection: %d/%d frames had valid faces",
        len(crops), len(frames),
    )

    if len(crops) == 0:
        raise RuntimeError(
            "No faces detected in video. "
            "The video may not contain a visible face, "
            "or RetinaFace confidence threshold is too high."
        )

    # --- Build TALL grid ---
    tall_grid = build_tall_grid(
        crops       = crops,
        grid_rows   = preproc.tall_grid_rows,
        grid_cols   = preproc.tall_grid_cols,
        tile_size   = preproc.tall_face_size,
        output_size = 224,
    )

    # --- Prepare tensors ---
    # Face input: use the median crop as representative frame
    median_idx  = len(crops) // 2
    face_pil    = Image.fromarray(crops[median_idx])
    face_tensor = FACE_TRANSFORM(face_pil).unsqueeze(0).to(device)

    tall_pil    = Image.fromarray(tall_grid)
    tall_tensor = TALL_TRANSFORM(tall_pil).unsqueeze(0).to(device)

    # --- Run inference ---
    logger.info("Running inference...")
    with torch.no_grad():
        logit = model(face_tensor, tall_tensor)
        prob_fake = torch.sigmoid(logit).item()

    prob_real = 1.0 - prob_fake
    verdict   = "FAKE" if prob_fake >= threshold else "REAL"

    return {
        "verdict"     : verdict,
        "prob_fake"   : prob_fake,
        "prob_real"   : prob_real,
        "faces_found" : len(crops),
        "frames_used" : len(frames),
        "threshold"   : threshold,
    }


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepTrace -- single video deepfake detection"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file (mp4, avi, mov, mkv).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to model checkpoint (.pt). "
            "If not provided, uses the highest-AUC checkpoint in "
            "the configured checkpoint directory."
        ),
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=cfg.preprocessing.frames_per_video,
        help=f"Number of frames to sample. Default: {cfg.preprocessing.frames_per_video}",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on. Default: cuda if available.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Decision threshold for FAKE verdict. "
            "Lower = more sensitive (more FAKEs). "
            "Higher = more conservative. Default: 0.5"
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Auto-select best checkpoint by AUC if not provided
    checkpoint = args.checkpoint
    if checkpoint is None:
        import re
        candidates = glob.glob(str(CHECKPOINT_ROOT / "ensemble_epoch*_auc*.pt"))
        if not candidates:
            logger.error(
                "No checkpoints found in %s. "
                "Pass --checkpoint explicitly.",
                CHECKPOINT_ROOT,
            )
            sys.exit(1)

        def _extract_auc(p: str) -> float:
            m = re.search(r"auc([0-9.]+)", Path(p).stem)
            return float(m.group(1)) if m else 0.0

        checkpoint = max(candidates, key=_extract_auc)
        logger.info(
            "Auto-selected checkpoint: %s (AUC=%.4f)",
            checkpoint, _extract_auc(checkpoint),
        )

    print()
    print("=" * 50)
    print("DeepTrace -- Video Deepfake Detection")
    print("=" * 50)
    print(f"Video      : {args.video}")
    print(f"Checkpoint : {checkpoint}")
    print(f"Device     : {args.device}")
    print("=" * 50)

    try:
        result = predict(
            video_path = args.video,
            checkpoint = checkpoint,
            num_frames = args.frames,
            device     = args.device,
            threshold  = args.threshold,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("Inference failed: %s", exc)
        sys.exit(1)

    print()
    print("=" * 50)
    print("RESULT")
    print("=" * 50)
    print(f"Verdict    : {result['verdict']}")
    print(f"Fake prob  : {result['prob_fake'] * 100:.2f}%")
    print(f"Real prob  : {result['prob_real'] * 100:.2f}%")
    print(f"Threshold  : {result['threshold']:.2f}")
    print(f"Faces found: {result['faces_found']}/{result['frames_used']} frames")
    print("=" * 50)
    print()


if __name__ == "__main__":
    main()