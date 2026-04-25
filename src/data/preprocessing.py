"""
src/data/preprocessing.py
--------------------------
Face extraction, frame sampling, and TALL thumbnail grid creation.

Pipeline per dataset:
    FF++ and Celeb-DF:
        1. Sample N evenly spaced frames from each video
        2. Detect face in each frame using RetinaFace
        3. Crop and align face to fixed size
        4. Save individual face crops
        5. Tile N crops into TALL grid image
        6. Save TALL grid

    DFDC:
        Already pre-extracted face crops (PNG files).
        1. Load existing PNG crops grouped by video ID
        2. Resize to face_size
        3. Build TALL grid from sibling crops of same video
        4. Save TALL grid

Run this on Colab where datasets live on Drive.
All paths resolve automatically via config.py environment detection.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs.config import cfg, PREPROCESSED_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class VideoRecord:
    """Metadata for one processed video or image sequence."""

    source_path    : str
    label          : int    # 0 = real, 1 = fake
    dataset        : str    # ff_plus_plus / celeb_df / dfdc
    subset         : str    # e.g. ff_deepfakes, celeb_df_fake, dfdc_real
    face_dir       : str    # directory of saved face crop JPGs
    tall_path      : str    # path to saved TALL grid JPG
    num_faces      : int    # number of valid face crops extracted
    frames_sampled : int    # number of frames attempted


@dataclass
class FaceResult:
    """Result of face detection on one frame."""

    frame_idx  : int
    success    : bool
    crop       : Optional[np.ndarray] = None    # RGB HxWx3
    confidence : float = 0.0


# ---------------------------------------------------------------------------
# Frame Sampling
# ---------------------------------------------------------------------------

def sample_frames(
    video_path : str,
    num_frames : int,
) -> list[np.ndarray]:
    """
    Sample N evenly spaced frames from a video file.

    Skips first and last 5 percent to avoid transitions and
    black frames common in dataset videos.

    Args:
        video_path: Absolute path to MP4 file.
        num_frames: Number of frames to sample.

    Returns:
        List of BGR numpy arrays. May be shorter than num_frames
        if video has fewer decodable frames than requested.

    Raises:
        ValueError: If video cannot be opened or has zero frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        raise ValueError(f"Video has 0 frames: {video_path}")

    start   = max(0, int(total * 0.05))
    end     = min(total - 1, int(total * 0.95))
    indices = np.unique(np.linspace(start, end, num_frames, dtype=int))

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)

    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Face Detection
# ---------------------------------------------------------------------------

def detect_face(
    frame_bgr   : np.ndarray,
    frame_idx   : int,
    target_size : int,
    min_conf    : float,
    margin      : float,
) -> FaceResult:
    """
    Detect the highest-confidence face in one frame.

    Args:
        frame_bgr:   OpenCV BGR frame.
        frame_idx:   Frame index for logging only.
        target_size: Output crop size in pixels (square).
        min_conf:    Minimum RetinaFace confidence to accept detection.
        margin:      Fractional padding added around detected bounding box.

    Returns:
        FaceResult. crop is None if no valid face detected.
    """
    from retinaface import RetinaFace

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]

    try:
        detections = RetinaFace.detect_faces(frame_rgb)
    except Exception as exc:
        logger.debug("RetinaFace error frame %d: %s", frame_idx, exc)
        return FaceResult(frame_idx=frame_idx, success=False)

    if not isinstance(detections, dict) or len(detections) == 0:
        return FaceResult(frame_idx=frame_idx, success=False)

    best = max(detections, key=lambda k: detections[k].get("score", 0.0))
    det  = detections[best]
    conf = float(det.get("score", 0.0))

    if conf < min_conf:
        return FaceResult(frame_idx=frame_idx, success=False, confidence=conf)

    x1, y1, x2, y2 = [int(v) for v in det["facial_area"]]
    fw, fh = x2 - x1, y2 - y1

    x1 = max(0, int(x1 - margin * fw))
    y1 = max(0, int(y1 - margin * fh))
    x2 = min(w, int(x2 + margin * fw))
    y2 = min(h, int(y2 + margin * fh))

    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return FaceResult(frame_idx=frame_idx, success=False)

    crop = cv2.resize(
        crop, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4
    )
    return FaceResult(
        frame_idx=frame_idx, success=True, crop=crop, confidence=conf
    )


# ---------------------------------------------------------------------------
# TALL Thumbnail Grid
# ---------------------------------------------------------------------------

def build_tall_grid(
    crops       : list[np.ndarray],
    grid_rows   : int,
    grid_cols   : int,
    tile_size   : int,
    output_size : int,
) -> np.ndarray:
    """
    Tile face crops into a TALL thumbnail grid image.

    Repeats last crop to fill grid if fewer crops available than
    grid_rows * grid_cols. Final grid is resized to output_size
    for Swin Transformer input.

    Args:
        crops:       List of RGB numpy arrays.
        grid_rows:   Number of rows.
        grid_cols:   Number of columns.
        tile_size:   Pixel size of each tile inside the grid.
        output_size: Final image size after resize.

    Returns:
        RGB numpy array (output_size, output_size, 3).
    """
    total = grid_rows * grid_cols

    while len(crops) < total:
        crops.append(crops[-1].copy())
    crops = crops[:total]

    tiles = [
        cv2.resize(c, (tile_size, tile_size), interpolation=cv2.INTER_LANCZOS4)
        for c in crops
    ]

    rows = []
    for r in range(grid_rows):
        row = np.concatenate(tiles[r * grid_cols:(r + 1) * grid_cols], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)

    return cv2.resize(
        grid, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4
    )


# ---------------------------------------------------------------------------
# Single Video Processor (FF++ and Celeb-DF)
# ---------------------------------------------------------------------------

def process_video(
    video_path : str,
    face_dir   : str,
    tall_path  : str,
    label      : int,
) -> tuple[int, int]:
    """
    Full preprocessing pipeline for one video file.

    Skips processing if outputs already exist (safe to re-run).

    Args:
        video_path: Path to source MP4.
        face_dir:   Output directory for face crop JPGs.
        tall_path:  Output path for TALL grid JPG.
        label:      Ground truth label (unused here, for caller logging).

    Returns:
        (num_faces_extracted, num_frames_sampled)
    """
    preproc = cfg.preprocessing
    os.makedirs(face_dir, exist_ok=True)
    os.makedirs(os.path.dirname(tall_path), exist_ok=True)

    # Resume support -- skip if already done
    existing = list(Path(face_dir).glob("*.jpg"))
    if len(existing) > 0 and Path(tall_path).exists():
        return len(existing), preproc.frames_per_video

    try:
        frames = sample_frames(video_path, preproc.frames_per_video)
    except ValueError as exc:
        logger.warning("Frame sampling failed: %s | %s", video_path, exc)
        return 0, 0

    crops = []
    for i, frame in enumerate(frames):
        result = detect_face(
            frame_bgr=frame,
            frame_idx=i,
            target_size=preproc.face_size,
            min_conf=preproc.face_confidence,
            margin=preproc.face_margin,
        )
        if result.success and result.crop is not None:
            crop_path = os.path.join(face_dir, f"frame_{i:04d}.jpg")
            Image.fromarray(result.crop).save(
                crop_path, quality=preproc.jpg_quality
            )
            crops.append(result.crop)

    if len(crops) > 0:
        grid = build_tall_grid(
            crops=crops,
            grid_rows=preproc.tall_grid_rows,
            grid_cols=preproc.tall_grid_cols,
            tile_size=preproc.tall_face_size,
            output_size=preproc.face_size,
        )
        Image.fromarray(grid).save(tall_path, quality=preproc.jpg_quality)

    return len(crops), len(frames)


# ---------------------------------------------------------------------------
# DFDC Processor (pre-extracted PNG crops)
# ---------------------------------------------------------------------------

def process_dfdc_image(
    image_path : str,
    face_dir   : str,
    tall_path  : str,
) -> tuple[int, int]:
    """
    Process one DFDC pre-extracted face crop.

    Finds sibling frames from the same video sequence to build
    a coherent TALL grid instead of repeating one frame.

    Args:
        image_path: Path to representative PNG face crop.
        face_dir:   Output directory for resized JPG.
        tall_path:  Output path for TALL grid JPG.

    Returns:
        (num_crops_used, 1)
    """
    preproc = cfg.preprocessing
    os.makedirs(face_dir, exist_ok=True)
    os.makedirs(os.path.dirname(tall_path), exist_ok=True)

    existing = list(Path(face_dir).glob("*.jpg"))
    if len(existing) > 0 and Path(tall_path).exists():
        return len(existing), 1

    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(
            (preproc.face_size, preproc.face_size), Image.LANCZOS
        )
        crop_np = np.array(img)
    except Exception as exc:
        logger.warning("Failed to load DFDC image %s: %s", image_path, exc)
        return 0, 1

    crop_path = os.path.join(face_dir, "frame_0000.jpg")
    img.save(crop_path, quality=preproc.jpg_quality)

    # Find sibling frames from same video sequence
    # DFDC filenames: {video_id}_{frame}_{face}.png
    stem   = Path(image_path).stem
    parts  = stem.rsplit("_", 2)
    video_id   = parts[0] if len(parts) >= 2 else stem
    parent_dir = Path(image_path).parent
    siblings   = sorted(parent_dir.glob(f"{video_id}_*.png"))

    crops = []
    for sib in siblings[:preproc.tall_grid_size]:
        try:
            sib_np = np.array(
                Image.open(sib).convert("RGB").resize(
                    (preproc.face_size, preproc.face_size), Image.LANCZOS
                )
            )
            crops.append(sib_np)
        except Exception:
            continue

    if len(crops) == 0:
        crops = [crop_np]

    grid = build_tall_grid(
        crops=crops,
        grid_rows=preproc.tall_grid_rows,
        grid_cols=preproc.tall_grid_cols,
        tile_size=preproc.tall_face_size,
        output_size=preproc.face_size,
    )
    Image.fromarray(grid).save(tall_path, quality=preproc.jpg_quality)

    return len(crops), 1


# ---------------------------------------------------------------------------
# Dataset Preprocessors
# ---------------------------------------------------------------------------

class FFPlusPlusPreprocessor:
    """Preprocesses FaceForensics++ videos -- all manipulation types."""

    def __init__(self) -> None:
        self.paths   = cfg.paths
        self.out_dir = PREPROCESSED_ROOT / "ff_plus_plus"

    def _collect(self) -> list[tuple[str, int, str]]:
        entries = []

        if self.paths.ff_original.exists():
            for mp4 in sorted(self.paths.ff_original.glob("*.mp4")):
                entries.append((str(mp4), 0, "ff_original"))
        else:
            logger.warning("Not found: %s", self.paths.ff_original)

        fake_map = {
            "ff_deepfakes"       : self.paths.ff_deepfakes,
            "ff_face2face"       : self.paths.ff_face2face,
            "ff_faceswap"        : self.paths.ff_faceswap,
            "ff_neural_textures" : self.paths.ff_neural_textures,
        }
        for subset, d in fake_map.items():
            if d.exists():
                for mp4 in sorted(d.glob("*.mp4")):
                    entries.append((str(mp4), 1, subset))
            else:
                logger.warning("Not found: %s", d)

        logger.info("FF++ collected %d videos", len(entries))
        return entries

    def run(self) -> list[VideoRecord]:
        """Run preprocessing. Returns list of VideoRecord."""
        entries = self._collect()
        records, failed = [], 0

        for video_path, label, subset in tqdm(
            entries, desc="FF++", unit="video"
        ):
            stem      = Path(video_path).stem
            face_dir  = str(self.out_dir / "faces" / subset / stem)
            tall_path = str(self.out_dir / "tall"  / subset / f"{stem}.jpg")

            try:
                num_faces, num_frames = process_video(
                    video_path, face_dir, tall_path, label
                )
            except Exception as exc:
                logger.warning("Error %s: %s", video_path, exc)
                failed += 1
                continue

            if num_faces > 0:
                records.append(VideoRecord(
                    source_path=video_path, label=label,
                    dataset="ff_plus_plus", subset=subset,
                    face_dir=face_dir, tall_path=tall_path,
                    num_faces=num_faces, frames_sampled=num_frames,
                ))
            else:
                failed += 1

        logger.info("FF++ done: %d ok, %d failed", len(records), failed)
        return records


class CelebDFPreprocessor:
    """Preprocesses Celeb-DF v2 videos."""

    def __init__(self) -> None:
        self.base    = cfg.paths.celeb_df
        self.out_dir = PREPROCESSED_ROOT / "celeb_df"

    def _collect(self) -> list[tuple[str, int, str]]:
        entries = []

        for subset, dirname in {
            "celeb_real"   : "Celeb-real",
            "youtube_real" : "YouTube-real",
        }.items():
            d = self.base / dirname
            if d.exists():
                for mp4 in sorted(d.glob("*.mp4")):
                    entries.append((str(mp4), 0, subset))
            else:
                logger.warning("Not found: %s", d)

        d = self.base / "Celeb-synthesis"
        if d.exists():
            for mp4 in sorted(d.glob("*.mp4")):
                entries.append((str(mp4), 1, "celeb_fake"))
        else:
            logger.warning("Not found: %s", d)

        logger.info("Celeb-DF collected %d videos", len(entries))
        return entries

    def run(self) -> list[VideoRecord]:
        """Run preprocessing. Returns list of VideoRecord."""
        entries = self._collect()
        records, failed = [], 0

        for video_path, label, subset in tqdm(
            entries, desc="Celeb-DF", unit="video"
        ):
            stem      = Path(video_path).stem
            face_dir  = str(self.out_dir / "faces" / subset / stem)
            tall_path = str(self.out_dir / "tall"  / subset / f"{stem}.jpg")

            try:
                num_faces, num_frames = process_video(
                    video_path, face_dir, tall_path, label
                )
            except Exception as exc:
                logger.warning("Error %s: %s", video_path, exc)
                failed += 1
                continue

            if num_faces > 0:
                records.append(VideoRecord(
                    source_path=video_path, label=label,
                    dataset="celeb_df", subset=subset,
                    face_dir=face_dir, tall_path=tall_path,
                    num_faces=num_faces, frames_sampled=num_frames,
                ))
            else:
                failed += 1

        logger.info("Celeb-DF done: %d ok, %d failed", len(records), failed)
        return records


class DFDCPreprocessor:
    """
    Preprocesses DFDC pre-extracted face crops.

    Groups PNG files by video ID and takes one representative
    entry per video sequence to avoid duplicate records.
    """

    def __init__(self) -> None:
        self.base    = cfg.paths.dfdc / "train"
        self.out_dir = PREPROCESSED_ROOT / "dfdc"

    def _collect(self) -> list[tuple[str, int, str]]:
        entries = []

        def by_video(
            directory: Path, label: int, subset: str
        ) -> list[tuple[str, int, str]]:
            seen: set[str] = set()
            result = []
            if not directory.exists():
                logger.warning("Not found: %s", directory)
                return result
            for png in sorted(directory.glob("*.png")):
                parts    = png.stem.rsplit("_", 2)
                video_id = parts[0] if len(parts) >= 2 else png.stem
                if video_id not in seen:
                    seen.add(video_id)
                    result.append((str(png), label, subset))
            return result

        entries.extend(by_video(self.base / "real", 0, "dfdc_real"))
        entries.extend(by_video(self.base / "fake", 1, "dfdc_fake"))

        logger.info("DFDC collected %d unique sequences", len(entries))
        return entries

    def run(self) -> list[VideoRecord]:
        """Run preprocessing. Returns list of VideoRecord."""
        entries = self._collect()
        records, failed = [], 0

        for image_path, label, subset in tqdm(
            entries, desc="DFDC", unit="sequence"
        ):
            stem      = Path(image_path).stem
            face_dir  = str(self.out_dir / "faces" / subset / stem)
            tall_path = str(self.out_dir / "tall"  / subset / f"{stem}.jpg")

            try:
                num_crops, _ = process_dfdc_image(
                    image_path, face_dir, tall_path
                )
            except Exception as exc:
                logger.warning("Error %s: %s", image_path, exc)
                failed += 1
                continue

            if num_crops > 0:
                records.append(VideoRecord(
                    source_path=image_path, label=label,
                    dataset="dfdc", subset=subset,
                    face_dir=face_dir, tall_path=tall_path,
                    num_faces=num_crops, frames_sampled=num_crops,
                ))
            else:
                failed += 1

        logger.info("DFDC done: %d ok, %d failed", len(records), failed)
        return records


class DFDCVideoPreprocessor:
    """
    Preprocesses raw DFDC MP4 videos from downloaded training parts.

    Unlike DFDCPreprocessor (which handles pre-extracted PNG crops),
    this runs the full RetinaFace pipeline on raw videos -- same as
    FFPlusPlusPreprocessor and CelebDFPreprocessor.

    Expected input structure:
        <parts_root>/
            part18/
                dfdc_train_part_18/
                    metadata.json       <-- label file
                    aaabcd.mp4
                    ...
            part19/
                dfdc_train_part_19/
                    metadata.json
                    ...

    Output structure:
        <preprocessed_root>/dfdc_video/
            faces/<subset>/<video_stem>/  <-- face crop JPGs
            tall/<subset>/<video_stem>.jpg  <-- TALL grid
    """

    def __init__(self, parts_root: str) -> None:
        """
        Args:
            parts_root: Path to the directory containing downloaded DFDC parts.
                        e.g. /content/drive/MyDrive/DeepTrace/datasets/DFDC/parts
        """
        self.parts_root = Path(parts_root)
        self.out_dir    = PREPROCESSED_ROOT / "dfdc_video"

    def _collect(self) -> list[tuple[str, int, str]]:
        """
        Walk all part directories, read metadata.json, collect video paths
        with their labels.

        Returns:
            List of (video_path, label, subset) tuples.
            subset is 'dfdc_video_real' or 'dfdc_video_fake'.
        """
        entries = []

        if not self.parts_root.exists():
            logger.warning("DFDC parts root not found: %s", self.parts_root)
            return entries

        # Walk each part directory
        for part_dir in sorted(self.parts_root.iterdir()):
            if not part_dir.is_dir():
                continue

            # Videos may be directly in part_dir or one level deeper
            metadata_candidates = list(part_dir.rglob("metadata.json"))
            if not metadata_candidates:
                logger.warning("No metadata.json in %s -- skipping", part_dir)
                continue

            metadata_path = metadata_candidates[0]
            video_dir     = metadata_path.parent

            try:
                with open(metadata_path) as f:
                    import json
                    metadata = json.load(f)
            except Exception as exc:
                logger.warning("Failed to load metadata %s: %s", metadata_path, exc)
                continue

            part_real, part_fake = 0, 0
            for filename, info in metadata.items():
                label_str = info.get("label", "").upper()
                video_path = video_dir / filename

                if not video_path.exists():
                    continue

                if label_str == "REAL":
                    entries.append((str(video_path), 0, "dfdc_video_real"))
                    part_real += 1
                elif label_str == "FAKE":
                    entries.append((str(video_path), 1, "dfdc_video_fake"))
                    part_fake += 1

            logger.info(
                "Part %s: %d real, %d fake",
                part_dir.name, part_real, part_fake,
            )

        logger.info(
            "DFDC video collected %d total videos (%d parts)",
            len(entries),
            sum(1 for p in self.parts_root.iterdir() if p.is_dir()),
        )
        return entries

    def run(self) -> list[VideoRecord]:
        """
        Run full RetinaFace preprocessing on all collected videos.
        Resume-safe: skips already-processed videos.

        Returns:
            List of VideoRecord objects ready for manifest building.
        """
        entries = self._collect()
        if not entries:
            logger.warning("No DFDC video entries found. Check parts_root path.")
            return []

        records, failed = [], 0

        for video_path, label, subset in tqdm(
            entries, desc="DFDC-video", unit="video"
        ):
            stem      = Path(video_path).stem
            face_dir  = str(self.out_dir / "faces" / subset / stem)
            tall_path = str(self.out_dir / "tall"  / subset / f"{stem}.jpg")

            try:
                num_faces, num_frames = process_video(
                    video_path, face_dir, tall_path, label
                )
            except Exception as exc:
                logger.warning("Error %s: %s", video_path, exc)
                failed += 1
                continue

            if num_faces > 0:
                records.append(VideoRecord(
                    source_path=video_path, label=label,
                    dataset="dfdc", subset=subset,
                    face_dir=face_dir, tall_path=tall_path,
                    num_faces=num_faces, frames_sampled=num_frames,
                ))
            else:
                failed += 1

        logger.info(
            "DFDC video done: %d ok, %d failed", len(records), failed
        )
        return records


class DF40Preprocessor:
    """
    Preprocesses DF40 dataset subsets for DeepTrace training.

    DF40 contains image-based deepfakes (not videos). This preprocessor
    handles three structural patterns found across DF40 methods:

    Pattern A -- identity subfolders in fake, flat real:
        Method/
            fake/<identity>/*.png
            real/*.jpg

    Pattern B -- flat fake with GIF/PNG/JPG, flat real:
        Method/
            fake/*.gif  (or *.png, *.jpg)
            real/*.jpg

    Pattern C -- domain subfolders then identity subfolders, no real:
        Method/
            <domain>/<identity>/*.png
            (no real folder -- real images come from FF++/CDF)

    The preprocessor auto-detects which pattern applies at runtime.
    For patterns with real images, both real and fake are processed.
    For fake-only patterns (C), only fakes are processed and real
    images are sourced from the existing FF++ preprocessed records.

    Face detection runs RetinaFace on each image. The TALL grid is
    built by tiling multiple crops from the same identity folder
    (or by repeating the single crop for flat structures).

    Output structure mirrors existing preprocessors:
        <preprocessed_root>/df40/<method_name>/
            faces/<subset>/<identity_or_stem>/  <-- face crop JPGs
            tall/<subset>/<identity_or_stem>.jpg  <-- TALL grid JPG

    Args:
        method_root: Path to extracted DF40 method directory.
                     e.g. /content/drive/.../DF40/CollabDiff
        method_name: Short name used in manifests and output paths.
                     e.g. 'collabdiff', 'ddim', 'midjourney'
        max_fake:    Maximum number of fake samples to process.
                     Keeps training set small. Default: 2000.
        max_real:    Maximum number of real samples to process.
                     Only used when real folder exists. Default: 500.
    """

    def __init__(
        self,
        method_root : str,
        method_name : str,
        max_fake    : int = 2000,
        max_real    : int = 500,
    ) -> None:
        self.method_root = Path(method_root)
        self.method_name = method_name.lower().replace(" ", "_")
        self.max_fake    = max_fake
        self.max_real    = max_real
        self.out_dir     = PREPROCESSED_ROOT / "df40" / self.method_name

    # -------------------------------------------------------------------
    # Structure detection
    # -------------------------------------------------------------------

    def _detect_pattern(self) -> str:
        """
        Auto-detect the folder structure pattern.

        Returns:
            'A' -- identity subfolders in fake, real folder exists
            'B' -- flat fake files, real folder exists
            'C' -- domain/identity subfolders, no real folder
            'D' -- fake/frames/<clips> + real/<clips> (HeyGen style)
        """
        fake_dir = self.method_root / "fake"
        real_dir = self.method_root / "real"
        has_real = real_dir.exists() and any(real_dir.iterdir())

        if not fake_dir.exists():
            # Pattern C: no top-level fake/ -- domain subfolders
            logger.info("[%s] Pattern C detected (domain subfolders, no real)",
                        self.method_name)
            return "C"

        if not has_real:
            logger.info("[%s] Pattern C detected (fake/ exists, no real/)",
                        self.method_name)
            return "C"

        # Check for Pattern D: fake/frames/ subfolder (HeyGen style)
        frames_dir = fake_dir / "frames"
        if frames_dir.exists() and frames_dir.is_dir():
            logger.info("[%s] Pattern D detected (frames/ subdir + real/ clips)",
                        self.method_name)
            return "D"

        # Check if fake/ has subfolders or flat files
        fake_contents = list(fake_dir.iterdir())
        has_subfolders = any(p.is_dir() for p in fake_contents)

        if has_subfolders:
            logger.info("[%s] Pattern A detected (identity subfolders + real/)",
                        self.method_name)
            return "A"
        else:
            logger.info("[%s] Pattern B detected (flat fake files + real/)",
                        self.method_name)
            return "B"

    # -------------------------------------------------------------------
    # Image collection per pattern
    # -------------------------------------------------------------------

    def _collect_pattern_a(self) -> tuple[list, list]:
        """
        Collect (image_path, identity) tuples for Pattern A.
        fake/<identity>/*.png  +  real/*.jpg

        Returns:
            (fake_entries, real_entries)
            Each entry: (image_path, identity_stem)
        """
        fake_dir = self.method_root / "fake"
        real_dir = self.method_root / "real"

        fake_entries = []
        for identity_dir in sorted(fake_dir.iterdir()):
            if not identity_dir.is_dir():
                continue
            images = sorted(
                p for p in identity_dir.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif"}
            )
            for img_path in images:
                fake_entries.append((img_path, identity_dir.name))

        real_entries = []
        for img_path in sorted(real_dir.iterdir()):
            if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                real_entries.append((img_path, img_path.stem))

        return fake_entries, real_entries

    def _collect_pattern_b(self) -> tuple[list, list]:
        """
        Collect entries for Pattern B.
        fake/*.gif (or *.png/*.jpg)  +  real/*.jpg

        Returns:
            (fake_entries, real_entries)
        """
        fake_dir = self.method_root / "fake"
        real_dir = self.method_root / "real"

        fake_entries = [
            (p, p.stem)
            for p in sorted(fake_dir.iterdir())
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif"}
        ]

        real_entries = [
            (p, p.stem)
            for p in sorted(real_dir.iterdir())
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]

        return fake_entries, real_entries

    def _collect_pattern_c(self) -> tuple[list, list]:
        """
        Collect entries for Pattern C.
        <domain>/<identity>/*.png  OR  fake/<domain>/<identity>/*.png
        No real images.

        Returns:
            (fake_entries, [])
        """
        fake_entries = []

        # Check if there is a top-level fake/ dir or domain dirs directly
        fake_dir = self.method_root / "fake"
        if fake_dir.exists():
            search_root = fake_dir
        else:
            search_root = self.method_root

        for domain_or_identity in sorted(search_root.iterdir()):
            if not domain_or_identity.is_dir():
                continue
            # Check if this is a domain folder (contains identity subfolders)
            # or an identity folder (contains images directly)
            contents = list(domain_or_identity.iterdir())
            has_subfolders = any(p.is_dir() for p in contents)

            if has_subfolders:
                # Domain folder -- recurse into identity subfolders
                for identity_dir in sorted(domain_or_identity.iterdir()):
                    if not identity_dir.is_dir():
                        continue
                    images = sorted(
                        p for p in identity_dir.iterdir()
                        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                    )
                    stem = f"{domain_or_identity.name}_{identity_dir.name}"
                    for img_path in images:
                        fake_entries.append((img_path, stem))
            else:
                # Identity folder directly under search_root
                images = sorted(
                    p for p in contents
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                )
                for img_path in images:
                    fake_entries.append((img_path, domain_or_identity.name))

        return fake_entries, []

    def _collect_pattern_d(self) -> tuple[list, list]:
        """
        Collect entries for Pattern D (HeyGen style).
        fake/frames/<clip_id>/*.png  +  real/<clip_id>/*.png

        Returns:
            (fake_entries, real_entries)
            Each entry: (image_path, clip_stem)
        """
        frames_dir = self.method_root / "fake" / "frames"
        real_dir   = self.method_root / "real"

        fake_entries = []
        for clip_dir in sorted(frames_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            images = sorted(
                p for p in clip_dir.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
            for img_path in images:
                fake_entries.append((img_path, f"clip_{clip_dir.name}"))

        real_entries = []
        for clip_dir in sorted(real_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            images = sorted(
                p for p in clip_dir.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
            # Sanitize clip name -- Drive clip names have + and special chars
            safe_name = clip_dir.name.replace("+", "_").replace(" ", "_")[:50]
            for img_path in images:
                real_entries.append((img_path, safe_name))

        return fake_entries, real_entries

    # -------------------------------------------------------------------
    # Single image processing
    # -------------------------------------------------------------------

    def _process_image_group(
        self,
        image_paths : list[Path],
        identity    : str,
        label       : int,
        subset      : str,
    ) -> tuple[int, int]:
        """
        Process a group of images belonging to the same identity.

        Runs RetinaFace on each image, saves face crops, builds TALL grid.
        Resume-safe: skips if outputs already exist.

        Args:
            image_paths: List of image paths for this identity.
            identity:    Identity/stem string for output naming.
            label:       0 = real, 1 = fake.
            subset:      Subset name for output path.

        Returns:
            (num_faces, num_images_attempted)
        """
        face_dir  = str(self.out_dir / "faces" / subset / identity)
        tall_path = str(self.out_dir / "tall"  / subset / f"{identity}.jpg")

        # Resume-safe check
        face_dir_path = Path(face_dir)
        if face_dir_path.exists() and Path(tall_path).exists():
            existing = list(face_dir_path.glob("*.jpg"))
            if existing:
                return len(existing), len(image_paths)

        face_dir_path.mkdir(parents=True, exist_ok=True)
        Path(tall_path).parent.mkdir(parents=True, exist_ok=True)

        preproc = cfg.preprocessing
        crops: list[np.ndarray] = []

        for i, img_path in enumerate(image_paths):
            try:
                # Handle GIF by reading first frame
                if img_path.suffix.lower() == ".gif":
                    from PIL import Image as PILImage
                    pil_img = PILImage.open(img_path).convert("RGB")
                    frame_rgb = np.array(pil_img)
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = cv2.imread(str(img_path))
                    if frame_bgr is None:
                        continue
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            except Exception as exc:
                logger.debug("Failed to load %s: %s", img_path, exc)
                continue

            result = detect_face(
                frame_bgr   = frame_bgr,
                frame_idx   = i,
                target_size = preproc.face_size,
                min_conf    = preproc.face_confidence,
                margin      = preproc.face_margin,
            )

            if result.success and result.crop is not None:
                crop_path = face_dir_path / f"{i:04d}.jpg"
                crop_bgr  = cv2.cvtColor(result.crop, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(crop_path), crop_bgr,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                crops.append(result.crop)

        if not crops:
            return 0, len(image_paths)

        # Build TALL grid
        tall_grid = build_tall_grid(
            crops       = crops,
            grid_rows   = preproc.tall_grid_rows,
            grid_cols   = preproc.tall_grid_cols,
            tile_size   = preproc.tall_face_size,
            output_size = 224,
        )
        tall_bgr = cv2.cvtColor(tall_grid, cv2.COLOR_RGB2BGR)
        cv2.imwrite(tall_path, tall_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return len(crops), len(image_paths)

    # -------------------------------------------------------------------
    # Main run
    # -------------------------------------------------------------------

    def run(self) -> list[VideoRecord]:
        """
        Run preprocessing on this DF40 method.

        Auto-detects structure, collects image paths, runs RetinaFace,
        saves face crops and TALL grids, returns VideoRecord list.

        Returns:
            List of VideoRecord objects ready for manifest building.
        """
        if not self.method_root.exists():
            logger.warning(
                "DF40 method root not found: %s", self.method_root
            )
            return []

        pattern = self._detect_pattern()

        if pattern == "A":
            fake_entries, real_entries = self._collect_pattern_a()
        elif pattern == "B":
            fake_entries, real_entries = self._collect_pattern_b()
        elif pattern == "D":
            fake_entries, real_entries = self._collect_pattern_d()
        else:
            fake_entries, real_entries = self._collect_pattern_c()

        # Cap at max_fake and max_real
        # Shuffle before capping to get variety across identities
        import random
        random.seed(42)
        random.shuffle(fake_entries)
        random.shuffle(real_entries)
        fake_entries = fake_entries[:self.max_fake]
        real_entries = real_entries[:self.max_real]

        logger.info(
            "[%s] Pattern %s | fake: %d | real: %d",
            self.method_name, pattern,
            len(fake_entries), len(real_entries),
        )

        # Group fake entries by identity
        from collections import defaultdict
        fake_groups: dict[str, list[Path]] = defaultdict(list)
        for img_path, identity in fake_entries:
            fake_groups[identity].append(img_path)

        real_groups: dict[str, list[Path]] = defaultdict(list)
        for img_path, identity in real_entries:
            real_groups[identity].append(img_path)

        records: list[VideoRecord] = []
        failed = 0

        # Process fakes
        fake_subset = f"df40_{self.method_name}_fake"
        for identity, images in tqdm(
            fake_groups.items(),
            desc=f"DF40-{self.method_name}-fake",
            unit="identity",
        ):
            num_faces, num_imgs = self._process_image_group(
                image_paths = images,
                identity    = identity,
                label       = 1,
                subset      = fake_subset,
            )
            if num_faces > 0:
                face_dir  = str(
                    self.out_dir / "faces" / fake_subset / identity
                )
                tall_path = str(
                    self.out_dir / "tall" / fake_subset / f"{identity}.jpg"
                )
                records.append(VideoRecord(
                    source_path    = str(images[0].parent),
                    label          = 1,
                    dataset        = "df40",
                    subset         = fake_subset,
                    face_dir       = face_dir,
                    tall_path      = tall_path,
                    num_faces      = num_faces,
                    frames_sampled = num_imgs,
                ))
            else:
                failed += 1

        # Process reals (only for patterns A and B)
        if real_groups:
            real_subset = f"df40_{self.method_name}_real"
            for identity, images in tqdm(
                real_groups.items(),
                desc=f"DF40-{self.method_name}-real",
                unit="identity",
            ):
                num_faces, num_imgs = self._process_image_group(
                    image_paths = images,
                    identity    = identity,
                    label       = 0,
                    subset      = real_subset,
                )
                if num_faces > 0:
                    face_dir  = str(
                        self.out_dir / "faces" / real_subset / identity
                    )
                    tall_path = str(
                        self.out_dir / "tall" / real_subset / f"{identity}.jpg"
                    )
                    records.append(VideoRecord(
                        source_path    = str(images[0]),
                        label          = 0,
                        dataset        = "df40",
                        subset         = real_subset,
                        face_dir       = face_dir,
                        tall_path      = tall_path,
                        num_faces      = num_faces,
                        frames_sampled = num_imgs,
                    ))
                else:
                    failed += 1

        real_count = sum(1 for r in records if r.label == 0)
        fake_count = sum(1 for r in records if r.label == 1)
        logger.info(
            "[%s] Done: %d ok (%d real, %d fake) | %d failed",
            self.method_name, len(records),
            real_count, fake_count, failed,
        )
        return records