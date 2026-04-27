"""
preprocessing.py — Dataset preparation for all three models.

Run directly to build ready-to-train datasets:
    python preprocessing.py --model pose
    python preprocessing.py --model face
    python preprocessing.py --model lip
"""

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import config
from logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 — Pose / Keypoint  (YOLOv8-Pose fine-tune)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_pose_dataset(source_dir: str, output_dir: str) -> None:
    """
    Expects `source_dir` organised as:
        source_dir/
            images/   *.jpg | *.png
            labels/   *.txt  (YOLO-pose format: cls cx cy w h kp0x kp0y kp0v ...)

    Splits into train/val and writes a `dataset.yaml` for YOLOv8 training.
    """
    log.info("Preparing POSE dataset: %s → %s", source_dir, output_dir)
    img_dir = Path(source_dir) / "images"
    lbl_dir = Path(source_dir) / "labels"

    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not images:
        log.error("No images found in %s", img_dir)
        return

    train_imgs, val_imgs = train_test_split(
        images, test_size=config.TRAIN_VAL_SPLIT, random_state=42
    )

    for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
        (Path(output_dir) / split / "images").mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / split / "labels").mkdir(parents=True, exist_ok=True)
        for img_path in imgs:
            shutil.copy(img_path, Path(output_dir) / split / "images" / img_path.name)
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.copy(lbl_path, Path(output_dir) / split / "labels" / lbl_path.name)

    # Write dataset.yaml consumed by training.py
    yaml_content = (
        f"path: {os.path.abspath(output_dir)}\n"
        f"train: train/images\n"
        f"val:   val/images\n"
        f"nc: 1\n"
        f"names: ['person']\n"
        f"kpt_shape: [17, 3]\n"
    )
    yaml_path = Path(output_dir) / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    log.info("Pose dataset ready. YAML written to %s", yaml_path)


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 — Face Expression Classifier
# ─────────────────────────────────────────────────────────────────────────────

def prepare_face_dataset(source_dir: str, output_dir: str) -> None:
    """
    Expects `source_dir` with one sub-folder per expression label:
        source_dir/
            angry/      *.jpg
            happy/      *.jpg
            neutral/    *.jpg
            ...

    Resizes each image to FACE_INPUT_SIZE (48×48 grayscale),
    saves to output_dir/{train,val}/{class_name}/.
    Compatible with torchvision.datasets.ImageFolder.
    """
    log.info("Preparing FACE dataset: %s → %s", source_dir, output_dir)
    all_samples: list[tuple[Path, str]] = []

    for cls_name in config.FACE_CLASS_NAMES:
        cls_dir = Path(source_dir) / cls_name
        if not cls_dir.is_dir():
            log.warning("Missing class folder: %s — skipping", cls_dir)
            continue
        for img_path in cls_dir.glob("*.jpg"):
            all_samples.append((img_path, cls_name))
        for img_path in cls_dir.glob("*.png"):
            all_samples.append((img_path, cls_name))

    if not all_samples:
        log.error("No samples found in %s", source_dir)
        return

    train_s, val_s = train_test_split(
        all_samples, test_size=config.TRAIN_VAL_SPLIT, random_state=42,
        stratify=[s[1] for s in all_samples],
    )

    def _write(samples: list[tuple[Path, str]], split: str) -> None:
        for img_path, cls_name in samples:
            dest_dir = Path(output_dir) / split / cls_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            img = cv2.imread(str(img_path))
            if img is None:
                log.warning("Could not read %s — skipping", img_path)
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, config.FACE_INPUT_SIZE,
                                  interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dest_dir / img_path.name), resized)

    _write(train_s, "train")
    _write(val_s,   "val")
    log.info("Face dataset ready — %d train / %d val samples",
             len(train_s), len(val_s))


# ─────────────────────────────────────────────────────────────────────────────
# Model 3 — Lip Movement Classifier
# ─────────────────────────────────────────────────────────────────────────────

def prepare_lip_dataset(source_dir: str, output_dir: str) -> None:
    """
    Expects `source_dir` organised as:
        source_dir/
            still/           (sub-folders or individual frames)
            slight_movement/
            active_speech/

    Each sample is a single mouth-region crop.
    Resized to LIP_INPUT_SIZE (64×32).

    For video-based datasets, extract frames first with
    `extract_frames_from_video()` below.
    """
    log.info("Preparing LIP dataset: %s → %s", source_dir, output_dir)
    all_samples: list[tuple[Path, str]] = []

    for cls_name in config.LIP_CLASS_NAMES:
        cls_dir = Path(source_dir) / cls_name
        if not cls_dir.is_dir():
            log.warning("Missing class folder: %s — skipping", cls_dir)
            continue
        for img_path in list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")):
            all_samples.append((img_path, cls_name))

    if not all_samples:
        log.error("No lip samples found in %s", source_dir)
        return

    train_s, val_s = train_test_split(
        all_samples, test_size=config.TRAIN_VAL_SPLIT, random_state=42,
        stratify=[s[1] for s in all_samples],
    )

    def _write(samples: list[tuple[Path, str]], split: str) -> None:
        for img_path, cls_name in samples:
            dest_dir = Path(output_dir) / split / cls_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            img = cv2.imread(str(img_path))
            if img is None:
                log.warning("Could not read %s — skipping", img_path)
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, config.LIP_INPUT_SIZE,
                                  interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dest_dir / img_path.name), resized)

    _write(train_s, "train")
    _write(val_s,   "val")
    log.info("Lip dataset ready — %d train / %d val samples",
             len(train_s), len(val_s))


# ─────────────────────────────────────────────────────────────────────────────
# Utility — extract frames from a video file
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    every_n: int = 3,
) -> int:
    """
    Dump every `every_n`-th frame from a video into `output_dir`.
    Returns the number of saved frames.
    Useful for building the lip-movement dataset from bedside recordings.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Cannot open video: %s", video_path)
        return 0

    saved = 0
    idx   = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n == 0:
            out_path = os.path.join(output_dir, f"frame_{idx:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        idx += 1

    cap.release()
    log.info("Extracted %d frames from %s → %s", saved, video_path, output_dir)
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare datasets for training.")
    parser.add_argument(
        "--model",
        choices=["pose", "face", "lip"],
        required=True,
        help="Which model's dataset to prepare.",
    )
    args = parser.parse_args()

    if args.model == "pose":
        prepare_pose_dataset(
            source_dir=config.POSE_DATA_DIR,
            output_dir=os.path.join(config.DATA_DIR, "pose_processed"),
        )
    elif args.model == "face":
        prepare_face_dataset(
            source_dir=config.FACE_DATA_DIR,
            output_dir=os.path.join(config.DATA_DIR, "face_processed"),
        )
    elif args.model == "lip":
        prepare_lip_dataset(
            source_dir=config.LIP_DATA_DIR,
            output_dir=os.path.join(config.DATA_DIR, "lip_processed"),
        )
