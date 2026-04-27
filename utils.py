"""
utils.py — Shared helper utilities used across the pipeline.
"""

import os
import sqlite3
import time
from datetime import datetime
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np

import config
from logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Database helpers
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Create the SQLite events table if it does not exist.
    Only timestamps and alert metadata are stored — no raw video.
    """
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id   TEXT    NOT NULL,
                event_type  TEXT    NOT NULL,   -- e.g. 'inactivity', 'distress', 'lip_activity'
                track_id    INTEGER,             -- DeepSORT person ID
                confidence  REAL,
                timestamp   TEXT    NOT NULL     -- ISO-8601
            )
        """)
        conn.commit()
    log.info("Database initialised at %s", config.DB_PATH)


def log_event(
    camera_id: str,
    event_type: str,
    track_id: Optional[int] = None,
    confidence: Optional[float] = None,
) -> None:
    """
    Insert one flagged event row.  No raw frames or PII are written.
    """
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.execute(
            "INSERT INTO events (camera_id, event_type, track_id, confidence, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (camera_id, event_type, track_id, confidence, ts),
        )
        conn.commit()
    log.debug("Event logged: camera=%s type=%s track=%s conf=%.2f",
              camera_id, event_type, track_id, confidence or 0.0)


def fetch_events(limit: int = 100) -> List[Dict]:
    """Return the most recent `limit` events as a list of dicts."""
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Frame / image helpers
# ─────────────────────────────────────────────────────────────────────────────

def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize frame to (width × height) — keeps aspect via letterbox if needed."""
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def crop_roi(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    pad: int = 4,
) -> Optional[np.ndarray]:
    """
    Safely crop a region of interest from a frame.
    Returns None if the crop is degenerate (zero area or out of bounds).
    """
    h, w = frame.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to single-channel grayscale."""
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img  # Already grayscale


def normalise(img: np.ndarray) -> np.ndarray:
    """Float32 in [0, 1] — standard normalisation before classifier input."""
    return img.astype(np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# Keypoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_keypoint(keypoints: np.ndarray, name: str) -> Optional[Tuple[float, float]]:
    """
    Look up a COCO keypoint by name.
    `keypoints` shape: (17, 3)  — (x, y, confidence) per point.
    Returns (x, y) or None if confidence is below 0.4.
    """
    if name not in config.KP_NAMES:
        log.warning("Unknown keypoint name: %s", name)
        return None
    idx = config.KP_NAMES.index(name)
    x, y, conf = keypoints[idx]
    if conf < 0.4:
        return None
    return float(x), float(y)


def face_bbox_from_keypoints(
    keypoints: np.ndarray,
    frame_w: int,
    frame_h: int,
    scale: float = 1.4,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Derive a face bounding box from the five facial keypoints.
    Returns (x1, y1, x2, y2) in pixel coordinates, or None.
    """
    face_kps = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
    pts = [get_keypoint(keypoints, k) for k in face_kps]
    pts = [p for p in pts if p is not None]
    if len(pts) < 2:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx, cy = np.mean(xs), np.mean(ys)
    half = max(np.ptp(xs), np.ptp(ys)) * scale / 2
    x1, y1 = int(cx - half), int(cy - half)
    x2, y2 = int(cx + half), int(cy + half)
    return (
        max(0, x1), max(0, y1),
        min(frame_w, x2), min(frame_h, y2),
    )


def lip_bbox_from_keypoints(
    keypoints: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Approximate lip region using nose keypoint as anchor.
    Returns (x1, y1, x2, y2) or None.
    """
    nose = get_keypoint(keypoints, "nose")
    if nose is None:
        return None
    nx, ny = nose
    # Mouth sits roughly 20–50px below the nose at 480p
    scale = frame_h / 480
    x1 = int(nx - 28 * scale)
    y1 = int(ny + 18 * scale)
    x2 = int(nx + 28 * scale)
    y2 = int(ny + 50 * scale)
    return (
        max(0, x1), max(0, y1),
        min(frame_w, x2), min(frame_h, y2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# FPS counter
# ─────────────────────────────────────────────────────────────────────────────

class FPSCounter:
    """Rolling-window FPS counter.  Call `.tick()` each frame, read `.fps`."""

    def __init__(self, window: int = 30):
        self._times: List[float] = []
        self._window = window

    def tick(self) -> None:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])
