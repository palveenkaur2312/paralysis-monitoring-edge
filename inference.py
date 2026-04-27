"""
inference.py — Real-time inference pipeline.

Runs on live camera or a video file:
    python inference.py
    python inference.py --source /path/to/video.mp4

Pipeline per frame:
  1. YOLOv8-Pose  -> person bounding boxes + 17 keypoints
  2. DeepSORT     -> assigns persistent track IDs across frames
  3. Face crop    -> LightCNN  -> expression label + confidence
  4. Lip crop     -> LightCNN  -> lip movement label + confidence
  5. Behaviour    -> inactivity / distress streak / lip activity alerts
  6. SQLite       -> log flagged events (no raw video stored)
"""


import os
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Union
import cv2

import cv2
import numpy as np
import onnxruntime as ort

import config
import utils
from logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ONNX helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_onnx(path: str) -> ort.InferenceSession:
    providers = ["CPUExecutionProvider"]
    try:
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    session = ort.InferenceSession(path, providers=providers)
    log.info("Loaded ONNX model: %s  (providers=%s)", path, providers)
    return session


def _run_classifier(
    session: ort.InferenceSession,
    crop: np.ndarray,
    input_hw: Tuple[int, int],
    class_names: List[str],
    conf_thresh: float,
) -> Tuple[str, float]:
    """
    Pre-process a grayscale crop and run inference via an ONNX session.
    Returns (predicted_label, confidence) or ("unknown", 0.0) if below threshold.
    """
    h, w  = input_hw
    gray  = utils.to_grayscale(crop)
    sized = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
    blob  = sized.astype(np.float32) / 255.0
    blob  = (blob - 0.5) / 0.5
    blob  = blob[np.newaxis, np.newaxis, :, :]

    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: blob})[0][0]

    exp   = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    idx   = int(np.argmax(probs))
    conf  = float(probs[idx])

    if conf < conf_thresh:
        return "unknown", conf
    return class_names[idx], conf


# ─────────────────────────────────────────────────────────────────────────────
# Behaviour tracker  (per track_id)
# ─────────────────────────────────────────────────────────────────────────────

class BehaviourTracker:
    """
    Accumulates per-person state across frames and fires alerts.
    State is keyed by DeepSORT track_id.
    """

    def __init__(self):
        self._last_active: Dict[int, float] = {}
        self._expr_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=config.DISTRESS_EXPR_COUNT)
        )
        self._lip_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=config.LIP_ACTIVITY_WINDOW)
        )
        self._distress_labels = {"angry", "fear", "sad"}

    def update(
        self,
        track_id: int,
        keypoints: np.ndarray,
        expression: str,
        lip_state: str,
        timestamp: float,
    ) -> List[str]:
        """
        Update state for one tracked person.
        Returns a list of triggered alert type strings (may be empty).
        """
        alerts: List[str] = []

        centroid = self._kp_centroid(keypoints)
        if centroid is not None:
            self._last_active[track_id] = timestamp
        else:
            if track_id not in self._last_active:
                self._last_active[track_id] = timestamp

        idle_secs = timestamp - self._last_active.get(track_id, timestamp)
        if idle_secs >= config.INACTIVITY_SECONDS:
            alerts.append("inactivity")
            log.warning("ALERT inactivity  track_id=%d  idle=%.0fs", track_id, idle_secs)
            self._last_active[track_id] = timestamp

        self._expr_history[track_id].append(expression)
        history = list(self._expr_history[track_id])
        if (
            len(history) == config.DISTRESS_EXPR_COUNT
            and all(e in self._distress_labels for e in history)
        ):
            alerts.append("distress_expression")
            log.warning("ALERT distress_expression  track_id=%d  expr=%s",
                        track_id, expression)
            self._expr_history[track_id].clear()

        self._lip_history[track_id].append(lip_state)
        lip_buf = list(self._lip_history[track_id])
        active_count = sum(1 for s in lip_buf if s == "active_speech")
        if len(lip_buf) == config.LIP_ACTIVITY_WINDOW and active_count > (
            config.LIP_ACTIVITY_WINDOW * 0.6
        ):
            alerts.append("sustained_lip_activity")
            log.info("ALERT sustained_lip_activity  track_id=%d", track_id)
            self._lip_history[track_id].clear()

        return alerts

    @staticmethod
    def _kp_centroid(keypoints: np.ndarray) -> Optional[Tuple[float, float]]:
        """Mean (x, y) of all keypoints with confidence >= 0.4."""
        mask = keypoints[:, 2] >= 0.4
        pts  = keypoints[mask][:, :2]
        if len(pts) == 0:
            return None
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())


# ─────────────────────────────────────────────────────────────────────────────
# Main inference loop
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(source=None):
    """
    Main loop. Reads frames from source (default: config.CAMERA_ID),
    runs the full pipeline, and logs events to SQLite.

    On-screen overlay shows:
      - FPS (top-left)
      - Inference time in ms (top-left, below FPS)
      - Per-person: track ID, expression, lip state, alerts
      - Green skeleton keypoints on detected persons
    Press 'q' to quit.
    """
    if source is None:
        source = config.CAMERA_ID

    utils.init_db()

    # ── Load ONNX models ──────────────────────────────────────────────────
    for path in (config.FACE_ONNX_PATH, config.LIP_ONNX_PATH):
        if not os.path.exists(path):
            log.error("ONNX model not found: %s  —  run training.py first", path)
            return

    face_sess = _load_onnx(config.FACE_ONNX_PATH)
    lip_sess  = _load_onnx(config.LIP_ONNX_PATH)

    # ── Load YOLOv8-Pose ──────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics not installed.")
        return

    pose_weights = os.path.join(config.MODEL_DIR, "pose_run", "weights", "best.pt")
    if not os.path.exists(pose_weights):
        log.warning("Trained pose weights not found; falling back to base model.")
        pose_weights = config.YOLO_BASE_MODEL
    yolo = YOLO(pose_weights)

    # ── Load DeepSORT tracker ─────────────────────────────────────────────
    try:
        from deep_sort_realtime.deepsort_tracker import DeepSort
        tracker = DeepSort(
            max_age=config.DEEPSORT_MAX_AGE,
            n_init=config.DEEPSORT_N_INIT,
            max_cosine_distance=config.DEEPSORT_MAX_DIST,
            embedder="mobilenet",
            half=False,
            bgr=True,
        )
    except ImportError:
        log.error("deep-sort-realtime not installed: pip install deep-sort-realtime")
        return

    # ── Open video / camera ───────────────────────────────────────────────
    
    # This allows OpenCV to automatically choose the right backend for files or cameras
  
    import cv2
    cap = cv2.VideoCapture(args.source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        log.error("Cannot open source: %s", source)
        return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    behaviour   = BehaviourTracker()
    fps_counter = utils.FPSCounter()
    camera_id   = str(source)

    log.info("Inference started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            log.info("End of stream.")
            break

        # ── No color conversion needed — OpenCV reads BGR natively ────────
        h_frame, w_frame = frame.shape[:2]
        now = time.time()

        # ── Measure total frame inference time ────────────────────────────
        frame_t0 = time.perf_counter()

        # ── Step 1: YOLOv8-Pose ───────────────────────────────────────────
        results = yolo.predict(
            frame,
            conf=config.YOLO_CONF_THRESH,
            iou=config.YOLO_IOU_THRESH,
            imgsz=config.YOLO_IMG_SIZE,
            verbose=False,
        )

        detections = []
        kp_map = {}

        for result in results:
            if result.boxes is None or result.keypoints is None:
                continue
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confs_arr  = result.boxes.conf.cpu().numpy()
            kps_all    = result.keypoints.data.cpu().numpy()

            for i, (box, conf, kps) in enumerate(zip(boxes_xyxy, confs_arr, kps_all)):
                x1, y1, x2, y2 = box.astype(int)
                bw, bh = x2 - x1, y2 - y1
                detections.append(([x1, y1, bw, bh], float(conf), "person"))
                kp_map[i] = kps

        # ── Step 2: DeepSORT tracking ─────────────────────────────────────
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb     = track.to_ltrb()
            x1, y1, x2, y2 = [int(v) for v in ltrb]

            kps = _match_kps(kp_map, x1, y1, x2, y2)
            if kps is None:
                continue

            # ── Draw green skeleton ───────────────────────────────────────
            _draw_skeleton(frame, kps)

            # ── Step 3: Face expression ───────────────────────────────────
            face_bb = utils.face_bbox_from_keypoints(kps, w_frame, h_frame)
            expression, face_conf = "unknown", 0.0
            if face_bb:
                fx1, fy1, fx2, fy2 = face_bb
                face_crop = utils.crop_roi(frame, fx1, fy1, fx2, fy2)
                if face_crop is not None:
                    expression, face_conf = _run_classifier(
                        face_sess, face_crop,
                        input_hw=config.FACE_INPUT_SIZE,
                        class_names=config.FACE_CLASS_NAMES,
                        conf_thresh=config.FACE_CONF_THRESH,
                    )

            # ── Step 4: Lip movement ──────────────────────────────────────
            lip_bb = utils.lip_bbox_from_keypoints(kps, w_frame, h_frame)
            lip_state, lip_conf = "unknown", 0.0
            if lip_bb:
                lx1, ly1, lx2, ly2 = lip_bb
                lip_crop = utils.crop_roi(frame, lx1, ly1, lx2, ly2)
                if lip_crop is not None:
                    lip_state, lip_conf = _run_classifier(
                        lip_sess, lip_crop,
                        input_hw=config.LIP_INPUT_SIZE,
                        class_names=config.LIP_CLASS_NAMES,
                        conf_thresh=config.LIP_CONF_THRESH,
                    )

            # ── Step 5: Behaviour analysis ────────────────────────────────
            alerts = behaviour.update(track_id, kps, expression, lip_state, now)

            # ── Step 6: Log flagged events ────────────────────────────────
            for alert_type in alerts:
                utils.log_event(
                    camera_id=camera_id,
                    event_type=alert_type,
                    track_id=track_id,
                    confidence=max(face_conf, lip_conf),
                )

            # ── Per-person overlay ────────────────────────────────────────
            _draw_overlay(frame, track_id, x1, y1, x2, y2,
                          expression, face_conf, lip_state, lip_conf, alerts)

        # ── Compute inference time ────────────────────────────────────────
        infer_ms = (time.perf_counter() - frame_t0) * 1000

        # ── FPS + Inference time on frame (top-left) ──────────────────────
        fps_counter.tick()
        cv2.putText(frame, f"FPS: {fps_counter.fps:.1f}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Inference: {infer_ms:.1f} ms",
                    (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        log.debug("FPS=%.1f  InferenceTime=%.1fms", fps_counter.fps, infer_ms)

        cv2.imshow("Paralysis Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            log.info("User quit.")
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# Helper — draw COCO skeleton keypoints + limbs
# ─────────────────────────────────────────────────────────────────────────────

_LIMBS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

def _draw_skeleton(frame: np.ndarray, keypoints: np.ndarray) -> None:
    """Draw green skeleton dots and limb lines on frame."""
    for a, b in _LIMBS:
        xa, ya, ca = keypoints[a]
        xb, yb, cb = keypoints[b]
        if ca >= 0.4 and cb >= 0.4:
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)),
                     (0, 200, 0), 2)
    for kp in keypoints:
        x, y, c = float(kp[0]), float(kp[1]), float(kp[2])
        if c >= 0.4:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)


# ─────────────────────────────────────────────────────────────────────────────
# Helper — match keypoints to a tracker bounding box
# ─────────────────────────────────────────────────────────────────────────────

def _match_kps(kp_map, x1, y1, x2, y2):
    """
    Pick the detection whose centroid falls inside the tracker bounding box.
    Returns keypoints (17, 3) or None.
    """
    for kps in kp_map.values():
        mask = kps[:, 2] >= 0.3
        pts  = kps[mask][:, :2]
        if len(pts) == 0:
            continue
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return kps
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Helper — draw per-person text overlay
# ─────────────────────────────────────────────────────────────────────────────

def _draw_overlay(frame, track_id, x1, y1, x2, y2,
                  expression, face_conf, lip_state, lip_conf, alerts):
    colour = (0, 0, 255) if alerts else (0, 200, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    lines = [
        f"ID:{track_id}",
        f"Expr:{expression} ({face_conf:.2f})",
        f"Lip:{lip_state} ({lip_conf:.2f})",
    ] + [f"! ALERT: {a}" for a in alerts]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (x1 + 4, y1 + 18 + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run real-time inference.")
    parser.add_argument(
        "--source",
        default=None,
        help="Video file path or camera index (default: config.CAMERA_ID)",
    )
    args = parser.parse_args()

    source = args.source
    if source is not None and source.isdigit():
        source = int(source)

    run_inference(source=source)