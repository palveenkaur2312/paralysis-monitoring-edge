"""
config.py — Central configuration for the paralysis monitoring pipeline.
Edit values here; all other modules import from this file.
"""

import os

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
MODEL_DIR       = os.path.join(BASE_DIR, "models")
LOG_DIR         = os.path.join(BASE_DIR, "logs")
DB_PATH         = os.path.join(BASE_DIR, "events.db")

# Raw dataset folders (populate before training)
POSE_DATA_DIR   = os.path.join(DATA_DIR, "pose")
FACE_DATA_DIR   = os.path.join(DATA_DIR, "face_expression")
LIP_DATA_DIR    = os.path.join(DATA_DIR, "lip_movement")

# Trained / exported model weights
POSE_MODEL_PATH = os.path.join(MODEL_DIR, "pose_model.pt")
FACE_MODEL_PATH = os.path.join(MODEL_DIR, "face_model.pt")
LIP_MODEL_PATH  = os.path.join(MODEL_DIR, "lip_model.pt")

# Edge-optimised exports (ONNX → TFLite on Jetson/RPi)
POSE_ONNX_PATH  = os.path.join(MODEL_DIR, "pose_model.onnx")
FACE_ONNX_PATH  = os.path.join(MODEL_DIR, "face_model.onnx")
LIP_ONNX_PATH   = os.path.join(MODEL_DIR, "lip_model.onnx")

# ─────────────────────────────────────────────
# Camera / video source
# ─────────────────────────────────────────────
CAMERA_ID       = 0          # Integer index or RTSP URL string
CAMERA_FPS      = 30         # Expected capture FPS
FRAME_WIDTH     = 640        # Resize before inference (keeps latency low on edge)
FRAME_HEIGHT    = 480

# ─────────────────────────────────────────────
# Model 1 — YOLOv8 Pose (keypoint detection)
# ─────────────────────────────────────────────
YOLO_BASE_MODEL  = "yolov8n-pose.pt"   # nano = fastest on edge; swap to 's' for accuracy
YOLO_CONF_THRESH = 0.50                 # Minimum detection confidence
YOLO_IOU_THRESH  = 0.45
YOLO_IMG_SIZE    = 320                  # Input resolution; 320 is optimal for RPi/Jetson

# 17 COCO keypoint names (index → name)
KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# ─────────────────────────────────────────────
# Model 2 — Face Expression Classifier
# ─────────────────────────────────────────────
FACE_NUM_CLASSES   = 7        # angry, disgust, fear, happy, neutral, sad, surprise
FACE_CLASS_NAMES   = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
FACE_INPUT_SIZE    = (48, 48) # Grayscale crop fed into classifier
FACE_CONF_THRESH   = 0.25

# ─────────────────────────────────────────────
# Model 3 — Lip Movement Classifier
# ─────────────────────────────────────────────
LIP_NUM_CLASSES  = 3          # still, slight_movement, active_speech
LIP_CLASS_NAMES  = ["still", "slight_movement", "active_speech"]
LIP_INPUT_SIZE   = (64, 32)   # Width × Height crop of mouth region
LIP_CONF_THRESH  = 0.35
LIP_SEQUENCE_LEN = 15         # Frames per temporal window for movement classification

# ─────────────────────────────────────────────
# DeepSORT tracker
# ─────────────────────────────────────────────
DEEPSORT_MAX_AGE      = 30    # Frames before a lost track is deleted
DEEPSORT_N_INIT       = 3     # Detections before track is confirmed
DEEPSORT_MAX_DIST     = 0.4   # Cosine distance threshold

# ─────────────────────────────────────────────
# Behaviour thresholds  (tweak via dashboard)
# ─────────────────────────────────────────────
INACTIVITY_SECONDS    = 60    # Flag if no movement detected for this many seconds
DISTRESS_EXPR_COUNT   = 5     # N consecutive distress expressions → alert
LIP_ACTIVITY_WINDOW   = 30    # Frames to check for sustained lip activity

# ─────────────────────────────────────────────
# Flask dashboard
# ─────────────────────────────────────────────
FLASK_HOST    = "0.0.0.0"
FLASK_PORT    = 5000
FLASK_DEBUG   = False         # Always False in production

# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
TRAIN_EPOCHS       = 30
TRAIN_BATCH_SIZE   = 16       # Keep low for edge hardware training
TRAIN_LR           = 1e-3
TRAIN_VAL_SPLIT    = 0.15
TRAIN_DEVICE       = "cpu"    # "cuda" if available, else "cpu" / "mps"

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_LEVEL  = "INFO"           # DEBUG | INFO | WARNING | ERROR
LOG_FILE   = os.path.join(LOG_DIR, "pipeline.log")
LOG_MAX_MB = 10               # Rotate after this size
LOG_BACKUP = 3                # Keep this many rotated files
