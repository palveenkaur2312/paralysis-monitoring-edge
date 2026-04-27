"""
collect_data.py — Webcam data collector for all 3 models.

Controls:
  F  = face expression mode
  L  = lip movement mode
  P  = pose mode (records full frames)
  Q  = quit

In FACE mode, press 1-7 to label the current expression:
  1=angry  2=disgust  3=fear  4=happy  5=neutral  6=sad  7=surprise

In LIP mode, press 1-3 to label lip state:
  1=still  2=slight_movement  3=active_speech

In POSE mode, press SPACE to save the current frame.

Saved images go directly into the correct training folders.
"""

import os
import time
import cv2

import config
from logger import get_logger

log = get_logger(__name__)

# ── Folder setup ─────────────────────────────────────────────────────────────
FACE_DIRS = {str(i+1): os.path.join(config.FACE_DATA_DIR, name)
             for i, name in enumerate(config.FACE_CLASS_NAMES)}
LIP_DIRS  = {str(i+1): os.path.join(config.LIP_DATA_DIR, name)
             for i, name in enumerate(config.LIP_CLASS_NAMES)}
POSE_IMG_DIR = os.path.join(config.POSE_DATA_DIR, "images")

for d in list(FACE_DIRS.values()) + list(LIP_DIRS.values()) + [POSE_IMG_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def count_images(folder: str) -> int:
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith((".jpg", ".png"))])

def save_image(folder: str, prefix: str, frame) -> str:
    ts   = int(time.time() * 1000)
    path = os.path.join(folder, f"{prefix}_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log.error("Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mode         = "face"   # start in face mode
    last_saved   = ""
    saved_count  = 0

    print("\n=== DATA COLLECTOR ===")
    print("F=face mode  L=lip mode  P=pose mode  Q=quit")
    print("FACE:  1=angry 2=disgust 3=fear 4=happy 5=neutral 6=sad 7=surprise")
    print("LIP:   1=still 2=slight_movement 3=active_speech")
    print("POSE:  SPACE=save frame\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        h, w    = display.shape[:2]

        # ── HUD ──────────────────────────────────────────────────────────────
        mode_color = {"face": (0,200,0), "lip": (0,180,255), "pose": (200,0,255)}
        cv2.rectangle(display, (0, 0), (w, 90), (20, 20, 20), -1)
        cv2.putText(display, f"MODE: {mode.upper()}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    mode_color.get(mode, (255,255,255)), 2)

        if mode == "face":
            hint = "Press 1-7 to save labelled face crop"
            counts = "  ".join(f"{n}:{count_images(d)}"
                               for n, d in FACE_DIRS.items())
        elif mode == "lip":
            hint  = "Press 1-3 to save labelled lip crop"
            counts = "  ".join(f"{n}:{count_images(d)}"
                               for n, d in LIP_DIRS.items())
        else:
            hint   = "Press SPACE to save full frame"
            counts = f"saved: {count_images(POSE_IMG_DIR)}"

        cv2.putText(display, hint,   (10, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(display, counts, (10, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,220,150), 1)

        if last_saved:
            cv2.putText(display, f"Saved: {last_saved}",
                        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 150), 1)

        # ── ROI guides ───────────────────────────────────────────────────────
        cx, cy = w // 2, h // 2
        if mode == "face":
            # Green box — centre your face here
            cv2.rectangle(display, (cx-80, cy-100), (cx+80, cy+80),
                          (0, 255, 0), 2)
            cv2.putText(display, "Face here", (cx-60, cy-108),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        elif mode == "lip":
            # Blue box — centre your mouth here
            cv2.rectangle(display, (cx-50, cy-20), (cx+50, cy+30),
                          (255, 180, 0), 2)
            cv2.putText(display, "Mouth here", (cx-50, cy-28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,180,0), 1)

        cv2.imshow("Data Collector", display)
        key = cv2.waitKey(1) & 0xFF

        # ── Mode switching ───────────────────────────────────────────────────
        if key == ord('q'):
            break
        elif key == ord('f'):
            mode = "face";  print("→ Face expression mode")
        elif key == ord('l'):
            mode = "lip";   print("→ Lip movement mode")
        elif key == ord('p'):
            mode = "pose";  print("→ Pose mode")

        # ── Saving ───────────────────────────────────────────────────────────
        elif mode == "face" and chr(key) in FACE_DIRS:
            label     = config.FACE_CLASS_NAMES[int(chr(key)) - 1]
            folder    = FACE_DIRS[chr(key)]
            # Crop the face ROI
            crop      = frame[cy-100:cy+80, cx-80:cx+80]
            if crop.size > 0:
                gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                sized = cv2.resize(gray, config.FACE_INPUT_SIZE)
                path  = save_image(folder, label, sized)
                last_saved = f"{label} ({count_images(folder)} total)"
                print(f"  Saved {label} → {path}")

        elif mode == "lip" and chr(key) in LIP_DIRS:
            label     = config.LIP_CLASS_NAMES[int(chr(key)) - 1]
            folder    = LIP_DIRS[chr(key)]
            # Crop the lip ROI
            crop      = frame[cy-20:cy+30, cx-50:cx+50]
            if crop.size > 0:
                gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                sized = cv2.resize(gray, config.LIP_INPUT_SIZE)
                path  = save_image(folder, label, sized)
                last_saved = f"{label} ({count_images(folder)} total)"
                print(f"  Saved {label} → {path}")

        elif mode == "pose" and key == ord(' '):
            path       = save_image(POSE_IMG_DIR, "pose", frame)
            last_saved = f"pose frame ({count_images(POSE_IMG_DIR)} total)"
            print(f"  Saved pose frame → {path}")

    cap.release()
    cv2.destroyAllWindows()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== COLLECTION SUMMARY ===")
    print("FACE:")
    for i, name in enumerate(config.FACE_CLASS_NAMES):
        print(f"  {name}: {count_images(FACE_DIRS[str(i+1)])} images")
    print("LIP:")
    for i, name in enumerate(config.LIP_CLASS_NAMES):
        print(f"  {name}: {count_images(LIP_DIRS[str(i+1)])} images")
    print(f"POSE: {count_images(POSE_IMG_DIR)} images")
    print("\nDone! Now run: python preprocessing.py --model face")
    print("               python preprocessing.py --model lip")
    print("               python training.py --model all")

if __name__ == "__main__":
    main()
