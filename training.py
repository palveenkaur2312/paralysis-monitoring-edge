"""
training.py — Train (or fine-tune) the three models and export to ONNX.

Usage:
    python training.py --model pose
    python training.py --model face
    python training.py --model lip
    python training.py --model all
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config
from logger import get_logger

log = get_logger(__name__)

os.makedirs(config.MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 — YOLOv8 Pose  (fine-tune / train from scratch via Ultralytics)
# ─────────────────────────────────────────────────────────────────────────────

def train_pose_model() -> None:
    """
    Fine-tune YOLOv8-Pose on the prepared dataset.
    Requires: pip install ultralytics
    Dataset path expected: data/pose_processed/dataset.yaml
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics not installed. Run: pip install ultralytics")
        return

    dataset_yaml = os.path.join(config.DATA_DIR, "pose_processed", "dataset.yaml")
    if not os.path.exists(dataset_yaml):
        log.error("dataset.yaml not found. Run: python preprocessing.py --model pose")
        return

    log.info("Starting YOLOv8 pose training …")
    model = YOLO(config.YOLO_BASE_MODEL)   # Downloads weights on first run

    model.train(
        data=dataset_yaml,
        epochs=config.TRAIN_EPOCHS,
        imgsz=config.YOLO_IMG_SIZE,
        batch=config.TRAIN_BATCH_SIZE,
        device=config.TRAIN_DEVICE,
        project=config.MODEL_DIR,
        name="pose_run",
        exist_ok=True,
        # Edge optimisations
        half=False,       # FP16 only on CUDA; set True on Jetson with CUDA
        workers=2,        # Keep low on edge hardware
        cache=False,      # Saves RAM on RPi
    )

    # Export best weights to ONNX for edge deployment
    best_weights = os.path.join(config.MODEL_DIR, "pose_run", "weights", "best.pt")
    if os.path.exists(best_weights):
        export_model = YOLO(best_weights)
        export_model.export(format="onnx", imgsz=config.YOLO_IMG_SIZE, simplify=True)
        log.info("Pose model exported to ONNX: %s", best_weights.replace(".pt", ".onnx"))
    else:
        log.warning("best.pt not found after training; check training logs.")


# ─────────────────────────────────────────────────────────────────────────────
# Shared lightweight CNN for face & lip classifiers
# ─────────────────────────────────────────────────────────────────────────────

class LightCNN(nn.Module):
    """
    A small, edge-friendly CNN.
    ~120k parameters — runs at 60+ FPS on Jetson Nano.

    in_channels : 1 for grayscale, 3 for BGR
    num_classes : number of output categories
    input_hw    : (height, width) of the input crop
    """

    def __init__(self, in_channels: int, num_classes: int, input_hw: tuple[int, int]):
        super().__init__()
        h, w = input_hw
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # h/2, w/2
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),           # h/4, w/4
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # Always 4×4 regardless of input size
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────────────────────────────────────
# Generic training loop (shared by face & lip)
# ─────────────────────────────────────────────────────────────────────────────

def _train_classifier(
    model: nn.Module,
    data_dir: str,
    input_hw: tuple[int, int],
    model_save_path: str,
    onnx_save_path: str,
) -> None:
    """
    Standard supervised-classification training loop.
    data_dir must follow ImageFolder layout: data_dir/{train,val}/{class_name}/*.jpg
    input_hw: (height, width) of each crop — used for transforms and ONNX dummy input.
    """
    device = torch.device(config.TRAIN_DEVICE)
    h, w   = input_hw

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((h, w)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=config.TRAIN_BATCH_SIZE,
                               shuffle=True,  num_workers=2, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=config.TRAIN_BATCH_SIZE,
                               shuffle=False, num_workers=2, pin_memory=False)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.TRAIN_LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0.0

    for epoch in range(1, config.TRAIN_EPOCHS + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        val_acc = correct / total if total else 0.0
        log.info("Epoch %d/%d  loss=%.4f  val_acc=%.3f",
                 epoch, config.TRAIN_EPOCHS,
                 running_loss / len(train_loader), val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            log.info("  ✓ Best model saved (val_acc=%.3f)", best_val_acc)

        scheduler.step()

    log.info("Training complete. Best val_acc=%.3f", best_val_acc)

    # ── ONNX export ─────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(model_save_path, map_location="cpu"))
    model.eval()
    dummy = torch.zeros(1, 1, h, w)  # Grayscale
    torch.onnx.export(
        model, dummy, onnx_save_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}},
        opset_version=13,
    )
    log.info("ONNX model saved: %s", onnx_save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 — Face Expression Classifier
# ─────────────────────────────────────────────────────────────────────────────

def train_face_model() -> None:
    data_dir = os.path.join(config.DATA_DIR, "face_processed")
    if not os.path.isdir(data_dir):
        log.error("Face dataset not found. Run: python preprocessing.py --model face")
        return

    h, w = config.FACE_INPUT_SIZE   # (48, 48)
    model = LightCNN(in_channels=1, num_classes=config.FACE_NUM_CLASSES, input_hw=(h, w))
    log.info("Training Face Expression Classifier …")
    _train_classifier(
        model=model,
        data_dir=data_dir,
        input_hw=(h, w),
        model_save_path=config.FACE_MODEL_PATH,
        onnx_save_path=config.FACE_ONNX_PATH,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model 3 — Lip Movement Classifier
# ─────────────────────────────────────────────────────────────────────────────

def train_lip_model() -> None:
    data_dir = os.path.join(config.DATA_DIR, "lip_processed")
    if not os.path.isdir(data_dir):
        log.error("Lip dataset not found. Run: python preprocessing.py --model lip")
        return

    h, w = config.LIP_INPUT_SIZE    # (32, 64) — height first
    model = LightCNN(in_channels=1, num_classes=config.LIP_NUM_CLASSES, input_hw=(h, w))
    log.info("Training Lip Movement Classifier …")
    _train_classifier(
        model=model,
        data_dir=data_dir,
        input_hw=(h, w),
        model_save_path=config.LIP_MODEL_PATH,
        onnx_save_path=config.LIP_ONNX_PATH,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models.")
    parser.add_argument(
        "--model",
        choices=["pose", "face", "lip", "all"],
        default="all",
        help="Which model to train (default: all).",
    )
    args = parser.parse_args()

    if args.model in ("pose", "all"):
        train_pose_model()
    if args.model in ("face", "all"):
        train_face_model()
    if args.model in ("lip", "all"):
        train_lip_model()
