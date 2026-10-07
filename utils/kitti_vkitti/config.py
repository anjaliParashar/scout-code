"""Shared configuration for the KITTI ↔ Virtual KITTI 2 detection task."""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (machine-agnostic via DATA_ROOT)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(REPO_ROOT / "data" / "kitti_vkitti"))).resolve()
OUTPUT_ROOT = Path(os.environ.get("KITTI_VKITTI_OUTPUT", str(REPO_ROOT / "outputs" / "kitti_vkitti"))).resolve()

RAW_DIR = DATA_ROOT / "raw"
KITTI_DIR = DATA_ROOT / "kitti"
VKITTI_DIR = DATA_ROOT / "vkitti"

# ---------------------------------------------------------------------------
# Sequence pairing (KITTI tracking ↔ Virtual KITTI 2)
# ---------------------------------------------------------------------------

SEQUENCE_MAP = {
    "0001": "Scene01",
    "0002": "Scene02",
    "0006": "Scene06",
    "0018": "Scene18",
    "0020": "Scene20",
}

VKITTI_CONDITION = "clone"
VKITTI_CAMERA = "Camera_0"
VKITTI_CAMERA_ID = 0

# ---------------------------------------------------------------------------
# Filtering / matching
# ---------------------------------------------------------------------------

OBJECT_CLASS = "Car"
MIN_BBOX_HEIGHT = 25.0
DET_CONF_THRESH = 0.25
IOU_MATCH_THRESH = 0.5
FRAME_STRIDE = int(os.environ.get("FRAME_STRIDE", "10"))

# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

DEFAULT_MODEL = os.environ.get("YOLO_MODEL", "yolo11n.pt")
FALLBACK_MODEL = "yolov8n.pt"
COCO_CAR_CLASS_ID = 2  # ultralytics/COCO: car
SEED = 0

# ---------------------------------------------------------------------------
# Official download URLs
# ---------------------------------------------------------------------------

VKITTI_RGB_URL = (
    "https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar"
)
VKITTI_TEXTGT_URL = (
    "https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_textgt.tar.gz"
)

KITTI_AWS_BUCKET = "s3://avg-kitti"
KITTI_IMAGE_ZIP = "data_tracking_image_2.zip"
KITTI_LABEL_ZIP = "data_tracking_label_2.zip"

# Clone bbox frame counts observed from vkitti_2.0.3 textgt
VKITTI_CLONE_FRAME_COUNTS = {
    "Scene01": 426,  # frames 0..425
    "Scene02": 233,  # frames 0..232
    "Scene06": 270,  # frames 0..269
    "Scene18": 314,  # frames 25..338 (no early frames)
    "Scene20": 837,  # frames 0..836
}


def ensure_dirs() -> None:
    for p in (DATA_ROOT, RAW_DIR, KITTI_DIR, VKITTI_DIR, OUTPUT_ROOT):
        p.mkdir(parents=True, exist_ok=True)
