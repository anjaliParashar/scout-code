"""Parse KITTI tracking training labels (label_02)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import MIN_BBOX_HEIGHT, OBJECT_CLASS

# frame, track_id, type, truncation, occlusion, alpha,
# bbox_left, bbox_top, bbox_right, bbox_bottom, ...
KITTI_TYPE_IDX = 2
KITTI_BBOX_SLICE = slice(6, 10)


def _box_valid(
    left: float,
    top: float,
    right: float,
    bottom: float,
    image_wh: Optional[Tuple[int, int]],
    min_height: float,
) -> bool:
    h = bottom - top
    w = right - left
    if h < min_height or w <= 0 or h <= 0:
        return False
    if image_wh is not None:
        iw, ih = image_wh
        if left < 0 or top < 0 or right > iw or bottom > ih:
            # Allow tiny float overflow; reject clearly OOB boxes.
            if left < -1 or top < -1 or right > iw + 1 or bottom > ih + 1:
                return False
            left = max(0.0, left)
            top = max(0.0, top)
            right = min(float(iw), right)
            bottom = min(float(ih), bottom)
            if bottom - top < min_height or right - left <= 0:
                return False
    return True


def parse_kitti_label_file(
    path: str | Path,
    *,
    object_class: str = OBJECT_CLASS,
    min_bbox_height: float = MIN_BBOX_HEIGHT,
    image_wh: Optional[Tuple[int, int]] = None,
    frames: Optional[Sequence[int]] = None,
) -> Dict[int, np.ndarray]:
    """
    Parse a KITTI tracking label_02 file.

    Returns
    -------
    dict mapping frame_id -> (N, 4) float array of [left, top, right, bottom]
    for kept Car boxes.
    """
    path = Path(path)
    keep_frames = None if frames is None else set(int(f) for f in frames)
    out: Dict[int, List[List[float]]] = {}

    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            frame = int(float(parts[0]))
            if keep_frames is not None and frame not in keep_frames:
                continue
            obj_type = parts[KITTI_TYPE_IDX]
            if obj_type != object_class:
                continue
            left, top, right, bottom = (float(x) for x in parts[KITTI_BBOX_SLICE])
            if not _box_valid(left, top, right, bottom, image_wh, min_bbox_height):
                continue
            out.setdefault(frame, []).append([left, top, right, bottom])

    return {k: np.asarray(v, dtype=np.float64) for k, v in sorted(out.items())}


def count_frames_in_label_file(path: str | Path) -> int:
    """Return 1 + max frame index present in a label file (0 if empty)."""
    path = Path(path)
    max_f = -1
    with path.open("r") as f:
        for line in f:
            parts = line.split()
            if parts:
                max_f = max(max_f, int(float(parts[0])))
    return max_f + 1 if max_f >= 0 else 0
