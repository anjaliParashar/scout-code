"""Parse Virtual KITTI 2 text ground-truth (bbox.txt + info.txt)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .config import MIN_BBOX_HEIGHT, OBJECT_CLASS, VKITTI_CAMERA_ID


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
        if left < -1 or top < -1 or right > iw + 1 or bottom > ih + 1:
            return False
    return True


def load_car_track_ids(info_path: str | Path, object_class: str = OBJECT_CLASS) -> Set[int]:
    """
    Read info.txt and return trackIDs whose label is `object_class`.

    Expected columns include: trackID, label, ...
    """
    info_path = Path(info_path)
    df = pd.read_csv(info_path, sep=r"\s+", engine="python")
    # Normalise column names
    cols = {c.lower(): c for c in df.columns}
    track_col = cols.get("trackid") or cols.get("track_id")
    label_col = cols.get("label") or cols.get("class") or cols.get("type")
    if track_col is None or label_col is None:
        raise ValueError(f"Unexpected info.txt columns in {info_path}: {list(df.columns)}")
    mask = df[label_col].astype(str).str.lower() == object_class.lower()
    return set(int(x) for x in df.loc[mask, track_col].tolist())


def parse_vkitti_bbox_file(
    bbox_path: str | Path,
    info_path: str | Path,
    *,
    camera_id: int = VKITTI_CAMERA_ID,
    object_class: str = OBJECT_CLASS,
    min_bbox_height: float = MIN_BBOX_HEIGHT,
    image_wh: Optional[Tuple[int, int]] = None,
    frames: Optional[Sequence[int]] = None,
) -> Dict[int, np.ndarray]:
    """
    Parse Virtual KITTI 2 bbox.txt, keeping Car boxes for the given camera.

    Returns
    -------
    dict mapping frame_id -> (N, 4) float array of [left, top, right, bottom]
    """
    bbox_path = Path(bbox_path)
    car_ids = load_car_track_ids(info_path, object_class=object_class)
    keep_frames = None if frames is None else set(int(f) for f in frames)

    df = pd.read_csv(bbox_path, sep=r"\s+", engine="python")
    cols = {c.lower(): c for c in df.columns}

    required = ["frame", "cameraid", "trackid", "left", "right", "top", "bottom"]
    for r in required:
        if r not in cols:
            raise ValueError(f"Missing column '{r}' in {bbox_path}; got {list(df.columns)}")

    frame_c = cols["frame"]
    cam_c = cols["cameraid"]
    track_c = cols["trackid"]
    left_c, right_c, top_c, bottom_c = (
        cols["left"], cols["right"], cols["top"], cols["bottom"]
    )

    out: Dict[int, List[List[float]]] = {}
    for row in df.itertuples(index=False):
        frame = int(getattr(row, frame_c))
        if keep_frames is not None and frame not in keep_frames:
            continue
        if int(getattr(row, cam_c)) != int(camera_id):
            continue
        track = int(getattr(row, track_c))
        if track not in car_ids:
            continue
        left = float(getattr(row, left_c))
        right = float(getattr(row, right_c))
        top = float(getattr(row, top_c))
        bottom = float(getattr(row, bottom_c))
        if not _box_valid(left, top, right, bottom, image_wh, min_bbox_height):
            continue
        out.setdefault(frame, []).append([left, top, right, bottom])

    return {k: np.asarray(v, dtype=np.float64) for k, v in sorted(out.items())}
