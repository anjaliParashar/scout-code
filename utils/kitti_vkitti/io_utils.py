"""CSV I/O helpers that preserve zero-padded KITTI sequence IDs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_STRING_COLS = {
    "scenario_id": str,
    "sequence_id": str,
    "vkitti_scene_id": str,
    "real_image_path": str,
    "proxy_image_path": str,
}


def read_task_csv(path: str | Path) -> pd.DataFrame:
    """Read manifest/task CSV without coercing sequence_id '0001' → 1."""
    df = pd.read_csv(path, dtype=_STRING_COLS)
    if "sequence_id" in df.columns:
        df["sequence_id"] = df["sequence_id"].astype(str).str.zfill(4)
    if "scenario_id" in df.columns:
        df["scenario_id"] = df["scenario_id"].astype(str)
    return df
