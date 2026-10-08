#!/usr/bin/env python3
"""Export a small paired KITTI / Virtual KITTI set for the project-page demo."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(os.environ["KITTI_IMAGE_ROOT"])
CSV = ROOT / "data/kitti_vkitti/paired_detection_task_linked.csv"
OUT_DIR = ROOT / "docs/assets/kitti"
OUT_JSON = ROOT / "docs/assets/kitti_pool.json"
N_FRAMES = 72
FEATS = [
    "num_cars",
    "mean_bbox_area_ratio",
    "min_bbox_area_ratio",
    "max_bbox_area_ratio",
    "mean_bbox_height_ratio",
    "image_brightness",
]


def _pca(frame: pd.DataFrame) -> np.ndarray:
    x = frame[FEATS].to_numpy(np.float64)
    x = (x - x.mean(0)) / (x.std(0) + 1e-8)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    xy = x @ vt[:2].T
    return xy / (np.percentile(np.abs(xy), 98) + 1e-8)


def _choose(xy: np.ndarray, target: np.ndarray, n: int) -> list[int]:
    chosen = [int(np.argmax(target)), int(np.argmin(target))]
    while len(chosen) < n:
        rest = [i for i in range(len(xy)) if i not in chosen]
        dist = np.linalg.norm(xy[rest][:, None, :] - xy[chosen][None, :, :], axis=-1).min(1)
        score = dist + 0.35 * target[rest]
        chosen.append(rest[int(np.argmax(score))])
    return chosen


def _thumb(src: Path, dest: Path) -> None:
    image = Image.open(src).convert("RGB")
    image.thumbnail((520, 180), Image.Resampling.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest, format="JPEG", quality=72, optimize=True)


def main() -> None:
    frame = pd.read_csv(CSV)
    xy = _pca(frame)
    target = frame["target_failure"].to_numpy(np.float64)
    keep = []
    for index in _choose(xy, target, N_FRAMES * 2):
        real = SRC / frame.at[index, "real_image_path"]
        proxy = SRC / frame.at[index, "proxy_image_path"]
        if real.is_file() and proxy.is_file():
            keep.append(index)
        if len(keep) == N_FRAMES:
            break
    if len(keep) < N_FRAMES:
        raise SystemExit(f"only found {len(keep)} paired images")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for index in keep:
        row = frame.loc[index]
        sid = str(row["scenario_id"])
        real_name = f"{sid}_real.jpg"
        proxy_name = f"{sid}_proxy.jpg"
        _thumb(SRC / row["real_image_path"], OUT_DIR / real_name)
        _thumb(SRC / row["proxy_image_path"], OUT_DIR / proxy_name)
        records.append({
            "id": sid,
            "seq": str(row["sequence_id"]),
            "frame": int(row["frame_id"]),
            "x": round(float(xy[index, 0]), 4),
            "y": round(float(xy[index, 1]), 4),
            "proxy": round(float(row["proxy_failure"]), 4),
            "target": round(float(row["target_failure"]), 4),
            "cars": int(row["num_cars"]),
            "real": f"assets/kitti/{real_name}",
            "proxyImg": f"assets/kitti/{proxy_name}",
        })
    OUT_JSON.write_text(json.dumps({"frames": records}, indent=1))
    nbytes = sum(p.stat().st_size for p in OUT_DIR.glob("*.jpg"))
    print(f"wrote {len(records)} pairs, {nbytes/1e6:.1f} MB, {OUT_JSON}")


if __name__ == "__main__":
    main()
