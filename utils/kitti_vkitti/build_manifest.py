"""Build paired-frame manifest between KITTI and Virtual KITTI 2."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from .config import (
    FRAME_STRIDE,
    KITTI_DIR,
    OUTPUT_ROOT,
    SEQUENCE_MAP,
    VKITTI_CAMERA,
    VKITTI_CONDITION,
    VKITTI_DIR,
    ensure_dirs,
)
from .kitti_parser import parse_kitti_label_file
from .vkitti_parser import parse_vkitti_bbox_file

logger = logging.getLogger(__name__)


def _image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        return im.size  # (W, H)


def _brightness(path: Path) -> float:
    with Image.open(path) as im:
        gray = im.convert("L")
        arr = np.asarray(gray, dtype=np.float64)
    return float(arr.mean() / 255.0)


def _bbox_stats(boxes: np.ndarray, image_wh: Tuple[int, int]) -> Dict[str, float]:
    iw, ih = image_wh
    area_img = float(iw * ih)
    if len(boxes) == 0:
        return dict(
            num_cars=0,
            mean_bbox_area_ratio=0.0,
            min_bbox_area_ratio=0.0,
            max_bbox_area_ratio=0.0,
            mean_bbox_height_ratio=0.0,
        )
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights
    area_ratios = areas / area_img
    height_ratios = heights / float(ih)
    return dict(
        num_cars=int(len(boxes)),
        mean_bbox_area_ratio=float(area_ratios.mean()),
        min_bbox_area_ratio=float(area_ratios.min()),
        max_bbox_area_ratio=float(area_ratios.max()),
        mean_bbox_height_ratio=float(height_ratios.mean()),
    )


def find_kitti_paths(kitti_root: Path = KITTI_DIR) -> Tuple[Path, Path]:
    """Locate training/image_02 and training/label_02 under DATA_ROOT."""
    candidates = [
        kitti_root / "training",
        kitti_root / "kitti_tracking" / "training",
        kitti_root,
    ]
    for base in candidates:
        img = base / "image_02"
        lab = base / "label_02"
        if img.is_dir() and lab.is_dir():
            return img, lab
    raise FileNotFoundError(
        f"Could not find KITTI training/image_02 and label_02 under {kitti_root}. "
        "Place extracted tracking training data there, or run download_kitti.sh."
    )


def find_vkitti_scene_root(vkitti_root: Path, scene: str) -> Path:
    """
    Find SceneXX/clone/... regardless of archive top-level directory name.
    """
    # Direct layout
    direct = vkitti_root / scene / VKITTI_CONDITION
    if direct.is_dir():
        return vkitti_root / scene
    # Nested: */SceneXX/clone
    matches = list(vkitti_root.glob(f"**/{scene}/{VKITTI_CONDITION}"))
    if matches:
        return matches[0].parent
    raise FileNotFoundError(
        f"Could not find {scene}/{VKITTI_CONDITION} under {vkitti_root}"
    )


def vkitti_rgb_path(scene_root: Path, frame_id: int) -> Path:
    return (
        scene_root
        / VKITTI_CONDITION
        / "frames"
        / "rgb"
        / VKITTI_CAMERA
        / f"rgb_{frame_id:05d}.jpg"
    )


def vkitti_bbox_path(scene_root: Path) -> Path:
    return scene_root / VKITTI_CONDITION / "bbox.txt"


def vkitti_info_path(scene_root: Path) -> Path:
    return scene_root / VKITTI_CONDITION / "info.txt"


def kitti_image_path(image_root: Path, seq: str, frame_id: int) -> Path:
    return image_root / seq / f"{frame_id:06d}.png"


def build_paired_manifest(
    *,
    kitti_root: Path = KITTI_DIR,
    vkitti_root: Path = VKITTI_DIR,
    frame_stride: int = FRAME_STRIDE,
    output_csv: Optional[Path] = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Pair KITTI and Virtual KITTI frames by sequence map + zero-based frame index.

    Keeps frames with frame_id % frame_stride == 0 and ≥1 valid Car GT in both domains.
    Scenario features are computed from Virtual KITTI only.
    """
    ensure_dirs()
    output_csv = Path(output_csv) if output_csv else OUTPUT_ROOT / "paired_manifest.csv"
    if output_csv.exists() and not overwrite:
        logger.info("Manifest exists, skipping rebuild: %s", output_csv)
        from .io_utils import read_task_csv
        return read_task_csv(output_csv)

    image_root, label_root = find_kitti_paths(kitti_root)

    rows: List[dict] = []
    exclusion_counts = {
        "missing_real_image": 0,
        "missing_proxy_image": 0,
        "no_real_car_gt": 0,
        "no_proxy_car_gt": 0,
        "stride_skip": 0,
        "kept": 0,
    }

    for seq, scene in sorted(SEQUENCE_MAP.items()):
        scene_root = find_vkitti_scene_root(vkitti_root, scene)
        bbox_p = vkitti_bbox_path(scene_root)
        info_p = vkitti_info_path(scene_root)
        label_p = label_root / f"{seq}.txt"
        if not label_p.exists():
            raise FileNotFoundError(f"Missing KITTI label: {label_p}")
        if not bbox_p.exists() or not info_p.exists():
            raise FileNotFoundError(f"Missing VKITTI GT under {scene_root}/{VKITTI_CONDITION}")

        # Probe dimensions from first available images
        # Enumerate candidate frames from available VKITTI RGB files
        rgb_dir = scene_root / VKITTI_CONDITION / "frames" / "rgb" / VKITTI_CAMERA
        if not rgb_dir.is_dir():
            raise FileNotFoundError(f"Missing VKITTI RGB dir: {rgb_dir}")
        frame_ids = sorted(
            int(p.stem.split("_")[-1]) for p in rgb_dir.glob("rgb_*.jpg")
        )
        if not frame_ids:
            raise FileNotFoundError(f"No rgb_*.jpg in {rgb_dir}")

        # Verify correspondence on a representative frame
        rep = frame_ids[0]
        real_rep = kitti_image_path(image_root, seq, rep)
        proxy_rep = vkitti_rgb_path(scene_root, rep)
        if not real_rep.exists():
            raise FileNotFoundError(
                f"Alignment check failed: KITTI image missing for {seq} frame {rep}: {real_rep}"
            )
        if not proxy_rep.exists():
            raise FileNotFoundError(
                f"Alignment check failed: VKITTI image missing for {scene} frame {rep}: {proxy_rep}"
            )
        real_wh = _image_size(real_rep)
        proxy_wh = _image_size(proxy_rep)
        logger.info(
            "Sequence %s↔%s: %d frames, KITTI %sx%s, VKITTI %sx%s",
            seq, scene, len(frame_ids), *real_wh, *proxy_wh,
        )

        kitti_gt = parse_kitti_label_file(label_p, image_wh=real_wh, frames=frame_ids)
        vkitti_gt = parse_vkitti_bbox_file(
            bbox_p, info_p, image_wh=proxy_wh, frames=frame_ids,
        )

        for frame_id in frame_ids:
            if frame_id % frame_stride != 0:
                exclusion_counts["stride_skip"] += 1
                continue
            real_img = kitti_image_path(image_root, seq, frame_id)
            proxy_img = vkitti_rgb_path(scene_root, frame_id)
            if not real_img.exists():
                exclusion_counts["missing_real_image"] += 1
                continue
            if not proxy_img.exists():
                exclusion_counts["missing_proxy_image"] += 1
                continue
            real_boxes = kitti_gt.get(frame_id, np.zeros((0, 4)))
            proxy_boxes = vkitti_gt.get(frame_id, np.zeros((0, 4)))
            if len(real_boxes) == 0:
                exclusion_counts["no_real_car_gt"] += 1
                continue
            if len(proxy_boxes) == 0:
                exclusion_counts["no_proxy_car_gt"] += 1
                continue

            stats = _bbox_stats(proxy_boxes, proxy_wh)
            brightness = _brightness(proxy_img)
            scenario_id = f"{seq}_{frame_id:06d}"
            rows.append(dict(
                scenario_id=scenario_id,
                sequence_id=seq,
                vkitti_scene_id=scene,
                frame_id=int(frame_id),
                real_image_path=str(real_img.resolve()),
                proxy_image_path=str(proxy_img.resolve()),
                real_gt_count=int(len(real_boxes)),
                proxy_gt_count=int(len(proxy_boxes)),
                num_cars=stats["num_cars"],
                mean_bbox_area_ratio=stats["mean_bbox_area_ratio"],
                min_bbox_area_ratio=stats["min_bbox_area_ratio"],
                max_bbox_area_ratio=stats["max_bbox_area_ratio"],
                mean_bbox_height_ratio=stats["mean_bbox_height_ratio"],
                image_brightness=brightness,
            ))
            exclusion_counts["kept"] += 1

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError(f"No paired frames retained. Exclusion stats: {exclusion_counts}")

    df = df.sort_values(["sequence_id", "frame_id"]).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    excl_path = output_csv.with_name("manifest_exclusions.json")
    import json
    excl_path.write_text(json.dumps(exclusion_counts, indent=2))
    logger.info(
        "Wrote %d paired scenarios → %s  exclusions=%s",
        len(df), output_csv, exclusion_counts,
    )
    return df


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Build KITTI–VKITTI paired manifest")
    p.add_argument("--frame-stride", type=int, default=FRAME_STRIDE)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--output", type=str, default="")
    args = p.parse_args()
    build_paired_manifest(
        frame_stride=args.frame_stride,
        output_csv=Path(args.output) if args.output else None,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
