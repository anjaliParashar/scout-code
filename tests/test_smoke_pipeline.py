"""
End-to-end smoke test with tiny synthetic fixtures (no dataset downloads).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from utils.kitti_vkitti.box_matching import missed_car_rate
from utils.kitti_vkitti.kitti_parser import parse_kitti_label_file
from utils.kitti_vkitti.vkitti_parser import parse_vkitti_bbox_file
from utils.kitti_vkitti.build_manifest import _bbox_stats, _brightness


def _make_image(path: Path, color: tuple[int, int, int], size=(64, 48)):
    Image.new("RGB", size, color).save(path)


def test_smoke_pairing_matching_and_csv(tmp_path: Path):
    # Synthetic layout
    kitti_img = tmp_path / "kitti" / "training" / "image_02" / "0001"
    kitti_lab = tmp_path / "kitti" / "training" / "label_02"
    vk_clone = tmp_path / "vkitti" / "Scene01" / "clone"
    vk_rgb = vk_clone / "frames" / "rgb" / "Camera_0"
    for p in (kitti_img, kitti_lab, vk_rgb):
        p.mkdir(parents=True)

    # Two frames (0 and 10); frame 0 has cars in both; frame 10 only real → excluded
    _make_image(kitti_img / "000000.png", (40, 40, 40))
    _make_image(kitti_img / "000010.png", (40, 40, 40))
    _make_image(vk_rgb / "rgb_00000.jpg", (80, 80, 80))
    _make_image(vk_rgb / "rgb_00010.jpg", (80, 80, 80))

    (kitti_lab / "0001.txt").write_text(
        "0 1 Car 0 0 0.0 5 5 40 35 0 0 0 0 0 0 0 0\n"
        "10 2 Car 0 0 0.0 5 5 40 35 0 0 0 0 0 0 0 0\n"
    )
    (vk_clone / "info.txt").write_text("trackID label\n1 Car\n")
    (vk_clone / "bbox.txt").write_text(
        "frame cameraID trackID left right top bottom number_pixels "
        "truncation_ratio occupancy_ratio isMoving\n"
        "0 0 1 5 40 5 35 100 0 0 1\n"
        # frame 10 intentionally has no Car GT
    )

    kitti_gt = parse_kitti_label_file(
        kitti_lab / "0001.txt", image_wh=(64, 48), min_bbox_height=25,
    )
    vk_gt = parse_vkitti_bbox_file(
        vk_clone / "bbox.txt", vk_clone / "info.txt",
        image_wh=(64, 48), min_bbox_height=25,
    )
    assert 0 in kitti_gt and 0 in vk_gt
    assert 10 in kitti_gt and 10 not in vk_gt

    # Manual pairing (stride 10, require cars both sides)
    rows = []
    for frame in (0, 10):
        if frame % 10 != 0:
            continue
        if frame not in kitti_gt or frame not in vk_gt:
            continue
        if len(kitti_gt[frame]) == 0 or len(vk_gt[frame]) == 0:
            continue
        stats = _bbox_stats(vk_gt[frame], (64, 48))
        rows.append(dict(
            scenario_id=f"0001_{frame:06d}",
            sequence_id="0001",
            vkitti_scene_id="Scene01",
            frame_id=frame,
            real_image_path=str(kitti_img / f"{frame:06d}.png"),
            proxy_image_path=str(vk_rgb / f"rgb_{frame:05d}.jpg"),
            real_gt_count=len(kitti_gt[frame]),
            proxy_gt_count=len(vk_gt[frame]),
            **stats,
            image_brightness=_brightness(vk_rgb / f"rgb_{frame:05d}.jpg"),
        ))
    manifest = pd.DataFrame(rows)
    assert len(manifest) == 1
    assert manifest.iloc[0]["frame_id"] == 0

    # Manual detector predictions: miss the real car, detect the proxy car
    real_pred = np.zeros((0, 4))
    proxy_pred = np.array([[5, 5, 40, 35]], float)
    t_fail, t_tp, t_fp, t_fn = missed_car_rate(kitti_gt[0], real_pred)
    p_fail, p_tp, p_fp, p_fn = missed_car_rate(vk_gt[0], proxy_pred)
    assert t_fail == 1.0 and t_fn == 1
    assert p_fail == 0.0 and p_tp == 1

    task = manifest.copy()
    task["proxy_failure"] = p_fail
    task["target_failure"] = t_fail
    task["proxy_tp"] = p_tp
    task["proxy_fp"] = p_fp
    task["proxy_fn"] = p_fn
    task["target_tp"] = t_tp
    task["target_fp"] = t_fp
    task["target_fn"] = t_fn
    out_csv = tmp_path / "paired_detection_task.csv"
    task.to_csv(out_csv, index=False)
    reloaded = pd.read_csv(out_csv)
    assert len(reloaded) == 1
    assert reloaded.iloc[0]["target_failure"] == 1.0
    assert reloaded.iloc[0]["proxy_failure"] == 0.0

    # IoU sanity
    from utils.kitti_vkitti.box_matching import box_iou
    iou = box_iou(vk_gt[0], proxy_pred)[0, 0]
    assert iou >= 0.5
