"""Unit tests for KITTI / Virtual KITTI annotation parsers (no downloads)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.kitti_vkitti.kitti_parser import parse_kitti_label_file
from utils.kitti_vkitti.vkitti_parser import parse_vkitti_bbox_file, load_car_track_ids


def test_kitti_parser_filters_cars_and_height(tmp_path: Path):
    lab = tmp_path / "0001.txt"
    # frame, track, type, trunc, occ, alpha, l, t, r, b, ...
    lab.write_text(
        "0 1 Car 0 0 0.0 10 10 100 50 0 0 0 0 0 0 0 0\n"       # height 40 — keep
        "0 2 Pedestrian 0 0 0.0 10 10 40 80 0 0 0 0 0 0 0 0\n"  # wrong class
        "0 3 Car 0 0 0.0 10 10 100 30 0 0 0 0 0 0 0 0\n"       # height 20 — drop
        "0 4 DontCare 0 0 0.0 1 1 10 10 0 0 0 0 0 0 0 0\n"
        "1 5 Van 0 0 0.0 10 10 100 60 0 0 0 0 0 0 0 0\n"       # Van — drop
        "1 6 Car 0 0 0.0 5 5 80 45 0 0 0 0 0 0 0 0\n"          # height 40 — keep
    )
    out = parse_kitti_label_file(lab, image_wh=(1242, 375), min_bbox_height=25)
    assert set(out.keys()) == {0, 1}
    assert out[0].shape == (1, 4)
    np.testing.assert_allclose(out[0][0], [10, 10, 100, 50])
    assert out[1].shape == (1, 4)


def test_vkitti_parser_uses_info_and_camera(tmp_path: Path):
    info = tmp_path / "info.txt"
    bbox = tmp_path / "bbox.txt"
    info.write_text(
        "trackID label\n"
        "1 Car\n"
        "2 Pedestrian\n"
        "3 Car\n"
    )
    bbox.write_text(
        "frame cameraID trackID left right top bottom number_pixels "
        "truncation_ratio occupancy_ratio isMoving\n"
        "0 0 1 10 100 10 50 100 0 0 1\n"          # keep
        "0 0 2 10 40 10 80 100 0 0 1\n"           # pedestrian — drop
        "0 1 3 10 100 10 60 100 0 0 1\n"          # wrong camera — drop
        "0 0 3 10 90 10 20 100 0 0 1\n"           # height 10 — drop
        "10 0 1 20 120 20 70 100 0 0 1\n"         # keep
    )
    cars = load_car_track_ids(info)
    assert cars == {1, 3}
    out = parse_vkitti_bbox_file(
        bbox, info, camera_id=0, image_wh=(1242, 375), min_bbox_height=25,
    )
    assert set(out.keys()) == {0, 10}
    assert out[0].shape == (1, 4)
    np.testing.assert_allclose(out[0][0], [10, 10, 100, 50])
    assert out[10].shape == (1, 4)
