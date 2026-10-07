"""Unit tests for IoU / Hungarian matching / missed-car failure."""

from __future__ import annotations

import numpy as np

from utils.kitti_vkitti.box_matching import box_iou, hungarian_match, missed_car_rate


def test_box_iou_identical():
    a = np.array([[0, 0, 10, 10]], float)
    iou = box_iou(a, a)
    assert iou.shape == (1, 1)
    assert abs(iou[0, 0] - 1.0) < 1e-9


def test_box_iou_no_overlap():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[20, 20, 30, 30]], float)
    assert box_iou(a, b)[0, 0] == 0.0


def test_hungarian_perfect_match():
    gt = np.array([[0, 0, 10, 10], [20, 20, 40, 40]], float)
    pr = np.array([[1, 1, 11, 11], [21, 21, 41, 41]], float)
    tp, fp, fn, matches = hungarian_match(gt, pr, iou_thresh=0.5)
    assert tp == 2 and fp == 0 and fn == 0
    assert matches.shape == (2, 2)


def test_missed_car_rate():
    gt = np.array([[0, 0, 10, 10], [50, 50, 80, 80]], float)
    # Only first GT matched; second missed; one FP
    pr = np.array([[0, 0, 10, 10], [200, 200, 210, 210]], float)
    fail, tp, fp, fn = missed_car_rate(gt, pr, iou_thresh=0.5)
    assert tp == 1 and fn == 1 and fp == 1
    assert abs(fail - 0.5) < 1e-9


def test_missed_car_all_detected():
    gt = np.array([[0, 0, 10, 10]], float)
    pr = np.array([[0, 0, 10, 10]], float)
    fail, tp, fp, fn = missed_car_rate(gt, pr)
    assert fail == 0.0 and tp == 1 and fn == 0 and fp == 0


def test_missed_car_all_missed():
    gt = np.array([[0, 0, 10, 10]], float)
    pr = np.zeros((0, 4))
    fail, tp, fp, fn = missed_car_rate(gt, pr)
    assert fail == 1.0 and tp == 0 and fn == 1 and fp == 0
