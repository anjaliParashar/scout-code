"""IoU and Hungarian matching for axis-aligned boxes."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import IOU_MATCH_THRESH


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Pairwise IoU between two sets of boxes [left, top, right, bottom].

    Returns (Na, Nb) IoU matrix.
    """
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    al, at, ar, ab = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bl, bt, br, bb = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    inter_l = np.maximum(al, bl)
    inter_t = np.maximum(at, bt)
    inter_r = np.minimum(ar, br)
    inter_b = np.minimum(ab, bb)
    inter_w = np.maximum(0.0, inter_r - inter_l)
    inter_h = np.maximum(0.0, inter_b - inter_t)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, ar - al) * np.maximum(0.0, ab - at)
    area_b = np.maximum(0.0, br - bl) * np.maximum(0.0, bb - bt)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def hungarian_match(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    iou_thresh: float = IOU_MATCH_THRESH,
) -> Tuple[int, int, int, np.ndarray]:
    """
    Match predicted boxes to GT via Hungarian assignment on IoU.

    A match is valid only when IoU >= iou_thresh.

    Returns
    -------
    tp, fp, fn, matches
        matches is (M, 2) array of (gt_idx, pred_idx) for valid matches.
    """
    gt = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    pr = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    n_gt, n_pr = len(gt), len(pr)
    if n_gt == 0:
        return 0, n_pr, 0, np.zeros((0, 2), dtype=int)
    if n_pr == 0:
        return 0, 0, n_gt, np.zeros((0, 2), dtype=int)

    iou = box_iou(gt, pr)
    # Maximise IoU ≡ minimise (1 - IoU)
    cost = 1.0 - iou
    gt_idx, pr_idx = linear_sum_assignment(cost)

    matches = []
    for gi, pi in zip(gt_idx, pr_idx):
        if iou[gi, pi] >= iou_thresh:
            matches.append((int(gi), int(pi)))
    matches_arr = np.asarray(matches, dtype=int).reshape(-1, 2)
    tp = len(matches_arr)
    fn = n_gt - tp
    fp = n_pr - tp
    return tp, fp, fn, matches_arr


def missed_car_rate(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    iou_thresh: float = IOU_MATCH_THRESH,
) -> Tuple[float, int, int, int]:
    """
    Primary failure metric: FN / n_GT.

    Returns (failure, tp, fp, fn). failure is NaN if n_GT == 0.
    """
    tp, fp, fn, _ = hungarian_match(gt_boxes, pred_boxes, iou_thresh=iou_thresh)
    n_gt = len(np.asarray(gt_boxes).reshape(-1, 4))
    if n_gt == 0:
        return float("nan"), tp, fp, fn
    return float(fn) / float(n_gt), tp, fp, fn
