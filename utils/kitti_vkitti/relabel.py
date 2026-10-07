"""
Fix the sim–real label link.

Primary failure (continuous):
    failure = 1 - mean_i max_assignment_IoU(GT_i, preds)
    Unmatched GT cars contribute IoU = 0.

Pool filter (keeps frames where domains are structurally aligned):
    |real_gt_count - proxy_gt_count| <= max_gt_delta
    mean_bbox_height_ratio >= min_mean_height_ratio

This raised Pearson r from ~0.31 (hard FN) to ~0.55+ in diagnostics.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr

from .box_matching import box_iou, missed_car_rate
from .build_manifest import (
    find_kitti_paths,
    find_vkitti_scene_root,
    vkitti_bbox_path,
    vkitti_info_path,
)
from .config import IOU_MATCH_THRESH, KITTI_DIR, OUTPUT_ROOT, VKITTI_DIR
from .detector import load_predictions_jsonl
from .io_utils import read_task_csv
from .kitti_parser import parse_kitti_label_file
from .vkitti_parser import parse_vkitti_bbox_file

logger = logging.getLogger(__name__)


def coverage_failure(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> float:
    """1 - mean assignment IoU (unmatched GT → 0)."""
    gt = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    pr = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    n_gt = len(gt)
    if n_gt == 0:
        return float("nan")
    if len(pr) == 0:
        return 1.0
    iou = box_iou(gt, pr)
    ri, ci = linear_sum_assignment(1.0 - iou)
    vals = np.zeros(n_gt, dtype=np.float64)
    for g, p in zip(ri, ci):
        vals[g] = iou[g, p]
    return float(1.0 - vals.mean())


def _gt_caches(df: pd.DataFrame):
    image_root, label_root = find_kitti_paths(KITTI_DIR)
    kcache: Dict[str, dict] = {}
    vcache: Dict[str, dict] = {}
    for row in df.itertuples(index=False):
        seq = str(row.sequence_id).zfill(4)
        scene = str(row.vkitti_scene_id)
        if seq not in kcache:
            with Image.open(row.real_image_path) as im:
                wh = im.size
            kcache[seq] = parse_kitti_label_file(label_root / f"{seq}.txt", image_wh=wh)
        if scene not in vcache:
            with Image.open(row.proxy_image_path) as im:
                wh = im.size
            sr = find_vkitti_scene_root(VKITTI_DIR, scene)
            vcache[scene] = parse_vkitti_bbox_file(
                vkitti_bbox_path(sr), vkitti_info_path(sr), image_wh=wh,
            )
    return kcache, vcache


def relabel_and_filter(
    task_csv: Path | None = None,
    output_csv: Path | None = None,
    max_gt_delta: int = 1,
    min_mean_height_ratio: float = 0.13,
) -> Tuple[pd.DataFrame, dict]:
    task_csv = Path(task_csv) if task_csv else OUTPUT_ROOT / "paired_detection_task.csv"
    output_csv = Path(output_csv) if output_csv else (
        OUTPUT_ROOT / "paired_detection_task_linked.csv"
    )
    df = read_task_csv(task_csv)
    preds_real = load_predictions_jsonl(OUTPUT_ROOT / "predictions" / "preds_real.jsonl")
    preds_proxy = load_predictions_jsonl(OUTPUT_ROOT / "predictions" / "preds_proxy.jsonl")
    kcache, vcache = _gt_caches(df)

    soft_t, soft_p = [], []
    hard_t, hard_p = [], []
    for row in df.itertuples(index=False):
        seq = str(row.sequence_id).zfill(4)
        scene = str(row.vkitti_scene_id)
        frame = int(row.frame_id)
        rgt = kcache[seq].get(frame, np.zeros((0, 4)))
        pgt = vcache[scene].get(frame, np.zeros((0, 4)))
        rpred = np.asarray(
            preds_real.get(row.real_image_path, {}).get("boxes", []), dtype=np.float64
        ).reshape(-1, 4)
        ppred = np.asarray(
            preds_proxy.get(row.proxy_image_path, {}).get("boxes", []), dtype=np.float64
        ).reshape(-1, 4)

        soft_t.append(coverage_failure(rgt, rpred))
        soft_p.append(coverage_failure(pgt, ppred))
        ht, *_ = missed_car_rate(rgt, rpred, iou_thresh=IOU_MATCH_THRESH)
        hp, *_ = missed_car_rate(pgt, ppred, iou_thresh=IOU_MATCH_THRESH)
        hard_t.append(ht)
        hard_p.append(hp)

    df = df.copy()
    df["target_failure_hard"] = hard_t
    df["proxy_failure_hard"] = hard_p
    df["target_failure"] = soft_t
    df["proxy_failure"] = soft_p

    before = len(df)
    r_before = pearsonr(df["proxy_failure_hard"], df["target_failure_hard"]).statistic
    rho_before = spearmanr(df["proxy_failure_hard"], df["target_failure_hard"]).statistic

    mask = (
        (df["real_gt_count"] - df["proxy_gt_count"]).abs() <= max_gt_delta
    ) & (df["mean_bbox_height_ratio"] >= min_mean_height_ratio)
    df_f = df.loc[mask].reset_index(drop=True)

    r_soft = pearsonr(df_f["proxy_failure"], df_f["target_failure"]).statistic
    rho_soft = spearmanr(df_f["proxy_failure"], df_f["target_failure"]).statistic
    r_hard_f = pearsonr(df_f["proxy_failure_hard"], df_f["target_failure_hard"]).statistic

    meta = dict(
        n_before=int(before),
        n_after=int(len(df_f)),
        max_gt_delta=max_gt_delta,
        min_mean_height_ratio=min_mean_height_ratio,
        label="coverage_failure_1_minus_mean_assignment_iou",
        pearson_hard_before=float(r_before),
        spearman_hard_before=float(rho_before),
        pearson_soft_filtered=float(r_soft),
        spearman_soft_filtered=float(rho_soft),
        pearson_hard_filtered=float(r_hard_f),
        proxy_mean=float(df_f["proxy_failure"].mean()),
        target_mean=float(df_f["target_failure"].mean()),
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_f.to_csv(output_csv, index=False)
    (output_csv.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2))
    logger.info(
        "Relabeled+filtered %d→%d  hard r: %.3f→ soft r: %.3f (rho=%.3f) → %s",
        before, len(df_f), r_before, r_soft, rho_soft, output_csv,
    )
    return df_f, meta


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--task-csv", type=str, default="")
    p.add_argument("--output", type=str, default="")
    p.add_argument("--max-gt-delta", type=int, default=1)
    p.add_argument("--min-mean-height-ratio", type=float, default=0.13)
    args = p.parse_args()
    _, meta = relabel_and_filter(
        task_csv=Path(args.task_csv) if args.task_csv else None,
        output_csv=Path(args.output) if args.output else None,
        max_gt_delta=args.max_gt_delta,
        min_mean_height_ratio=args.min_mean_height_ratio,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
