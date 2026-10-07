"""Compute missed-car failure metrics for paired real/proxy images."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .box_matching import missed_car_rate
from .config import (
    DET_CONF_THRESH,
    FRAME_STRIDE,
    IOU_MATCH_THRESH,
    KITTI_DIR,
    OUTPUT_ROOT,
    VKITTI_DIR,
    ensure_dirs,
)
from .detector import load_predictions_jsonl, predict_images
from .kitti_parser import parse_kitti_label_file
from .vkitti_parser import parse_vkitti_bbox_file
from .build_manifest import (
    find_kitti_paths,
    find_vkitti_scene_root,
    vkitti_bbox_path,
    vkitti_info_path,
)
from .io_utils import read_task_csv

logger = logging.getLogger(__name__)


from .enrich_features import FEATURE_COLS  # noqa: E402  — enriched detector features


def _gt_for_manifest_row(
    row,
    kitti_gt_cache: Dict[str, dict],
    vkitti_gt_cache: Dict[str, dict],
    image_root: Path,
    label_root: Path,
    vkitti_root: Path,
) -> tuple:
    seq = str(row.sequence_id).zfill(4)
    scene = str(row.vkitti_scene_id)
    frame = int(row.frame_id)

    if seq not in kitti_gt_cache:
        from PIL import Image
        lab = label_root / f"{seq}.txt"
        # Use image size from real image
        with Image.open(row.real_image_path) as im:
            wh = im.size
        kitti_gt_cache[seq] = parse_kitti_label_file(lab, image_wh=wh)

    if scene not in vkitti_gt_cache:
        from PIL import Image
        scene_root = find_vkitti_scene_root(vkitti_root, scene)
        with Image.open(row.proxy_image_path) as im:
            wh = im.size
        vkitti_gt_cache[scene] = parse_vkitti_bbox_file(
            vkitti_bbox_path(scene_root),
            vkitti_info_path(scene_root),
            image_wh=wh,
        )

    real_gt = kitti_gt_cache[seq].get(frame, np.zeros((0, 4)))
    proxy_gt = vkitti_gt_cache[scene].get(frame, np.zeros((0, 4)))
    return real_gt, proxy_gt


def compute_paired_detection_task(
    *,
    manifest_csv: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    run_detector: bool = True,
    overwrite: bool = False,
    device: Optional[str] = None,
) -> pd.DataFrame:
    ensure_dirs()
    manifest_csv = Path(manifest_csv) if manifest_csv else OUTPUT_ROOT / "paired_manifest.csv"
    output_csv = Path(output_csv) if output_csv else OUTPUT_ROOT / "paired_detection_task.csv"

    if output_csv.exists() and not overwrite:
        logger.info("Task CSV exists, skipping: %s", output_csv)
        return read_task_csv(output_csv)

    if not manifest_csv.exists():
        raise FileNotFoundError(
            f"Missing manifest {manifest_csv}. Run build_manifest first."
        )
    df = read_task_csv(manifest_csv)
    if len(df) == 0:
        raise RuntimeError("Empty paired manifest.")

    pred_real_path = OUTPUT_ROOT / "predictions" / "preds_real.jsonl"
    pred_proxy_path = OUTPUT_ROOT / "predictions" / "preds_proxy.jsonl"

    if run_detector:
        predict_images(
            df["real_image_path"].tolist(),
            domain="real",
            scenario_ids=df["scenario_id"].tolist(),
            cache_path=pred_real_path,
            device=device,
            conf_thresh=DET_CONF_THRESH,
            overwrite=False,
        )
        predict_images(
            df["proxy_image_path"].tolist(),
            domain="proxy",
            scenario_ids=df["scenario_id"].tolist(),
            cache_path=pred_proxy_path,
            device=device,
            conf_thresh=DET_CONF_THRESH,
            overwrite=False,
        )

    preds_real = load_predictions_jsonl(pred_real_path)
    preds_proxy = load_predictions_jsonl(pred_proxy_path)

    image_root, label_root = find_kitti_paths(KITTI_DIR)
    kitti_gt_cache: Dict[str, dict] = {}
    vkitti_gt_cache: Dict[str, dict] = {}

    rows = []
    for row in df.itertuples(index=False):
        real_gt, proxy_gt = _gt_for_manifest_row(
            row, kitti_gt_cache, vkitti_gt_cache, image_root, label_root, VKITTI_DIR,
        )
        pr = preds_real.get(row.real_image_path, {})
        pp = preds_proxy.get(row.proxy_image_path, {})
        real_pred = np.asarray(pr.get("boxes", []), dtype=np.float64).reshape(-1, 4)
        proxy_pred = np.asarray(pp.get("boxes", []), dtype=np.float64).reshape(-1, 4)

        t_fail, t_tp, t_fp, t_fn = missed_car_rate(
            real_gt, real_pred, iou_thresh=IOU_MATCH_THRESH,
        )
        p_fail, p_tp, p_fp, p_fn = missed_car_rate(
            proxy_gt, proxy_pred, iou_thresh=IOU_MATCH_THRESH,
        )

        rec = {c: getattr(row, c) for c in df.columns}
        rec.update(dict(
            proxy_failure=p_fail,
            target_failure=t_fail,
            proxy_tp=p_tp,
            proxy_fp=p_fp,
            proxy_fn=p_fn,
            target_tp=t_tp,
            target_fp=t_fp,
            target_fn=t_fn,
            det_conf_thresh=DET_CONF_THRESH,
            iou_match_thresh=IOU_MATCH_THRESH,
            frame_stride=FRAME_STRIDE,
        ))
        rows.append(rec)

    out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    meta = {
        "n_scenarios": int(len(out)),
        "det_conf_thresh": DET_CONF_THRESH,
        "iou_match_thresh": IOU_MATCH_THRESH,
        "frame_stride": FRAME_STRIDE,
        "mean_proxy_failure": float(out["proxy_failure"].mean()),
        "mean_target_failure": float(out["target_failure"].mean()),
    }
    (OUTPUT_ROOT / "task_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Wrote paired detection task (%d rows) → %s", len(out), output_csv)
    return out


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Scenario features for SCOUT (from Virtual KITTI columns)."""
    cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[cols].to_numpy(np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, default="")
    p.add_argument("--output", type=str, default="")
    p.add_argument("--no-detector", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", type=str, default="")
    args = p.parse_args()
    compute_paired_detection_task(
        manifest_csv=Path(args.manifest) if args.manifest else None,
        output_csv=Path(args.output) if args.output else None,
        run_detector=not args.no_detector,
        overwrite=args.overwrite,
        device=args.device or None,
    )


if __name__ == "__main__":
    main()
