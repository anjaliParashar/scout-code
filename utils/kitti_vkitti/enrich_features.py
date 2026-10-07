"""Add detector-side proxy features to the paired detection task CSV."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import OUTPUT_ROOT
from .detector import load_predictions_jsonl
from .io_utils import read_task_csv

logger = logging.getLogger(__name__)

# Features used by SCOUT (scenario geometry + detector diagnostics).
# Do NOT include proxy_failure / FN / TP — those leak the CV label into X.
FEATURE_COLS = [
    "num_cars",
    "mean_bbox_area_ratio",
    "min_bbox_area_ratio",
    "max_bbox_area_ratio",
    "mean_bbox_height_ratio",
    "image_brightness",
    "proxy_n_pred",
    "proxy_mean_score",
    "proxy_max_score",
    "proxy_mean_pred_area_ratio",
    "proxy_fp",
]


def _pred_stats(rec: dict, image_wh: tuple[int, int]) -> dict:
    boxes = np.asarray(rec.get("boxes", []), dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(rec.get("scores", []), dtype=np.float64).ravel()
    iw, ih = image_wh
    area_img = float(max(iw * ih, 1))
    if len(boxes) == 0:
        return dict(
            proxy_n_pred=0,
            proxy_mean_score=0.0,
            proxy_max_score=0.0,
            proxy_mean_pred_area_ratio=0.0,
        )
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    return dict(
        proxy_n_pred=int(len(boxes)),
        proxy_mean_score=float(scores.mean()) if len(scores) else 0.0,
        proxy_max_score=float(scores.max()) if len(scores) else 0.0,
        proxy_mean_pred_area_ratio=float(areas.mean() / area_img),
    )


def enrich_task_csv(
    task_csv: Path | None = None,
    pred_proxy: Path | None = None,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    task_csv = Path(task_csv) if task_csv else OUTPUT_ROOT / "paired_detection_task.csv"
    pred_proxy = Path(pred_proxy) if pred_proxy else (
        OUTPUT_ROOT / "predictions" / "preds_proxy.jsonl"
    )
    output_csv = Path(output_csv) if output_csv else task_csv

    df = read_task_csv(task_csv)
    preds = load_predictions_jsonl(pred_proxy)
    if not preds:
        raise FileNotFoundError(f"No proxy predictions in {pred_proxy}")

    # Cache image sizes per unique path
    size_cache: dict[str, tuple[int, int]] = {}
    rows = []
    for row in df.itertuples(index=False):
        path = row.proxy_image_path
        if path not in size_cache:
            with Image.open(path) as im:
                size_cache[path] = im.size
        rec = preds.get(path, {})
        stats = _pred_stats(rec, size_cache[path])
        rows.append(stats)

    feat = pd.DataFrame(rows)
    for c in feat.columns:
        df[c] = feat[c].to_numpy()

    # Ensure TP/FP/FN present (from original metrics stage)
    for c in ("proxy_tp", "proxy_fp", "proxy_fn", "proxy_failure"):
        if c not in df.columns:
            raise RuntimeError(f"Missing column {c} in {task_csv}")

    df.to_csv(output_csv, index=False)
    logger.info(
        "Enriched %d rows → %s  (feature cols: %s)",
        len(df), output_csv, FEATURE_COLS,
    )

    # Quick correlation diagnostic with new features available
    from scipy.stats import pearsonr
    r = pearsonr(df["proxy_failure"], df["target_failure"]).statistic
    logger.info("proxy↔target failure Pearson r=%.4f", r)
    return df


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--task-csv", type=str, default="")
    p.add_argument("--output", type=str, default="")
    args = p.parse_args()
    enrich_task_csv(
        task_csv=Path(args.task_csv) if args.task_csv else None,
        output_csv=Path(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()
