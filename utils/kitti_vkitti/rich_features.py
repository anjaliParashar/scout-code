"""Rich proxy-domain feature matrix for SCOUT."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

from .build_manifest import find_vkitti_scene_root, vkitti_bbox_path, vkitti_info_path
from .config import OUTPUT_ROOT, VKITTI_DIR
from .detector import load_predictions_jsonl
from .io_utils import read_task_csv

logger = logging.getLogger(__name__)

RICH_FEATURE_NAMES = [
    "n_cars", "n_small", "n_med", "n_large",
    "h_mean", "h_min", "h_max", "h_std",
    "area_mean", "area_max", "area_std",
    "x_mean", "x_std", "y_mean", "y_std",
    "trunc_mean", "occ_mean",
    "n_pred", "score_mean", "score_max", "score_std",
    "pred_h_mean", "pred_area_mean", "pred_x_mean", "pred_x_std",
    "n_score_lo", "n_score_mid", "n_score_hi",
    "proxy_fp",
    "bright_mean", "bright_std", "edge_energy",
    "frame_norm",
    # PLS components toward proxy_failure (legal: proxy known for all pool)
    "pls0", "pls1", "pls2", "pls3",
]


def _vkitti_objects_by_frame():
    out = defaultdict(list)
    for scene_dir in sorted(Path(VKITTI_DIR).glob("Scene*")):
        scene = scene_dir.name
        try:
            sr = find_vkitti_scene_root(VKITTI_DIR, scene)
        except FileNotFoundError:
            continue
        info = pd.read_csv(vkitti_info_path(sr), sep=r"\s+", engine="python")
        cars = set(info.loc[info["label"].astype(str).str.lower() == "car", "trackID"].astype(int))
        bb = pd.read_csv(vkitti_bbox_path(sr), sep=r"\s+", engine="python")
        bb = bb[(bb["cameraID"] == 0) & (bb["trackID"].isin(cars))]
        for rec in bb.itertuples(index=False):
            h = float(rec.bottom - rec.top)
            if h < 25:
                continue
            out[(scene, int(rec.frame))].append(dict(
                h=h,
                w=float(rec.right - rec.left),
                area=float((rec.right - rec.left) * h),
                xc=float((rec.left + rec.right) / 2),
                yc=float((rec.top + rec.bottom) / 2),
                trunc=float(getattr(rec, "truncation_ratio", 0.0)),
                occ=float(getattr(rec, "occupancy_ratio", 0.0)),
            ))
    return out


def build_rich_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    preds = load_predictions_jsonl(OUTPUT_ROOT / "predictions" / "preds_proxy.jsonl")
    vk_objs = _vkitti_objects_by_frame()
    rows = []
    for row in df.itertuples(index=False):
        scene = str(row.vkitti_scene_id)
        frame = int(row.frame_id)
        with Image.open(row.proxy_image_path) as im:
            gray = np.asarray(im.convert("L"), dtype=np.float32)
            iw, ih = im.size
        area_img = float(max(iw * ih, 1))
        vos = vk_objs.get((scene, frame), [])
        if vos:
            hs = np.array([o["h"] for o in vos]) / ih
            ars = np.array([o["area"] for o in vos]) / area_img
            xs = np.array([o["xc"] for o in vos]) / iw
            ys = np.array([o["yc"] for o in vos]) / ih
            truncs = np.array([o["trunc"] for o in vos])
            occs = np.array([o["occ"] for o in vos])
            n_small = float(((hs * ih) < 40).sum())
            n_med = float((((hs * ih) >= 40) & ((hs * ih) < 80)).sum())
            n_large = float(((hs * ih) >= 80).sum())
        else:
            hs = ars = xs = ys = truncs = occs = np.array([0.0])
            n_small = n_med = n_large = 0.0

        prec = preds.get(row.proxy_image_path, {})
        pboxes = np.asarray(prec.get("boxes", []), dtype=np.float64).reshape(-1, 4)
        pscores = np.asarray(prec.get("scores", []), dtype=np.float64).ravel()
        if len(pboxes):
            ph = (pboxes[:, 3] - pboxes[:, 1]) / ih
            pa = ((pboxes[:, 2] - pboxes[:, 0]) * (pboxes[:, 3] - pboxes[:, 1])) / area_img
            px = ((pboxes[:, 0] + pboxes[:, 2]) / 2) / iw
        else:
            ph = pa = px = np.array([0.0])
            pscores = np.array([0.0])

        gy, gx = np.gradient(gray)
        edge = float(np.mean(np.hypot(gx, gy)) / 255.0)
        rows.append([
            float(len(vos)), n_small, n_med, n_large,
            float(hs.mean()), float(hs.min()), float(hs.max()),
            float(hs.std() if len(hs) > 1 else 0.0),
            float(ars.mean()), float(ars.max()), float(ars.std() if len(ars) > 1 else 0.0),
            float(xs.mean()), float(xs.std() if len(xs) > 1 else 0.0),
            float(ys.mean()), float(ys.std() if len(ys) > 1 else 0.0),
            float(truncs.mean()), float(occs.mean()),
            float(len(pboxes)),
            float(pscores.mean()), float(pscores.max()),
            float(pscores.std() if len(pscores) > 1 else 0.0),
            float(ph.mean()), float(pa.mean()),
            float(px.mean()), float(px.std() if len(px) > 1 else 0.0),
            float((pscores < 0.4).sum()),
            float(((pscores >= 0.4) & (pscores < 0.7)).sum()),
            float((pscores >= 0.7).sum()),
            float(getattr(row, "proxy_fp", 0.0)),
            float(gray.mean() / 255.0), float(gray.std() / 255.0), edge,
            float(frame) / 1000.0,
        ])

    base = np.asarray(rows, dtype=np.float64)
    base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
    y_proxy = df["proxy_failure"].to_numpy(np.float64)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(base)
    pls = PLSRegression(n_components=4)
    pls.fit(Xs, y_proxy)
    Z = pls.transform(Xs)
    X = np.concatenate([base, Z], axis=1)
    names = list(RICH_FEATURE_NAMES)
    assert X.shape[1] == len(names), (X.shape, len(names))
    return X, names


def write_rich_features(
    task_csv: Path | None = None,
    output_npy: Path | None = None,
    output_names: Path | None = None,
) -> tuple[np.ndarray, list[str]]:
    task_csv = Path(task_csv) if task_csv else OUTPUT_ROOT / "paired_detection_task_linked.csv"
    output_npy = Path(output_npy) if output_npy else OUTPUT_ROOT / "X_rich.npy"
    output_names = Path(output_names) if output_names else OUTPUT_ROOT / "X_rich_names.txt"
    df = read_task_csv(task_csv)
    X, names = build_rich_feature_matrix(df)
    # Also append sequence one-hot into saved matrix? keep in run_scout.
    np.save(output_npy, X)
    output_names.write_text("\n".join(names) + "\n")
    logger.info("Wrote rich features %s → %s", X.shape, output_npy)
    return X, names


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--task-csv", type=str, default="")
    args = p.parse_args()
    write_rich_features(task_csv=Path(args.task_csv) if args.task_csv else None)


if __name__ == "__main__":
    main()
