"""Headless diagnostics for the paired detection task."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .config import OUTPUT_ROOT, ensure_dirs
from .io_utils import read_task_csv

logger = logging.getLogger(__name__)


def failure_agreement(proxy: np.ndarray, target: np.ndarray, gamma: float) -> dict:
    p_fail = proxy >= gamma
    t_fail = target >= gamma
    return {
        "gamma": float(gamma),
        "proxy_safe_target_safe": int((~p_fail & ~t_fail).sum()),
        "proxy_fail_target_fail": int((p_fail & t_fail).sum()),
        "proxy_fail_target_safe": int((p_fail & ~t_fail).sum()),
        "proxy_safe_target_fail": int((~p_fail & t_fail).sum()),
        "agreement_rate": float(((p_fail & t_fail) | (~p_fail & ~t_fail)).mean()),
    }


def run_diagnostics(
    task_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> dict:
    ensure_dirs()
    output_dir = Path(output_dir) if output_dir else OUTPUT_ROOT
    task_csv = Path(task_csv) if task_csv else output_dir / "paired_detection_task.csv"
    json_path = output_dir / "diagnostics.json"
    md_path = output_dir / "diagnostics.md"
    scatter_path = output_dir / "proxy_vs_target_scatter.pdf"

    if json_path.exists() and scatter_path.exists() and not overwrite:
        logger.info("Diagnostics exist, skipping.")
        return json.loads(json_path.read_text())

    df = read_task_csv(task_csv)
    proxy = df["proxy_failure"].to_numpy(float)
    target = df["target_failure"].to_numpy(float)

    pear = pearsonr(proxy, target)
    spear = spearmanr(proxy, target)
    per_seq = df.groupby("sequence_id").size().to_dict()

    report = {
        "n_scenarios": int(len(df)),
        "scenarios_per_sequence": {str(k): int(v) for k, v in per_seq.items()},
        "proxy_failure_mean": float(np.mean(proxy)),
        "proxy_failure_std": float(np.std(proxy)),
        "target_failure_mean": float(np.mean(target)),
        "target_failure_std": float(np.std(target)),
        "pearson_r": float(pear.statistic),
        "pearson_p": float(pear.pvalue),
        "spearman_r": float(spear.statistic),
        "spearman_p": float(spear.pvalue),
        "agreement_gamma_0.25": failure_agreement(proxy, target, 0.25),
        "agreement_gamma_0.50": failure_agreement(proxy, target, 0.50),
    }

    # Scatter
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.scatter(proxy, target, s=18, alpha=0.65, edgecolors="none")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Proxy failure (VKITTI missed-car rate)")
    ax.set_ylabel("Target failure (KITTI missed-car rate)")
    ax.set_title(
        f"Proxy vs target  (n={len(df)}, "
        f"Pearson r={report['pearson_r']:.3f}, "
        f"Spearman ρ={report['spearman_r']:.3f})"
    )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(scatter_path)
    plt.close(fig)

    json_path.write_text(json.dumps(report, indent=2))

    a25 = report["agreement_gamma_0.25"]
    a50 = report["agreement_gamma_0.50"]
    md = f"""# KITTI–VKITTI detection task diagnostics

- Paired scenarios: **{report['n_scenarios']}**
- Per sequence: `{report['scenarios_per_sequence']}`

## Failure statistics

| Domain | Mean | Std |
|--------|------|-----|
| Proxy (VKITTI) | {report['proxy_failure_mean']:.4f} | {report['proxy_failure_std']:.4f} |
| Target (KITTI) | {report['target_failure_mean']:.4f} | {report['target_failure_std']:.4f} |

- Pearson r = {report['pearson_r']:.4f} (p={report['pearson_p']:.3g})
- Spearman ρ = {report['spearman_r']:.4f} (p={report['spearman_p']:.3g})

## Failure agreement

### γ = 0.25
- proxy safe / target safe: {a25['proxy_safe_target_safe']}
- proxy fail / target fail: {a25['proxy_fail_target_fail']}
- proxy fail / target safe: {a25['proxy_fail_target_safe']}
- proxy safe / target fail: {a25['proxy_safe_target_fail']}
- agreement rate: {a25['agreement_rate']:.3f}

### γ = 0.50
- proxy safe / target safe: {a50['proxy_safe_target_safe']}
- proxy fail / target fail: {a50['proxy_fail_target_fail']}
- proxy fail / target safe: {a50['proxy_fail_target_safe']}
- proxy safe / target fail: {a50['proxy_safe_target_fail']}
- agreement rate: {a50['agreement_rate']:.3f}

See `{scatter_path.name}` for the scatter plot.
"""
    md_path.write_text(md)
    logger.info("Wrote diagnostics → %s, %s, %s", json_path, md_path, scatter_path)
    return report


def save_qualitative_examples(
    task_csv: Optional[Path] = None,
    pred_real: Optional[Path] = None,
    pred_proxy: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    gamma: float = 0.5,
    n_per_bucket: int = 2,
) -> None:
    """Draw GT + predictions for a few agreement/disagreement examples."""
    import cv2

    from .detector import load_predictions_jsonl
    from .build_manifest import find_kitti_paths, find_vkitti_scene_root, vkitti_bbox_path, vkitti_info_path
    from .config import KITTI_DIR, VKITTI_DIR
    from .kitti_parser import parse_kitti_label_file
    from .vkitti_parser import parse_vkitti_bbox_file

    output_dir = Path(output_dir) if output_dir else OUTPUT_ROOT / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    task_csv = Path(task_csv) if task_csv else OUTPUT_ROOT / "paired_detection_task.csv"
    pred_real = Path(pred_real) if pred_real else OUTPUT_ROOT / "predictions" / "preds_real.jsonl"
    pred_proxy = Path(pred_proxy) if pred_proxy else OUTPUT_ROOT / "predictions" / "preds_proxy.jsonl"

    df = read_task_csv(task_csv)
    pr = load_predictions_jsonl(pred_real)
    pp = load_predictions_jsonl(pred_proxy)

    buckets = {
        "both_safe": (df["proxy_failure"] < gamma) & (df["target_failure"] < gamma),
        "both_fail": (df["proxy_failure"] >= gamma) & (df["target_failure"] >= gamma),
        "proxy_fail_target_safe": (df["proxy_failure"] >= gamma) & (df["target_failure"] < gamma),
        "proxy_safe_target_fail": (df["proxy_failure"] < gamma) & (df["target_failure"] >= gamma),
    }

    image_root, label_root = find_kitti_paths(KITTI_DIR)
    kitti_cache, vkitti_cache = {}, {}

    def _draw(img_path, gt, pred, title_suffix):
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        for box in np.asarray(gt).reshape(-1, 4):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for box in np.asarray(pred).reshape(-1, 4):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, title_suffix, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img

    for name, mask in buckets.items():
        sub = df.loc[mask].head(n_per_bucket)
        for _, row in sub.iterrows():
            seq = str(row.sequence_id).zfill(4)
            scene, frame = str(row.vkitti_scene_id), int(row.frame_id)
            if seq not in kitti_cache:
                from PIL import Image
                with Image.open(row.real_image_path) as im:
                    wh = im.size
                kitti_cache[seq] = parse_kitti_label_file(label_root / f"{seq}.txt", image_wh=wh)
            if scene not in vkitti_cache:
                from PIL import Image
                scene_root = find_vkitti_scene_root(VKITTI_DIR, scene)
                with Image.open(row.proxy_image_path) as im:
                    wh = im.size
                vkitti_cache[scene] = parse_vkitti_bbox_file(
                    vkitti_bbox_path(scene_root), vkitti_info_path(scene_root), image_wh=wh,
                )
            real_gt = kitti_cache[seq].get(frame, np.zeros((0, 4)))
            proxy_gt = vkitti_cache[scene].get(frame, np.zeros((0, 4)))
            real_pred = np.asarray(pr.get(row.real_image_path, {}).get("boxes", [])).reshape(-1, 4)
            proxy_pred = np.asarray(pp.get(row.proxy_image_path, {}).get("boxes", [])).reshape(-1, 4)

            left = _draw(row.real_image_path, real_gt, real_pred, f"REAL y={row.target_failure:.2f}")
            right = _draw(row.proxy_image_path, proxy_gt, proxy_pred, f"PROXY y={row.proxy_failure:.2f}")
            if left is None or right is None:
                continue
            h = min(left.shape[0], right.shape[0])
            left, right = left[:h], right[:h]
            combo = np.concatenate([left, right], axis=1)
            out_p = output_dir / f"{name}_{row.scenario_id}.jpg"
            cv2.imwrite(str(out_p), combo)
            logger.info("Wrote example %s", out_p)


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--task-csv", type=str, default="")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--examples", action="store_true")
    args = p.parse_args()
    run_diagnostics(
        task_csv=Path(args.task_csv) if args.task_csv else None,
        overwrite=args.overwrite,
    )
    if args.examples:
        save_qualitative_examples()


if __name__ == "__main__":
    main()
