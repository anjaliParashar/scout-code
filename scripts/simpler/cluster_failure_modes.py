#!/usr/bin/env python3
"""
scripts/simpler/cluster_failure_modes.py
-----------------------------------------
K-means clustering of failure scenarios collected across all seeds to
identify representative failure modes and map each cluster back to
meaningful scenario descriptions.

Pipeline
--------
1. Collect all AL_{t}_{j} directories across all seeds and methods.
2. Load paired_result.json from each — keep only FAILURE scenarios
   (target.success == False).
3. Build a feature vector from the scenario fields that vary:
     [obj_x, obj_y,
      cam_dx, cam_dy, cam_dz, cam_roll, cam_pitch, cam_yaw,
      crop_shift_x, crop_shift_y,
      brightness, contrast, gamma,
      proxy_brightness_gain, target_brightness_gain,
      proxy_contrast_gain,  target_contrast_gain,
      blur, noise_std, occlusion_frac,
      n_distractors]
4. Standardise features and run KMeans with K = n_clusters.
5. For each cluster, print:
     - centroid description (human-readable interpretation of each feature)
     - representative scenario (closest to centroid)
     - failure rate, task distribution, seed distribution
6. Save cluster assignments and a summary CSV.

Usage
-----
python scripts/simpler/cluster_failure_modes.py \
    --seeds-root results \
    --seeds      0 1 2 \
    --methods    scout bnn_cv random gpc \
    --n-clusters 5 \
    --n-show     60 \
    --output-dir ./outputs/simpler/failure_clusters
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster      import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Feature extraction from paired_result.json
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "obj_x", "obj_y",
    "cam_dx", "cam_dy", "cam_dz",
    "cam_roll", "cam_pitch", "cam_yaw",
    "crop_shift_x", "crop_shift_y",
    "brightness", "contrast", "gamma_ctrl",
    "proxy_brightness_gain", "target_brightness_gain",
    "proxy_contrast_gain",   "target_contrast_gain",
    "blur", "noise_std", "occlusion_frac",
    "n_distractors",
]

# Human-readable thresholds for centroid description
_FEATURE_DESC = {
    "obj_x":                   ("object x", "left",        "right"),
    "obj_y":                   ("object y", "near",         "far"),
    "cam_dx":                  ("camera dx", "shifted left", "shifted right"),
    "cam_dy":                  ("camera dy", "shifted back", "shifted fwd"),
    "cam_dz":                  ("camera dz", "lower",        "higher"),
    "cam_roll":                ("cam roll",  "tilted left",  "tilted right"),
    "cam_pitch":               ("cam pitch", "tilted down",  "tilted up"),
    "cam_yaw":                 ("cam yaw",   "yawed left",   "yawed right"),
    "crop_shift_x":            ("crop x",    "left crop",    "right crop"),
    "crop_shift_y":            ("crop y",    "top crop",     "bottom crop"),
    "brightness":              ("brightness","dim",          "bright"),
    "contrast":                ("contrast",  "low contrast", "high contrast"),
    "gamma_ctrl":              ("gamma",     "dark",         "washed out"),
    "proxy_brightness_gain":   ("proxy bright","proxy dim",  "proxy bright"),
    "target_brightness_gain":  ("tgt bright","target dim",   "target bright"),
    "proxy_contrast_gain":     ("proxy cont","proxy flat",   "proxy sharp"),
    "target_contrast_gain":    ("tgt cont",  "target flat",  "target sharp"),
    "blur":                    ("blur",      "sharp",        "blurry"),
    "noise_std":               ("noise",     "clean",        "noisy"),
    "occlusion_frac":          ("occlusion", "unoccluded",   "occluded"),
    "n_distractors":           ("distractors","no distractors","many distractors"),
}


def _extract_features(data: dict) -> np.ndarray | None:
    """Extract a fixed-length feature vector from a paired_result.json dict."""
    try:
        sc  = data["scenario"]
        cam = sc.get("camera", {})
        vis = sc.get("visual_control", {})
        vm  = sc.get("visual_map", {})
        gap = sc.get("fixed_gap", {})
        cl  = sc.get("clutter", {})

        obj_x = sc.get("object_init", {}).get("x", 0.0)
        obj_y = sc.get("object_init", {}).get("y", 0.0)

        n_dist = len(cl.get("distractors", []))

        vec = np.array([
            obj_x,
            obj_y,
            cam.get("dx", 0.0),
            cam.get("dy", 0.0),
            cam.get("dz", 0.0),
            cam.get("roll",  0.0),
            cam.get("pitch", 0.0),
            cam.get("yaw",   0.0),
            cam.get("crop_shift_x", 0.0),
            cam.get("crop_shift_y", 0.0),
            vis.get("brightness", 1.0),
            vis.get("contrast",   1.0),
            vis.get("gamma",      1.0),
            vm.get("proxy_brightness_gain",  1.0),
            vm.get("target_brightness_gain", 1.0),
            vm.get("proxy_contrast_gain",    1.0),
            vm.get("target_contrast_gain",   1.0),
            gap.get("blur",           0.0),
            gap.get("noise_std",      0.0),
            gap.get("occlusion_frac", 0.0),
            float(n_dist),
        ], dtype=np.float64)
        return vec
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data collection across seeds / methods
# ---------------------------------------------------------------------------

def _find_paired_results(method_dir: Path, method: str,
                          n_show: int) -> list[Path]:
    """Return paired_result.json paths under AL_{t}_{j} dirs, sorted."""
    if method in ("scout", "bams", "bnn_cv", "gpc"):
        prefix = "AL_"
    else:   # random
        prefix = "random_"

    al_dirs: dict = {}
    for p in method_dir.glob(f"{prefix}*"):
        parts = p.name.split("_")
        if len(parts) >= 3:
            try:
                t = int(parts[1])
                al_dirs.setdefault(t, []).append(p)
            except ValueError:
                pass

    flat = [p for sub in [sorted(al_dirs[t]) for t in sorted(al_dirs)]
              for p in sub]
    if n_show > 0:
        flat = flat[:n_show]

    results = []
    for d in flat:
        matches = list(d.glob("*/rt1/*/scenario_*/paired_result.json"))
        results.extend(matches)
    return results


def collect_failures(
    seeds_root: str,
    seeds:      list[int],
    methods:    list[str],
    n_show:     int,
) -> tuple[np.ndarray, list[dict]]:
    """
    Load all failure scenarios across seeds and methods.

    Returns
    -------
    X       : (N_fail, D) raw feature matrix
    records : list of metadata dicts (one per failure)
    """
    X_list  = []
    records = []

    for s in seeds:
        for method in methods:
            method_dir = Path(f"{seeds_root}_{s}") / method
            if not method_dir.is_dir():
                print(f"  [WARN] Not found: {method_dir}")
                continue

            paths = _find_paired_results(method_dir, method, n_show)
            n_fail = 0
            for p in paths:
                try:
                    with open(p) as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"  [WARN] Cannot read {p}: {e}")
                    continue

                if data["target"]["success"]:
                    continue   # keep failures only

                vec = _extract_features(data)
                if vec is None:
                    continue

                X_list.append(vec)
                records.append(dict(
                    seed          = s,
                    method        = method,
                    task          = data["scenario"].get("task", "unknown"),
                    episode_id    = data["scenario"].get("episode_id", -1),
                    scenario_id   = data.get("scenario_id", -1),
                    json_path     = str(p),
                    timesteps     = data["target"].get("timesteps", -1),
                    proxy_success = data["proxy"].get("success", None),
                    distractor_config = (data["scenario"]
                                         .get("clutter", {})
                                         .get("distractor_config", "none")),
                ))
                n_fail += 1

            print(f"  seed={s}  method={method}: "
                  f"{len(paths)} rollouts  {n_fail} failures")

    if not X_list:
        raise ValueError("No failure scenarios found.")

    return np.array(X_list, dtype=np.float64), records


# ---------------------------------------------------------------------------
# Human-readable centroid description
# ---------------------------------------------------------------------------

def _describe_centroid(centroid_raw: np.ndarray,
                        centroid_std:  np.ndarray) -> str:
    """
    Turn a centroid in original feature space into a bullet-point description.
    Only mentions features where the centroid is notably above/below zero
    (more than 0.5 std from the population mean, which is 0 after standardisation).
    """
    lines = []
    for i, name in enumerate(FEATURE_NAMES):
        val   = centroid_raw[i]
        sigma = centroid_std[i]
        if sigma < 1e-9:
            continue
        z = val / sigma   # z-score relative to full dataset std
        label, lo, hi = _FEATURE_DESC[name]
        if z > 0.6:
            lines.append(f"  • {label}: {hi}  (z={z:+.2f})")
        elif z < -0.6:
            lines.append(f"  • {label}: {lo}  (z={z:+.2f})")
    return "\n".join(lines) if lines else "  • (near average on all features)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect failures ───────────────────────────────────────────────────
    print(f"[INFO] Collecting failures from {len(args.seeds)} seeds, "
          f"{len(args.methods)} methods …")
    X_raw, records = collect_failures(
        seeds_root = args.seeds_root,
        seeds      = args.seeds,
        methods    = args.methods,
        n_show     = args.n_show,
    )
    N = len(records)
    print(f"\n[INFO] Total failures: {N}")

    # ── Standardise ───────────────────────────────────────────────────────
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    feat_std = scaler.scale_   # per-feature std for description

    # ── KMeans ────────────────────────────────────────────────────────────
    k  = min(args.n_clusters, N)
    print(f"[INFO] Running KMeans (k={k}) …")
    km = KMeans(n_clusters=k, random_state=args.seed,
                n_init="auto", max_iter=500)
    labels = km.fit_predict(X_scaled)

    # ── PCA for visualisation ──────────────────────────────────────────────
    pca  = PCA(n_components=2, random_state=args.seed)
    X_2d = pca.fit_transform(X_scaled)
    centers_2d = pca.transform(km.cluster_centers_)

    # ── Cluster colours ───────────────────────────────────────────────────
    cmap   = plt.cm.get_cmap("tab10", k)
    colors = [cmap(c) for c in range(k)]

    # ── PCA scatter ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    for c in range(k):
        mask = labels == c
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=colors[c], s=50, alpha=0.6,
                   label=f"Cluster {c+1} (n={mask.sum()})")
    for c in range(k):
        ax.scatter(*centers_2d[c], color=colors[c],
                   s=250, marker="*", edgecolors="black",
                   linewidths=1.0, zorder=5)
        ax.annotate(f"C{c+1}", centers_2d[c],
                    fontsize=9, fontweight="bold",
                    ha="center", va="bottom")
    ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)", fontsize=11)
    ax.set_title("Failure scenario clusters (PCA projection)\n"
                 f"k={k}, N={N} failures across {len(args.seeds)} seeds "
                 f"× {len(args.methods)} methods", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    scatter_path = out_dir / "cluster_scatter.pdf"
    fig.savefig(scatter_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {scatter_path}")

    # ── Per-cluster analysis ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"FAILURE MODE ANALYSIS  (k={k}, N={N})")
    print(f"{'='*70}")

    summary_rows = []

    for c in range(k):
        mask      = labels == c
        idx       = np.where(mask)[0]
        n_c       = int(mask.sum())
        cluster_X = X_raw[mask]

        # Centroid in original feature space
        centroid_raw = scaler.inverse_transform(
            km.cluster_centers_[c:c+1])[0]

        # Representative: closest to centroid in scaled space
        dists        = np.linalg.norm(X_scaled[mask] -
                                       km.cluster_centers_[c], axis=1)
        rep_local    = int(np.argmin(dists))
        rep_global   = idx[rep_local]
        rep          = records[rep_global]

        # Task distribution
        tasks        = [records[i]["task"] for i in idx]
        task_counts  = {}
        for t in tasks:
            task_counts[t] = task_counts.get(t, 0) + 1
        top_tasks    = sorted(task_counts, key=task_counts.get, reverse=True)

        # Seed distribution
        seed_counts  = {}
        for i in idx:
            s = records[i]["seed"]
            seed_counts[s] = seed_counts.get(s, 0) + 1

        # Method distribution
        method_counts = {}
        for i in idx:
            m = records[i]["method"]
            method_counts[m] = method_counts.get(m, 0) + 1

        # Proxy success rate (did sim predict success but target failed?)
        proxy_ok = sum(1 for i in idx
                       if records[i]["proxy_success"] is True)
        proxy_rate = proxy_ok / n_c if n_c > 0 else 0.0

        print(f"\n{'─'*70}")
        print(f"Cluster {c+1}  (n={n_c}, {100*n_c/N:.1f}% of failures)")
        print(f"{'─'*70}")
        print(f"Centroid feature profile:")
        print(_describe_centroid(centroid_raw, feat_std))
        print(f"\nTask distribution:")
        for t in top_tasks[:3]:
            print(f"  {task_counts[t]:3d}x  {t}")
        print(f"\nMethod distribution:  "
              + "  ".join(f"{m}={method_counts[m]}" for m in
                          sorted(method_counts, key=method_counts.get, reverse=True)))
        print(f"Seed distribution:    "
              + "  ".join(f"seed{s}={seed_counts[s]}" for s in sorted(seed_counts)))
        print(f"Proxy success rate:   {100*proxy_rate:.1f}%  "
              f"(sim said OK but robot failed)")
        print(f"\nRepresentative scenario:")
        print(f"  task       : {rep['task']}")
        print(f"  episode_id : {rep['episode_id']}")
        print(f"  seed       : {rep['seed']}  method: {rep['method']}")
        print(f"  timesteps  : {rep['timesteps']}")
        print(f"  distractors: {rep['distractor_config']}")
        print(f"  json       : {rep['json_path']}")

        summary_rows.append(dict(
            cluster        = c + 1,
            n              = n_c,
            pct            = round(100 * n_c / N, 1),
            top_task       = top_tasks[0] if top_tasks else "",
            proxy_succ_pct = round(100 * proxy_rate, 1),
            rep_task       = rep["task"],
            rep_episode    = rep["episode_id"],
            rep_seed       = rep["seed"],
            rep_method     = rep["method"],
            rep_json       = rep["json_path"],
        ))

    print(f"\n{'='*70}")

    # ── Save cluster assignments ───────────────────────────────────────────
    import csv
    csv_path = out_dir / "cluster_assignments.csv"
    fieldnames = ["cluster", "seed", "method", "task", "episode_id",
                  "scenario_id", "timesteps", "proxy_success",
                  "distractor_config", "json_path"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, rec in enumerate(records):
            row = {k: rec[k] for k in fieldnames if k in rec}
            row["cluster"] = int(labels[i]) + 1
            w.writerow(row)
    print(f"[INFO] Cluster assignments → {csv_path}")

    # Save summary
    sum_path = out_dir / "cluster_summary.csv"
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[INFO] Cluster summary     → {sum_path}")

    # Save feature matrix and labels
    np.save(out_dir / "X_failures.npy", X_raw)
    np.save(out_dir / "labels.npy",     labels)
    print(f"[INFO] Feature matrix      → {out_dir / 'X_failures.npy'}")
    print(f"\n[DONE]  outputs → {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="K-means failure mode analysis for SimplerEnv AL results."
    )
    p.add_argument("--seeds-root",  type=str, required=True,
                   help="Prefix for seed dirs. Seed s → {seeds-root}_{s}/{method}/")
    p.add_argument("--seeds",       type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--methods",     type=str, nargs="+",
                   default=["scout", "bnn_cv", "random", "gpc"])
    p.add_argument("--n-clusters",  type=int, default=5,
                   help="Number of KMeans clusters")
    p.add_argument("--n-show",      type=int, default=60,
                   help="Max AL dirs per method per seed (0 = all)")
    p.add_argument("--seed",        type=int, default=42,
                   help="KMeans random seed")
    p.add_argument("--output-dir",  type=str,
                   default="./outputs/simpler/failure_clusters")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
