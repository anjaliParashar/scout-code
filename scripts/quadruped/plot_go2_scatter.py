#!/usr/bin/env python3
"""
scripts/quadruped/plot_go2_scatter.py
---------------------------------------
2D scatter plots comparing SCOUT vs BAMS — failure scenarios only.
Three projections: (vx,vy), (vx,wz), (vy,wz).
Colour = method,  shape = seed.

Example
-------
python scripts/quadruped/plot_go2_scatter.py \
    --results-root results/go2_hardware \
    --threshold 0.7 \
    --output-dir results/quadruped/plots
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

_METHOD_STYLE = {
    "scout": dict(color="#e41a1c", label="Our"),
    "bams":  dict(color="#377eb8", label="BAMS"),
}
_METHOD_MARKERS = {"scout": ["o", "^"], "bams": ["s", "D"]}

_PROJECTIONS = [
    (0, 1, r"$v_x$", r"$v_y$"),
    (0, 2, r"$v_x$", r"$w_z$"),
    (1, 2, r"$v_y$", r"$w_z$"),
]

def _save(fig, path: Path, dpi: int = 200):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_seed(pkl_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    pkl_files = sorted(pkl_dir.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files in {pkl_dir}")
    X_list, y_list = [], []
    for f in pkl_files:
        with open(f, "rb") as fh:
            data = pickle.load(fh)
        X_list.append([data["command"]["vx"],
                        data["command"]["vy"],
                        data["command"]["wz"]])
        y_list.append(data["error_summary"]["mean_abs_err_sum"])
    return (np.array(X_list, dtype=np.float64),
            np.array(y_list, dtype=np.float64))


def load_method(results_root, method_base, seed_suffixes=("", "_2")):
    seeds = []
    for suf in seed_suffixes:
        d = results_root / f"{method_base}{suf}"
        if not d.is_dir():
            print(f"  [WARN] Not found: {d}")
            continue
        print(f"  Loading {d.name} …")
        X, y = load_seed(d)
        fail_n = int((y >= 0).sum())   # placeholder; threshold applied later
        print(f"    N={len(y)}  y=[{y.min():.3f}, {y.max():.3f}]")
        seeds.append((X, y))
    return seeds


# ---------------------------------------------------------------------------
# Main plot — failure points only
# ---------------------------------------------------------------------------

def plot_failure_scatter(method_seeds, threshold, output_path):
    """
    1×3 grid, one cell per 2D velocity projection.
    Only points with y >= threshold are shown.
    Colour = method,  shape = seed.
    """
    methods  = [m for m in ("scout", "bams") if m in method_seeds]
    n_seeds  = max(len(v) for v in method_seeds.values())
    marker   = "o"   # same marker for all — colour distinguishes method

    for s_idx in range(n_seeds):
        fig, axes     = plt.subplots(1, 3, figsize=(14, 4.5))
        legend_handles = []

        for method in methods:
            seeds = method_seeds[method]
            if s_idx >= len(seeds):
                continue
            X, y  = seeds[s_idx]
            fail  = y >= threshold
            color = _METHOD_STYLE[method]["color"]
            label = _METHOD_STYLE[method]["label"]

            if not fail.any():
                print(f"  [WARN] {method} seed {s_idx+1}: "
                      f"no failures at threshold={threshold}")
                continue

            for col, (xi, yi, xl, yl) in enumerate(_PROJECTIONS):
                axes[col].scatter(
                    X[fail, xi], X[fail, yi],
                    c=color, s=120, alpha=0.85,
                    marker=marker, edgecolors="none",
                    zorder=3,
                )

            legend_handles.append(Line2D(
                [0], [0], marker=marker, color="w",
                markerfacecolor=color, markersize=10,
                label=label,
            ))

        for col, (xi, yi, xl, yl) in enumerate(_PROJECTIONS):
            axes[col].set_xlabel(xl, fontsize=25)
            axes[col].set_ylabel(yl, fontsize=25)
            # axes[col].set_title(f"{xl} vs {yl}", fontsize=12)
            axes[col].grid(alpha=0.25)
            axes[col].tick_params(labelsize=20)

        if legend_handles:
            fig.legend(handles=legend_handles, fontsize=20,
                       loc="lower center", ncol=len(legend_handles),
                       framealpha=0.9, bbox_to_anchor=(0.5, -0.1))

        # fig.suptitle(
        #     f"Failure scenarios (y $\\geq$ {threshold}) — Seed {s_idx + 1}",
        #     fontsize=13, y=1.02)
        plt.tight_layout()

        stem      = output_path.stem
        suffix    = output_path.suffix
        seed_path = output_path.with_name(f"{stem}_seed{s_idx + 1}{suffix}")
        _save(fig, seed_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.results_root)

    method_seeds = {}
    for method in ("scout", "bams"):
        print(f"\n[INFO] Loading {method} …")
        seeds = load_method(root, method)
        if seeds:
            method_seeds[method] = seeds

    if not method_seeds:
        print("[ERROR] No data loaded.")
        return

    plot_failure_scatter(
        method_seeds, args.threshold,
        out_dir / "failure_scatter.pdf",
    )
    plot_failure_scatter(
        method_seeds, args.threshold,
        out_dir / "failure_scatter.png",
    )
    print("[DONE]")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=str,
                   default="results/go2_hardware")
    p.add_argument("--threshold",    type=float, default=0.7)
    p.add_argument("--output-dir",   type=str,
                   default="results/quadruped/plots")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
