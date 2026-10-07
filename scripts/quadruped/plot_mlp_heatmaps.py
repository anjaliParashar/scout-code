#!/usr/bin/env python3
"""
scripts/quadruped/plot_mlp_heatmaps.py
----------------------------------------
Heatmaps of the MLP surrogate's predicted tracking error y over all three
pairwise 2D projections of the velocity space:
    (vx, vy)   — wz fixed at its median
    (vx, wz)   — vy fixed at its median
    (vy, wz)   — vx fixed at its median

The third axis is fixed at the median of the hardware-evaluated commands
so the slice is representative of the data collected.

For each projection, a 2D grid is swept.  The MLP is queried at each
grid point with MC-dropout to get both the posterior mean and uncertainty.

Four panels per method:
    A : posterior mean  (main prediction)
    B : posterior std   (epistemic uncertainty)

Hardware-evaluated points are overlaid on each panel as red stars.

Supports any checkpoint saved by train_mlp / save_checkpoint in train_utils.py.

Example
-------
python scripts/quadruped/plot_mlp_heatmaps.py \
    --ckpt_path    results/quadruped/hardware.pt \
    --hardware_glob "results/go2_hardware/scout/*.pkl" \
    --output_dir   outputs/quadruped/heatmaps \
    --method_name  SCOUT

"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---------- import the project utilities ----------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.quadruped.train_utils import load_mlp_checkpoint, predict_mlp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, path, dpi=200):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def load_hardware(glob_pattern):
    """Load (X, y) from hardware pkl files."""
    from glob import glob
    paths = sorted(glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No pkls found: {glob_pattern}")
    X_list, y_list = [], []
    for p in paths:
        with open(p, "rb") as f:
            data = pickle.load(f)
        X_list.append([data["command"]["vx"],
                        data["command"]["vy"],
                        data["command"]["wz"]])
        y_list.append(data["error_summary"]["mean_abs_err_sum"])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ---------------------------------------------------------------------------
# Grid prediction
# ---------------------------------------------------------------------------

def make_slice_grid(
    axis_i: int,          # first axis index  (x-axis of heatmap)
    axis_j: int,          # second axis index (y-axis of heatmap)
    fixed_vals: dict,     # {axis_k: fixed_value}
    bounds: dict,         # {0: (min,max), 1: (min,max), 2: (min,max)}
    n_grid: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a 2D grid over (axis_i, axis_j) with the third axis fixed.

    Returns
    -------
    XI  : (n_grid, n_grid) meshgrid for axis_i
    XJ  : (n_grid, n_grid) meshgrid for axis_j
    X   : (n_grid*n_grid, 3) flattened input matrix for the MLP
    """
    vi = np.linspace(*bounds[axis_i], n_grid)
    vj = np.linspace(*bounds[axis_j], n_grid)
    XI, XJ = np.meshgrid(vi, vj)

    X = np.zeros((n_grid * n_grid, 3), dtype=np.float32)
    X[:, axis_i] = XI.ravel()
    X[:, axis_j] = XJ.ravel()
    for k, val in fixed_vals.items():
        X[:, k] = float(val)

    return XI, XJ, X


def mlp_predict_grid(model, ckpt, X_grid, device, n_mc=50):
    """
    Run MC-dropout on X_grid.
    Returns mean (N,) and std (N,).
    """
    mean, std = predict_mlp(
        model, ckpt, X_grid,
        device=device, mc_dropout=True, n_mc=n_mc,
    )
    return np.asarray(mean).ravel(), np.asarray(std).ravel()


# ---------------------------------------------------------------------------
# Plotting — one pair of panels (mean + std) for one projection
# ---------------------------------------------------------------------------

def plot_projection(
    ax_mean,
    XI, XJ,
    mean_grid, std_grid,
    X_hw, y_hw,
    hw_axis_i, hw_axis_j,
    xl, yl,
    fixed_label,
    vmin, vmax, std_max,
    threshold,
    n_grid,
):
    shape = (n_grid, n_grid)

    # Mean
    im_m = ax_mean.contourf(
        XI, XJ, mean_grid.reshape(shape),
        levels=40, cmap="RdYlGn_r", vmin=vmin, vmax=vmax,
    )
    # Failure contour
    ax_mean.contour(
        XI, XJ, mean_grid.reshape(shape),
        levels=[threshold], colors="black", linewidths=2.0,
    )
    # Hardware points
    ax_mean.scatter(
        X_hw[:, hw_axis_i], X_hw[:, hw_axis_j],
        c="red", marker="*", s=150, zorder=5,
        edgecolors="white", linewidths=0.5,
    )
    ax_mean.set_xlabel(xl, fontsize=20)
    ax_mean.set_ylabel(yl, fontsize=20)
    ax_mean.set_title(f"Mean — {xl} vs {yl}\n({fixed_label})", fontsize=20)
    ax_mean.tick_params(labelsize=20)
    plt.colorbar(im_m, ax=ax_mean, fraction=0.046, pad=0.04).ax.tick_params(labelsize=20)

    # Std
    # im_s = ax_std.contourf(
    #     XI, XJ, std_grid.reshape(shape),
    #     levels=40, cmap="YlOrRd", vmin=0, vmax=std_max,
    # )
    # ax_std.scatter(
    #     X_hw[:, hw_axis_i], X_hw[:, hw_axis_j],
    #     c="red", marker="*", s=150, zorder=5,
    #     edgecolors="white", linewidths=0.5,
    # )
    # ax_std.set_xlabel(xl, fontsize=10)
    # ax_std.set_ylabel(yl, fontsize=10)
    # ax_std.set_title(f"Std — {xl} vs {yl}\n({fixed_label})", fontsize=9)
    # ax_std.tick_params(labelsize=7)
    # plt.colorbar(im_s, ax=ax_std, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load MLP ─────────────────────────────────────────────────────────
    print(f"[INFO] Loading MLP from {args.ckpt_path} …")
    model, ckpt, device = load_mlp_checkpoint(args.ckpt_path)
    model.eval()
    print(f"[INFO] Device: {device}")

    # ── Load hardware data ────────────────────────────────────────────────
    print(f"[INFO] Loading hardware data from {args.hardware_glob} …")
    X_hw, y_hw = load_hardware(args.hardware_glob)
    print(f"[INFO] Hardware: N={len(X_hw)}  "
          f"y=[{y_hw.min():.3f}, {y_hw.max():.3f}]")

    # ── Velocity bounds and fixed-axis medians ────────────────────────────
    bounds = {
        0: (args.vx_min, args.vx_max),
        1: (args.vy_min, args.vy_max),
        2: (args.wz_min, args.wz_max),
    }
    medians = {
        0: float(np.median(X_hw[:, 0])),
        1: float(np.median(X_hw[:, 1])),
        2: float(np.median(X_hw[:, 2])),
    }
    axis_labels = {0: r"$v_x$", 1: r"$v_y$", 2: r"$w_z$"}

    # Three projections: (i, j, fixed_k)
    projections = [
        (0, 1, 2),   # vx-vy,  wz fixed
        (0, 2, 1),   # vx-wz,  vy fixed
        (1, 2, 0),   # vy-wz,  vx fixed
    ]

    # ── Predict on all grids to get shared colour scale ───────────────────
    print(f"[INFO] Predicting on {args.n_grid}×{args.n_grid} grids …")
    all_means, all_stds = [], []
    grid_cache = {}

    for axis_i, axis_j, axis_k in projections:
        XI, XJ, X_grid = make_slice_grid(
            axis_i, axis_j,
            fixed_vals={axis_k: medians[axis_k]},
            bounds=bounds, n_grid=args.n_grid,
        )
        mean_g, std_g = mlp_predict_grid(
            model, ckpt, X_grid, device, n_mc=args.n_mc
        )
        grid_cache[(axis_i, axis_j)] = (XI, XJ, mean_g, std_g)
        all_means.append(mean_g)
        all_stds.append(std_g)

    all_means = np.concatenate(all_means)
    vmin = float(np.quantile(all_means, 0.01))
    vmax = float(np.quantile(all_means, 0.99))
    std_max = float(np.quantile(np.concatenate(all_stds), 0.98))

    # ── Figure: 2 rows × 3 cols  (mean / std rows, projections cols) ─────
    fig, axes = plt.subplots(1, 3, figsize=(15,4))

    for col, (axis_i, axis_j, axis_k) in enumerate(projections):
        XI, XJ, mean_g, std_g = grid_cache[(axis_i, axis_j)]
        xl    = axis_labels[axis_i]
        yl    = axis_labels[axis_j]
        kl    = axis_labels[axis_k]
        fixed = f"{kl} = {medians[axis_k]:.2f}"

        plot_projection(
            ax_mean    = axes[col],
            # ax_std     = axes[1][col],
            XI=XI, XJ=XJ,
            mean_grid  = mean_g,
            std_grid   = std_g,
            X_hw       = X_hw,
            y_hw       = y_hw,
            hw_axis_i  = axis_i,
            hw_axis_j  = axis_j,
            xl=xl, yl=yl,
            fixed_label = fixed,
            vmin=vmin, vmax=vmax, std_max=std_max,
            threshold   = args.threshold,
            n_grid      = args.n_grid,
        )

    # fig.suptitle(
    #     f"{args.method_name} — MLP surrogate heatmaps\n"
    #     f"(black contour = failure boundary γ={args.threshold}  |  "
    #     f"★ = hardware observations)",
    #     fontsize=12, y=1.01,
    # )
    # plt.tight_layout()
    _save(fig, out_dir / f"mlp_heatmap_{args.method_name.lower()}.pdf")
    _save(fig, out_dir / f"mlp_heatmap_{args.method_name.lower()}.png")

    # ── Mean-only figure (for paper) ──────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

    for col, (axis_i, axis_j, axis_k) in enumerate(projections):
        XI, XJ, mean_g, std_g = grid_cache[(axis_i, axis_j)]
        xl    = axis_labels[axis_i]
        yl    = axis_labels[axis_j]
        kl    = axis_labels[axis_k]
        shape = (args.n_grid, args.n_grid)

        im = axes2[col].contourf(
            XI, XJ, mean_g.reshape(shape),
            levels=40, cmap="RdYlGn_r", vmin=vmin, vmax=vmax,
        )
        axes2[col].contour(
            XI, XJ, mean_g.reshape(shape),
            levels=[args.threshold], colors="black", linewidths=2.0,
        )
        axes2[col].scatter(
            X_hw[:, axis_i], X_hw[:, axis_j],
            c="red", marker="*", s=160, zorder=5,
            edgecolors="white", linewidths=0.5,
        )
        axes2[col].set_xlabel(xl, fontsize=12)
        axes2[col].set_ylabel(yl, fontsize=12)
        axes2[col].set_title(
            f"{xl} vs {yl}  ({kl} = {medians[axis_k]:.2f})", fontsize=11,
        )
        axes2[col].tick_params(labelsize=9)
        axes2[col].grid(alpha=0.2)

    # Shared colourbar
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm   = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=norm)
    sm.set_array([])
    cbar = fig2.colorbar(sm, ax=axes2.tolist(), fraction=0.018, pad=0.02)
    cbar.set_label("Predicted tracking error y", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    fig2.suptitle(
        f"{args.method_name} — MLP surrogate mean prediction\n"
        f"(black = failure boundary γ={args.threshold}  |  ★ = hardware data)",
        fontsize=12, y=1.02,
    )
    # plt.tight_layout()
    _save(fig2, out_dir / f"mlp_mean_{args.method_name.lower()}.pdf")
    _save(fig2, out_dir / f"mlp_mean_{args.method_name.lower()}.png")

    print(f"[DONE]  outputs → {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="MLP surrogate heatmaps for Go2 velocity tracking."
    )
    p.add_argument("--ckpt_path",     type=str, required=True,
                   help="Path to MLP checkpoint .pt file")
    p.add_argument("--hardware_glob", type=str, required=True,
                   help="Glob for hardware pkl files, e.g. 'results/scout/*.pkl'")
    p.add_argument("--output_dir",    type=str,
                   default="results/quadruped/heatmaps")
    p.add_argument("--method_name",   type=str, default="SCOUT")
    p.add_argument("--threshold",     type=float, default=0.7,
                   help="Failure boundary value for black contour")
    p.add_argument("--n_grid",        type=int,   default=60,
                   help="Grid resolution per axis (60 = fast, 120 = publication)")
    p.add_argument("--n_mc",          type=int,   default=50,
                   help="MC-dropout samples for posterior mean/std")
    # Velocity bounds
    p.add_argument("--vx_min", type=float, default=-0.4)
    p.add_argument("--vx_max", type=float, default=1.0)
    p.add_argument("--vy_min", type=float, default=-0.8)
    p.add_argument("--vy_max", type=float, default=0.8)
    p.add_argument("--wz_min", type=float, default=-0.8)
    p.add_argument("--wz_max", type=float, default=0.8)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
