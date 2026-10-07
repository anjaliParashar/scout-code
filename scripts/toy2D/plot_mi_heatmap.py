#!/usr/bin/env python3
"""
scripts/toy2D/plot_mi_heatmap.py
----------------------------------
MI heatmap for the toy2D domain given a set of already-evaluated
random datapoints as the support D_t.

The heatmap shows I(Z; R_x | D_t, x) evaluated on the full [-3,3]² grid,
with the support points overlaid as red circles — exactly matching the
visual convention of 1_mi_test.py (tsne_mi.png) but in the native 2D
input space instead of t-SNE.

Four panels are shown:
  A : MI I(Z; R_x | D_t, x)       — main acquisition criterion
  B : p_r1 = P(R_x=1)             — marginal novelty probability
  C : d_min to nearest cluster     — pure distance metric
  D : failure mask (f_real ≤ γ)   — ground truth for reference

Support points (D_t) are overlaid as red circles on every panel.
Diamond boundaries and Gaussian input density are also overlaid.

Example
-------
# 20 random support points, default MI hyperparameters
python scripts/toy2D/plot_mi_heatmap.py \
    --n-sample   20 \
    --output-dir outputs/toy2D/mi \
    --seed       42

# Vary support size to see how MI changes
python scripts/toy2D/plot_mi_heatmap.py --n-sample 5  --output-dir outputs/toy2D/mi_n5
python scripts/toy2D/plot_mi_heatmap.py --n-sample 50 --output-dir outputs/toy2D/mi_n50
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import multivariate_normal

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
import matplotlib
fontsize = 40
parameters = {
    'font.family': 'Times New Roman',
    'axes.labelsize': fontsize,
    'axes.titlesize': fontsize,
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize,
    'legend.fontsize': fontsize
}
plt.rcParams.update(parameters)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# Okabe–Ito (colorblind-friendly)
OI = {
    # Core (high contrast, safest)
    "blue":          "#0072B2",
    "orange":        "#E69F00",
    "bluish_green":  "#009E73",
    "vermillion":    "#D55E00",
    "purple":        "#CC79A7",
    "skyblue":       "#56B4E9",

    # Neutrals
    "black":         "#000000",
    "grey":          "#999999",

    # Optional extensions (still colorblind-friendly)
    "dark_blue":     "#003B5C",
    "olive":         "#7A7A00",
    "brown":         "#8C510A",
}
MORE_CVD = {
    "sky_blue":   "#56B4E9",  # lighter blue, very distinct from #0072B2
    "bluish_green":"#009E73", # teal/green (safe vs red–green confusion)
    "vermillion": "#D55E00",  # red-orange, strong contrast
    "purple":     "#CC79A7",  # magenta-purple
}
PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#CC79A7",  # purple
    "#000000",  # black
    "#999999",  # gray (use sparingly)
]
from utils.toy2D.domain import (
    real_mean_fn, sample_designs,
    make_grid,
    C1, C2, R1, R2, GAMMA,
    MU_X, SIGMA_X,
)
from utils.toy2D.mi_toy import fit_support_and_compute_mi


# ---------------------------------------------------------------------------
# Drawing helpers (identical to plot_domain_heatmaps.py)
# ---------------------------------------------------------------------------

def _diamond_verts(center, radius):
    cx, cy = center
    verts = np.array([
        [cx + radius, cy],
        [cx,          cy + radius],
        [cx - radius, cy],
        [cx,          cy - radius],
        [cx + radius, cy],
    ])
    return verts[:, 0], verts[:, 1]


def _add_diamonds(ax, lw=1.8, color="white", ls="--", zorder=6):
    for center, radius in [(C1, R1), (C2, R2)]:
        bx, by = _diamond_verts(center, radius)
        ax.plot(bx, by, color=color, lw=lw, ls=ls, zorder=zorder)
        ax.plot(center[0], center[1], "+",
                color=color, ms=7, mew=1.8, zorder=zorder + 1)


def _add_input_density(ax, alpha=0.20, zorder=1):
    g = np.linspace(-3, 3, 200)
    XX, YY = np.meshgrid(g, g)
    pos  = np.stack([XX, YY], axis=-1)
    dens = multivariate_normal(mean=MU_X, cov=SIGMA_X).pdf(pos)
    ax.contour(XX, YY, dens, levels=5,
               colors="grey", alpha=alpha, linewidths=0.7, zorder=zorder)


def _add_failure_boundary(ax, XX, YY, real_vals, lw=1.5, zorder=5):
    ax.contour(XX, YY, real_vals, levels=[GAMMA],
               colors="black", linewidths=lw, zorder=zorder)


def _overlay_support(ax, X_support, zorder=8):
    """Red circles — matching 1_mi_test.py / plot_utils._overlay_initial."""
    ax.scatter(
        X_support[:, 0], X_support[:, 1],
        s=180, facecolors="red", edgecolors="black",
        linewidths=1.0, zorder=zorder, label=f"Data observed (n={len(X_support)})",
    )


def _style(ax, title, ylabel="$x_2$"):
    # ax.set_title(title, fontsize=20, pad=4)
    ax.set_xlabel("$x_1$", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=16)


def _cbar(fig, ax, im, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, fontsize=20)
    cb.ax.tick_params(labelsize=15)
    return cb


def _save(fig, path, dpi=200):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Sample support D_t ────────────────────────────────────────────────
    X_support = sample_designs(args.n_sample, seed=args.seed)
    print(f"[INFO] Support D_t: {len(X_support)} random designs  (seed={args.seed})")

    # ── Build query grid ──────────────────────────────────────────────────
    XX, YY, X_grid = make_grid(args.n_grid)
    shape = XX.shape

    # ── Ground truth for reference ────────────────────────────────────────
    real_vals = real_mean_fn(X_grid).reshape(shape)
    fail_vals = (real_vals <= GAMMA).astype(float)

    # ── Compute MI on the full grid ───────────────────────────────────────
    print(f"[INFO] Computing MI on {args.n_grid}² = {args.n_grid**2:,} grid points …")
    mi_vals, p_r1_vals, dmin_vals, cluster_info, pZ, km = fit_support_and_compute_mi(
        X_support      = X_support.astype(np.float64),
        X_query        = X_grid.astype(np.float64),
        n_clusters     = args.n_clusters,
        radius_quantile = args.radius_quantile,
        pi_new         = args.pi_new,
        eps_exist      = args.eps_exist,
        tau_scale      = args.tau_scale,
        novelty_radius = args.novelty_radius,
        gamma          = args.gamma,
        random_state   = args.seed,
    )
    mi_map   = mi_vals.reshape(shape)
    p_r1_map = p_r1_vals.reshape(shape)
    dmin_map = dmin_vals.reshape(shape)

    print(f"[INFO] MI   min={mi_vals.min():.4f}  "
          f"median={np.median(mi_vals):.4f}  max={mi_vals.max():.4f}")
    print(f"[INFO] p_r1 min={p_r1_vals.min():.4f}  "
          f"median={np.median(p_r1_vals):.4f}  max={p_r1_vals.max():.4f}")

    # =========================================================================
    # Figure 1: 2×2 panel
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # A — MI
    im = axes[0, 0].contourf(XX, YY, mi_map, levels=40)
    _add_failure_boundary(axes[0, 0], XX, YY, real_vals)
    _add_diamonds(axes[0, 0], color="white")
    _add_input_density(axes[0, 0])
    _overlay_support(axes[0, 0], X_support)
    axes[0, 0].legend(fontsize=20, loc="lower right", framealpha=0.8)
    _cbar(fig, axes[0, 0], im, r"MI")
    _style(axes[0, 0],
           rf"MI  $I(Z;\,R_x\mid D_t,x)$  —  n={args.n_sample} support pts"
           "\n(black = real failure boundary  |  red circles = $D_t$)")

    # B — p_r1
    im = axes[0, 1].contourf(XX, YY, p_r1_map, levels=40, cmap="plasma")
    _add_failure_boundary(axes[0, 1], XX, YY, real_vals)
    _add_diamonds(axes[0, 1], color="white")
    _add_input_density(axes[0, 1])
    _overlay_support(axes[0, 1], X_support)
    _cbar(fig, axes[0, 1], im, r"$P(R_x=1)$ — marginal novelty")
    _style(axes[0, 1], r"$P(R_x=1)$ — marginal novelty probability")

    # C — d_min
    im = axes[1, 0].contourf(XX, YY, dmin_map, levels=40, cmap="YlOrRd")
    _add_failure_boundary(axes[1, 0], XX, YY, real_vals)
    _add_diamonds(axes[1, 0], color="white")
    _add_input_density(axes[1, 0])
    _overlay_support(axes[1, 0], X_support)
    # Overlay cluster centres
    centers = np.array([c["center"] for c in cluster_info])
    axes[1, 0].scatter(centers[:, 0], centers[:, 1],
                       marker="x", s=90, c="black", lw=1.5,
                       zorder=9, label="cluster centres")
    axes[1, 0].legend(fontsize=7, loc="lower right", framealpha=0.8)
    _cbar(fig, axes[1, 0], im, r"$d_\min$ to nearest cluster ball")
    _style(axes[1, 0],
           r"$d_\min(x, \mathcal{D}_t)$ — distance to nearest cluster"
           "\n(× = KMeans cluster centres)")

    # D — failure reference
    im = axes[1, 1].contourf(XX, YY, fail_vals, levels=40,
                              cmap="Reds", vmin=0, vmax=1)
    axes[1, 1].contour(XX, YY, real_vals, levels=[GAMMA],
                       colors="darkred", linewidths=2.0, zorder=5)
    _add_diamonds(axes[1, 1], color="white")
    _add_input_density(axes[1, 1])
    _overlay_support(axes[1, 1], X_support)
    _cbar(fig, axes[1, 1], im, r"failure  $[f_\mathrm{real}(x) \leq \gamma=0]$")
    _style(axes[1, 1],
           r"Failure reference  $f_\mathrm{real}(x) \leq \gamma = 0$"
           "\n(red = failure, white = safe)")

    fig.suptitle(
        rf"toy2D MI heatmap  —  $|D_t|$ = {args.n_sample}  "
        rf"(seed={args.seed},  K={min(args.n_clusters, args.n_sample)} clusters)",
        fontsize=12, y=1.01,
    )
    # plt.tight_layout()
    _save(fig, out / f"mi_heatmap_n{args.n_sample}.pdf")
    _save(fig, out / f"mi_heatmap_n{args.n_sample}.png")

    # =========================================================================
    # Figure 2: MI standalone (publication-quality single panel)
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    im2 = ax2.contourf(XX, YY, mi_map, levels=40, cmap="RdYlGn")
    _add_failure_boundary(ax2, XX, YY, real_vals, lw=2.0)
    _add_diamonds(ax2, color="white", lw=2.0)
    _add_input_density(ax2)
    _overlay_support(ax2, X_support)
    ax2.legend(fontsize=9, loc="lower right", framealpha=0.85)
    cb2 = fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label(r"MI  $I(Z;\,R_x \mid D_t,\,x)$", fontsize=10)
    cb2.ax.tick_params(labelsize=8)
    ax2.set_title(
        rf"MI  $I(Z;\,R_x \mid D_t,\,x)$  —  $|D_t|$ = {args.n_sample}"
        "\n(black contour = failure boundary  |"
        "  red circles = evaluated support $D_t$)",
        fontsize=10,
    )
    ax2.set_xlabel("x₁", fontsize=10)
    ax2.set_ylabel("x₂", fontsize=10)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_aspect("equal")
    ax2.tick_params(labelsize=8)
    # plt.tight_layout()
    _save(fig2, out / f"mi_standalone_n{args.n_sample}.pdf")
    _save(fig2, out / f"mi_standalone_n{args.n_sample}.png")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n[SUMMARY]")
    print(f"  K = {min(args.n_clusters, args.n_sample)} clusters")
    print(f"  π_new = {args.pi_new}")
    print(f"  MI high (top 5% of grid): "
          f"{np.sum(mi_vals >= np.quantile(mi_vals, 0.95)):,} grid points")
    print(f"  Overlap of high-MI with failure region: "
          f"{np.sum((mi_vals >= np.quantile(mi_vals, 0.95)) & (fail_vals.ravel() > 0.5)):,} pts")
    print(f"\n[DONE]  outputs → {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "MI heatmap on the toy2D domain for a given random support D_t."
        )
    )
    p.add_argument("--n-sample",        type=int,   default=20,
                   help="Number of already-evaluated random support points")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--n-grid",          type=int,   default=120,
                   help="Grid resolution (n×n over [-3,3]²). "
                        "120 is fast; use 300 for publication quality.")
    p.add_argument("--output-dir",      type=str,   default="outputs/toy2D/mi")

    # MI hyperparameters — same defaults as mi_toy.py / 1_mi_test.py
    p.add_argument("--n-clusters",      type=int,   default=5)
    p.add_argument("--radius-quantile", type=float, default=0.90)
    p.add_argument("--pi-new",          type=float, default=0.18)
    p.add_argument("--eps-exist",       type=float, default=0.04)
    p.add_argument("--tau-scale",       type=float, default=1.3)
    p.add_argument("--novelty-radius",  type=float, default=0.75)
    p.add_argument("--gamma",           type=float, default=0.25)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
