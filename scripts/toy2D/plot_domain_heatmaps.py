#!/usr/bin/env python3
"""
scripts/toy2D/plot_domain_heatmaps.py
---------------------------------------
Visualise the toy2D domain as heatmaps:

  Figure 1 — 2×2 panel:
    A : real_mean_fn(x)       — ground-truth real system
    B : sim_mean_fn(x)        — simulation surrogate
    C : gap  Δ = real − sim   — diverging coolwarm
    D : failure mask           — f_real(x) ≤ γ

  Figure 2 — side-by-side sim vs real (shared colour scale)
    Consistent style with 0_visualize_sim_real.py

  Figure 3 — gap standalone

All panels overlay diamond boundaries and Gaussian input-density contours.

Example
-------
python scripts/toy2D/plot_domain_heatmaps.py \
    --output-dir outputs/toy2D/domain \
    --n-grid     300
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

from utils.toy2D.domain import (
    real_mean_fn, sim_mean_fn,
    make_grid,
    C1, C2, R1, R2, GAMMA,
    MU_X, SIGMA_X,
)

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
# ---------------------------------------------------------------------------
# Drawing helpers
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


def _add_diamonds(ax, lw=2.0, color="white", ls="--", zorder=6):
    for center, radius in [(C1, R1), (C2, R2)]:
        bx, by = _diamond_verts(center, radius)
        ax.plot(bx, by, color=color, lw=lw, ls=ls, zorder=zorder)
        ax.plot(center[0], center[1], "+",
                color=color, ms=8, mew=2.0, zorder=zorder + 1)


def _add_input_density(ax, alpha=0.22, zorder=1):
    g = np.linspace(-3, 3, 200)
    XX, YY = np.meshgrid(g, g)
    pos  = np.stack([XX, YY], axis=-1)
    dens = multivariate_normal(mean=MU_X, cov=SIGMA_X).pdf(pos)
    ax.contour(XX, YY, dens, levels=5,
               colors="grey", alpha=alpha, linewidths=0.7, zorder=zorder)


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

    XX, YY, X_grid = make_grid(args.n_grid)
    shape = XX.shape

    real_vals = real_mean_fn(X_grid).reshape(shape)
    sim_vals  = sim_mean_fn(X_grid).reshape(shape)
    gap_vals  = real_vals - sim_vals
    fail_vals = (real_vals <= GAMMA).astype(float)

    levels = 40
    vmin_s = min(real_vals.min(), sim_vals.min())
    vmax_s = max(real_vals.max(), sim_vals.max())
    shared_lvls = np.linspace(vmin_s, vmax_s, levels)
    vgap = float(np.abs(gap_vals).max())

    # =========================================================================
    # Figure 1: 2×2 panel
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # A — real
    im = axes[0, 0].contourf(XX, YY, real_vals,
                              levels=shared_lvls)
    axes[0, 0].contour(XX, YY, real_vals, levels=[GAMMA],
                       colors="black", linewidths=2.5, zorder=5)
    _add_diamonds(axes[0, 0], color="white")
    _add_input_density(axes[0, 0])
    _cbar(fig, axes[0, 0], im, "$y_r$")
    _style(axes[0, 0],
           "Real system  f_real(x)\n(black = failure boundary γ=0)")

    # B — sim
    im = axes[0, 1].contourf(XX, YY, sim_vals,
                              levels=shared_lvls)
    axes[0, 1].contour(XX, YY, real_vals, levels=[GAMMA],
                       colors="black", linewidths=2.5, linestyles="--", zorder=5)
    _add_diamonds(axes[0, 1], color="white")
    _add_input_density(axes[0, 1])
    _cbar(fig, axes[0, 1], im, "$y_s$")
    _style(axes[0, 1],
           "Sim surrogate  f_sim(x)\n(dashed = real failure boundary)")

    # C — gap
    im = axes[1, 0].contourf(XX, YY, gap_vals, levels=levels,
                              cmap="coolwarm", vmin=-vgap, vmax=vgap)
    axes[1, 0].contour(XX, YY, real_vals, levels=[GAMMA],
                       colors="black", linewidths=1.8, zorder=5)
    _add_diamonds(axes[1, 0], color="black", ls="--")
    _add_input_density(axes[1, 0])
    _cbar(fig, axes[1, 0], im, "Δ = f_real − f_sim")
    _style(axes[1, 0],
           "Sim/real gap  Δ = f_real − f_sim\n"
           "(blue = sim overestimates, red = underestimates)")

    # D — failure mask
    im = axes[1, 1].contourf(XX, YY, fail_vals, levels=40,
                              cmap="Reds", vmin=0, vmax=1)
    axes[1, 1].contour(XX, YY, real_vals, levels=[GAMMA],
                       colors="darkred", linewidths=2.0, zorder=5)
    _add_diamonds(axes[1, 1], color="white")
    _add_input_density(axes[1, 1])
    _cbar(fig, axes[1, 1], im, "failure  [f_real ≤ γ=0]")
    _style(axes[1, 1],
           "Failure region  f_real(x) ≤ γ = 0\n(red = fail, white = safe)")

    fig.suptitle("toy2D domain — ground-truth heatmaps",
                 fontsize=13, y=1.01)
    # plt.tight_layout()
    _save(fig, out / "domain_heatmaps_2x2.pdf")
    _save(fig, out / "domain_heatmaps_2x2.png")

    # =========================================================================
    # Figure 2: side-by-side sim vs real — shared colour scale
    # Matches style of 0_visualize_sim_real.py
    # =========================================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))

    for ax, vals, title in [
        (axes2[0], sim_vals,  "Sim surrogate  f_sim(x)"),
        (axes2[1], real_vals, "Real system  f_real(x)"),
    ]:
        ax.contourf(XX, YY, vals, levels=shared_lvls, cmap="RdYlGn")
        ax.contour(XX, YY, real_vals, levels=[GAMMA],
                   colors="black", linewidths=2.2, zorder=5)
        _add_diamonds(ax, color="white", lw=1.8)
        _add_input_density(ax)
        _style(ax, title)

    axes2[1].set_ylabel("")   # remove duplicate ylabel

    # Shared colourbar
    norm = mcolors.Normalize(vmin=vmin_s, vmax=vmax_s)
    sm   = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
    sm.set_array([])
    cbar = fig2.colorbar(sm, ax=axes2.tolist(), fraction=0.025, pad=0.02)
    cbar.set_label(
        "f(x) — shared scale  [red = low / failure,  green = high / safe]",
        fontsize=10,
    )
    cbar.ax.tick_params(labelsize=8)

    fig2.suptitle(
        "toy2D: sim surrogate vs real system  (shared colour scale)\n"
        "black contour = real failure boundary  |  "
        "white dashed = diamond  |  grey = input density",
        fontsize=11, y=1.01,
    )
    # plt.tight_layout()
    _save(fig2, out / "sim_vs_real.pdf")
    _save(fig2, out / "sim_vs_real.png")

    # =========================================================================
    # Figure 3: gap standalone
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(8, 7))
    im3 = ax3.contourf(XX, YY, gap_vals, levels=levels,
                        cmap="coolwarm", vmin=-vgap, vmax=vgap)
    ax3.contour(XX, YY, real_vals, levels=[GAMMA],
                colors="black", linewidths=2.0, zorder=5)
    _add_diamonds(ax3, color="black", ls="--")
    _add_input_density(ax3)
    _cbar(fig3, ax3, im3,
          "Δ = f_real − f_sim\n(blue = sim overestimates, red = underestimates)")
    _style(ax3, "Sim/real gap  Δ = f_real − f_sim")
    # plt.tight_layout()
    _save(fig3, out / "gap_heatmap.pdf")
    _save(fig3, out / "gap_heatmap.png")

    # ── Summary ───────────────────────────────────────────────────────────
    fail_frac = float(fail_vals.mean())
    print(f"\n[SUMMARY]")
    print(f"  Grid: {args.n_grid}×{args.n_grid} over [-3,3]²")
    print(f"  real_mean_fn:  [{real_vals.min():.4f}, {real_vals.max():.4f}]")
    print(f"  sim_mean_fn:   [{sim_vals.min():.4f}, {sim_vals.max():.4f}]")
    print(f"  gap:           [{gap_vals.min():.4f}, {gap_vals.max():.4f}]")
    print(f"  Failure fraction (f_real ≤ {GAMMA}): "
          f"{fail_frac:.4f}  ({100*fail_frac:.2f}% of grid)")
    print(f"\n[DONE]  outputs → {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Heatmaps of toy2D sim and real domain functions."
    )
    p.add_argument("--output-dir", type=str, default="outputs/toy2D/domain")
    p.add_argument("--n-grid",     type=int, default=300,
                   help="Grid resolution (n×n points over [-3,3]²)")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
