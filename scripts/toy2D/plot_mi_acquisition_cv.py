#!/usr/bin/env python3
"""
scripts/toy2D/plot_mi_acquisition.py
--------------------------------------
Given a random dataset D_t (same as plot_mi_heatmap.py), this script:

  1. Trains a GP on D_t using real (noisy) observations.
  2. Computes MI on the full [-3,3]² grid given D_t as support.
  3. Selects the top-k grid points by MI as the candidate acquisitions.
  4. Produces two figures:

     Figure 1 — GP posterior (real model)
       Left  : GP posterior mean + uncertainty contours
                D_t training points overlaid as red circles
                MI-selected candidates overlaid as gold stars
       Right : GP posterior std (epistemic uncertainty)
                same overlays

     Figure 2 — Cheap proxy evaluation
       Left  : sim_mean_fn(x) heatmap (the full proxy field)
                MI-selected candidates overlaid as gold stars
                sim values at candidates shown as text annotations
       Right : real_mean_fn(x) heatmap for comparison
                same candidate overlay

All figures share the diamond boundary, failure boundary, and input
density overlays from plot_domain_heatmaps.py / plot_mi_heatmap.py.

The script uses the exact same D_t as plot_mi_heatmap.py when called
with the same --n-sample and --seed — so the two sets of figures are
directly comparable.

Example
-------
python scripts/toy2D/plot_mi_acquisition_cv.py \
    --n-sample   10 \
    --n-select   1000  \
    --seed       42 \
    --output-dir outputs/toy2D/mi_acquisition_cv
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
    sample_designs, sample_real,
    make_grid,
    C1, C2, R1, R2, GAMMA,
    MU_X, SIGMA_X,
)
from utils.toy2D.gp_utils   import train_gp, gp_predict
from utils.toy2D.mi_toy     import fit_support_and_compute_mi, compute_cv_batch


# ---------------------------------------------------------------------------
# Shared drawing helpers (identical to plot_mi_heatmap.py)
# ---------------------------------------------------------------------------

def _diamond_verts(center, radius):
    cx, cy = center
    v = np.array([
        [cx + radius, cy], [cx, cy + radius],
        [cx - radius, cy], [cx, cy - radius],
        [cx + radius, cy],
    ])
    return v[:, 0], v[:, 1]


def _add_diamonds(ax, color="white", lw=1.8, ls="--", zorder=6):
    for c, r in [(C1, R1), (C2, R2)]:
        bx, by = _diamond_verts(c, r)
        ax.plot(bx, by, color=color, lw=lw, ls=ls, zorder=zorder)
        ax.plot(c[0], c[1], "+", color=color, ms=7, mew=1.8, zorder=zorder+1)


def _add_failure_boundary(ax, XX, YY, real_vals, color="black", lw=2.0):
    ax.contour(XX, YY, real_vals, levels=[GAMMA],
               colors=color, linewidths=lw, zorder=5)


def _add_input_density(ax, alpha=0.18):
    g = np.linspace(-3, 3, 200)
    XX, YY = np.meshgrid(g, g)
    pos  = np.stack([XX, YY], axis=-1)
    dens = multivariate_normal(mean=MU_X, cov=SIGMA_X).pdf(pos)
    ax.contour(XX, YY, dens, levels=5,
               colors="grey", alpha=alpha, linewidths=0.7, zorder=1)


def _style(ax, title, ylabel="x₂"):
    # ax.set_title(title, fontsize=10, pad=4)
    ax.set_xlabel("x₁", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    ax.set_aspect("equal"); ax.tick_params(labelsize=6)


def _cbar(fig, ax, im, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, fontsize=7); cb.ax.tick_params(labelsize=6)


def _save(fig, path, dpi=200):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _overlay_support(ax, X_support, label=True):
    """Red circles — D_t training points."""
    ax.scatter(X_support[:, 0], X_support[:, 1],
               s=160, facecolors="red", edgecolors="black",
               linewidths=0.9, zorder=8)#,
            #    label=f"$D_t$ (n={len(X_support)})" if label else None)


def _overlay_selected(ax, X_sel, label=True):
    """Gold stars — MI-selected candidates."""
    # ax.scatter(X_sel[:, 0], X_sel[:, 1],
    #            s=280, marker="*", facecolors="gold",
    #            edgecolors="black", linewidths=0.8, zorder=9,
    #            label=f"MI-selected (n={len(X_sel)})" if label else None)
    ax.scatter(X_sel[:, 0], X_sel[:, 1],
               s=10,alpha=0.2)
            #    label=f"MI-selected (n={len(X_sel)})" if label else None)


def _overlay_cv_selected(ax, X_cv, label=True):
    """Cyan pentagons — CV-selected candidates."""
    ax.scatter(X_cv[:, 0], X_cv[:, 1],
               s=280, marker="p", facecolors="cyan",
               edgecolors="black", linewidths=0.8, zorder=9,
               label=f"Target Eval" if label else None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Sample D_t (same as plot_mi_heatmap.py) ───────────────────────────
    X_support = sample_designs(args.n_sample, seed=args.seed)
    y_real    = sample_real(X_support, seed=args.seed)
    print(f"[INFO] D_t: {len(X_support)} support pts  "
          f"y_real range [{y_real.min():.3f}, {y_real.max():.3f}]")

    # ── Build grid ────────────────────────────────────────────────────────
    XX, YY, X_grid = make_grid(args.n_grid)
    shape = XX.shape

    real_vals = real_mean_fn(X_grid).reshape(shape)
    sim_vals  = sim_mean_fn(X_grid).reshape(shape)

    # ── Train GP on D_t ──────────────────────────────────────────────────
    print("[INFO] Training GP on D_t …")
    gp = train_gp(X_support.astype(np.float64),
                  y_real.astype(np.float64))
    gp_mean_flat, gp_var_flat = gp_predict(gp, X_grid.astype(np.float64))
    gp_mean = gp_mean_flat.reshape(shape)
    gp_std  = np.sqrt(np.maximum(gp_var_flat, 1e-12)).reshape(shape)
    print(f"[INFO] GP posterior mean range "
          f"[{gp_mean.min():.3f}, {gp_mean.max():.3f}]")

    # ── Compute MI on grid ────────────────────────────────────────────────
    print(f"[INFO] Computing MI on {args.n_grid}² grid points …")
    mi_vals, _, _, _, _, _ = fit_support_and_compute_mi(
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

    # ── Select top-k by MI ────────────────────────────────────────────────
    n_sel    = min(args.n_select, len(mi_vals))
    top_k    = np.argsort(mi_vals)[-n_sel:][::-1]   # descending
    X_sel    = X_grid[top_k]                         # (n_sel, 2)
    mi_sel   = mi_vals[top_k]
    sim_sel  = sim_mean_fn(X_sel.astype(np.float32))
    real_sel = real_mean_fn(X_sel.astype(np.float32))

    print(f"[INFO] Top-{n_sel} MI candidates:")
    for i in range(n_sel):
        print(f"  x=({X_sel[i,0]:.2f}, {X_sel[i,1]:.2f})  "
              f"MI={mi_sel[i]:.4f}  "
              f"sim={sim_sel[i]:.3f}  real={real_sel[i]:.3f}")

    # ── Compute CV on MI shortlist, select top-k CV ───────────────────────
    # Use a larger MI shortlist (3× n_sel) so CV has meaningful candidates
    n_shortlist  = min(args.n_select , len(mi_vals))
    short_local  = np.argsort(mi_vals)[-n_shortlist:]  # top MI shortlist
    X_short      = X_grid[short_local]                 # (S, 2)
    print(f"[INFO] Computing CV on MI shortlist ({n_shortlist} pts) …")
    cv_vals_short = compute_cv_batch(
        gp          = gp,
        X_query     = X_short.astype(np.float64),
        R_local     = args.cv_radius,
        n_pair      = args.cv_n_pair,
        k_unpaired  = args.cv_k_unpaired,
        n_f_draws   = args.cv_n_draws,
    )   # (S,) — NaN where CV failed
    # Replace NaN with median so argsort is stable
    cv_vals_short = np.where(
        np.isfinite(cv_vals_short),
        cv_vals_short,
        float(np.nanmedian(cv_vals_short)),
    )
    # Select top-k by LOWEST CV mean (lowest = most likely to fail)
    
    top_cv_local = np.argsort(cv_vals_short)[-5:]  # ascending
    X_cv_sel     = X_short[top_cv_local]               # (n_sel, 2)
    cv_sel       = cv_vals_short[top_cv_local]
    sim_cv_sel   = sim_mean_fn(X_cv_sel.astype(np.float32))
    real_cv_sel  = real_mean_fn(X_cv_sel.astype(np.float32))
    print(f"[INFO] Top-{5} CV candidates (lowest mu_CV):")
    for i in range(5):
        print(f"  x=({X_cv_sel[i,0]:.2f}, {X_cv_sel[i,1]:.2f})  "
              f"CV={cv_sel[i]:.4f}  "
              f"sim={sim_cv_sel[i]:.3f}  real={real_cv_sel[i]:.3f}")

    # =========================================================================
    # Figure 1: GP posterior  (left=mean, right=std)
    # =========================================================================
    fig1, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Left: GP mean ─────────────────────────────────────────────────────
    vmin_m = min(gp_mean.min(), real_vals.min())
    vmax_m = max(gp_mean.max(), real_vals.max())
    lvls   = np.linspace(vmin_m, vmax_m, 40)

    im = axes[0].contourf(XX, YY, gp_mean, levels=lvls, cmap="RdYlGn")
    # Uncertainty contours (std iso-lines)
    axes[0].contour(XX, YY, gp_std,
                    levels=5, colors="steelblue",
                    linewidths=0.8, alpha=0.7, zorder=4,
                    linestyles="--")
    _add_failure_boundary(axes[0], XX, YY, real_vals)
    _add_diamonds(axes[0], color="white")
    _add_input_density(axes[0])
    _overlay_support(axes[0], X_support)
    _overlay_selected(axes[0], X_sel)
    axes[0].legend(fontsize=7, loc="lower right", framealpha=0.85)
    _cbar(fig1, axes[0], im, "GP posterior mean")
    _style(axes[0],
           f"GP posterior mean  (trained on $|D_t|$={args.n_sample})\n"
           "dashed blue = std contours  |  gold ★ = MI-selected")

    # ── Right: GP std ─────────────────────────────────────────────────────
    im2 = axes[1].contourf(XX, YY, gp_std, levels=40, cmap="YlOrRd")
    _add_failure_boundary(axes[1], XX, YY, real_vals)
    _add_diamonds(axes[1], color="black", ls="--")
    _add_input_density(axes[1])
    _overlay_support(axes[1], X_support, label=False)
    _overlay_selected(axes[1], X_sel, label=False)
    _cbar(fig1, axes[1], im2, "GP posterior std  (epistemic uncertainty)")
    _style(axes[1], "GP posterior std\n(high = uncertain → high MI region)", ylabel="")
    axes[1].set_ylabel("")

    fig1.suptitle(
        rf"GP model trained on $D_t$ (n={args.n_sample})  —  "
        rf"gold ★ = top-{n_sel} MI-selected candidates",
        fontsize=12, y=1.01,
    )
    # plt.tight_layout()
    _save(fig1, out / f"gp_posterior_n{args.n_sample}_k{n_sel}.pdf")
    _save(fig1, out / f"gp_posterior_n{args.n_sample}_k{n_sel}.png")

    # =========================================================================
    # Figure 2: Sim heatmap (MI candidates) | Real heatmap (MI + CV candidates)
    # =========================================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))

    vmin_s = min(real_vals.min(), sim_vals.min())
    vmax_s = max(real_vals.max(), sim_vals.max())
    shared = np.linspace(vmin_s, vmax_s, 40)

    # ── Left: sim heatmap + MI candidates ────────────────────────────────
    axes2[0].contourf(XX, YY, sim_vals, levels=shared)
    _add_failure_boundary(axes2[0], XX, YY, real_vals)
    _add_diamonds(axes2[0], color="white")
    _add_input_density(axes2[0])
    _overlay_support(axes2[0], X_support)
    _overlay_selected(axes2[0], X_sel)              # MI candidates only
    # for i in range(n_sel):                          # sim value annotations
    #     axes2[0].annotate(
    #         f"{sim_sel[i]:.2f}",
    #         xy=(X_sel[i, 0], X_sel[i, 1]),
    #         xytext=(X_sel[i, 0] + 0.15, X_sel[i, 1] + 0.15),
    #         fontsize=6.5, fontweight="bold",
    #         bbox=dict(fc="white", ec="none", alpha=0.6, pad=1.0), zorder=10)
    axes2[0].legend(fontsize=7, loc="lower right", framealpha=0.85)
    _cbar(fig2, axes2[0],
          plt.cm.ScalarMappable(
              norm=mcolors.Normalize(vmin=vmin_s, vmax=vmax_s)),
          "f_sim(x)")
    _style(axes2[0],
           f"Sim proxy  f_sim(x)\n"
           f"gold ★ = top-{n_sel} MI  |  labels = sim value")

    # ── Right: real heatmap + MI candidates + CV candidates ──────────────
    axes2[1].contourf(XX, YY, real_vals, levels=shared)
    _add_failure_boundary(axes2[1], XX, YY, real_vals)
    _add_diamonds(axes2[1], color="white")
    _add_input_density(axes2[1])
    _overlay_support(axes2[1], X_support)
    # _overlay_selected(axes2[1], X_sel)              # MI candidates
    _overlay_cv_selected(axes2[1], X_cv_sel)        # CV candidates
    # for i in range(n_sel):                          # real value at MI pts
    #     axes2[1].annotate(
    #         f"{real_sel[i]:.2f}",
    #         xy=(X_sel[i, 0], X_sel[i, 1]),
    #         xytext=(X_sel[i, 0] + 0.15, X_sel[i, 1] + 0.15),
    #         fontsize=6.5, fontweight="bold", color="darkred",
    #         bbox=dict(fc="white", ec="none", alpha=0.6, pad=1.0), zorder=10)
    # for i in range(5):                          # CV value at CV pts
    #     axes2[1].annotate(
    #         f"{cv_sel[i]:.2f}",
    #         xy=(X_cv_sel[i, 0], X_cv_sel[i, 1]),
    #         xytext=(X_cv_sel[i, 0] - 0.15, X_cv_sel[i, 1] - 0.25),
    #         fontsize=6.5, fontweight="bold", color="teal",
    #         bbox=dict(fc="white", ec="none", alpha=0.6, pad=1.0), zorder=10)
    axes2[1].legend(fontsize=20, loc="lower right", framealpha=1.0)
    _cbar(fig2, axes2[1],
          plt.cm.ScalarMappable(
              norm=mcolors.Normalize(vmin=vmin_s, vmax=vmax_s), cmap="RdYlGn"),
          "f_real(x)")
    _style(axes2[1],
           f"Real system  f_real(x)\n"
           f"gold ★ = top-{n_sel} MI  |  cyan ⬠ = top-{n_sel} CV  "
           f"|  dark-red = real val  |  teal = CV val",
           ylabel="")
    axes2[1].set_ylabel("")

    fig2.suptitle(
        rf"Sim (MI candidates) vs Real (MI + CV candidates)  "
        rf"—  $D_t$ n={args.n_sample},  top-{n_sel} each",
        fontsize=11, y=1.01,
    )
    # plt.tight_layout()
    _save(fig2, out / f"proxy_vs_real_n{args.n_sample}_k{n_sel}.pdf")
    _save(fig2, out / f"proxy_vs_real_n{args.n_sample}_k{n_sel}.png")

    # =========================================================================
    # Figure 3: Combined 2×2
    # =========================================================================
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 12))

    # Top-left: GP mean
    im = axes3[0, 0].contourf(XX, YY, gp_mean, levels=lvls, cmap="RdYlGn")
    axes3[0, 0].contour(XX, YY, gp_std, levels=5, colors="steelblue",
                        linewidths=0.8, alpha=0.6, zorder=4, linestyles="--")
    _add_failure_boundary(axes3[0, 0], XX, YY, real_vals)
    _add_diamonds(axes3[0, 0], color="white")
    _add_input_density(axes3[0, 0])
    _overlay_support(axes3[0, 0], X_support)
    _overlay_selected(axes3[0, 0], X_sel)
    axes3[0, 0].legend(fontsize=6, loc="lower right", framealpha=0.85)
    _cbar(fig3, axes3[0, 0], im, "GP posterior mean")
    _style(axes3[0, 0], "GP posterior mean\n(blue dashed = std contours)")

    # Top-right: GP std
    im2 = axes3[0, 1].contourf(XX, YY, gp_std, levels=40, cmap="YlOrRd")
    _add_failure_boundary(axes3[0, 1], XX, YY, real_vals)
    _add_diamonds(axes3[0, 1], color="black", ls="--")
    _add_input_density(axes3[0, 1])
    _overlay_support(axes3[0, 1], X_support, label=False)
    _overlay_selected(axes3[0, 1], X_sel, label=False)
    _cbar(fig3, axes3[0, 1], im2, "GP posterior std")
    _style(axes3[0, 1], "GP posterior std", ylabel="")

    # Bottom-left: sim proxy
    axes3[1, 0].contourf(XX, YY, sim_vals, levels=shared, cmap="RdYlGn")
    _add_failure_boundary(axes3[1, 0], XX, YY, real_vals)
    _add_diamonds(axes3[1, 0], color="white")
    _add_input_density(axes3[1, 0])
    _overlay_support(axes3[1, 0], X_support, label=False)
    _overlay_selected(axes3[1, 0], X_sel, label=False)
    for i in range(n_sel):
        axes3[1, 0].annotate(f"{sim_sel[i]:.2f}",
            xy=(X_sel[i, 0], X_sel[i, 1]),
            xytext=(X_sel[i, 0]+0.15, X_sel[i, 1]+0.15),
            fontsize=6, fontweight="bold",
            bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.8), zorder=10)
    _style(axes3[1, 0], "Sim proxy  f_sim(x)\n(labels = sim value at ★)")

    # Bottom-right: real
    axes3[1, 1].contourf(XX, YY, real_vals, levels=shared, cmap="RdYlGn")
    _add_failure_boundary(axes3[1, 1], XX, YY, real_vals)
    _add_diamonds(axes3[1, 1], color="white")
    _add_input_density(axes3[1, 1])
    _overlay_support(axes3[1, 1], X_support, label=False)
    _overlay_selected(axes3[1, 1], X_sel, label=False)
    for i in range(n_sel):
        axes3[1, 1].annotate(f"{real_sel[i]:.2f}",
            xy=(X_sel[i, 0], X_sel[i, 1]),
            xytext=(X_sel[i, 0]+0.15, X_sel[i, 1]+0.15),
            fontsize=6, fontweight="bold",
            bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.8), zorder=10)
    _style(axes3[1, 1], "Real system  f_real(x)\n(labels = real value at ★)",
           ylabel="")

    fig3.suptitle(
        rf"toy2D: GP model + proxy evaluation at MI-selected candidates  "
        rf"($D_t$ n={args.n_sample},  top-{n_sel} by MI)",
        fontsize=12, y=1.01,
    )
    # plt.tight_layout()
    _save(fig3, out / f"combined_n{args.n_sample}_k{n_sel}.pdf")
    _save(fig3, out / f"combined_n{args.n_sample}_k{n_sel}.png")

    print(f"\n[DONE]  outputs → {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "GP model + MI-selected candidates with sim/real proxy comparison."
        )
    )
    p.add_argument("--n-sample",        type=int,   default=20,
                   help="Size of D_t (must match plot_mi_heatmap.py)")
    p.add_argument("--n-select",        type=int,   default=5,
                   help="Top-k grid points to select by MI")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--n-grid",          type=int,   default=120,
                   help="Grid resolution (120 fast, 300 publication quality)")
    p.add_argument("--output-dir",      type=str,   default="outputs/toy2D/mi_acquisition")

    # MI hyperparameters — same defaults as plot_mi_heatmap.py
    p.add_argument("--n-clusters",      type=int,   default=5)
    p.add_argument("--radius-quantile", type=float, default=0.90)
    p.add_argument("--pi-new",          type=float, default=0.18)
    p.add_argument("--eps-exist",       type=float, default=0.04)
    p.add_argument("--tau-scale",       type=float, default=1.3)
    p.add_argument("--novelty-radius",  type=float, default=0.75)
    p.add_argument("--gamma",           type=float, default=0.25)
    # CV hyperparameters
    p.add_argument("--cv-radius",       type=float, default=0.35,
                   help="Local ball radius for paired/unpaired CV samples")
    p.add_argument("--cv-n-pair",       type=int,   default=20,
                   help="Paired sim samples per candidate")
    p.add_argument("--cv-k-unpaired",   type=int,   default=80,
                   help="Unpaired sim samples per candidate")
    p.add_argument("--cv-n-draws",      type=int,   default=16,
                   help="GP posterior draws for f in CV estimator")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
