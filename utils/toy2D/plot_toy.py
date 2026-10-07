#!/usr/bin/env python3
"""
utils/toy2D/plot_toy.py
-----------------------
Plotting helpers for all toy2D experiments.

Public API
----------
plot_gp_mean_heatmap   — single strategy GP-mean heatmap with acquired points
plot_comparison_grid   — 2×N grid comparing multiple strategies
plot_best_seen_curve   — line chart of best observed real value per round
plot_discovery_summary — text + bar chart of diamond-membership results
"""

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from utils.toy2D.domain import make_grid, latent_target_fn, diamond_membership

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
# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _draw_diamonds(ax):
    """Overlay the true failure boundary (latent == 0 contour)."""
    XX, YY, X_grid = make_grid(120)
    latent = latent_target_fn(X_grid).reshape(XX.shape)
    ax.contour(XX, YY, latent, levels=[0.0], colors="k", linewidths=1.8,
               linestyles="--")


# ---------------------------------------------------------------------------
# Single-strategy heatmap
# ---------------------------------------------------------------------------

def plot_gp_mean_heatmap(
    gp,
    X_init:    np.ndarray,
    X_new:     np.ndarray,
    title:     str,
    output_path: Optional[Path] = None,
    n_rounds:  int = 0,
    figsize:   tuple = (7, 6),
):
    """
    GP-mean heatmap for one strategy after n_rounds of AL.

    Parameters
    ----------
    gp          : fitted BoTorch GP
    X_init      : (n_init, 2) initial sample
    X_new       : (n_acquired, 2) points acquired by AL
    title       : plot title
    output_path : save path; if None, calls plt.show()
    n_rounds    : for the legend label
    """
    from utils.toy2D.gp_utils import gp_predict

    XX, YY, X_grid = make_grid(120)
    mean, _ = gp_predict(gp, X_grid)
    mean    = mean.reshape(XX.shape)

    fig, ax = plt.subplots(figsize=figsize)
    ax.contourf(XX, YY, mean, levels=30, cmap="viridis")
    
    _draw_diamonds(ax)

    ax.scatter(X_init[:, 0], X_init[:, 1], s=35, color="white",
               edgecolors="k", zorder=4, label="initial")
    if len(X_new) > 0:
        im = ax.scatter(X_new[:, 0], X_new[:, 1], s=50, marker="*",
                   color="red", zorder=5, label=f"acquired ({n_rounds} rounds)")
    plt.colorbar(im, ax=ax, label="GP mean")
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    # plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved: {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Multi-strategy comparison grid
# ---------------------------------------------------------------------------

def plot_comparison_grid(
    results:       dict,
    strategies:    Sequence[str],
    n_init:        int,
    n_rounds:      int,
    output_path:   Optional[Path] = None,
    ncols:         int = 3,
):
    """
    Grid of GP-mean heatmaps, one per strategy.

    results[strategy] must contain keys:
      gp_final, X_init, X_final
    """
    from utils.toy2D.gp_utils import gp_predict

    XX, YY, X_grid = make_grid(120)
    latent = latent_target_fn(X_grid).reshape(XX.shape)

    nrows = (len(strategies) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18,6))
    axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, s in zip(axes_flat, strategies):
        gp     = results[s]["gp_final"]
        X_init = results[s]["X_init"]
        X_all  = results[s]["X_final"]
        X_new  = X_all[n_init:]

        mean, _ = gp_predict(gp, X_grid)
        mean = mean.reshape(XX.shape)

        ax.contourf(XX, YY, mean, levels=30, cmap="viridis")
        ax.contour(XX, YY, latent, levels=[0.0], colors="k",
                   linewidths=1.5, linestyles="--")
        # ax.scatter(X_init[:, 0], X_init[:, 1], s=50, color="black",
        #            edgecolors="k", zorder=4, label="initial")
        im = ax.scatter(X_new[:, 0], X_new[:, 1], s=250, marker="*",
                   c= np.arange(len(X_new)), cmap = 'Reds',zorder=5, label=f"acquired")
        # plt.colorbar(im, ax=ax,fontsize=15)
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=25)
        cbar.set_label("No. of evaluations", fontsize=30)
        # ax.set_title(f"{s.upper()} | GP fit after {n_rounds} rounds")
        # ax.tick_params(axis="x", labelsize=25)  
        # ax.tick_params(axis="y", labelsize=25)
        
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_xlabel("$x_1$",fontsize=20); ax.set_ylabel("$x_2$",fontsize=20)
        # ax.legend(loc="lower center", fontsize=20)

    for ax in axes_flat[len(strategies):]:
        ax.set_visible(False)

    # plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved: {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Best-seen curve
# ---------------------------------------------------------------------------

def plot_best_seen_curve(
    results:     dict,
    strategies:  Sequence[str],
    n_rounds:    int,
    output_path: Optional[Path] = None,
):
    """Line chart of best observed real value per round for each strategy."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for s in strategies:
        brs = results[s]["best_real_seen"]
        ax.plot(np.arange(1, len(brs) + 1), brs, marker="o", label=s)
    ax.set_xlabel("Round")
    ax.set_ylabel("Best real value seen")
    ax.set_title(f"Best observed real value vs round  ({n_rounds} rounds)")
    ax.grid(alpha=0.3)
    ax.legend()
    # plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved: {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Discovery summary
# ---------------------------------------------------------------------------

def print_discovery_summary(results: dict, strategies: Sequence[str]):
    """Print a table showing which diamond modes each strategy found."""
    print("\nDiscovery summary:")
    print(f"{'strategy':>10} | {'total pts':>10} | diamond1 | diamond2 |  both")
    print("-" * 58)
    for s in strategies:
        Xf = results[s]["X_final"]
        m  = diamond_membership(Xf)
        d1 = bool(np.any(m == 1))
        d2 = bool(np.any(m == 2))
        both = d1 and d2
        print(
            f"{s:>10} | {len(Xf):>10} | {str(d1):>8} | {str(d2):>8} | {str(both):>5}"
        )


def plot_discovery_bar(
    results:     dict,
    strategies:  Sequence[str],
    output_path: Optional[Path] = None,
):
    """Bar chart: fraction of acquired points inside each diamond, per strategy."""
    d1_fracs, d2_fracs = [], []
    for s in strategies:
        Xf  = results[s]["X_final"]
        n   = results[s].get("n_init", 0)
        Xaq = Xf[n:]
        if len(Xaq) == 0:
            d1_fracs.append(0); d2_fracs.append(0); continue
        m  = diamond_membership(Xaq)
        d1_fracs.append(np.mean(m == 1))
        d2_fracs.append(np.mean(m == 2))

    x   = np.arange(len(strategies))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(strategies) * 1.5), 4))
    ax.bar(x - w/2, d1_fracs, width=w, label="diamond 1", color="steelblue")
    ax.bar(x + w/2, d2_fracs, width=w, label="diamond 2", color="coral")
    ax.set_xticks(x); ax.set_xticklabels(strategies, rotation=15, ha="right")
    ax.set_ylabel("Fraction of acquired points inside diamond")
    ax.set_title("Discovery rate per diamond region")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3, axis="y")
    # plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved: {output_path}")
    else:
        plt.show()
