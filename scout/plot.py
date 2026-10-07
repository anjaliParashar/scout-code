#!/usr/bin/env python3
"""
utils/plot_utils.py
-------------------
Plotting helpers for t-SNE visualisations used across all three scripts.

Design principles
-----------------
*  Every plot function accepts a pre-built `tsne_df` (DataFrame with columns
   tsne_1, tsne_2 and colour columns) plus an optional `hook_global_idx`
   array — the hook layer is rendered identically on every figure.

*  Hooks (low-target anchors) are always drawn last, in red, so they are
   visible regardless of the colour scheme of the background scatter.

*  The cumulative-selection plot in script 3 overlays:
     - all data (background, grey or coloured by target)
     - initial random sample (blue outline)
     - acquired so far (orange diamonds)
     - hook anchors (red stars)
   making it easy to see whether acquisition is converging toward the hooks.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Private drawing helpers
# ---------------------------------------------------------------------------

def _scatter_continuous(ax, tsne_df, col, cmap="viridis"):
    vals  = pd.to_numeric(tsne_df[col], errors="coerce")
    valid = vals.notna()
    sc = ax.scatter(
        tsne_df.loc[valid, "tsne_1"], tsne_df.loc[valid, "tsne_2"],
        c=vals.loc[valid], s=14, alpha=0.7, cmap=cmap,
    )
    if (~valid).any():
        ax.scatter(
            tsne_df.loc[~valid, "tsne_1"], tsne_df.loc[~valid, "tsne_2"],
            s=8, alpha=0.2, color="lightgray",
        )
    return sc


def _overlay_initial(ax, tsne_df):
    if "is_initial_sample" not in tsne_df.columns:
        return
    m = tsne_df["is_initial_sample"].astype(bool)
    if m.any():
        ax.scatter(
            tsne_df.loc[m, "tsne_1"], tsne_df.loc[m, "tsne_2"],
            s=65, facecolors="none", edgecolors="black",
            linewidths=1.2, label="initial sample",
        )


def _overlay_previous_train(ax, tsne_df):
    if "is_previous_train" not in tsne_df.columns:
        return
    m = tsne_df["is_previous_train"].astype(bool)
    if m.any():
        ax.scatter(
            tsne_df.loc[m, "tsne_1"], tsne_df.loc[m, "tsne_2"],
            s=45, marker="x", linewidths=1.1,
            color="dimgray", label="prev. training pts",
        )


def _overlay_selected_iter(ax, tsne_df):
    if "is_selected_this_iter" not in tsne_df.columns:
        return
    m = tsne_df["is_selected_this_iter"].astype(bool)
    if m.any():
        ax.scatter(
            tsne_df.loc[m, "tsne_1"], tsne_df.loc[m, "tsne_2"],
            s=170, facecolors="none", edgecolors="tab:orange",
            linewidths=2.2, label="selected this iter",
        )


def _overlay_hooks(ax, Z_hooks: Optional[np.ndarray]):
    """Draw red-star hook markers.  Z_hooks is (n_hooks, 2)."""
    if Z_hooks is None or len(Z_hooks) == 0:
        return
    ax.scatter(
        Z_hooks[:, 0], Z_hooks[:, 1],
        s=230, marker="*", color="red", edgecolors="darkred",
        linewidths=0.7, zorder=6, label="hooks (low target value)",
    )


def _finish(ax, title, path, legend=True):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    if legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left",
                  fontsize=7, frameon=True)
    ax.get_figure().savefig(path, dpi=250, bbox_inches="tight")
    plt.close(ax.get_figure())
    print(f"[INFO] Saved: {path}")


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def plot_continuous(
    tsne_df: pd.DataFrame,
    color_col: str,
    output_path: Path,
    title: str,
    colorbar_label: Optional[str] = None,
    cmap: str = "viridis",
    highlight_initial: bool = True,
    Z_hooks: Optional[np.ndarray] = None,
):
    """t-SNE scatter coloured by a continuous column, with optional hooks."""
    if color_col not in tsne_df.columns:
        print(f"[WARN] '{color_col}' not in tsne_df. Skipping.")
        return
    fig, ax = plt.subplots(figsize=(11, 8))
    sc = _scatter_continuous(ax, tsne_df, color_col, cmap)
    plt.colorbar(sc, ax=ax).set_label(colorbar_label or color_col)
    if highlight_initial:
        _overlay_initial(ax, tsne_df)
    _overlay_hooks(ax, Z_hooks)
    _finish(ax, title, output_path)


def plot_categorical(
    tsne_df: pd.DataFrame,
    color_col: str,
    output_path: Path,
    title: str,
    highlight_initial: bool = True,
    Z_hooks: Optional[np.ndarray] = None,
):
    """t-SNE scatter coloured by a categorical column."""
    fig, ax = plt.subplots(figsize=(11, 8))
    vals = tsne_df[color_col].astype(str)
    for cat in sorted(vals.unique()):
        m = vals == cat
        ax.scatter(
            tsne_df.loc[m, "tsne_1"], tsne_df.loc[m, "tsne_2"],
            s=13, alpha=0.75, label=str(cat),
        )
    if highlight_initial:
        _overlay_initial(ax, tsne_df)
    _overlay_hooks(ax, Z_hooks)
    _finish(ax, title, output_path)


def plot_acquisition(
    tsne_df: pd.DataFrame,
    color_col: str,
    output_path: Path,
    title: str,
    colorbar_label: Optional[str] = None,
    cmap: str = "plasma",
    Z_hooks: Optional[np.ndarray] = None,
):
    """
    Single-iteration acquisition plot.
    Overlays previous training, selected-this-iter, and hooks.
    """
    if color_col not in tsne_df.columns:
        print(f"[WARN] '{color_col}' not in tsne_df. Skipping.")
        return
    fig, ax = plt.subplots(figsize=(11, 8))
    sc = _scatter_continuous(ax, tsne_df, color_col, cmap)
    plt.colorbar(sc, ax=ax).set_label(colorbar_label or color_col)
    _overlay_previous_train(ax, tsne_df)
    _overlay_selected_iter(ax, tsne_df)
    _overlay_hooks(ax, Z_hooks)
    _finish(ax, title, output_path)


def plot_cumulative_selection(
    Z: np.ndarray,
    vis_idx: np.ndarray,
    train_idx_so_far: np.ndarray,
    initial_train_idx: np.ndarray,
    y_target: np.ndarray,
    hook_global_idx: np.ndarray,
    output_path: Path,
    iteration: int,
    target_col_label: str = "target metric",
):
    """
    Show ALL data points coloured by target value (background), with:
      - initial sample (blue outline circles)
      - all acquired points so far, excluding initial (orange diamonds)
      - hook anchors (red stars)

    This allows you to see whether the acquisition is converging toward the
    failure (low-target) hook regions as iterations proceed.

    Parameters
    ----------
    Z                  : (len(vis_idx), 2) t-SNE coordinates.
    vis_idx            : global indices corresponding to rows of Z.
    train_idx_so_far   : global indices of ALL selected points so far
                         (initial + all acquired iterations).
    initial_train_idx  : global indices of the initial random sample.
    y_target           : (n_total,) target values for ALL rows.
    hook_global_idx    : global indices of hook points.
    output_path        : where to save the figure.
    iteration          : current AL iteration number (for the title).
    target_col_label   : label for the colour bar.
    """
    fig, ax = plt.subplots(figsize=(13, 9))

    # ---- Background: all vis points coloured by target value ----
    y_vis = y_target[vis_idx]
    sc = ax.scatter(
        Z[:, 0], Z[:, 1],
        c=y_vis, cmap="RdYlGn",          # green = high TTC (safe), red = low target value (failure)
        s=10, alpha=0.30, vmin=np.nanpercentile(y_vis, 2),
        vmax=np.nanpercentile(y_vis, 98),
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(f"{target_col_label} (all visible points)", fontsize=8)

    # ---- Build a local→global lookup for vis_idx ----
    g2l = {g: l for l, g in enumerate(vis_idx)}

    # ---- Initial sample ----
    init_local = np.array([g2l[g] for g in initial_train_idx if g in g2l], int)
    if len(init_local) > 0:
        ax.scatter(
            Z[init_local, 0], Z[init_local, 1],
            s=60, facecolors="none", edgecolors="steelblue",
            linewidths=1.5, label=f"initial sample (n={len(init_local)})",
        )

    # ---- Acquired points (all iterations so far, excluding initial) ----
    acq_global = np.setdiff1d(train_idx_so_far, initial_train_idx)
    acq_local  = np.array([g2l[g] for g in acq_global if g in g2l], int)
    if len(acq_local) > 0:
        ax.scatter(
            Z[acq_local, 0], Z[acq_local, 1],
            s=80, marker="D", color="tab:orange", alpha=0.85,
            label=f"acquired so far (n={len(acq_local)})",
        )

    # ---- Hooks ----
    hook_local = np.array([g2l[g] for g in hook_global_idx if g in g2l], int)
    if len(hook_local) > 0:
        ax.scatter(
            Z[hook_local, 0], Z[hook_local, 1],
            s=240, marker="*", color="red", edgecolors="darkred",
            linewidths=0.8, zorder=7,
            label=f"hooks — low target value (n={len(hook_local)})",
        )

    ax.set_title(
        f"Cumulative selection after iteration {iteration:02d}\n"
        f"(are acquired points reaching the red-star failure hooks?)",
        fontsize=11,
    )
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, frameon=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved cumulative plot: {output_path}")
