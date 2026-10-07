#!/usr/bin/env python3
"""
Stepwise figures for SCOUT on the 2-D toy problem.

1. Real target field and the shifted proxy field.
2. A small set of real labels.
3. Mutual-information shortlist around that support.
4. Control-variate scores on the shortlist, and the batch they select.

Example
-------
python scripts/toy2D/stepwise_viz.py --output-dir outputs/toy2D/stepwise
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.toy2D.domain import (  # noqa: E402
    C1, C2, R1, R2,
    make_grid, real_mean_fn, sample_designs, sample_real, sim_mean_fn,
)
from utils.toy2D.gp_utils import gp_predict, train_gp  # noqa: E402
from utils.toy2D.mi_toy import compute_cv_batch, fit_support_and_compute_mi  # noqa: E402

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


def _diamond(center, radius):
    cx, cy = float(center[0]), float(center[1])
    return np.array([
        [cx + radius, cy],
        [cx, cy + radius],
        [cx - radius, cy],
        [cx, cy - radius],
        [cx + radius, cy],
    ])


def _outline(ax):
    for center, radius in ((C1, R1), (C2, R2)):
        verts = _diamond(center, radius)
        ax.plot(verts[:, 0], verts[:, 1], color="k", lw=1.0, alpha=0.85)


def _heat(ax, grid, values, title):
    n = int(np.sqrt(len(grid)))
    image = values.reshape(n, n)
    im = ax.imshow(
        image, origin="lower", extent=[-3, 3, -3, 3],
        cmap="magma", aspect="equal",
    )
    _outline(ax)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    return im


def build_figures(
    seed: int = 7,
    n_init: int = 16,
    grid_n: int = 56,
    n_shortlist: int = 24,
    n_batch: int = 5,
):
    """Return four matplotlib figures, one per SCOUT step."""
    xx, yy, grid = make_grid(grid_n)
    real = real_mean_fn(grid)
    proxy = sim_mean_fn(grid)

    x_init = sample_designs(n_init, seed=seed)
    y_init = sample_real(x_init, seed=seed + 1)

    gp = train_gp(x_init.astype(np.float64), y_init.astype(np.float64))
    gp_mean, gp_var = gp_predict(gp, grid.astype(np.float64))
    # Candidates are random designs, matching the toy acquisition scripts.
    # Scoring the square grid instead piles the shortlist on the boundary.
    rng = np.random.default_rng(seed + 3)
    pool = rng.uniform(-2.6, 2.6, size=(500, 2)).astype(np.float64)
    mi, _, _, *_rest = fit_support_and_compute_mi(
        X_support=x_init.astype(np.float64),
        X_query=pool,
        n_clusters=min(6, n_init),
        radius_quantile=0.90,
        pi_new=0.18,
        eps_exist=0.04,
        tau_scale=1.3,
        novelty_radius=0.75,
        gamma=0.25,
        random_state=seed,
    )
    mi = np.nan_to_num(mi)
    short_local = np.argsort(-mi)[:n_shortlist]
    short_x = pool[short_local]

    cv = compute_cv_batch(
        gp, short_x.astype(np.float32),
        R_local=0.25, n_pair=40, k_unpaired=48, n_f_draws=12, clip=2.0,
    )
    cv = np.nan_to_num(cv, nan=-np.inf)
    batch_local = np.argsort(-cv)[:n_batch]

    figures = []

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True)
    im0 = _heat(axes[0], grid, real, "Step 1a  ·  real target")
    im1 = _heat(axes[1], grid, proxy, "Step 1b  ·  proxy (shifted)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    figures.append(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    im = _heat(ax, grid, real, "Step 2  ·  initial real labels")
    ax.scatter(x_init[:, 0], x_init[:, 1], c="white", edgecolors="k", s=36, zorder=3)
    fig.colorbar(im, ax=ax, fraction=0.046)
    figures.append(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True)
    im0 = _heat(axes[0], grid, gp_mean, "Step 3a  ·  GP mean")
    im1 = _heat(axes[1], grid, np.sqrt(np.maximum(gp_var, 0.0)), "Step 3b  ·  GP std")
    for ax in axes:
        ax.scatter(x_init[:, 0], x_init[:, 1], c="white", edgecolors="k", s=22, zorder=3)
    axes[1].scatter(
        short_x[:, 0], short_x[:, 1],
        marker="*", c="#FFD700", edgecolors="k", s=70, zorder=4, label="MI shortlist",
    )
    axes[1].legend(loc="lower right", fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    figures.append(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    im = _heat(ax, grid, real, "Step 4  ·  CV re-ranks the shortlist")
    sc = ax.scatter(
        short_x[:, 0], short_x[:, 1],
        c=np.where(np.isfinite(cv), cv, np.nan),
        cmap="viridis", edgecolors="k", s=42, zorder=3,
    )
    chosen = short_x[batch_local]
    ax.scatter(
        chosen[:, 0], chosen[:, 1],
        marker="*", c="#FFD700", edgecolors="k", s=160, zorder=4, label="next batch",
    )
    ax.scatter(x_init[:, 0], x_init[:, 1], c="white", edgecolors="k", s=22, zorder=2)
    ax.legend(loc="lower right", fontsize=8)
    fig.colorbar(sc, ax=ax, fraction=0.046, label="CV mean")
    figures.append(fig)
    return figures


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-init", type=int, default=16)
    p.add_argument("--grid-n", type=int, default=56)
    p.add_argument("--output-dir", type=str, default="outputs/toy2D/stepwise")
    args = p.parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = ["01_fields", "02_initial_labels", "03_mutual_information", "04_cv_batch"]
    figures = build_figures(seed=args.seed, n_init=args.n_init, grid_n=args.grid_n)
    for name, fig in zip(names, figures):
        path = out / f"{name}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        print(path)
    return figures


if __name__ == "__main__":
    main()
