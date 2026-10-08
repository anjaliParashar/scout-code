#!/usr/bin/env python3
"""Render the README hero: static sim field beside the target as it is learned."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import minimize

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.toy2D.domain import (  # noqa: E402
    C1, C2, R1, R2,
    make_grid, real_mean_fn, sample_designs, sample_real, sim_mean_fn,
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


def _outline(ax):
    for center, radius in ((C1, R1), (C2, R2)):
        cx, cy = float(center[0]), float(center[1])
        verts = np.array([
            [cx + radius, cy], [cx, cy + radius],
            [cx - radius, cy], [cx, cy - radius], [cx + radius, cy],
        ])
        ax.plot(verts[:, 0], verts[:, 1], color="k", lw=1.1)


def _bare(ax, title):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_box_aspect(1)
    ax.set_title(title, pad=6)


def _heat(ax, grid, values, vmin, vmax, title):
    n = int(np.sqrt(len(grid)))
    ax.imshow(
        np.asarray(values).reshape(n, n),
        origin="lower", extent=[-3, 3, -3, 3],
        cmap="magma", aspect="equal", vmin=vmin, vmax=vmax,
    )
    _outline(ax)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    _bare(ax, title)


def _kernel(a, b, lengthscale):
    d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
    return np.exp(-0.5 * d2 / lengthscale ** 2)


def _posterior(x_train, y_train, x_query, lengthscale, noise, mean):
    centered = y_train - mean
    k_tt = _kernel(x_train, x_train, lengthscale)
    k_tt.flat[:: len(x_train) + 1] += noise ** 2 + 1e-6
    k_qt = _kernel(x_query, x_train, lengthscale)
    return mean + k_qt @ np.linalg.solve(k_tt, centered)


def _nll(theta, x_train, y_train):
    lengthscale, noise = np.exp(theta)
    k = _kernel(x_train, x_train, lengthscale)
    k.flat[:: len(x_train) + 1] += noise ** 2 + 1e-6
    sign, logdet = np.linalg.slogdet(k)
    if sign <= 0:
        return 1e6
    alpha = np.linalg.solve(k, y_train)
    n = len(y_train)
    return float(0.5 * y_train @ alpha + 0.5 * logdet + 0.5 * n * np.log(2 * np.pi))


def _grab(fig) -> Image.Image:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(rgba[:, :, :3])


def _designs(n_bg: int, n_each: int, seed: int) -> np.ndarray:
    """Background labels first, then points inside each real diamond."""
    rng = np.random.default_rng(seed)
    background = sample_designs(n_bg, seed=seed).astype(np.float64)
    in_c1 = np.clip(C1 + rng.normal(scale=0.12, size=(n_each, 2)), -3.0, 3.0)
    in_c2 = np.clip(C2 + rng.normal(scale=0.12, size=(n_each, 2)), -3.0, 3.0)
    ordered = [background]
    for i in range(n_each):
        ordered.append(in_c2[i:i + 1])
        ordered.append(in_c1[i:i + 1])
    return np.vstack(ordered)


def render(path: Path, seed: int = 7) -> None:
    grid = make_grid(80)[2].astype(np.float64)
    sim = sim_mean_fn(grid).astype(np.float64)
    truth = real_mean_fn(grid).astype(np.float64)
    vmin = float(min(sim.min(), truth.min()))
    vmax = float(max(sim.max(), truth.max()))

    x_all = _designs(n_bg=12, n_each=4, seed=seed)
    n_init = len(x_all)
    y_all = sample_real(x_all, seed=seed + 1).astype(np.float64)

    theta_hat = minimize(
        _nll, np.array([np.log(0.55), np.log(0.12)]), args=(x_all, y_all),
        method="L-BFGS-B", options={"maxiter": 40},
    ).x
    lengthscale = float(np.clip(np.exp(theta_hat[0]), 0.40, 0.70))
    noise = float(np.exp(theta_hat[1]))

    fig = plt.figure(figsize=(8.2, 4.45), layout="constrained")
    grid_spec = fig.add_gridspec(
        2, 2, height_ratios=[1.0, 0.07], wspace=0.06, hspace=0.02,
    )
    ax_sim = fig.add_subplot(grid_spec[0, 0])
    ax_tgt = fig.add_subplot(grid_spec[0, 1])
    ax_bar = fig.add_subplot(grid_spec[1, :])
    frames: list[Image.Image] = []
    durations: list[int] = []

    def snapshot(k: int, pred, hold_ms: int) -> None:
        ax_sim.clear()
        ax_tgt.clear()
        ax_bar.clear()
        _heat(ax_sim, grid, sim, vmin, vmax, "Simulated system")
        if k == 0:
            n = int(np.sqrt(len(grid)))
            ax_tgt.imshow(
                np.full((n, n), 0.93),
                origin="lower", extent=[-3, 3, -3, 3],
                cmap="gray", vmin=0, vmax=1, aspect="equal",
            )
            _outline(ax_tgt)
            ax_tgt.set_xlim(-3, 3)
            ax_tgt.set_ylim(-3, 3)
            _bare(ax_tgt, "Target system")
        else:
            _heat(ax_tgt, grid, pred, vmin, vmax, "Target system")
        if k:
            ax_tgt.scatter(
                x_all[:k, 0], x_all[:k, 1],
                c="white", edgecolors="k", s=36, zorder=3,
            )
        ax_bar.barh(0, n_init, height=0.6, color="#E6E6E6")
        ax_bar.barh(0, k, height=0.6, color="#0072B2")
        ax_bar.set_xlim(0, n_init)
        ax_bar.set_ylim(-0.55, 0.55)
        ax_bar.set_yticks([])
        ax_bar.set_xticks([])
        for spine in ax_bar.spines.values():
            spine.set_visible(False)
        ax_bar.set_xlabel(f"target budget    {k} / {n_init}")
        frames.append(_grab(fig))
        durations.append(hold_ms)

    snapshot(0, np.zeros(len(grid)), 420)
    for k in range(1, n_init + 1):
        pred = _posterior(
            x_all[:k], y_all[:k], grid, lengthscale, noise, float(y_all[:k].mean()),
        )
        hold = 950 if k == n_init else 180
        snapshot(k, pred, hold)

    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path, save_all=True, append_images=frames[1:],
        duration=durations, loop=0, optimize=True, disposal=2,
    )
    plt.close(fig)
    print(f"wrote {path}  frames={len(frames)}  bytes={path.stat().st_size}")


if __name__ == "__main__":
    render(_ROOT / "docs" / "toy2d.gif")
