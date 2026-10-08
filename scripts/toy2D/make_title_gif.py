#!/usr/bin/env python3
"""Render the README hero: sample real labels, then train a surrogate on them."""

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
    make_grid, real_mean_fn, sample_designs, sample_real,
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


def _heat(ax, grid, values, vmin, vmax):
    n = int(np.sqrt(len(grid)))
    im = ax.imshow(
        np.asarray(values).reshape(n, n),
        origin="lower", extent=[-3, 3, -3, 3],
        cmap="magma", aspect="equal", vmin=vmin, vmax=vmax,
    )
    _outline(ax)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    return im


def _kernel(a, b, lengthscale):
    d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
    return np.exp(-0.5 * d2 / lengthscale ** 2)


def _posterior(x_train, y_train, x_query, lengthscale, noise):
    k_tt = _kernel(x_train, x_train, lengthscale)
    k_tt.flat[:: len(x_train) + 1] += noise ** 2 + 1e-6
    k_qt = _kernel(x_query, x_train, lengthscale)
    mean = k_qt @ np.linalg.solve(k_tt, y_train)
    return mean


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


def render(path: Path, seed: int = 7, n_init: int = 16) -> None:
    grid = make_grid(72)[2]
    truth = real_mean_fn(grid).astype(np.float64)
    vmin, vmax = float(truth.min()), float(truth.max())
    # Uniform designs plus a few draws inside each diamond, shuffled, so the
    # surrogate has high-score labels to fit and the field visibly sharpens.
    rng = np.random.default_rng(seed)
    x_all = np.vstack([
        sample_designs(n_init - 6, seed=seed),
        C1 + rng.normal(scale=0.16, size=(3, 2)),
        C2 + rng.normal(scale=0.14, size=(3, 2)),
    ]).astype(np.float64)
    x_all = x_all[rng.permutation(len(x_all))]
    x_all = np.clip(x_all, -3.0, 3.0)
    y_all = sample_real(x_all, seed=seed + 1).astype(np.float64)

    fig, axes = plt.subplots(
        1, 2, figsize=(8.6, 4.15),
        gridspec_kw={"width_ratios": [1.15, 0.85]},
        constrained_layout=True,
    )
    frames: list[Image.Image] = []
    durations: list[int] = []

    train_ylim = None

    def snapshot(title, shown, loss_x, loss_y, phase, hold_ms):
        ax, side = axes
        ax.clear()
        side.clear()
        im = _heat(ax, grid, shown, vmin, vmax)
        if phase == "sample":
            k = len(loss_y)
            if k:
                ax.scatter(
                    x_all[:k, 0], x_all[:k, 1],
                    c="white", edgecolors="k", s=36, zorder=3,
                )
            side.plot(np.arange(1, k + 1), loss_y, color="#0072B2", lw=2, marker="o", ms=4)
            side.set_xlim(0.5, n_init + 0.5)
            side.set_ylim(vmin - 0.15, vmax + 0.15)
            side.set_xlabel("label index")
            side.set_ylabel("observed target")
            side.set_title(f"Collected labels  {k}/{n_init}")
        else:
            ax.scatter(x_all[:, 0], x_all[:, 1], c="white", edgecolors="k", s=28, zorder=3)
            side.plot(loss_x, loss_y, color="#D55E00", lw=2)
            side.set_xlabel("training step")
            side.set_ylabel("negative log likelihood")
            side.set_title("Surrogate fit")
            if train_ylim is not None:
                side.set_ylim(*train_ylim)
                side.set_xlim(0, max(len(history) - 1, 1))
        ax.set_title(title)
        frames.append(_grab(fig))
        durations.append(hold_ms)

    # Phase 1: reveal the field, then add one real label at a time.
    snapshot("Sampling real labels", truth, [], [], "sample", 280)
    collected = []
    for k in range(1, n_init + 1):
        collected.append(float(y_all[k - 1]))
        snapshot("Sampling real labels", truth, [], collected, "sample", 160)
    snapshot("Sampling real labels", truth, [], collected, "sample", 450)

    # Phase 2: walk hyperparameters from a long, uncertain kernel to the fitted one.
    theta_hat = minimize(
        _nll, np.array([np.log(0.45), np.log(0.12)]), args=(x_all, y_all),
        method="L-BFGS-B", options={"maxiter": 40},
    ).x
    theta_start = np.array([np.log(2.4), np.log(0.85)])
    history = []
    n_steps = 18
    for t in np.linspace(0.0, 1.0, n_steps):
        theta = (1.0 - t) * theta_start + t * theta_hat
        history.append((np.exp(theta), _nll(theta, x_all, y_all)))
    nlls = [item[1] for item in history]
    pad = 0.08 * (max(nlls) - min(nlls) + 1e-6)
    train_ylim = (min(nlls) - pad, max(nlls) + pad)

    for step, ((lengthscale, noise), nll) in enumerate(history):
        pred = _posterior(x_all, y_all, grid.astype(np.float64), float(lengthscale), float(noise))
        xs = list(range(step + 1))
        ys = [item[1] for item in history[: step + 1]]
        title = "Training the surrogate" if step < len(history) - 1 else "Fitted surrogate"
        snapshot(title, pred, xs, ys, "train", 170 if step < len(history) - 1 else 800)

    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path, save_all=True, append_images=frames[1:],
        duration=durations, loop=0, optimize=True, disposal=2,
    )
    plt.close(fig)
    print(f"wrote {path}  frames={len(frames)}  bytes={path.stat().st_size}")


if __name__ == "__main__":
    render(_ROOT / "docs" / "toy2d.gif")
