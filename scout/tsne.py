#!/usr/bin/env python3
"""
utils/tsne_utils.py
-------------------
Canonical t-SNE computation with disk caching.

All three scripts call  `get_or_compute_tsne(X_all, cache_path)`  so that
every plot in the project uses the *same* 2-D coordinates.

Typical usage
-------------
    from utils.tsne_utils import get_or_compute_tsne

    Z = get_or_compute_tsne(
        X_all,                          # (n, d) scaled embedding
        cache_path="outputs/tsne_Z.pkl",
        perplexity=40,
        random_state=42,
    )
    # Z has shape (n, 2) and is identical across all scripts.
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Default path used when no override is given.
DEFAULT_TSNE_CACHE = Path("outputs/tsne_Z.pkl")


# ---------------------------------------------------------------------------
# Core t-SNE runner
# ---------------------------------------------------------------------------

def run_tsne(
    X: np.ndarray,
    perplexity: float = 40,
    random_state: int = 0,
) -> np.ndarray:
    """
    PCA (≤50 components) → t-SNE(2).

    Parameters
    ----------
    X             : (n, d) array, must have n ≥ 3.
    perplexity    : t-SNE perplexity (auto-clamped to valid range).
    random_state  : reproducibility seed.

    Returns
    -------
    Z : (n, 2) t-SNE coordinates.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n < 3:
        raise ValueError(f"t-SNE needs ≥ 3 points; got {n}.")

    perplexity = float(min(perplexity, max(2, (n - 1) // 3)))
    pca_dim = min(50, X.shape[1], n - 1)

    X_r = (
        PCA(n_components=pca_dim, random_state=random_state).fit_transform(X)
        if pca_dim >= 2
        else X
    )
    return TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    ).fit_transform(X_r)


# ---------------------------------------------------------------------------
# Cached entry-point (called by every script)
# ---------------------------------------------------------------------------

def get_or_compute_tsne(
    X: np.ndarray,
    cache_path: Path = DEFAULT_TSNE_CACHE,
    perplexity: float = 40,
    random_state: int = 0,
    force_recompute: bool = False,
) -> np.ndarray:
    """
    Return cached t-SNE embedding if available, otherwise compute and cache it.

    The cache stores Z for *all* rows of X.  If the cached Z has a different
    number of rows than X (dataset changed), it is recomputed automatically.

    Parameters
    ----------
    X               : (n, d) scaled embedding matrix for ALL data points.
    cache_path      : path to the pickle cache.
    perplexity      : t-SNE perplexity.
    random_state    : seed.
    force_recompute : ignore existing cache.

    Returns
    -------
    Z : (n, 2) t-SNE coordinates, consistent across all scripts.
    """
    cache_path = Path(cache_path)

    if not force_recompute and cache_path.exists():
        print(f"[INFO] Loading cached t-SNE from {cache_path}")
        with open(cache_path, "rb") as fh:
            Z = pickle.load(fh)
        if isinstance(Z, np.ndarray) and Z.shape[0] == X.shape[0]:
            return Z
        print(
            f"[WARN] Cached Z has {Z.shape[0] if hasattr(Z,'shape') else '?'} rows "
            f"but X has {X.shape[0]}. Recomputing."
        )

    print(
        f"[INFO] Computing t-SNE on {X.shape[0]:,} × {X.shape[1]} matrix "
        f"(perplexity={perplexity}) …"
    )
    Z = run_tsne(X, perplexity=perplexity, random_state=random_state)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(Z, fh)
    print(f"[INFO] t-SNE cached → {cache_path}")
    return Z


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def choose_vis_indices(
    n: int,
    anchor_idx: np.ndarray,
    max_points: int,
    random_state: int,
) -> np.ndarray:
    """
    Return up to `max_points` indices for visualisation.
    Always includes `anchor_idx` (initial sample / training set), then
    fills with random extras up to `max_points`.
    """
    rng = np.random.default_rng(random_state)
    anchor_idx = np.asarray(anchor_idx, dtype=int)
    if max_points <= 0 or max_points >= n:
        return np.arange(n)
    remaining = np.setdiff1d(np.arange(n), anchor_idx)
    extra = max(0, min(max_points - len(anchor_idx), len(remaining)))
    extra_idx = rng.choice(remaining, size=extra, replace=False)
    return np.concatenate([anchor_idx, extra_idx])
