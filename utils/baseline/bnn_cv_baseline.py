#!/usr/bin/env python3
"""
utils/baseline/bnn_cv_baseline.py
-----------------------------------
μ_CV baseline with BNN surrogate — no MI filtering step.

Embedding-space acquisition (closed-loop SCOUT structure, minus MI)
--------------------------------------------------------------
Each iteration:
  1. Train BNN on the current labelled set.
  2. Run compute_local_cv_parallel on the full candidate pool — this uses
     kNN neighbours from the actual embedding space as paired/unpaired sim
     samples (proxy values), the same neighbourhood estimator the closed-loop loop uses on its shortlist.
  3. Select the batch_size candidates with the LOWEST cv_mean
     (low target value = failure).

Speed fix vs. old version
--------------------------
Old: serial loop over candidates using _local_ball random samples (toy2D).
New: compute_local_cv_parallel — joblib-parallelised, batched BNN draws,
     kNN neighbours from the real embedding, same neighbourhood control-variate step as the closed-loop loop.

Toy2D acquisition
-----------------
compute_bnn_cv_batch is kept for the toy2D scripts (baseline_bnn_cv.py),
which need local-ball sim sampling because there is no kNN dataset.
"""

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from scout.bnn import HeteroBNNEmbedding, train_bnn, batched_posterior_draws
from scout.cv import compute_local_cv_parallel


# ---------------------------------------------------------------------------
# BNN training wrapper
# ---------------------------------------------------------------------------

def train_bnn_surrogate(
    X_np:         np.ndarray,
    y_np:         np.ndarray,
    input_dim:    int   = None,
    epochs:       int   = 1200,
    lr:           float = 1e-3,
    weight_decay: float = 1e-6,
    p_drop:       float = 0.10,
    device:       str   = "cpu",
) -> HeteroBNNEmbedding:
    """Fit a HeteroBNNEmbedding on (X_np, y_np)."""
    if input_dim is None:
        input_dim = X_np.shape[1]
    model = HeteroBNNEmbedding(input_dim=input_dim, p_drop=p_drop).to(device)
    train_bnn(model, X_np, y_np, epochs=epochs, lr=lr,
              weight_decay=weight_decay, device=device)
    return model


# ---------------------------------------------------------------------------
# Embedding-space acquisition — uses compute_local_cv_parallel
# ---------------------------------------------------------------------------

def bnn_cv_acquisition_embedding(
    model:            HeteroBNNEmbedding,
    pool_global_idx:  np.ndarray,   # global indices of pool points in X_all
    X_all:            np.ndarray,   # full embedding matrix (all N points)
    y_cv:             np.ndarray,   # sim CV column (proxy values), shape (N,)
    batch_size:       int,
    device:           str   = "cpu",
    n_pair_local:     int   = 10,
    k_unpaired:       int   = 20,
    cv_radius:        float = 1.5,
    cv_fallback_k:    int   = 80,
    n_jobs:           int   = -1,
    batch_size_draw:  int   = 4096,
) -> np.ndarray:
    """
    Select batch_size pool points with the lowest CV-estimated mean.

    Uses compute_local_cv_parallel — the same parallelised kNN-based CV
    estimator used by the closed-loop loop, so the scores match.
    The only difference from that loop is that there is no
    MI pre-filtering: CV is computed directly on the full candidate pool.

    Parameters
    ----------
    model           : fitted HeteroBNNEmbedding (BNN posterior draws = f)
    pool_global_idx : global indices of candidate pool points in X_all
    X_all           : (N, d) full standardised embedding matrix
    y_cv            : (N,) proxy values for all N points (paired sim values)
    batch_size      : number of points to select
    device          : torch device
    n_pair_local    : paired draws per point (n in CV estimator)
    k_unpaired      : unpaired sim pool size per point (k in CV estimator)
    cv_radius       : kNN ball radius for finding paired neighbours
    cv_fallback_k   : hard-kNN fallback when radius ball is too small
    n_jobs          : joblib parallelism (-1 = all cores)
    batch_size_draw : BNN draw batch size

    Returns
    -------
    batch_local_idx : (batch_size,) int array into pool_global_idx
    """
    X_pool = X_all[pool_global_idx]

    cv_mean, _, _, _ = compute_local_cv_parallel(
        model            = model,
        X_query          = X_pool,
        query_global_idx = pool_global_idx,
        X_target         = X_all,
        y_cv             = y_cv,
        n_pair_local     = n_pair_local,
        k_unpaired       = k_unpaired,
        radius           = cv_radius,
        fallback_k       = cv_fallback_k,
        n_jobs           = n_jobs,
        device           = device,
        batch_size       = batch_size_draw,
    )

    # Select lowest cv_mean — low target value = failure
    valid = np.where(np.isfinite(cv_mean))[0]
    if len(valid) == 0:
        return np.random.choice(len(pool_global_idx), batch_size, replace=False)

    order = valid[np.argsort(cv_mean[valid])]          # ascending: lowest first
    return order[:batch_size]


# ---------------------------------------------------------------------------
# Toy2D acquisition — local-ball sim sampling (kept for toy2D scripts)
# ---------------------------------------------------------------------------

def _local_ball(x0: np.ndarray, radius: float, n: int) -> np.ndarray:
    d    = x0.shape[0]
    dirs = np.random.randn(n, d).astype(np.float32)
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-8)
    r    = radius * np.sqrt(np.random.rand(n, 1).astype(np.float32))
    return np.clip(x0[None, :] + r * dirs, -3.0, 3.0).astype(np.float32)


def compute_bnn_cv_batch(
    model:       HeteroBNNEmbedding,
    X_query:     np.ndarray,
    sim_fn,                          # callable (N,d) -> (N,) sim scores
    device:      str   = "cpu",
    R_local:     float = 0.35,
    n_pair:      int   = 20,
    k_unpaired:  int   = 120,
    n_f_draws:   int   = 32,
    clip:        float = 1.0,
) -> np.ndarray:
    """
    CV-estimated mean for every point in X_query using BNN draws + local-ball sim.
    Used by toy2D baseline scripts only.

    Returns (N,) array of CV means (NaN on failure, clipped to `clip`).
    """
    from utils.toy2D.mi_toy import control_variates_estimator

    out = np.full(len(X_query), np.nan)
    for i, x0 in enumerate(X_query):
        u   = _local_ball(x0, R_local, n_pair)
        v   = _local_ball(x0, R_local, k_unpaired)
        g_u = sim_fn(u).astype(np.float64)
        g_v = sim_fn(v).astype(np.float64)

        f_draws = batched_posterior_draws(model, u, n_draws=n_f_draws, device=device)
        vals = []
        for s in range(n_f_draws):
            try:
                res = control_variates_estimator(
                    f=f_draws[s].astype(np.float64), g=g_u, g_unpaired=g_v,
                )
                vals.append(res.mu_hat)
            except ValueError:
                pass
        if vals:
            out[i] = min(np.mean(vals), clip)
    return out


def bnn_cv_acquisition(
    pool_X:     np.ndarray,
    model:      HeteroBNNEmbedding,
    batch_size: int,
    sim_fn,                          # callable for toy2D sim scores
    device:     str   = "cpu",
    maximise:   bool  = True,
    R_local:    float = 0.35,
    n_pair:     int   = 20,
    k_unpaired: int   = 120,
    n_f_draws:  int   = 32,
    cv_clip:    float = 1.0,
) -> np.ndarray:
    """
    Toy2D acquisition: select by BNN-CV mean using local-ball sim sampling.
    For an embedding-space pool use bnn_cv_acquisition_embedding instead.
    """
    cv_vals = compute_bnn_cv_batch(
        model, pool_X, sim_fn, device=device,
        R_local=R_local, n_pair=n_pair,
        k_unpaired=k_unpaired, n_f_draws=n_f_draws, clip=cv_clip,
    )
    valid = np.where(np.isfinite(cv_vals))[0]
    if len(valid) == 0:
        return np.random.choice(len(pool_X), batch_size, replace=False)
    order = valid[np.argsort(cv_vals[valid])[::-1 if maximise else 1]]
    return order[:batch_size]
