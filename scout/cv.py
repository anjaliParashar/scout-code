#!/usr/bin/env python3
"""
scout/cv.py
-----------------
Control-variates (CV) mean estimator for paired real / sim data.

Core estimator
--------------
    mu_hat = mean(f − β·g) + β·θ̂

where
  f        = BNN posterior draws at x          (real side)
  g        = paired proxy values near x       (sim/control side)
  θ̂        = mean of a larger unpaired sim pool
  β̂ ≈ cov(g, f) / var(g)

The CV estimator reduces the variance of the target mean by exploiting the
sim/real correlation.
"""

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors

from scout.bnn import HeteroBNNEmbedding, batched_posterior_draws


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

@dataclass
class CVResult:
    mu_hat: float
    var_mu_hat: float
    beta: float


def control_variates_estimator(
    f: np.ndarray,
    g: np.ndarray,
    g_unpaired: np.ndarray,
) -> CVResult:
    """
    f : (n,)  paired BNN draws (real side)
    g : (n,)  paired sim observations
    g_unpaired : (k,) larger unpaired sim sample
    """
    n, k = len(f), len(g_unpaired)
    if n <= 1 or k <= 1 or len(g) != n:
        raise ValueError("Invalid CV sample sizes.")
    var_f        = np.var(f)
    theta_hat    = np.mean(g_unpaired)
    cov_gf       = np.cov(g, f)[0, 1]
    var_g        = np.var(g)
    var_g_unp    = np.var(g_unpaired)
    if np.isclose(var_g, 0):
        raise ValueError("var(g) == 0.")
    beta    = k / (k + n) * cov_gf / var_g
    mu_hat  = np.mean(f - beta * g) + beta * theta_hat
    var_hat = (
        (var_f + beta**2 * var_g - 2 * beta * cov_gf) / n
        + beta**2 * var_g_unp / k
    )
    return CVResult(float(mu_hat), float(max(var_hat, 1e-10)), float(beta))


# ---------------------------------------------------------------------------
# Per-point worker (called in parallel)
# ---------------------------------------------------------------------------

def _cv_worker(
    local_i: int,
    global_idx: int,
    f_draws: np.ndarray,           # (n_draws,)
    paired_nbrs: np.ndarray,
    unpaired_nbrs: np.ndarray,
    fallback_nbrs: np.ndarray,
    y_cv: np.ndarray,
) -> dict:
    f        = np.asarray(f_draws, dtype=float).ravel()
    g        = y_cv[paired_nbrs.astype(int)].astype(float)
    g_unp    = y_cv[unpaired_nbrs.astype(int)].astype(float)

    # If paired sample is degenerate, try fallback neighbours
    if len(g) < 2 or np.isclose(np.var(g), 0.0):
        if len(fallback_nbrs) >= 2:
            g = y_cv[fallback_nbrs.astype(int)[: len(f)]].astype(float)
        else:
            return dict(local_i=local_i, global_idx=int(global_idx),
                        cv_mean=np.nan, cv_var=np.nan, beta=np.nan, cv_corr=np.nan)

    n = min(len(f), len(g))
    f, g = f[:n], g[:n]
    if len(g_unp) < 2:
        g_unp = g.copy()

    cv_corr = (
        0.0 if np.std(f) < 1e-12 or np.std(g) < 1e-12
        else float(np.corrcoef(f, g)[0, 1])
    )
    try:
        res = control_variates_estimator(f=f, g=g, g_unpaired=g_unp)
        return dict(local_i=local_i, global_idx=int(global_idx),
                    cv_mean=res.mu_hat, cv_var=res.var_mu_hat,
                    beta=res.beta, cv_corr=cv_corr)
    except Exception:
        return dict(local_i=local_i, global_idx=int(global_idx),
                    cv_mean=np.nan, cv_var=np.nan, beta=np.nan, cv_corr=cv_corr)


# ---------------------------------------------------------------------------
# Parallel CV estimation over a query set
# ---------------------------------------------------------------------------

def compute_local_cv_parallel(
    model: HeteroBNNEmbedding,
    X_query: np.ndarray,
    query_global_idx: np.ndarray,
    X_target: np.ndarray,
    y_cv: np.ndarray,
    n_pair_local: int = 10,
    k_unpaired: int   = 20,
    radius: float     = 1.5,
    fallback_k: int   = 80,
    n_jobs: int       = -1,
    device: str       = "cuda",
    batch_size: int   = 4096,
) -> tuple:
    """
    Compute CV estimates for every point in X_query in parallel.

    Returns
    -------
    cv_mean  : (N,)
    cv_var   : (N,)
    beta     : (N,)
    cv_corr  : (N,)
    """
    X_query          = np.asarray(X_query,          dtype=float)
    X_target         = np.asarray(X_target,         dtype=float)
    y_cv             = np.asarray(y_cv,              dtype=float)
    query_global_idx = np.asarray(query_global_idx,  dtype=int)
    N                = len(X_query)

    # BNN draws: shape (n_pair_local, N)
    f_draws = batched_posterior_draws(
        model, X_query, n_pair_local, device, batch_size
    )

    # Radius neighbours for paired / unpaired samples
    nn_r    = NearestNeighbors(radius=radius,    n_jobs=-1).fit(X_target)
    rad_nbr = nn_r.radius_neighbors(X_query, radius=radius, return_distance=False)

    k_tot   = min(max(n_pair_local, k_unpaired, fallback_k), len(X_target))
    nn_k    = NearestNeighbors(n_neighbors=k_tot, n_jobs=-1).fit(X_target)
    knn_nbr = nn_k.kneighbors(X_query, return_distance=False)

    rng = np.random.default_rng(0)
    paired_list, unpaired_list, fallback_list = [], [], []
    for i in range(N):
        neigh = np.asarray(rad_nbr[i], int)
        neigh = neigh[neigh != query_global_idx[i]]
        if len(neigh) < max(2, n_pair_local):
            neigh = knn_nbr[i][knn_nbr[i] != query_global_idx[i]]
        if len(neigh) == 0:
            neigh = knn_nbr[i]
        paired   = rng.choice(neigh, n_pair_local,  replace=len(neigh) < n_pair_local)
        unpaired = rng.choice(neigh, k_unpaired,    replace=len(neigh) < k_unpaired)
        fallback = knn_nbr[i][knn_nbr[i] != query_global_idx[i]]
        paired_list.append(paired)
        unpaired_list.append(unpaired)
        fallback_list.append(fallback)

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_cv_worker)(
            i, query_global_idx[i], f_draws[:, i],
            paired_list[i], unpaired_list[i], fallback_list[i], y_cv,
        )
        for i in range(N)
    )

    cv_mean  = np.array([r["cv_mean"]  for r in results], float)
    cv_var   = np.array([r["cv_var"]   for r in results], float)
    beta     = np.array([r["beta"]     for r in results], float)
    cv_corr  = np.array([r["cv_corr"]  for r in results], float)

    for arr, default in [(cv_mean, 0.0), (cv_var, 1.0), (beta, 0.0), (cv_corr, 0.0)]:
        bad = ~np.isfinite(arr)
        if bad.any():
            good = np.isfinite(arr)
            arr[bad] = np.nanmedian(arr[good]) if good.any() else default

    return cv_mean, cv_var, beta, cv_corr
