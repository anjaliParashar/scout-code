#!/usr/bin/env python3
"""
utils/cv_utils.py
-----------------
Control-variates (CV) mean estimator for paired target / proxy MLP data.

Core estimator
--------------
    mu_hat = mean(f - beta * g) + beta * theta_hat

where
  f        = target MLP predictions on local paired points
  g        = proxy MLP predictions on the same local paired points
  theta_hat= mean proxy MLP prediction on a larger local unpaired set
  beta     = shrinkage-adjusted cov(g, f) / var(g)

This version is deterministic: each MLP produces one prediction per sample.
It does not use BNN posterior draws.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors

from utils.quadruped.train_utils import load_mlp_checkpoint, predict_mlp


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
    f          : (n,) target-side values on paired local samples
    g          : (n,) proxy-side values on the same paired local samples
    g_unpaired : (k,) proxy-side values on a larger local unpaired sample
    """
    f = np.asarray(f, dtype=float).ravel()
    g = np.asarray(g, dtype=float).ravel()
    g_unpaired = np.asarray(g_unpaired, dtype=float).ravel()

    n, k = len(f), len(g_unpaired)
    if n <= 1 or k <= 1 or len(g) != n:
        raise ValueError("Invalid CV sample sizes.")

    var_f = np.var(f)
    theta_hat = np.mean(g_unpaired)
    cov_gf = np.cov(g, f)[0, 1]
    var_g = np.var(g)
    var_g_unp = np.var(g_unpaired)

    if np.isclose(var_g, 0.0):
        raise ValueError("var(g) == 0.")

    beta = k / (k + n) * cov_gf / var_g
    mu_hat = np.mean(f - beta * g) + beta * theta_hat
    var_hat = (
        (var_f + beta**2 * var_g - 2.0 * beta * cov_gf) / n
        + beta**2 * var_g_unp / k
    )
    return CVResult(float(mu_hat), float(max(var_hat, 1e-10)), float(beta))


# ---------------------------------------------------------------------------
# Per-point worker
# ---------------------------------------------------------------------------

def _cv_worker(
    local_i: int,
    global_idx: int,
    paired_nbrs: np.ndarray,
    unpaired_nbrs: np.ndarray,
    fallback_nbrs: np.ndarray,
    target_pred_all: np.ndarray,
    proxy_pred_all: np.ndarray,
) -> dict:
    paired_nbrs = np.asarray(paired_nbrs, dtype=int)
    unpaired_nbrs = np.asarray(unpaired_nbrs, dtype=int)
    fallback_nbrs = np.asarray(fallback_nbrs, dtype=int)

    f = target_pred_all[paired_nbrs].astype(float)
    g = proxy_pred_all[paired_nbrs].astype(float)
    g_unp = proxy_pred_all[unpaired_nbrs].astype(float)

    # If paired proxy values are degenerate, try fallback neighbours.
    if len(g) < 2 or np.isclose(np.var(g), 0.0):
        if len(fallback_nbrs) >= 2:
            m = min(len(f), len(fallback_nbrs))
            fb = fallback_nbrs[:m]
            f = target_pred_all[fb].astype(float)
            g = proxy_pred_all[fb].astype(float)
        else:
            return dict(
                local_i=local_i,
                global_idx=int(global_idx),
                cv_mean=np.nan,
                cv_var=np.nan,
                beta=np.nan,
                cv_corr=np.nan,
            )

    n = min(len(f), len(g))
    f, g = f[:n], g[:n]
    if len(g_unp) < 2:
        g_unp = g.copy()

    cv_corr = (
        0.0
        if np.std(f) < 1e-12 or np.std(g) < 1e-12
        else float(np.corrcoef(f, g)[0, 1])
    )

    try:
        res = control_variates_estimator(f=f, g=g, g_unpaired=g_unp)
        return dict(
            local_i=local_i,
            global_idx=int(global_idx),
            cv_mean=res.mu_hat,
            cv_var=res.var_mu_hat,
            beta=res.beta,
            cv_corr=cv_corr,
        )
    except Exception:
        return dict(
            local_i=local_i,
            global_idx=int(global_idx),
            cv_mean=np.nan,
            cv_var=np.nan,
            beta=np.nan,
            cv_corr=cv_corr,
        )


# ---------------------------------------------------------------------------
# Parallel CV estimation over a query set
# ---------------------------------------------------------------------------

def compute_local_cv_parallel(
    target_model,
    target_ckpt: dict,
    X_query: np.ndarray,
    query_global_idx: np.ndarray,
    X_target: np.ndarray,
    proxy_path: str,
    n_pair_local: int = 10,
    k_unpaired: int = 20,
    radius: float = 1.5,
    fallback_k: int = 80,
    n_jobs: int = -1,
    device: str = "cuda",
    batch_size: int = 4096,
) -> tuple:
    """
    Compute deterministic MLP-based CV estimates for every point in X_query.

    Parameters
    ----------
    target_model, target_ckpt:
        Target/hardware MLP and checkpoint metadata.
    X_query:
        MI-selected query candidates, shape (N_query, d).
    query_global_idx:
        Global indices for X_query if they are part of X_target; pass -1 for
        externally sampled candidates.
    X_target:
        Current target-support commands used to define local neighborhoods,
        shape (N_target, d).
    proxy_path:
        Path to the proxy/sim MLP checkpoint. The proxy model is loaded inside
        this function using load_mlp_checkpoint(proxy_path, device=str(device)).

    Returns
    -------
    cv_mean : (N_query,)
    cv_var  : (N_query,)
    beta    : (N_query,)
    cv_corr : (N_query,)
    """
    X_query = np.asarray(X_query, dtype=float)
    X_target = np.asarray(X_target, dtype=float)
    query_global_idx = np.asarray(query_global_idx, dtype=int)
    N = len(X_query)

    if N == 0:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, empty
    if len(X_target) == 0:
        raise ValueError("X_target must contain at least one support point.")

    device_str = str(device)

    proxy_model, proxy_ckpt, _ = load_mlp_checkpoint(proxy_path, device=device_str)

    target_pred_all, _ = predict_mlp(
        target_model,
        target_ckpt,
        X_target,
        device=device_str,
        mc_dropout=False,
    )
    proxy_pred_all, _ = predict_mlp(
        proxy_model,
        proxy_ckpt,
        X_target,
        device=device_str,
        mc_dropout=False,
    )

    target_pred_all = np.asarray(target_pred_all, dtype=float).reshape(-1)
    proxy_pred_all = np.asarray(proxy_pred_all, dtype=float).reshape(-1)

    if len(target_pred_all) != len(X_target) or len(proxy_pred_all) != len(X_target):
        raise ValueError("MLP prediction length does not match X_target length.")

    # Radius neighbours for paired / unpaired local proxy-target samples.
    nn_r = NearestNeighbors(radius=radius, n_jobs=-1).fit(X_target)
    rad_nbr = nn_r.radius_neighbors(X_query, radius=radius, return_distance=False)

    k_tot = min(max(n_pair_local, k_unpaired, fallback_k), len(X_target))
    nn_k = NearestNeighbors(n_neighbors=k_tot, n_jobs=-1).fit(X_target)
    knn_nbr = nn_k.kneighbors(X_query, return_distance=False)

    rng = np.random.default_rng(0)
    paired_list, unpaired_list, fallback_list = [], [], []
    for i in range(N):
        neigh = np.asarray(rad_nbr[i], dtype=int)
        if 0 <= query_global_idx[i] < len(X_target):
            neigh = neigh[neigh != query_global_idx[i]]

        if len(neigh) < max(2, n_pair_local):
            neigh = np.asarray(knn_nbr[i], dtype=int)
            if 0 <= query_global_idx[i] < len(X_target):
                neigh = neigh[neigh != query_global_idx[i]]

        if len(neigh) == 0:
            neigh = np.asarray(knn_nbr[i], dtype=int)

        paired = rng.choice(neigh, n_pair_local, replace=len(neigh) < n_pair_local)
        unpaired = rng.choice(neigh, k_unpaired, replace=len(neigh) < k_unpaired)

        fallback = np.asarray(knn_nbr[i], dtype=int)
        if 0 <= query_global_idx[i] < len(X_target):
            fallback = fallback[fallback != query_global_idx[i]]

        paired_list.append(paired)
        unpaired_list.append(unpaired)
        fallback_list.append(fallback)

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_cv_worker)(
            i,
            query_global_idx[i],
            paired_list[i],
            unpaired_list[i],
            fallback_list[i],
            target_pred_all,
            proxy_pred_all,
        )
        for i in range(N)
    )

    cv_mean = np.array([r["cv_mean"] for r in results], float)
    cv_var = np.array([r["cv_var"] for r in results], float)
    beta = np.array([r["beta"] for r in results], float)
    cv_corr = np.array([r["cv_corr"] for r in results], float)

    for arr, default in [
        (cv_mean, 0.0),
        (cv_var, 1.0),
        (beta, 0.0),
        (cv_corr, 0.0),
    ]:
        bad = ~np.isfinite(arr)
        if bad.any():
            good = np.isfinite(arr)
            arr[bad] = np.nanmedian(arr[good]) if good.any() else default

    return cv_mean, cv_var, beta, cv_corr


# Backward-compatible alias expected by scout_utils.py.
compute_cv_parallel = compute_local_cv_parallel
