#!/usr/bin/env python3

"""
SCOUT-style acquisition utilities for Go2 velocity commands.

Inputs:
  - target/hardware MLP trained on collected hardware data
  - proxy/sim MLP trained on the larger sim CSV
  - current target X support
  - candidate command pool

Selection:
  1. Compute support-novelty MI over candidate pool.
  2. Shortlist valid high-MI points.
  3. Estimate local CV values on the MI-selected shortlist.
  4. Return the top-2 command candidates by CV value.

This follows the MI-shortlist + proxy/CV-selection shape of run_mi_cv.py,
where MI is computed first and CV is evaluated on the high-MI shortlist.
Here, proxy MLP predictions replace local CV proxy samples. The MI helper
mirrors the KMeans support-state mechanism in mi_toy.py. :contentReference[oaicite:1]{index=1}
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans

from utils.quadruped.train_utils import predict_mlp

try:
    from utils.quadruped.cv_utils import compute_cv_parallel
except ImportError:
    from utils.quadruped.cv_utils import compute_local_cv_parallel as compute_cv_parallel

import matplotlib.pyplot as plt
# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

def normalize01(x, eps: float = 1e-8):
    x = np.asarray(x, dtype=np.float64)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if hi - lo < eps:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def sample_command_pool(
    n: int,
    vx_range=(0.0, 0.8),
    vy_range=(-0.2, 0.2),
    wz_range=(-0.5, 0.5),
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    vx = rng.uniform(vx_range[0], vx_range[1], size=n)
    vy = rng.uniform(vy_range[0], vy_range[1], size=n)
    wz = rng.uniform(wz_range[0], wz_range[1], size=n)

    return np.stack([vx, vy, wz], axis=-1).astype(np.float32)


def remove_near_existing(
    pool_X: np.ndarray,
    X_existing: np.ndarray,
    min_dist: float = 1e-3,
) -> np.ndarray:
    if len(X_existing) == 0:
        return pool_X

    keep = []
    for x in pool_X:
        d = np.linalg.norm(X_existing - x[None, :], axis=1)
        keep.append(np.min(d) > min_dist)

    return pool_X[np.asarray(keep, dtype=bool)]


# ---------------------------------------------------------------------
# MI implementation
# ---------------------------------------------------------------------

def _normalize_prob(p, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    s = p.sum()
    return p / s if s >= eps else np.ones_like(p) / len(p)


def _kl(p, q, eps=1e-12):
    p = np.clip(_normalize_prob(p, eps), eps, 1.0)
    q = np.clip(_normalize_prob(q, eps), eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def _build_cluster_info(X_support, labels, centers, radius_quantile=0.90):
    info = []
    for k in range(len(centers)):
        pts = X_support[labels == k]
        if len(pts) == 0:
            radius, scale = 1e-6, 1e-6
        else:
            dists = np.linalg.norm(pts - centers[k], axis=1)
            radius = max(float(np.quantile(dists, radius_quantile)), 1e-6)
            scale = max(float(np.std(dists) + 1e-6), 1e-6)
        info.append(
            {
                "cluster": k,
                "center": centers[k],
                "size": len(pts),
                "radius": radius,
                "scale": scale,
            }
        )
    return info


def _dist_to_ball(x, c):
    return max(0.0, float(np.linalg.norm(x - c["center"])) - c["radius"])


def _dmin(x, cluster_info):
    return min(_dist_to_ball(x, c) for c in cluster_info)


def _q_existing(x, c, eps_exist=0.04, tau_scale=1.3):
    d = _dist_to_ball(x, c)
    tau = tau_scale * max(c["scale"], 1e-6)
    return float(eps_exist * (1.0 - np.exp(-0.5 * (d / max(tau, 1e-6)) ** 2)))


def _q_new(x, cluster_info, novelty_radius=0.25, gamma=0.10):
    return float(_sigmoid((_dmin(x, cluster_info) - novelty_radius) / gamma))


def _likelihood_R1(x, cluster_info, eps_exist, tau_scale, novelty_radius, gamma):
    q = [_q_existing(x, c, eps_exist, tau_scale) for c in cluster_info]
    q.append(_q_new(x, cluster_info, novelty_radius, gamma))
    return np.asarray(q, dtype=np.float64)


def _posterior(x, r, pZ, cluster_info, eps_exist, tau_scale, novelty_radius, gamma):
    q1 = _likelihood_R1(x, cluster_info, eps_exist, tau_scale, novelty_radius, gamma)
    lk = q1 if r == 1 else (1.0 - q1)
    return _normalize_prob(lk * pZ)


def compute_support_mi(
    X_support: np.ndarray,
    X_query: np.ndarray,
    n_clusters: int = 5,
    radius_quantile: float = 0.90,
    pi_new: float = 0.20,
    eps_exist: float = 0.04,
    tau_scale: float = 1.3,
    novelty_radius: float = 0.25,
    gamma: float = 0.10,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute support-state MI I(Z;R_x) over candidate commands.

    Returns:
      mi_vals, dmin_vals
    """
    X_support = np.asarray(X_support, dtype=np.float32)
    X_query = np.asarray(X_query, dtype=np.float32)

    if len(X_support) < 2:
        # With too little support, prefer far/spread points by a flat MI.
        return np.ones(len(X_query), dtype=np.float32), np.ones(len(X_query), dtype=np.float32)

    n_eff = min(n_clusters, len(X_support))
    km = KMeans(n_clusters=n_eff, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X_support)

    cluster_info = _build_cluster_info(
        X_support,
        labels,
        km.cluster_centers_,
        radius_quantile=radius_quantile,
    )

    sizes = np.asarray([max(1, c["size"]) for c in cluster_info], dtype=float)
    pZ = _normalize_prob(
        np.concatenate([(1.0 - pi_new) * sizes / sizes.sum(), [pi_new]])
    )

    mi_vals = np.zeros(len(X_query), dtype=np.float64)
    dmin_vals = np.zeros(len(X_query), dtype=np.float64)

    for i, x in enumerate(X_query):
        q1 = _likelihood_R1(x, cluster_info, eps_exist, tau_scale, novelty_radius, gamma)
        p_r1 = float(np.sum(pZ * q1))
        p_r0 = 1.0 - p_r1

        post1 = _posterior(x, 1, pZ, cluster_info, eps_exist, tau_scale, novelty_radius, gamma)
        post0 = _posterior(x, 0, pZ, cluster_info, eps_exist, tau_scale, novelty_radius, gamma)

        mi = p_r1 * _kl(post1, pZ) + p_r0 * _kl(post0, pZ)

        mi_vals[i] = mi
        dmin_vals[i] = _dmin(x, cluster_info)

    return mi_vals.astype(np.float32), dmin_vals.astype(np.float32)


# ---------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------

@dataclass
class ScoutConfig:
    pool_size: int = 2000
    top_mi_frac: float = 0.12
    min_top: int = 100

    n_clusters: int = 8
    radius_quantile: float = 0.90
    pi_new: float = 0.20
    eps_exist: float = 0.04
    tau_scale: float = 1.0
    novelty_radius: float = 0.05
    gamma: float = 1.0

    proxy_weight: float = 1.0
    mi_weight: float = 0.5
    target_mean_weight: float = 0.0
    target_uncertainty_weight: float = 0.25

    cv_n_pair_local: int = 100
    cv_k_unpaired: int = 200
    cv_radius: float = 0.1
    cv_fallback_k: int = 80
    cv_n_jobs: int = -1
    cv_batch_size: int = 4096
    n_cv_select: int = 2
    proxy_path: str | None = None

    min_dist_from_existing: float = 1e-3
    seed: int = 0

    vx_min: float = 0.0
    vx_max: float = 0.8
    vy_min: float = -0.2
    vy_max: float = 0.2
    wz_min: float = -0.5
    wz_max: float = 0.5


def select_next_candidate(
    target_model,
    target_ckpt: dict,
    proxy_model,
    proxy_ckpt: dict | None,
    X_target: np.ndarray,
    X_sim:np.array,
    cfg: ScoutConfig,
    device=None,
    proxy_path: str | None = None,
) -> dict:
    """
    Select the next command candidates for hardware evaluation.

    X_target:
      current hardware-evaluated command set, shape (N, 3)

    Returns dict with the best candidate for backward compatibility, plus
    the top-2 CV-selected candidates and diagnostics.
    """
    pool = sample_command_pool(
        n=cfg.pool_size,
        vx_range=(cfg.vx_min, cfg.vx_max),
        vy_range=(cfg.vy_min, cfg.vy_max),
        wz_range=(cfg.wz_min, cfg.wz_max),
        seed=cfg.seed,
    )
    pool = remove_near_existing(pool, X_target, min_dist=cfg.min_dist_from_existing)
    mi_vals, dmin_vals = compute_support_mi(
        X_support=X_target,
        X_query=pool,
        n_clusters=cfg.n_clusters,
        radius_quantile=cfg.radius_quantile,
        pi_new=cfg.pi_new,
        eps_exist=cfg.eps_exist,
        tau_scale=cfg.tau_scale,
        novelty_radius=cfg.novelty_radius,
        gamma=cfg.gamma,
        random_state=cfg.seed,
    )

    valid = np.where(np.isfinite(mi_vals))[0]
    if len(valid) == 0:
        raise ValueError("No valid candidates after MI computation.")

    n_top = max(cfg.min_top, int(cfg.top_mi_frac * len(valid)))
    n_top = min(n_top, len(valid))

    # MI shortlist: only valid candidates are passed to the CV stage.
    top_idx = valid[np.argsort(mi_vals[valid])[::-1][:n_top]]
    mi_value = mi_vals[top_idx]
    X_top = pool[top_idx]
    fig,axes=plt.subplots(3,1,figsize=(5,15))
    axes[0].scatter(X_top[:,0],X_top[:,1],c=mi_value)
    axes[0].scatter(X_target[:,0],X_target[:,1],color='red',marker='*')
    axes[0].set_title('Vx and Vy')

    axes[1].scatter(X_top[:,0],X_top[:,2],c=mi_value)
    axes[1].scatter(X_target[:,0],X_target[:,2],color='red',marker='*')
    axes[1].set_title('Vx and Wz')

    axes[2].scatter(X_top[:,1],X_top[:,2],c=mi_value)
    axes[2].scatter(X_target[:,1],X_target[:,2],color='red',marker='*')
    axes[2].set_title('Vy and Wz')
    plt.savefig('scripts/quadruped/mi_running.png')
    # CV stage: use deterministic MLP predictions. The CV utility loads
    # the proxy/sim MLP from proxy_path and computes local paired/unpaired
    # proxy predictions internally.
    
    active_proxy_path = proxy_path if proxy_path is not None else cfg.proxy_path
    if active_proxy_path is None:
        raise ValueError(
            "proxy_path must be provided either as select_next_candidate(..., proxy_path=...) "
            "or cfg.proxy_path so compute_cv_parallel can load the proxy MLP."
        )

    query_global_idx = -np.ones(len(X_top), dtype=int)
    device_str = str(device) if device is not None else "cuda"

    cv_mean, cv_var, cv_beta, cv_corr = compute_cv_parallel(
        target_model=target_model,
        target_ckpt=target_ckpt,
        X_query=X_top,
        query_global_idx=query_global_idx,
        X_target=pool,
        proxy_path=active_proxy_path,
        n_pair_local=cfg.cv_n_pair_local,
        k_unpaired=cfg.cv_k_unpaired,
        radius=cfg.cv_radius,
        fallback_k=cfg.cv_fallback_k,
        n_jobs=cfg.cv_n_jobs,
        device=device_str,
        batch_size=cfg.cv_batch_size,
    )

    score = np.asarray(cv_mean, dtype=float)

    fig,axes=plt.subplots(3,1,figsize=(5,15))
    axes[0].scatter(X_top[:,0],X_top[:,1],c=cv_mean)
    axes[0].scatter(X_target[:,0],X_target[:,1],color='red',marker='*')
    axes[0].set_title('Vx and Vy')

    
    axes[1].scatter(X_top[:,0],X_top[:,2],c=cv_mean)
    axes[1].scatter(X_target[:,0],X_target[:,2],color='red',marker='*')
    axes[1].set_title('Vx and Wz')

    axes[2].scatter(X_top[:,1],X_top[:,2],c=cv_mean)
    axes[2].scatter(X_target[:,1],X_target[:,2],color='red',marker='*')
    axes[2].set_title('Vy and Wz')
    plt.savefig('scripts/quadruped/cv_mean.png')

    finite_score = np.isfinite(score)
    if not finite_score.any():
        raise ValueError("CV scoring failed: all CV scores are non-finite.")

    # Keep invalid CV estimates from being selected.
    score = score.copy()
    score[~finite_score] = -np.inf

    n_select = min(int(cfg.n_cv_select), len(score))
    n_select = max(1, n_select)
    n_select=4
    best_local = np.argsort(score)[::-1][:n_select]
    best_idx = top_idx[best_local]
    candidates = pool[best_idx]
    candidate = candidates[0]
    print("CANDIDATES",candidates)

    if proxy_model is not None and proxy_ckpt is not None:
        proxy_mean, _ = predict_mlp(
            proxy_model,
            proxy_ckpt,
            candidates,
            device=device,
            mc_dropout=False,
        )
    else:
        proxy_mean = np.full(len(candidates), np.nan, dtype=float)

    target_mean, target_std = predict_mlp(
        target_model,
        target_ckpt,
        candidates,
        device=device,
        mc_dropout=True,
        n_mc=64,
    )

    proxy_mean = np.asarray(proxy_mean).reshape(-1)
    target_mean = np.asarray(target_mean).reshape(-1)
    target_std = np.asarray(target_std).reshape(-1)

    return {
        # Backward-compatible single best candidate.
        "candidate": candidate,
        "candidate_vx": float(candidate[0]),
        "candidate_vy": float(candidate[1]),
        "candidate_wz": float(candidate[2]),
        "score": float(score[best_local[0]]),
        "mi": float(mi_vals[best_idx[0]]),
        "dmin": float(dmin_vals[best_idx[0]]),
        "proxy_pred": float(proxy_mean[0]),
        "target_pred": float(target_mean[0]),
        "target_uncertainty": float(target_std[0]),

        # New top-k CV-selected result.
        "candidates": candidates,
        "candidate_indices": best_idx,
        "candidate_scores": score[best_local],
        "candidate_cv_mean": cv_mean[best_local],
        "candidate_cv_var": cv_var[best_local],
        "candidate_cv_beta": cv_beta[best_local],
        "candidate_cv_corr": cv_corr[best_local],
        "candidate_mi": mi_vals[best_idx],
        "candidate_dmin": dmin_vals[best_idx],

        # Full diagnostics.
        "pool": pool,
        "mi_vals": mi_vals,
        "dmin_vals": dmin_vals,
        "top_idx": top_idx,
        "top_scores": score,
        "top_cv_mean": cv_mean,
        "top_cv_var": cv_var,
        "top_cv_beta": cv_beta,
        "top_cv_corr": cv_corr,
    }
