#!/usr/bin/env python3
"""
utils/quadruped/scout_utils.py
---------------------------------
SCOUT-style acquisition utilities for Go2 velocity commands.

Changes vs original
--------------------
1. Elbow method for automatic K selection in MI clustering.
   K is chosen as the sharpest second-difference of KMeans inertia.
   The n_clusters field in ScoutConfig is now used as k_max for the search.

2. Distance-based prior p(Z=z_k|x) — Option A, no pi_new hyperparameter:
       w_k(x) = exp(-||x - c_k||² / (2 l_k²))
       p(Z=z_k|x)   = w_k / (W + 1)
       p(Z=z_new|x) = 1   / (W + 1)
   where l_k = RMS within-cluster spread and W = Σ_k w_k.
   pi_new field in ScoutConfig is kept for backward compat but ignored.

3. Cluster-and-pick batch selection:
   After CV estimation, KMeans the MI shortlist into n_cv_select clusters
   and pick the highest CV point from each cluster.  This guarantees
   spatial diversity in the selected batch.
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


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

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
    vx  = rng.uniform(vx_range[0], vx_range[1], size=n)
    vy  = rng.uniform(vy_range[0], vy_range[1], size=n)
    wz  = rng.uniform(wz_range[0], wz_range[1], size=n)
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


# ---------------------------------------------------------------------------
# Change 1 — Elbow method for automatic K
# ---------------------------------------------------------------------------

def _elbow_k(
    X:            np.ndarray,
    k_min:        int = 2,
    k_max:        int = 20,
    random_state: int = 0,
) -> int:
    """
    Select K via the largest second difference of KMeans inertia.

    Fits KMeans for k = k_min..k_max, records inertia, picks the k at
    the sharpest change of slope (the elbow).
    Falls back to k_min when support is too small or curve is flat.
    """
    M     = len(X)
    k_max = min(k_max, M - 1)
    k_min = max(k_min, 1)
    if k_max <= k_min:
        return max(k_min, 1)

    ks       = list(range(k_min, k_max + 1))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state,
                    n_init="auto", max_iter=200)
        km.fit(X)
        inertias.append(km.inertia_)

    inertias = np.array(inertias)
    if len(inertias) < 3:
        return ks[0]

    # Largest second difference = sharpest elbow
    d2     = np.diff(np.diff(inertias))   # length = len(ks) - 2
    return ks[int(np.argmax(d2)) + 1]


# ---------------------------------------------------------------------------
# Change 2 — Distance-based prior  p(Z=z_k|x)  (Option A, no pi_new)
# ---------------------------------------------------------------------------

def _prior_at(x: np.ndarray, cluster_info: list) -> np.ndarray:
    """
    Compute p(Z|x) using the pseudo-cluster formulation:

        w_k(x) = exp(-||x - c_k||² / (2 l_k²))
        W(x)   = Σ_k w_k(x)
        p(Z=z_k|x)   = w_k / (W + 1)
        p(Z=z_new|x) = 1   / (W + 1)

    Returns (K+1,) normalised vector.
    """
    K = len(cluster_info)
    if K == 0:
        return np.array([1.0])

    w = np.zeros(K, dtype=np.float64)
    for k, c in enumerate(cluster_info):
        d_sq = float(np.sum((x.astype(np.float64) - c["center"]) ** 2))
        l_k  = max(c["lengthscale"], 1e-6)
        w[k] = np.exp(min(0.0, -d_sq / (2.0 * l_k ** 2)))

    W     = float(w.sum())
    denom = W + 1.0
    pZ    = np.concatenate([w / denom, [1.0 / denom]])
    # Safety normalisation
    pZ    = np.clip(pZ, 0.0, None)
    s     = pZ.sum()
    return pZ / s if s > 1e-12 else np.ones_like(pZ) / len(pZ)


# ---------------------------------------------------------------------------
# MI implementation
# ---------------------------------------------------------------------------

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
    """
    Build per-cluster descriptors.  Adds 'lengthscale' = RMS within-cluster
    distance, used by the distance-based prior.
    """
    info = []
    for k in range(len(centers)):
        pts = X_support[labels == k]
        if len(pts) == 0:
            radius, scale, lengthscale = 1e-6, 1e-6, 1e-6
        else:
            dists       = np.linalg.norm(pts - centers[k], axis=1)
            radius      = max(float(np.quantile(dists, radius_quantile)), 1e-6)
            scale       = max(float(np.std(dists)) + 1e-6,               1e-6)
            lengthscale = max(float(np.sqrt(np.mean(dists ** 2) + 1e-12)), 1e-6)
        info.append(dict(
            cluster=k, center=centers[k], size=len(pts),
            radius=radius, scale=scale, lengthscale=lengthscale,
        ))
    return info


def _dist_to_ball(x, c):
    return max(0.0, float(np.linalg.norm(x - c["center"])) - c["radius"])


def _dmin(x, cluster_info):
    return min(_dist_to_ball(x, c) for c in cluster_info)


def _q_existing(x, c, eps_exist=0.04, tau_scale=1.3):
    d   = _dist_to_ball(x, c)
    tau = tau_scale * max(c["scale"], 1e-6)
    return float(eps_exist * (1.0 - np.exp(-0.5 * (d / max(tau, 1e-6)) ** 2)))


def _q_new(x, cluster_info, novelty_radius=0.25, gamma=0.10):
    return float(_sigmoid((_dmin(x, cluster_info) - novelty_radius) / gamma))


def _likelihood_R1(x, cluster_info, eps_exist, tau_scale, novelty_radius, gamma):
    q = [_q_existing(x, c, eps_exist, tau_scale) for c in cluster_info]
    q.append(_q_new(x, cluster_info, novelty_radius, gamma))
    return np.asarray(q, dtype=np.float64)


def _posterior(x, pZ, r, cluster_info, eps_exist, tau_scale, novelty_radius, gamma):
    q1 = _likelihood_R1(x, cluster_info, eps_exist, tau_scale, novelty_radius, gamma)
    lk = q1 if r == 1 else (1.0 - q1)
    return _normalize_prob(lk * pZ)

def compute_support_mi(
    X_support:       np.ndarray,
    X_query:         np.ndarray,
    n_clusters:      int   = 20,
    radius_quantile: float = 0.90,
    pi_new:          float = None,    # ignored — prior is parameter-free
    eps_exist:       float = 0.04,
    tau_scale:       float = 1.3,
    novelty_radius:  float = None,    # ignored — computed from data
    gamma:           float = None,    # ignored — computed from data
    random_state:    int   = 0,
) -> tuple:
    from sklearn.neighbors import NearestNeighbors

    X_support = np.asarray(X_support, dtype=np.float64)
    X_query   = np.asarray(X_query,   dtype=np.float64)

    if len(X_support) < 2:
        return (np.ones(len(X_query), dtype=np.float32),
                np.ones(len(X_query), dtype=np.float32))

    # ── Elbow K ───────────────────────────────────────────────────────────
    k_max = max(2, n_clusters)
    K     = _elbow_k(X_support, k_min=2, k_max=k_max,
                     random_state=random_state)
    K     = max(K, 1)

    km     = KMeans(n_clusters=K, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X_support)
    cluster_info = _build_cluster_info(
        X_support, labels, km.cluster_centers_, radius_quantile
    )

    # ── Data-adaptive novelty threshold ───────────────────────────────────
    n_nbrs   = min(2, len(X_support))
    nn       = NearestNeighbors(n_neighbors=n_nbrs).fit(X_support)
    dists, _ = nn.kneighbors(X_support)
    nn_dists = dists[:, -1]           # distance to nearest *other* support pt
    rho      = float(np.median(nn_dists))
    gam      = max(float(np.std(nn_dists)), 1e-6)
    # ─────────────────────────────────────────────────────────────────────

    mi_vals   = np.zeros(len(X_query), dtype=np.float64)
    dmin_vals = np.zeros(len(X_query), dtype=np.float64)

    for i, x in enumerate(X_query):
        pZ    = _prior_at(x, cluster_info)
        q1    = _likelihood_R1(x, cluster_info, eps_exist, tau_scale, rho, gam)
        p_r1  = float(np.sum(pZ * q1))
        p_r0  = 1.0 - p_r1
        post1 = _posterior(x, pZ, 1, cluster_info, eps_exist, tau_scale, rho, gam)
        post0 = _posterior(x, pZ, 0, cluster_info, eps_exist, tau_scale, rho, gam)
        mi    = p_r1 * _kl(post1, pZ) + p_r0 * _kl(post0, pZ)

        mi_vals[i]   = mi
        dmin_vals[i] = _dmin(x, cluster_info)

    return mi_vals.astype(np.float32), dmin_vals.astype(np.float32)
# def compute_support_mi(
#     X_support:       np.ndarray,
#     X_query:         np.ndarray,
#     # n_clusters now used as k_max for elbow search (0 = auto)
#     n_clusters:      int   = 20,
#     radius_quantile: float = 0.90,
#     # pi_new kept for backward compat — ignored (prior is parameter-free)
#     pi_new:          float = None,
#     eps_exist:       float = 0.04,
#     tau_scale:       float = 1.3,
#     novelty_radius:  float = 0.25,
#     gamma:           float = 0.10,
#     random_state:    int   = 0,
# ) -> tuple:
#     """
#     Compute support-state MI I(Z;R_x) over candidate commands.

#     Changes vs original
#     --------------------
#     1. K is selected automatically via elbow method (n_clusters = k_max).
#     2. Prior p(Z|x) is distance-based per query point — no shared pZ vector.

#     Returns: mi_vals, dmin_vals
#     """
#     X_support = np.asarray(X_support, dtype=np.float64)
#     X_query   = np.asarray(X_query,   dtype=np.float64)

#     if len(X_support) < 2:
#         return (np.ones(len(X_query), dtype=np.float32),
#                 np.ones(len(X_query), dtype=np.float32))

#     # ── Elbow K ───────────────────────────────────────────────────────────
#     k_max = max(2, n_clusters)
#     K     = _elbow_k(X_support, k_min=2, k_max=k_max,
#                      random_state=random_state)
#     K     = max(K, 1)

#     km     = KMeans(n_clusters=K, random_state=random_state, n_init="auto")
#     labels = km.fit_predict(X_support)
#     cluster_info = _build_cluster_info(
#         X_support, labels, km.cluster_centers_, radius_quantile
#     )

#     mi_vals   = np.zeros(len(X_query), dtype=np.float64)
#     dmin_vals = np.zeros(len(X_query), dtype=np.float64)

#     for i, x in enumerate(X_query):
#         # ── Distance-based prior at this specific x ────────────────────────
#         pZ = _prior_at(x, cluster_info)

#         q1    = _likelihood_R1(x, cluster_info, eps_exist, tau_scale,
#                                novelty_radius, gamma)
#         p_r1  = float(np.sum(pZ * q1))
#         p_r0  = 1.0 - p_r1
#         post1 = _posterior(x, pZ, 1, cluster_info, eps_exist, tau_scale,
#                            novelty_radius, gamma)
#         post0 = _posterior(x, pZ, 0, cluster_info, eps_exist, tau_scale,
#                            novelty_radius, gamma)
#         mi    = p_r1 * _kl(post1, pZ) + p_r0 * _kl(post0, pZ)

#         mi_vals[i]   = mi
#         dmin_vals[i] = _dmin(x, cluster_info)

#     return mi_vals.astype(np.float32), dmin_vals.astype(np.float32)


# ---------------------------------------------------------------------------
# Change 3 — Cluster-and-pick batch selection
# ---------------------------------------------------------------------------

def _cluster_and_pick(
    X_top:        np.ndarray,   # (S, d) MI shortlist features
    top_idx:      np.ndarray,   # (S,)   global pool indices
    cv_mean:      np.ndarray,   # (S,)   CV mean for each shortlist point
    n_select:     int,
    random_state: int = 0,
) -> tuple:
    """
    Select n_select spatially diverse candidates from the MI shortlist.

    Steps:
      1. KMeans shortlist into n_select clusters.
      2. From each cluster, pick the point with the highest CV mean.

    Returns
    -------
    best_local : (n_select,) local indices into top_idx / cv_mean
    best_idx   : (n_select,) global pool indices
    """
    S = len(top_idx)
    k = min(n_select, S)

    if k == 0:
        return np.array([], int), np.array([], int)

    if k >= S:
        # Shortlist already small — pick top-k by CV globally
        local = np.argsort(cv_mean)[::-1][:k]
        return local, top_idx[local]

    km = KMeans(n_clusters=k, random_state=random_state,
                n_init="auto", max_iter=300)
    cluster_labels = km.fit_predict(X_top)   # (S,)

    best_local = []
    for c in range(k):
        in_c = np.where(cluster_labels == c)[0]
        if len(in_c) == 0:
            continue
        best_local.append(int(in_c[np.argmax(cv_mean[in_c])]))

    best_local = np.array(best_local, int)
    return best_local, top_idx[best_local]


# ---------------------------------------------------------------------------
# ScoutConfig
# ---------------------------------------------------------------------------

@dataclass
class ScoutConfig:
    pool_size:   int   = 2000
    top_mi_frac: float = 0.12
    min_top:     int   = 500

    # n_clusters is now k_max for the elbow search
    n_clusters:      int   = 8
    radius_quantile: float = 0.90
    # pi_new kept for backward compat — no longer used
    pi_new:          float = 0.20
    eps_exist:       float = 0.04
    tau_scale:       float = 1.0
    novelty_radius:  float = 0.05
    gamma:           float = 1.0

    proxy_weight:              float = 1.0
    mi_weight:                 float = 0.5
    target_mean_weight:        float = 0.0
    target_uncertainty_weight: float = 0.25

    cv_n_pair_local: int   = 100
    cv_k_unpaired:   int   = 200
    cv_radius:       float = 0.1
    cv_fallback_k:   int   = 80
    cv_n_jobs:       int   = -1
    cv_batch_size:   int   = 4096
    n_cv_select:     int   = 5
    proxy_path:      str | None = None

    min_dist_from_existing: float = 1e-3
    seed:                   int   = 0

    vx_min: float = -0.4
    vx_max: float = 1.0
    vy_min: float = -0.8
    vy_max: float = 0.8
    wz_min: float = -0.8
    wz_max: float = 0.8


# ---------------------------------------------------------------------------
# Main acquisition function
# ---------------------------------------------------------------------------

def select_next_candidate(
    target_model,
    target_ckpt: dict,
    proxy_model,
    proxy_ckpt:  dict | None,
    X_target:    np.ndarray,
    X_sim:       np.ndarray,
    cfg:         ScoutConfig,
    device       = None,
    proxy_path:  str | None = None,
) -> dict:
    """
    Select the next command candidates for hardware evaluation.

    Returns dict with the best candidate (backward-compat) plus
    top-n_cv_select candidates selected via cluster-and-pick.
    """
    pool = sample_command_pool(
        n        = cfg.pool_size,
        vx_range = (cfg.vx_min, cfg.vx_max),
        vy_range = (cfg.vy_min, cfg.vy_max),
        wz_range = (cfg.wz_min, cfg.wz_max),
        seed     = cfg.seed,
    )
    pool = remove_near_existing(pool, X_target,
                                min_dist=cfg.min_dist_from_existing)

    # ── MI (elbow K, distance-based prior) ───────────────────────────────
    mi_vals, dmin_vals = compute_support_mi(
        X_support       = X_target,
        X_query         = pool,
        n_clusters      = cfg.n_clusters,    # used as k_max
        radius_quantile = cfg.radius_quantile,
        eps_exist       = cfg.eps_exist,
        tau_scale       = cfg.tau_scale,
        novelty_radius  = cfg.novelty_radius,
        gamma           = cfg.gamma,
        random_state    = cfg.seed,
    )

    valid = np.where(np.isfinite(mi_vals))[0]
    if len(valid) == 0:
        raise ValueError("No valid candidates after MI computation.")

    n_top   = max(cfg.min_top, int(cfg.top_mi_frac * len(valid)))
    n_top   = min(n_top, len(valid))
    top_idx = valid[np.argsort(mi_vals[valid])[::-1][:n_top]]
    X_top   = pool[top_idx]
    mi_top  = mi_vals[top_idx]

    # ── MI scatter plot ────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(5, 15))
    axes[0].scatter(X_top[:, 0], X_top[:, 1], c=mi_top); axes[0].scatter(X_target[:, 0], X_target[:, 1], color='red', marker='*'); axes[0].set_title('Vx and Vy')
    axes[1].scatter(X_top[:, 0], X_top[:, 2], c=mi_top); axes[1].scatter(X_target[:, 0], X_target[:, 2], color='red', marker='*'); axes[1].set_title('Vx and Wz')
    axes[2].scatter(X_top[:, 1], X_top[:, 2], c=mi_top); axes[2].scatter(X_target[:, 1], X_target[:, 2], color='red', marker='*'); axes[2].set_title('Vy and Wz')
    plt.savefig('scripts/quadruped/mi_running.png')
    plt.close(fig)

    # ── CV on MI shortlist ────────────────────────────────────────────────
    active_proxy_path = proxy_path if proxy_path is not None else cfg.proxy_path
    if active_proxy_path is None:
        raise ValueError(
            "proxy_path must be provided via select_next_candidate(..., proxy_path=...) "
            "or cfg.proxy_path."
        )

    query_global_idx = -np.ones(len(X_top), dtype=int)
    device_str       = str(device) if device is not None else "cuda"

    cv_mean, cv_var, cv_beta, cv_corr = compute_cv_parallel(
        target_model     = target_model,
        target_ckpt      = target_ckpt,
        X_query          = X_top,
        query_global_idx = query_global_idx,
        X_target         = pool,
        proxy_path       = active_proxy_path,
        n_pair_local     = cfg.cv_n_pair_local,
        k_unpaired       = cfg.cv_k_unpaired,
        radius           = cfg.cv_radius,
        fallback_k       = cfg.cv_fallback_k,
        n_jobs           = cfg.cv_n_jobs,
        device           = device_str,
        batch_size       = cfg.cv_batch_size,
    )
    cv_corr = cv_corr[cv_mean>=0.6]
    top_idx = top_idx[cv_mean>=0.6]
    X_top = pool[top_idx]
    cv_mean = cv_mean[cv_mean>=0.6]
    score = np.asarray(cv_mean, dtype=float)
 

    # ── CV scatter plot ────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(5, 15))
    im = axes[0].scatter(X_top[:, 0], X_top[:, 1], c=cv_mean); axes[0].scatter(X_target[:, 0], X_target[:, 1], color='red', marker='*'); axes[0].set_title('Vx and Vy')
    axes[1].scatter(X_top[:, 0], X_top[:, 2], c=cv_mean); axes[1].scatter(X_target[:, 0], X_target[:, 2], color='red', marker='*'); axes[1].set_title('Vx and Wz')
    axes[2].scatter(X_top[:, 1], X_top[:, 2], c=cv_mean); axes[2].scatter(X_target[:, 1], X_target[:, 2], color='red', marker='*'); axes[2].set_title('Vy and Wz')
    plt.colorbar(im)
    plt.savefig('scripts/quadruped/cv_mean.png')
    plt.close(fig)

   

    # ── Change 3: cluster-and-pick selection ─────────────────────────────
    finite_score = np.isfinite(score)
    if not finite_score.any():
        raise ValueError("CV scoring failed: all CV scores are non-finite.")
    score_clean = score.copy()
    score_clean[~finite_score] = -np.inf

    n_select = max(1, min(int(cfg.n_cv_select), len(score)))

    best_local, best_idx = _cluster_and_pick(
        X_top        = X_top,
        top_idx      = top_idx,
        cv_mean      = score_clean,
        n_select     = n_select,
        random_state = cfg.seed,
    )


    candidates = pool[best_idx]
    candidate  = candidates[0]
    print("CANDIDATES", candidates)

    # ── Surrogate predictions for diagnostics ─────────────────────────────
    if proxy_model is not None and proxy_ckpt is not None:
        proxy_mean, _ = predict_mlp(
            proxy_model, proxy_ckpt, candidates,
            device=device, mc_dropout=False,
        )
    else:
        proxy_mean = np.full(len(candidates), np.nan, dtype=float)

    target_mean, target_std = predict_mlp(
        target_model, target_ckpt, candidates,
        device=device, mc_dropout=True, n_mc=64,
    )

    proxy_mean  = np.asarray(proxy_mean).reshape(-1)
    target_mean = np.asarray(target_mean).reshape(-1)
    target_std  = np.asarray(target_std).reshape(-1)

    return {
        # Backward-compatible single best
        "candidate":           candidates,
        "candidate_vx":        float(candidate[0]),
        "candidate_vy":        float(candidate[1]),
        "candidate_wz":        float(candidate[2]),
        "score":               float(score_clean[best_local[0]]),
        "mi":                  float(mi_vals[best_idx[0]]),
        "dmin":                float(dmin_vals[best_idx[0]]),
        "proxy_pred":          float(proxy_mean[0]),
        "target_pred":         float(target_mean[0]),
        "target_uncertainty":  float(target_std[0]),
        # Top-k cluster-and-pick results
        "candidates":          candidates,
        "candidate_indices":   best_idx,
        "candidate_scores":    score_clean[best_local],
        "candidate_cv_mean":   cv_mean[best_local],
        "candidate_cv_var":    cv_var[best_local],
        "candidate_cv_beta":   cv_beta[best_local],
        "candidate_cv_corr":   cv_corr[best_local],
        "candidate_mi":        mi_vals[best_idx],
        "candidate_dmin":      dmin_vals[best_idx],
        # Full diagnostics
        "pool":         pool,
        "mi_vals":      mi_vals,
        "dmin_vals":    dmin_vals,
        "top_idx":      top_idx,
        "top_scores":   score,
        "top_cv_mean":  cv_mean,
        "top_cv_var":   cv_var,
        "top_cv_beta":  cv_beta,
        "top_cv_corr":  cv_corr,
    }
