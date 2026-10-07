#!/usr/bin/env python3
"""
scout/mi.py
-------------------------
Novelty-oriented support-state mutual information estimator.

Implements  I(Z ; R_x | D_t, x)  in high-dimensional embedding space.

Design
------
1. K chosen automatically via elbow method — no n_clusters hyperparameter.

2. Prior p(Z | x, D_t) is fully data-driven (Option A — no pi_new):

       w_k(x) = exp( -‖x − c_k‖² / (2 l_k²) )   k = 1..K
       W(x)   = Σ_k w_k(x)

       p(Z = z_k   | x) = w_k(x) / (W(x) + 1)
       p(Z = z_new | x) = 1      / (W(x) + 1)

   z_new is a pseudo-cluster with unit weight in the original scale.
   When x is near a cluster:   W >> 1  →  p(z_new) ≈ 0  (well-explained)
   When x is far from all:     W << 1  →  p(z_new) ≈ 1  (novel)

   l_k = RMS within-cluster spread (data-adaptive lengthscale per cluster).

3. Batch selection is cluster-and-pick in scout_baseline.py.

No pi_new, no tau_scale_prior.  Only likelihood parameters remain:
  eps_exist, tau_scale, novelty_radius, gamma.

All functions are pure numpy — no torch dependency.
"""

import numpy as np
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Probability utilities
# ---------------------------------------------------------------------------

def _normalize(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    s = p.sum()
    return p / s if s >= eps else np.ones_like(p) / len(p)


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(_normalize(p, eps), eps, 1.0)
    q = np.clip(_normalize(q, eps), eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


# ---------------------------------------------------------------------------
# Elbow method — automatic K selection
# ---------------------------------------------------------------------------

def elbow_k(
    X:            np.ndarray,
    k_min:        int = 2,
    k_max:        int = 20,
    random_state: int = 0,
) -> int:
    """
    Select K via the largest second difference of KMeans inertia.

    Fits KMeans for k = k_min..k_max, records inertia, picks the k at
    the sharpest change of slope (the "elbow").  Falls back to k_min
    if the support is too small or the curve is flat.
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

    # Second difference: largest value = sharpest elbow
    d2     = np.diff(np.diff(inertias))   # length = len(ks) - 2
    # d2[i] corresponds to ks[i+1]
    return ks[int(np.argmax(d2)) + 1]


# ---------------------------------------------------------------------------
# Cluster descriptors
# ---------------------------------------------------------------------------

def build_cluster_info(
    X_support:       np.ndarray,
    labels:          np.ndarray,
    centers:         np.ndarray,
    radius_quantile: float = 0.90,
) -> list:
    """
    Build per-cluster descriptor dicts with keys:
        cluster, center, size, radius, scale, lengthscale.

    radius      : quantile of within-cluster distances (ball boundary)
    scale       : std of within-cluster distances (likelihood decay)
    lengthscale : RMS of within-cluster distances  l_k = sqrt(mean ‖x_i-c_k‖²)
                  Used in the distance-based prior.
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


def _dist_to_ball(x: np.ndarray, c: dict) -> float:
    return max(0.0, float(np.linalg.norm(x - c["center"])) - c["radius"])


def _dmin(x: np.ndarray, cluster_info: list) -> float:
    return min(_dist_to_ball(x, c) for c in cluster_info)


# ---------------------------------------------------------------------------
# Option A — fully data-driven prior  (no pi_new)
# ---------------------------------------------------------------------------

def _prior(x: np.ndarray, cluster_info: list) -> np.ndarray:
    """
    Compute p(Z | x) using the pseudo-cluster formulation.

        w_k(x) = exp( -‖x − c_k‖² / (2 l_k²) )
        W(x)   = Σ_k w_k(x)

        p(Z = z_k   | x) = w_k / (W + 1)
        p(Z = z_new | x) = 1   / (W + 1)

    Returns (K+1,) normalised vector.
    """
    K = len(cluster_info)
    if K == 0:
        return np.array([1.0])

    w = np.zeros(K, dtype=np.float64)
    for k, c in enumerate(cluster_info):
        d_sq = float(np.sum((x - c["center"]) ** 2))
        l_k  = max(c["lengthscale"], 1e-6)
        # clip to avoid overflow — max exponent 0 (when d=0)
        w[k] = np.exp(min(0.0, -d_sq / (2.0 * l_k ** 2)))

    W     = float(w.sum())
    denom = W + 1.0                         # pseudo-cluster has weight 1

    pZ    = np.concatenate([w / denom, [1.0 / denom]])
    return _normalize(pZ)                   # safety normalisation


# ---------------------------------------------------------------------------
# Likelihood  P(R_x = 1 | Z = z)
# ---------------------------------------------------------------------------

def _q_existing(x, c, eps_exist=0.04, tau_scale=1.3) -> float:
    d   = _dist_to_ball(x, c)
    tau = tau_scale * max(c["scale"], 1e-6)
    return float(eps_exist * (1.0 - np.exp(-0.5 * (d / tau) ** 2)))


def _q_new(x, cluster_info, novelty_radius=0.75, gamma=0.25) -> float:
    return float(_sigmoid((_dmin(x, cluster_info) - novelty_radius) / gamma))


def _likelihood_R1(x, cluster_info, eps_exist, tau_scale, novelty_radius, gamma):
    q = [_q_existing(x, c, eps_exist, tau_scale) for c in cluster_info]
    q.append(_q_new(x, cluster_info, novelty_radius, gamma))
    return np.asarray(q, dtype=np.float64)


def _posterior(x, r, pZ, cluster_info, eps_exist, tau_scale, novelty_radius, gamma):
    q1 = _likelihood_R1(x, cluster_info, eps_exist, tau_scale, novelty_radius, gamma)
    lk = q1 if r == 1 else (1.0 - q1)
    return _normalize(lk * pZ)


# ---------------------------------------------------------------------------
# MI for a single point
# ---------------------------------------------------------------------------

def _mi_single(
    x, cluster_info, eps_exist, tau_scale, novelty_radius, gamma,
) -> dict:
    """
    pZ is re-computed at x via _prior() so it adapts to the query location.
    """
    pZ    = _prior(x, cluster_info)
    q1    = _likelihood_R1(x, cluster_info, eps_exist, tau_scale,
                           novelty_radius, gamma)
    p_r1  = float(np.sum(pZ * q1))
    p_r0  = 1.0 - p_r1
    post1 = _posterior(x, 1, pZ, cluster_info,
                       eps_exist, tau_scale, novelty_radius, gamma)
    post0 = _posterior(x, 0, pZ, cluster_info,
                       eps_exist, tau_scale, novelty_radius, gamma)
    
    mi    = p_r1 * _kl(post1, pZ) + p_r0 * _kl(post0, pZ)
    # breakpoint()
    return dict(mi=float(mi), p_r1=float(p_r1), pZ=pZ)


# ---------------------------------------------------------------------------
# Batch MI
# ---------------------------------------------------------------------------

def compute_mi_batch(
    X_query:        np.ndarray,
    cluster_info:   list,
    eps_exist:      float = 0.04,
    tau_scale:      float = 1.3,
    novelty_radius: float = 0.75,
    gamma:          float = 0.25,
) -> tuple:
    """
    Compute MI for every row in X_query.
    Prior is re-evaluated per query point — no shared pZ vector.

    Returns
    -------
    mi_vals, p_r1_vals, dmin_vals  each (N,)
    """
    N         = len(X_query)
    mi_vals   = np.zeros(N, dtype=np.float64)
    p_r1_vals = np.zeros(N, dtype=np.float64)
    dmin_vals = np.zeros(N, dtype=np.float64)
    for i, x in enumerate(X_query):
        out          = _mi_single(x, cluster_info,
                                  eps_exist, tau_scale, novelty_radius, gamma)
        # breakpoint()
        mi_vals[i]   = out["mi"]
        p_r1_vals[i] = out["p_r1"]
        dmin_vals[i] = _dmin(x, cluster_info)
    return mi_vals, p_r1_vals, dmin_vals


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fit_support_and_compute_mi(
    X_support:       np.ndarray,
    X_query:         np.ndarray,
    # K selection — 0 = auto elbow, >0 = fixed
    n_clusters:      int   = 0,
    k_min:           int   = 2,
    k_max:           int   = 5,
    # cluster geometry
    radius_quantile: float = 0.90,
    # likelihood hyperparameters (prior is now parameter-free)
    eps_exist:       float = 0.04,
    tau_scale:       float = 1.3,
    novelty_radius:  float = 0.75,
    gamma:           float = 0.25,
    random_state:    int   = 0,
    # backward-compat stubs — accepted but ignored
    pi_new:          float = None,
    tau_scale_prior: float = None,
) -> tuple:
    """
    Fit KMeans on X_support, then compute MI for every point in X_query.

    Parameters
    ----------
    n_clusters      : 0 = auto (elbow), >0 = fixed K
    k_min / k_max   : elbow search bounds (only used when n_clusters=0)
    radius_quantile : quantile for cluster ball radius
    eps_exist       : max likelihood of revelation near existing cluster
    tau_scale       : Gaussian decay width for existing-cluster likelihood
    novelty_radius  : novelty threshold ρ for q_new sigmoid
    gamma           : softness of q_new sigmoid
    pi_new          : IGNORED (backward-compat stub — prior is now parameter-free)

    Returns
    -------
    mi_vals, p_r1_vals, dmin_vals  — shape (len(X_query),)
    cluster_info, pZ_centroid, km
    pZ_centroid is the prior evaluated at the centroid of X_query (diagnostic).
    """
    # ── K selection ───────────────────────────────────────────────────────
    if n_clusters is None or n_clusters <= 0:
        K = elbow_k(X_support, k_min=k_min, k_max=k_max,
                    random_state=random_state)
    else:
        K = min(int(n_clusters), len(X_support))
    K = max(K, 1)
    
    # breakpoint()
    # ── KMeans ────────────────────────────────────────────────────────────
    km     = KMeans(n_clusters=K, random_state=random_state,
                    n_init="auto", max_iter=300)
    labels = km.fit_predict(X_support)
    cluster_info = build_cluster_info(
        X_support, labels, km.cluster_centers_, radius_quantile,
    )
    # breakpoint()
    # ── MI on query ───────────────────────────────────────────────────────
    mi, p_r1, dmin = compute_mi_batch(
        X_query, cluster_info, eps_exist, tau_scale, novelty_radius, gamma,
    )
    # breakpoint()

    # Diagnostic: prior at query centroid
    pZ_centroid = _prior(X_query.mean(axis=0), cluster_info)

    return mi, p_r1, dmin, cluster_info, pZ_centroid, km
