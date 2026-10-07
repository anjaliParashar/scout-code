#!/usr/bin/env python3
"""
utils/toy2D/mi_toy.py
---------------------
MI and control-variate estimators for the toy2D experiments.

The MI implementation follows scout/mi.py:
  - same function names and signatures
  - same dict keys  (cluster, center, size, radius, scale)
  - same argument-passing style  (eps_exist, tau_scale, novelty_radius, gamma
    are always passed explicitly — never read from module-level constants)
  - same 3-tuple return from compute_mi_batch  (mi_vals, p_r1_vals, dmin_vals)
  - same unified entry point  fit_support_and_compute_mi

The only toy2D-specific additions below the MI block are:
  cluster_by_radius           — greedy radius clustering for batch-diversity selection
  compute_cv_batch            — local-ball CV using GP posterior draws + sim_mean_fn
  control_variates_estimator  — standard CV estimator (same math as scout.cv)
"""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from utils.toy2D.domain import sim_mean_fn
from utils.toy2D.gp_utils import gp_sample_functions


# ============================================================
# § 1  Probability utilities          (shared estimator)
# ============================================================

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


# ============================================================
# § 2  Cluster support descriptors    (shared estimator)
# ============================================================

def build_cluster_info(
    X_support:       np.ndarray,
    labels:          np.ndarray,
    centers:         np.ndarray,
    radius_quantile: float = 0.90,
) -> list:
    """
    Build per-cluster descriptor dicts with keys:
        cluster, center, size, radius, scale.

    Distance from x to cluster k = max(0, ||x - c_k|| - radius_k).

    Signature is identical to scout.mi.build_cluster_info.
    """
    info = []
    for k in range(len(centers)):
        pts = X_support[labels == k]
        if len(pts) == 0:
            radius, scale = 1e-6, 1e-6
        else:
            dists  = np.linalg.norm(pts - centers[k], axis=1)
            radius = max(float(np.quantile(dists, radius_quantile)), 1e-6)
            scale  = max(float(np.std(dists) + 1e-6),               1e-6)
        info.append(
            dict(cluster=k, center=centers[k], size=len(pts),
                 radius=radius, scale=scale)
        )
    return info


def _dist_to_ball(x: np.ndarray, c: dict) -> float:
    return max(0.0, float(np.linalg.norm(x - c["center"])) - c["radius"])


def _dmin(x: np.ndarray, cluster_info: list) -> float:
    return min(_dist_to_ball(x, c) for c in cluster_info)


# ============================================================
# § 3  Likelihood functions           (shared estimator)
# ============================================================

def _q_existing(x, c, eps_exist=0.04, tau_scale=1.3) -> float:
    """Near cluster -> small; far away -> approaches eps_exist."""
    d   = _dist_to_ball(x, c)
    tau = tau_scale * max(c["scale"], 1e-6)
    return float(eps_exist * (1.0 - np.exp(-0.5 * (d / max(tau, 1e-6)) ** 2)))


def _q_new(x, cluster_info, novelty_radius=0.75, gamma=0.25) -> float:
    """Increases as x moves away from all cluster supports."""
    return float(_sigmoid((np.array(_dmin(x, cluster_info)) - novelty_radius) / gamma))


def _likelihood_R1(x, cluster_info, eps_exist, tau_scale, novelty_radius, gamma):
    q = [_q_existing(x, c, eps_exist, tau_scale) for c in cluster_info]
    q.append(_q_new(x, cluster_info, novelty_radius, gamma))
    return np.asarray(q, dtype=np.float64)


def _posterior(x, r, pZ, cluster_info, eps_exist, tau_scale, novelty_radius, gamma):
    q1 = _likelihood_R1(x, cluster_info, eps_exist, tau_scale, novelty_radius, gamma)
    lk = q1 if r == 1 else (1.0 - q1)
    return _normalize(lk * pZ)


# ============================================================
# § 4  MI for a single point          (shared estimator)
# ============================================================

def _mi_single(x, pZ, cluster_info, eps_exist, tau_scale, novelty_radius, gamma) -> dict:
    q1   = _likelihood_R1(x, cluster_info, eps_exist, tau_scale, novelty_radius, gamma)
    p_r1 = float(np.sum(pZ * q1))
    p_r0 = 1.0 - p_r1
    post1 = _posterior(x, 1, pZ, cluster_info, eps_exist, tau_scale, novelty_radius, gamma)
    post0 = _posterior(x, 0, pZ, cluster_info, eps_exist, tau_scale, novelty_radius, gamma)
    mi = p_r1 * _kl(post1, pZ) + p_r0 * _kl(post0, pZ)
    return dict(mi=float(mi), p_r1=float(p_r1),
                post_r1=post1, post_r0=post0)


# ============================================================
# § 5  Batch MI computation           (shared estimator)
# ============================================================

def compute_mi_batch(
    X_query:        np.ndarray,
    pZ:             np.ndarray,
    cluster_info:   list,
    eps_exist:      float = 0.04,
    tau_scale:      float = 1.3,
    novelty_radius: float = 0.75,
    gamma:          float = 0.25,
) -> tuple:
    """
    Compute MI for every row in X_query.

    Returns
    -------
    mi_vals   : (N,)
    p_r1_vals : (N,)
    dmin_vals : (N,)

    Signature matches scout.mi.compute_mi_batch.
    """
    N = len(X_query)
    mi_vals   = np.zeros(N)
    p_r1_vals = np.zeros(N)
    dmin_vals = np.zeros(N)
    for i, x in enumerate(X_query):
        out = _mi_single(x, pZ, cluster_info, eps_exist, tau_scale, novelty_radius, gamma)
        mi_vals[i]   = out["mi"]
        p_r1_vals[i] = out["p_r1"]
        dmin_vals[i] = _dmin(x, cluster_info)
    return mi_vals, p_r1_vals, dmin_vals


def fit_support_and_compute_mi(
    X_support:      np.ndarray,
    X_query:        np.ndarray,
    n_clusters:     int,
    radius_quantile: float,
    pi_new:         float,
    eps_exist:      float,
    tau_scale:      float,
    novelty_radius: float,
    gamma:          float,
    random_state:   int,
) -> tuple:
    """
    Fit KMeans on X_support, build cluster_info + prior, compute MI for X_query.

    Returns
    -------
    mi_vals, p_r1_vals, dmin_vals  — all shape (len(X_query),)
    cluster_info, pZ, km           — reusable objects for the same iteration

    Signature and behaviour are identical to scout.mi.fit_support_and_compute_mi.
    """
    n_eff  = min(n_clusters, len(X_support))
    km     = KMeans(n_clusters=n_eff, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X_support)
    cluster_info = build_cluster_info(
        X_support, labels, km.cluster_centers_, radius_quantile
    )
    sizes = np.array([max(1, c["size"]) for c in cluster_info], dtype=float)
    pZ    = _normalize(
        np.concatenate([(1.0 - pi_new) * sizes / sizes.sum(), [pi_new]])
    )
    mi, p_r1, dmin = compute_mi_batch(
        X_query, pZ, cluster_info, eps_exist, tau_scale, novelty_radius, gamma
    )
    return mi, p_r1, dmin, cluster_info, pZ, km


# ============================================================
# § 6  Toy2D-only: batch-diversity clustering
# ============================================================

def cluster_by_radius(x: np.ndarray, eps: float = 0.55) -> list:
    """
    Greedy radius-based clustering used only in the batch-selection step to
    diversify acquired points across the high-MI shortlist.

    NOT part of the MI estimator — the MI prior uses KMeans via
    fit_support_and_compute_mi above.

    Returns a list of (Ni, d) float64 arrays, one per cluster.
    """
    x      = np.asarray(x, dtype=np.float64)
    unused = set(range(len(x)))
    clusters = []
    while unused:
        seed     = unused.pop()
        cluster  = [seed]
        frontier = [seed]
        while frontier:
            i   = frontier.pop()
            xi  = x[i]
            add = [j for j in list(unused) if np.linalg.norm(xi - x[j]) <= eps]
            for j in add:
                unused.discard(j)
                frontier.append(j)
                cluster.append(j)
        clusters.append(x[cluster])
    return clusters


# ============================================================
# § 7  Toy2D-only: control-variate estimator + CV batch
# ============================================================

@dataclass
class CVResult:
    mu_hat:     float
    var_mu_hat: float
    beta:       float


def control_variates_estimator(
    f: np.ndarray, g: np.ndarray, g_unpaired: np.ndarray
) -> CVResult:
    """
    Standard control-variates estimator.

    f          : (n,) paired real-side GP draws
    g          : (n,) paired simulation values
    g_unpaired : (k,) unpaired simulation pool
    """
    n, k = len(f), len(g_unpaired)
    if n <= 1 or k <= 1:
        raise ValueError("Need n > 1 and k > 1 for CV estimator.")
    var_f   = np.var(f, ddof=1)
    theta   = np.mean(g_unpaired)
    cov_gf  = np.cov(g, f, ddof=1)[0, 1]
    var_g   = np.var(g, ddof=1)
    var_gu  = np.var(g_unpaired, ddof=1)
    if np.isclose(var_g, 0.0):
        raise ValueError("var(g) = 0")
    beta    = k / (k + n) * cov_gf / var_g
    mu_hat  = np.mean(f - beta * g) + beta * theta
    var_hat = (
        (var_f + beta**2 * var_g - 2 * beta * cov_gf) / n
        + beta**2 * var_gu / k
    )
    return CVResult(float(mu_hat), float(max(var_hat, 1e-10)), float(beta))


def _local_ball_samples(x0: np.ndarray, radius: float, n: int) -> np.ndarray:
    """Sample n points uniformly inside the L2-ball of given radius around x0."""
    d    = x0.shape[0]
    dirs = np.random.randn(n, d).astype(np.float32)
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-8)
    r    = radius * np.sqrt(np.random.rand(n, 1).astype(np.float32))
    return np.clip(x0[None, :] + r * dirs, -3.0, 3.0).astype(np.float32)


def compute_cv_batch(
    gp,
    X_query:    np.ndarray,
    R_local:    float = 0.1,
    n_pair:     int   = 100,
    k_unpaired: int   = 120,
    n_f_draws:  int   = 32,
    clip:       float = 1.0,
) -> np.ndarray:
    """
    Compute CV-estimated mean for every point in X_query.

    Uses local-ball sim samples as paired/unpaired control and GP posterior
    draws as the real-side estimate.

    Returns
    -------
    out : (N,) array of CV-estimated means (clipped to `clip`, NaN on failure)
    """
    out = np.full(len(X_query), np.nan)
    for i, x0 in enumerate(X_query):
        u   = _local_ball_samples(x0, R_local, n_pair)
        v   = _local_ball_samples(x0, R_local, k_unpaired)
        g_u = sim_mean_fn(u).astype(np.float64)
        g_v = sim_mean_fn(v).astype(np.float64)
        f_draws = gp_sample_functions(gp, u, n_draws=n_f_draws)  # (n_f_draws, n_pair)
        vals = []
        for s in range(n_f_draws):
            try:
                res = control_variates_estimator(f_draws[s], g_u, g_v)
                vals.append(res.mu_hat)
            except ValueError:
                pass
        if vals:
            out[i] = min(np.mean(vals), clip)
    return out
