#!/usr/bin/env python3
"""
bams_baseline_augmented.py

BAS/BAMS-style acquisitions for already-encoded scenarios.

This version updates the multi-fidelity BAMS acquisition to use a SINGLE
augmented multi-fidelity GP over inputs [z, m], where:

    z : encoded scenario, shape (d,)
    m : scalar fidelity code, e.g. low=0.0, high=1.0

Key correction:
---------------
The multi-fidelity acquisition no longer scores low-fidelity candidates using a
separate low-fidelity beta-reduction objective. Instead, both high- and
low-fidelity actions are scored by their expected reduction in HIGH-FIDELITY
target uncertainty over the empirical pool.

This is the BAMS-style behavior:
    query action = (z_k, m)
    target variables = {(z_i, high)} for all pool points i
    score = reduction in high-fidelity point-variance / beta objective per cost

The runner is responsible for:
    1. Training/retraining the augmented GP after data acquisition.
    2. Passing that GP as gp_mf into bams_acquisition(...).

Expected runner-side augmented training data:
---------------------------------------------
    Z_high_aug = np.concatenate([Z_high, np.ones((len(Z_high), 1))], axis=1)
    Z_low_aug  = np.concatenate([Z_low,  np.zeros((len(Z_low),  1))], axis=1)

    X_train_aug = np.concatenate([Z_high_aug, Z_low_aug], axis=0)
    y_train     = np.concatenate([y_high, y_low], axis=0)

    gp_mf = train_gp(X_train_aug, y_train)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.special import ndtr, owens_t
from sklearn.cluster import KMeans


# -----------------------------------------------------------------------------
# Numerics for beta/J score
# -----------------------------------------------------------------------------

def _phi2(a: np.ndarray, b: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Vectorized bivariate normal CDF Φ₂(a, b; rho) using Owen's T."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    rho = np.clip(np.asarray(rho, dtype=np.float64), -1.0 + 1e-9, 1.0 - 1e-9)

    denom = np.sqrt(np.maximum(1.0 - rho**2, 1e-12))

    val = (
        ndtr(a) * ndtr(b)
        + 2.0 * owens_t(a, (b - rho * a) / denom)
        + 2.0 * owens_t(b, (a - rho * b) / denom)
    )

    return np.clip(val, 0.0, 1.0)


def _dphi2_drho_symmetric(s: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """
    d/d rho Φ₂(s, -s; rho) = bivariate normal pdf at (s, -s).

    This is used as a first-order ranking approximation.
    Exact beta is recomputed after committing a point.
    """
    s = np.asarray(s, dtype=np.float64)
    rho = np.clip(np.asarray(rho, dtype=np.float64), -1.0 + 1e-9, 1.0 - 1e-9)

    denom = np.maximum(1.0 - rho**2, 1e-12)

    return (
        1.0 / (2.0 * np.pi * np.sqrt(denom))
        * np.exp(-(s**2) / (1.0 - rho))
    )


def _as_numpy_1d(x) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


# -----------------------------------------------------------------------------
# GP posterior helpers
# -----------------------------------------------------------------------------

def _posterior_from_gp(gp, X_t: torch.Tensor):
    """
    Get posterior mean/covariance in a way that works for most BoTorch/GPyTorch GPs.
    """
    with torch.no_grad():
        if hasattr(gp, "posterior"):
            posterior = gp.posterior(X_t)
            mean = posterior.mean.squeeze(-1)
            cov = posterior.mvn.covariance_matrix
            return mean, cov

        gp.eval()
        if hasattr(gp, "likelihood"):
            gp.likelihood.eval()

        mvn = gp(X_t)
        mean = mvn.mean
        cov = mvn.covariance_matrix
        return mean, cov


def _build_gp_cache(gp, pool_Z: np.ndarray, jitter: float = 1e-8) -> Dict[str, np.ndarray]:
    """
    Cache posterior mean, marginal variance, and posterior covariance on pool_Z.
    Used by single-fidelity BAS.
    """
    pool_Z = np.asarray(pool_Z, dtype=np.float64)

    device = next(gp.parameters()).device
    dtype = next(gp.parameters()).dtype

    X_t = torch.as_tensor(pool_Z, dtype=dtype, device=device)

    mean_t, cov_t = _posterior_from_gp(gp, X_t)

    mean = _as_numpy_1d(mean_t)
    C = cov_t.detach().cpu().double().numpy()
    C = 0.5 * (C + C.T)

    var = np.maximum(np.diag(C), jitter)
    C[np.diag_indices_from(C)] = var

    return {
        "post_mean": mean,
        "post_var": var,
        "C": C,
    }


# -----------------------------------------------------------------------------
# Single-fidelity BAS state
# -----------------------------------------------------------------------------

class _FidelityState:
    """State for scoring a sequential batch under one GP/fidelity."""

    def __init__(
        self,
        cache: Dict[str, np.ndarray],
        threshold: float,
        weights: Optional[np.ndarray] = None,
    ):
        self.C = cache["C"]
        self.var = np.maximum(cache["post_var"], 1e-12)
        self.std = np.sqrt(self.var)
        self.s_hat = (float(threshold) - cache["post_mean"]) / self.std
        self.N = len(self.s_hat)

        if weights is None:
            self.weights = np.ones(self.N, dtype=np.float64) / self.N
        else:
            weights = np.asarray(weights, dtype=np.float64).reshape(-1)
            if len(weights) != self.N:
                raise ValueError(f"weights length must be {self.N}, got {len(weights)}")
            weights = np.clip(weights, 0.0, None)
            self.weights = weights / max(weights.sum(), 1e-12)

        self.rho_now = np.full(self.N, -1.0 + 1e-9, dtype=np.float64)

        p = ndtr(self.s_hat)
        self.beta_now = p * (1.0 - p)

        self.sel_idx: List[int] = []
        self.L_sel: Optional[np.ndarray] = None
        self.C_pool_sel: Optional[np.ndarray] = None

    def score_candidate(self, k: int) -> Tuple[float, np.ndarray]:
        """
        Score adding pool index k.

        Returns:
            delta_J: positive scalar beta/J reduction
            rho_new: updated rho vector if k is committed
        """
        k = int(k)
        m = len(self.sel_idx)

        if m == 0:
            cov_res_col = self.C[:, k].copy()
            schur_k = float(self.C[k, k])
        else:
            assert self.L_sel is not None
            assert self.C_pool_sel is not None

            c_k_sel = self.C_pool_sel[k, :]

            v = np.linalg.solve(self.L_sel, c_k_sel)
            w = np.linalg.solve(self.L_sel.T, v)

            cov_res_col = self.C[:, k] - self.C_pool_sel @ w
            schur_k = float(self.C[k, k] - np.dot(v, v))

        schur_k = max(schur_k, 1e-10)

        delta_t = -(cov_res_col**2) / (self.var * schur_k)
        rho_new = np.clip(self.rho_now + delta_t, -1.0 + 1e-9, 1.0 - 1e-9)

        d_beta_d_rho = _dphi2_drho_symmetric(self.s_hat, self.rho_now)
        delta_beta = d_beta_d_rho * (rho_new - self.rho_now)

        delta_J = -float(np.sum(self.weights * delta_beta))

        return max(delta_J, 0.0), rho_new

    def commit(self, k: int, rho_new: np.ndarray) -> None:
        """Commit k and update exact beta."""
        k = int(k)
        m = len(self.sel_idx)

        if m == 0:
            schur_k = max(float(self.C[k, k]), 1e-10)
            self.L_sel = np.array([[np.sqrt(schur_k)]], dtype=np.float64)
            self.C_pool_sel = self.C[:, [k]].copy()
        else:
            assert self.L_sel is not None
            assert self.C_pool_sel is not None

            c_k_sel = self.C_pool_sel[k, :]

            v = np.linalg.solve(self.L_sel, c_k_sel)
            schur_k = max(float(self.C[k, k] - np.dot(v, v)), 1e-10)

            L_new = np.zeros((m + 1, m + 1), dtype=np.float64)
            L_new[:m, :m] = self.L_sel
            L_new[m, :m] = v
            L_new[m, m] = np.sqrt(schur_k)

            self.L_sel = L_new
            self.C_pool_sel = np.concatenate(
                [self.C_pool_sel, self.C[:, [k]]],
                axis=1,
            )

        self.sel_idx.append(k)
        self.rho_now = np.clip(rho_new, -1.0 + 1e-9, 1.0 - 1e-9)
        self.beta_now = _phi2(self.s_hat, -self.s_hat, self.rho_now)


@dataclass(frozen=True)
class _QueuedCandidate:
    idx: int
    fidelity: str
    raw_gain: float
    score_per_cost: float
    cost: float
    cluster: int


# -----------------------------------------------------------------------------
# Shared queue / clustering helpers
# -----------------------------------------------------------------------------

def _cluster_pool(
    pool_Z: np.ndarray,
    n_clusters: int,
    random_state: int,
) -> Dict[int, np.ndarray]:
    pool_Z = np.asarray(pool_Z, dtype=np.float64)
    N = len(pool_Z)

    if N == 0:
        raise ValueError("pool_Z is empty.")

    n_eff = min(int(n_clusters), N)

    labels = KMeans(
        n_clusters=n_eff,
        random_state=random_state,
        n_init="auto",
    ).fit_predict(pool_Z)

    return {s: np.where(labels == s)[0] for s in range(n_eff)}


def _budget_for_cluster(
    total_budget: float,
    cluster_size: int,
    N: int,
    eta: float,
) -> float:
    return max(
        1.0,
        float(eta) * float(total_budget) * float(cluster_size) / float(N),
    )


def _global_greedy_from_queues(
    queues: Dict[int, List[_QueuedCandidate]],
    total_budget: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge per-cluster queues with a global budget.

    Returns:
        selected_idx
        selected_fidelity
        selected_cost
    """
    ptr = {s: 0 for s in queues}

    selected_idx: List[int] = []
    selected_fid: List[str] = []
    selected_cost: List[float] = []

    spent = 0.0

    while spent < total_budget - 1e-12:
        front: List[_QueuedCandidate] = []

        for s, q in queues.items():
            p = ptr[s]

            while p < len(q) and spent + q[p].cost > total_budget + 1e-12:
                p += 1

            ptr[s] = p

            if p < len(q):
                front.append(q[p])

        if not front:
            break

        best = max(front, key=lambda c: c.score_per_cost)

        selected_idx.append(best.idx)
        selected_fid.append(best.fidelity)
        selected_cost.append(best.cost)

        spent += best.cost
        ptr[best.cluster] += 1

    return (
        np.asarray(selected_idx, dtype=int),
        np.asarray(selected_fid, dtype=object),
        np.asarray(selected_cost, dtype=np.float64),
    )


# -----------------------------------------------------------------------------
# Public API 1: single-fidelity BAS
# -----------------------------------------------------------------------------

def bas_acquisition(
    pool_Z: np.ndarray,
    gp_high,
    batch_size: int,
    threshold: float,
    n_clusters: int = 6,
    eta: float = 1.5,
    random_state: int = 0,
    exclude_idx: Optional[Iterable[int]] = None,
    pool_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Single-fidelity BAS acquisition.

    This part is intentionally close to your original implementation.
    """
    pool_Z = np.asarray(pool_Z, dtype=np.float64)
    N = len(pool_Z)

    excluded = set([] if exclude_idx is None else [int(i) for i in exclude_idx])

    cache = _build_gp_cache(gp_high, pool_Z)
    clusters = _cluster_pool(pool_Z, n_clusters=n_clusters, random_state=random_state)

    queues: Dict[int, List[_QueuedCandidate]] = {}

    for s, cluster_idx in clusters.items():
        state = _FidelityState(
            cache=cache,
            threshold=threshold,
            weights=pool_weights,
        )

        cluster_budget = min(
            _budget_for_cluster(batch_size, len(cluster_idx), N, eta),
            float(batch_size),
        )

        used = set(excluded)
        q: List[_QueuedCandidate] = []
        spent_s = 0.0

        while spent_s < cluster_budget - 1e-12:
            candidates = [int(i) for i in cluster_idx if int(i) not in used]

            if not candidates:
                break

            best_idx = None
            best_gain = -np.inf
            best_rho = None

            for k in candidates:
                gain, rho_new = state.score_candidate(k)

                if gain > best_gain:
                    best_idx = k
                    best_gain = gain
                    best_rho = rho_new

            if best_idx is None or best_rho is None:
                break

            q.append(
                _QueuedCandidate(
                    idx=best_idx,
                    fidelity="high",
                    raw_gain=float(best_gain),
                    score_per_cost=float(best_gain),
                    cost=1.0,
                    cluster=s,
                )
            )

            used.add(best_idx)
            state.commit(best_idx, best_rho)
            spent_s += 1.0

        queues[s] = q

    selected_idx, _, _ = _global_greedy_from_queues(
        queues,
        total_budget=float(batch_size),
    )

    return selected_idx


# -----------------------------------------------------------------------------
# Augmented multi-fidelity GP cache
# -----------------------------------------------------------------------------

def _augment_with_fidelity(
    pool_Z: np.ndarray,
    fidelity_value: float,
) -> np.ndarray:
    """
    Build augmented inputs [z, m].

    pool_Z: (N, d)
    return: (N, d + 1)
    """
    pool_Z = np.asarray(pool_Z, dtype=np.float64)
    m = np.full((len(pool_Z), 1), float(fidelity_value), dtype=np.float64)
    return np.concatenate([pool_Z, m], axis=1)


def _build_mf_augmented_cache(
    gp_mf,
    pool_Z: np.ndarray,
    high_fidelity_value: float = 1.0,
    low_fidelity_value: float = 0.0,
    jitter: float = 1e-8,
) -> Dict[str, np.ndarray]:
    """
    Build posterior covariance cache for the augmented action space.

    Action indexing convention:
        0      ... N-1   : high-fidelity actions (z_i, H)
        N      ... 2N-1  : low-fidelity actions  (z_i, L)

    Target variables:
        high-fidelity pool values (z_i, H), i=0..N-1

    The key object is:
        C_target_action = Cov(F_H(pool), F_action(pool, fidelity))

    This allows low-fidelity actions to be scored by their covariance with the
    high-fidelity target pool.
    """
    pool_Z = np.asarray(pool_Z, dtype=np.float64)
    N = len(pool_Z)

    X_high = _augment_with_fidelity(pool_Z, high_fidelity_value)
    X_low = _augment_with_fidelity(pool_Z, low_fidelity_value)

    X_aug = np.concatenate([X_high, X_low], axis=0)

    device = next(gp_mf.parameters()).device
    dtype = next(gp_mf.parameters()).dtype

    X_t = torch.as_tensor(X_aug, dtype=dtype, device=device)

    mean_t, cov_t = _posterior_from_gp(gp_mf, X_t)

    mean_all = _as_numpy_1d(mean_t)

    C_all = cov_t.detach().cpu().double().numpy()
    C_all = 0.5 * (C_all + C_all.T)

    diag = np.maximum(np.diag(C_all), jitter)
    C_all[np.diag_indices_from(C_all)] = diag

    high_action_idx = np.arange(N, dtype=int)
    low_action_idx = np.arange(N, 2 * N, dtype=int)

    target_idx = high_action_idx.copy()

    target_mean = mean_all[target_idx]
    target_C = C_all[np.ix_(target_idx, target_idx)]

    target_var = np.maximum(np.diag(target_C), jitter)
    target_C[np.diag_indices_from(target_C)] = target_var

    C_target_action = C_all[np.ix_(target_idx, np.arange(2 * N, dtype=int))]

    return {
        "N": N,
        "X_aug": X_aug,
        "mean_all": mean_all,
        "C_all": C_all,

        "high_action_idx": high_action_idx,
        "low_action_idx": low_action_idx,

        "target_idx": target_idx,
        "target_mean": target_mean,
        "target_C": target_C,
        "target_var": target_var,
        "C_target_action": C_target_action,
    }


class _TargetBAMSState:
    """
    Sequential BAMS state over the high-fidelity target pool.

    Candidate actions may be high or low fidelity, but every candidate is scored
    by reduction in high-fidelity target beta/J.

    This is the main replacement for the old:
        high_state = _FidelityState(high_cache, ...)
        low_state  = _FidelityState(low_cache, ...)
    """

    def __init__(
        self,
        mf_cache: Dict[str, np.ndarray],
        threshold: float,
        weights: Optional[np.ndarray] = None,
    ):
        self.C_action = mf_cache["C_all"]
        self.C_target_action = mf_cache["C_target_action"]

        self.target_mean = mf_cache["target_mean"]
        self.target_var = np.maximum(mf_cache["target_var"], 1e-12)
        self.target_std = np.sqrt(self.target_var)

        self.N_target = len(self.target_mean)
        self.N_action = self.C_action.shape[0]

        self.s_hat = (float(threshold) - self.target_mean) / self.target_std

        if weights is None:
            self.weights = np.ones(self.N_target, dtype=np.float64) / self.N_target
        else:
            weights = np.asarray(weights, dtype=np.float64).reshape(-1)
            if len(weights) != self.N_target:
                raise ValueError(
                    f"weights length must be {self.N_target}, got {len(weights)}"
                )
            weights = np.clip(weights, 0.0, None)
            self.weights = weights / max(weights.sum(), 1e-12)

        self.rho_now = np.full(self.N_target, -1.0 + 1e-9, dtype=np.float64)

        p = ndtr(self.s_hat)
        self.beta_now = p * (1.0 - p)

        self.sel_action_idx: List[int] = []
        self.L_sel: Optional[np.ndarray] = None
        self.C_target_sel: Optional[np.ndarray] = None
        self.C_action_sel: Optional[np.ndarray] = None

    def score_action(self, action_idx: int) -> Tuple[float, np.ndarray]:
        """
        Score observing augmented action action_idx.

        The score is high-fidelity target uncertainty reduction, regardless of
        whether the action is low or high fidelity.
        """
        action_idx = int(action_idx)

        m = len(self.sel_action_idx)

        if m == 0:
            cov_res_col = self.C_target_action[:, action_idx].copy()
            schur = float(self.C_action[action_idx, action_idx])
        else:
            assert self.L_sel is not None
            assert self.C_target_sel is not None
            assert self.C_action_sel is not None

            # Cov(selected actions, proposed action)
            k_Sq = self.C_action_sel[action_idx, :]

            v = np.linalg.solve(self.L_sel, k_Sq)
            w = np.linalg.solve(self.L_sel.T, v)

            # Residual Cov(target high pool, proposed action | selected actions)
            cov_res_col = self.C_target_action[:, action_idx] - self.C_target_sel @ w

            schur = float(self.C_action[action_idx, action_idx] - np.dot(v, v))

        schur = max(schur, 1e-10)

        # Conditional variance reduction over high-fidelity target pool.
        delta_t = -(cov_res_col**2) / (self.target_var * schur)

        rho_new = np.clip(
            self.rho_now + delta_t,
            -1.0 + 1e-9,
            1.0 - 1e-9,
        )

        d_beta_d_rho = _dphi2_drho_symmetric(self.s_hat, self.rho_now)
        delta_beta = d_beta_d_rho * (rho_new - self.rho_now)

        delta_J = -float(np.sum(self.weights * delta_beta))

        return max(delta_J, 0.0), rho_new

    def commit(self, action_idx: int, rho_new: np.ndarray) -> None:
        """
        Commit an augmented action and update the selected-action covariance.
        """
        action_idx = int(action_idx)
        m = len(self.sel_action_idx)

        if m == 0:
            schur = max(float(self.C_action[action_idx, action_idx]), 1e-10)

            self.L_sel = np.array([[np.sqrt(schur)]], dtype=np.float64)
            self.C_target_sel = self.C_target_action[:, [action_idx]].copy()
            self.C_action_sel = self.C_action[:, [action_idx]].copy()
        else:
            assert self.L_sel is not None
            assert self.C_target_sel is not None
            assert self.C_action_sel is not None

            k_Sq = self.C_action_sel[action_idx, :]

            v = np.linalg.solve(self.L_sel, k_Sq)
            schur = max(float(self.C_action[action_idx, action_idx] - np.dot(v, v)), 1e-10)

            L_new = np.zeros((m + 1, m + 1), dtype=np.float64)
            L_new[:m, :m] = self.L_sel
            L_new[m, :m] = v
            L_new[m, m] = np.sqrt(schur)

            self.L_sel = L_new

            self.C_target_sel = np.concatenate(
                [self.C_target_sel, self.C_target_action[:, [action_idx]]],
                axis=1,
            )

            self.C_action_sel = np.concatenate(
                [self.C_action_sel, self.C_action[:, [action_idx]]],
                axis=1,
            )

        self.sel_action_idx.append(action_idx)
        self.rho_now = np.clip(rho_new, -1.0 + 1e-9, 1.0 - 1e-9)
        self.beta_now = _phi2(self.s_hat, -self.s_hat, self.rho_now)


def _pool_idx_to_action_idx(
    pool_idx: int,
    fidelity: str,
    N: int,
) -> int:
    """
    Action indexing:
        high action for pool index i = i
        low action for pool index i  = N + i
    """
    pool_idx = int(pool_idx)

    if fidelity == "high":
        return pool_idx

    if fidelity == "low":
        return N + pool_idx

    raise ValueError(f"Unknown fidelity: {fidelity}")


def _action_idx_to_pool_idx_and_fidelity(
    action_idx: int,
    N: int,
) -> Tuple[int, str]:
    action_idx = int(action_idx)

    if action_idx < N:
        return action_idx, "high"

    return action_idx - N, "low"


# -----------------------------------------------------------------------------
# Public API 2: multi-fidelity BAMS over augmented GP
# -----------------------------------------------------------------------------

def bams_acquisition(
    pool_Z: np.ndarray,
    gp_mf,
    total_budget: float,
    high_threshold: float,
    high_cost: float = 1.0,
    low_cost: float = 0.2,
    n_clusters: int = 6,
    eta: float = 1.5,
    random_state: int = 0,
    high_fidelity_value: float = 1.0,
    low_fidelity_value: float = 0.0,
    low_fidelity_available_idx: Optional[Sequence[int]] = None,
    exclude_high_idx: Optional[Iterable[int]] = None,
    exclude_low_idx: Optional[Iterable[int]] = None,
    pool_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Multi-fidelity BAMS acquisition using one augmented GP over (z, fidelity).

    Both high- and low-fidelity actions are scored by how much they reduce
    uncertainty in the high-fidelity target objective over the empirical pool.

    Parameters
    ----------
    pool_Z:
        Encoded scenarios, shape (N, d).

    gp_mf:
        Joint multi-fidelity GP trained on augmented inputs [z, m].

    total_budget:
        Budget in high-fidelity cost units.

    high_threshold:
        Adverse-event threshold for the high-fidelity target objective.

    high_cost, low_cost:
        Evaluation costs.

    high_fidelity_value, low_fidelity_value:
        Numeric fidelity encodings used when training gp_mf.

    low_fidelity_available_idx:
        Optional subset of pool indices where low-fidelity query is allowed.

    exclude_high_idx, exclude_low_idx:
        Already-acquired pool indices per fidelity.

    pool_weights:
        Optional empirical weights over pool_Z. If omitted, uniform weights.

    Returns
    -------
    selected_idx:
        Pool indices selected.

    selected_fidelity:
        Array with entries "high" or "low".

    selected_cost:
        Cost of each selected action.
    """
    if high_cost <= 0 or low_cost <= 0:
        raise ValueError("high_cost and low_cost must be positive.")

    if low_cost >= high_cost:
        print("[WARN] low_cost >= high_cost; low fidelity is not cheaper.", flush=True)

    pool_Z = np.asarray(pool_Z, dtype=np.float64)
    N = len(pool_Z)

    if N == 0:
        raise ValueError("pool_Z is empty.")

    lf_allowed = (
        set(range(N))
        if low_fidelity_available_idx is None
        else set(int(i) for i in low_fidelity_available_idx)
    )

    excluded_high = set([] if exclude_high_idx is None else [int(i) for i in exclude_high_idx])
    excluded_low = set([] if exclude_low_idx is None else [int(i) for i in exclude_low_idx])

    mf_cache = _build_mf_augmented_cache(
        gp_mf=gp_mf,
        pool_Z=pool_Z,
        high_fidelity_value=high_fidelity_value,
        low_fidelity_value=low_fidelity_value,
    )

    clusters = _cluster_pool(
        pool_Z,
        n_clusters=n_clusters,
        random_state=random_state,
    )

    queues: Dict[int, List[_QueuedCandidate]] = {}

    for s, cluster_idx in clusters.items():
        state = _TargetBAMSState(
            mf_cache=mf_cache,
            threshold=high_threshold,
            weights=pool_weights,
        )

        cluster_budget = min(
            _budget_for_cluster(total_budget, len(cluster_idx), N, eta),
            float(total_budget),
        )

        used_high = set(excluded_high)
        used_low = set(excluded_low)

        q: List[_QueuedCandidate] = []
        spent_s = 0.0

        while spent_s < cluster_budget - 1e-12:
            best: Optional[_QueuedCandidate] = None
            best_rho: Optional[np.ndarray] = None

            # Candidate action: evaluate high fidelity at pool point k.
            for k in cluster_idx:
                k = int(k)

                if k in used_high:
                    continue

                if spent_s + high_cost > cluster_budget + 1e-12:
                    continue

                action_idx = _pool_idx_to_action_idx(k, "high", N)
                gain, rho_new = state.score_action(action_idx)

                cand = _QueuedCandidate(
                    idx=k,
                    fidelity="high",
                    raw_gain=float(gain),
                    score_per_cost=float(gain / high_cost),
                    cost=float(high_cost),
                    cluster=s,
                )

                if best is None or cand.score_per_cost > best.score_per_cost:
                    best = cand
                    best_rho = rho_new

            # Candidate action: evaluate low fidelity at pool point k.
            # This is scored by high-fidelity target uncertainty reduction
            # through the augmented GP cross-covariance.
            for k in cluster_idx:
                k = int(k)

                if k not in lf_allowed:
                    continue

                if k in used_low:
                    continue

                if spent_s + low_cost > cluster_budget + 1e-12:
                    continue

                action_idx = _pool_idx_to_action_idx(k, "low", N)
                gain, rho_new = state.score_action(action_idx)

                cand = _QueuedCandidate(
                    idx=k,
                    fidelity="low",
                    raw_gain=float(gain),
                    score_per_cost=float(gain / low_cost),
                    cost=float(low_cost),
                    cluster=s,
                )

                if best is None or cand.score_per_cost > best.score_per_cost:
                    best = cand
                    best_rho = rho_new

            if best is None or best_rho is None:
                break

            action_idx = _pool_idx_to_action_idx(best.idx, best.fidelity, N)
            state.commit(action_idx, best_rho)

            q.append(best)

            if best.fidelity == "high":
                used_high.add(best.idx)
            else:
                used_low.add(best.idx)

            spent_s += best.cost

        queues[s] = q

    return _global_greedy_from_queues(
        queues,
        total_budget=float(total_budget),
    )


# -----------------------------------------------------------------------------
# Optional backward-compatibility wrapper
# -----------------------------------------------------------------------------

def bams_acquisition_independent_fidelity_legacy(
    pool_Z: np.ndarray,
    gp_high,
    gp_low,
    total_budget: float,
    high_threshold: float,
    low_threshold: Optional[float] = None,
    high_cost: float = 1.0,
    low_cost: float = 0.2,
    n_clusters: int = 6,
    eta: float = 1.5,
    random_state: int = 0,
    low_fidelity_available_idx: Optional[Sequence[int]] = None,
    exclude_high_idx: Optional[Iterable[int]] = None,
    exclude_low_idx: Optional[Iterable[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Legacy version of your previous independent-fidelity implementation.

    This is retained only if you need to reproduce older results. It is NOT the
    corrected BAMS acquisition because low-fidelity actions reduce low-fidelity
    uncertainty rather than high-fidelity target uncertainty.
    """
    if high_cost <= 0 or low_cost <= 0:
        raise ValueError("high_cost and low_cost must be positive.")

    if low_cost >= high_cost:
        print("[WARN] low_cost >= high_cost; low fidelity is not cheaper.", flush=True)

    pool_Z = np.asarray(pool_Z, dtype=np.float64)
    N = len(pool_Z)

    low_threshold = high_threshold if low_threshold is None else low_threshold

    lf_allowed = (
        set(range(N))
        if low_fidelity_available_idx is None
        else set(int(i) for i in low_fidelity_available_idx)
    )

    excluded_high = set([] if exclude_high_idx is None else [int(i) for i in exclude_high_idx])
    excluded_low = set([] if exclude_low_idx is None else [int(i) for i in exclude_low_idx])

    high_cache = _build_gp_cache(gp_high, pool_Z)
    low_cache = _build_gp_cache(gp_low, pool_Z)

    clusters = _cluster_pool(pool_Z, n_clusters=n_clusters, random_state=random_state)

    queues: Dict[int, List[_QueuedCandidate]] = {}

    for s, cluster_idx in clusters.items():
        high_state = _FidelityState(high_cache, threshold=high_threshold)
        low_state = _FidelityState(low_cache, threshold=low_threshold)

        cluster_budget = min(
            _budget_for_cluster(total_budget, len(cluster_idx), N, eta),
            float(total_budget),
        )

        used_high = set(excluded_high)
        used_low = set(excluded_low)

        spent_s = 0.0
        q: List[_QueuedCandidate] = []

        while spent_s < cluster_budget - 1e-12:
            best: Optional[_QueuedCandidate] = None
            best_rho: Optional[np.ndarray] = None

            for k in cluster_idx:
                k = int(k)
                if k in used_high:
                    continue
                if spent_s + high_cost > cluster_budget + 1e-12:
                    continue

                gain, rho_new = high_state.score_candidate(k)

                cand = _QueuedCandidate(
                    idx=k,
                    fidelity="high",
                    raw_gain=float(gain),
                    score_per_cost=float(gain / high_cost),
                    cost=float(high_cost),
                    cluster=s,
                )

                if best is None or cand.score_per_cost > best.score_per_cost:
                    best = cand
                    best_rho = rho_new

            for k in cluster_idx:
                k = int(k)
                if k not in lf_allowed or k in used_low:
                    continue
                if spent_s + low_cost > cluster_budget + 1e-12:
                    continue

                gain, rho_new = low_state.score_candidate(k)

                cand = _QueuedCandidate(
                    idx=k,
                    fidelity="low",
                    raw_gain=float(gain),
                    score_per_cost=float(gain / low_cost),
                    cost=float(low_cost),
                    cluster=s,
                )

                if best is None or cand.score_per_cost > best.score_per_cost:
                    best = cand
                    best_rho = rho_new

            if best is None or best_rho is None:
                break

            q.append(best)

            if best.fidelity == "high":
                used_high.add(best.idx)
                high_state.commit(best.idx, best_rho)
            else:
                used_low.add(best.idx)
                low_state.commit(best.idx, best_rho)

            spent_s += best.cost

        queues[s] = q

    return _global_greedy_from_queues(queues, total_budget=float(total_budget))


# -----------------------------------------------------------------------------
# Minimal usage note
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print(
        "Import this file and call bas_acquisition(...) or bams_acquisition(...).\n"
        "For bams_acquisition, pass a joint augmented-fidelity GP trained on [z, m]."
    )