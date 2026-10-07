#!/usr/bin/env python3
"""
scout/al.py
-----------------
Active-learning acquisition logic, lambda schedule, and hook computation.

KEY FIX vs. original:
    Acquisition scores lower target values as more severe.
    cv_mean is negated before normalisation so that low values score high.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def minmax01(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Map finite values of x to [0, 1]; imputes non-finite with median."""
    x = np.asarray(x, dtype=float)
    fin = np.isfinite(x)
    if not fin.any():
        return np.zeros_like(x)
    med = np.nanmedian(x[fin])
    x = np.where(fin, x, med)
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi - lo < eps else (x - lo) / (hi - lo + eps)


# ---------------------------------------------------------------------------
# Lambda schedule
# ---------------------------------------------------------------------------

def lambda_schedule(iteration: int) -> float:
    """
    Weight on MI in the acquisition blend.
    Early iterations (< 5)  : pure MI  (explore new support regions).
    Middle iterations (5-14) : 50/50 blend.
    Late iterations  (≥ 15) : pure CV  (exploit low-target failure regions).
    """
    if iteration < 5:
        return 1.0
    if iteration < 15:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Batch selection — TARGETS LOW cv_mean (failure = low target value)
# ---------------------------------------------------------------------------

def select_batch(
    iteration: int,
    shortlist_global_idx: np.ndarray,
    mi_vals: np.ndarray,
    cv_mean: np.ndarray,
    cv_corr: np.ndarray,
    batch_size: int  = 10,
    low_corr_pool: int = 50,
    corr_mode: str   = "abs",
) -> tuple:
    """
    Select a batch of `batch_size` points from the MI shortlist.

    Acquisition score (higher = better candidate):

        score = λ · norm(MI)  +  (1 − λ) · norm(−cv_mean)
                                           ^^^^^^^^^^^^^^^^
                Negate cv_mean so that LOW target values
                score HIGH and get selected.

    For iterations > 10 a two-stage filter is applied:
        1. Take top-50 by acquisition score (MI pool).
        2. Within that, keep the `low_corr_pool` points with lowest |corr|
           (CV estimate more reliable when corr is not saturated).
        3. From those, pick the `batch_size` with the LOWEST cv_mean.

    Returns
    -------
    selected_global_idx : global indices of chosen points
    acquisition         : full acquisition array (shortlist-aligned)
    chosen_local        : local indices within shortlist
    """
    # lam = lambda_schedule(iteration)

    # Negate cv_mean so low target values score high
    lam=1.0
    acquisition = lam * minmax01(mi_vals) + (1.0 - lam) * minmax01(-cv_mean)

    n = len(shortlist_global_idx)

    if iteration > 25:
        if corr_mode not in {"abs", "raw"}:
            raise ValueError("corr_mode must be 'abs' or 'raw'.")
        # Stage 1: MI-rich pool
        # pool = np.argsort(acquisition)[-min(50, n):]
        # # Stage 2: low-correlation subset (more reliable CV)
        # corr_score = np.abs(cv_corr) if corr_mode == "abs" else cv_corr
        # lo_corr    = np.argsort(corr_score[pool])[:min(low_corr_pool, len(pool))]
        # lo_corr_pool = pool[lo_corr]
        # # Stage 3: among those, pick LOWEST cv_mean (most likely low target value)
        # chosen_local = lo_corr_pool[np.argsort(cv_mean[lo_corr_pool])[:batch_size]]
        chosen_local = np.argsort(cv_mean)[0:batch_size]
    else:
     
        # Early: pure acquisition (MI-dominated exploration)
        chosen_local = np.argsort(acquisition)[-50:]

        chosen_local = np.argsort(cv_mean[chosen_local])[0:batch_size]



    return (
        np.asarray(shortlist_global_idx, int)[chosen_local],
        acquisition,
        chosen_local,
    )


# ---------------------------------------------------------------------------
# Hook computation: low-target anchor points spread across t-SNE space
# ---------------------------------------------------------------------------

def compute_hooks(
    Z: np.ndarray,
    y_target: np.ndarray,
    vis_idx: np.ndarray,
    n_hooks: int = 20,
    low_quantile: float = 0.05,
    min_sep_frac: float = 0.06,
) -> np.ndarray:
    """
    Select `n_hooks` global indices corresponding to low-target scenarios that
    are spatially spread across t-SNE space.

    Strategy
    --------
    1. Candidate pool = bottom `low_quantile` of y_target[vis_idx].
    2. Sort candidates by ascending y_target (worst failures first).
    3. Greedy spatial spread: keep a candidate only if it is farther than
       `min_sep_frac * t-SNE-diagonal` from all already-selected hooks.
    4. If fewer than n_hooks pass the distance test, relax and fill from
       the remaining candidates.

    Parameters
    ----------
    Z           : (len(vis_idx), 2) t-SNE coordinates.
    y_target    : (n_total,) target values for ALL rows.
    vis_idx     : global indices corresponding to rows of Z.
    n_hooks     : desired number of hooks.
    low_quantile: pool = bottom this fraction of target values.
    min_sep_frac: minimum inter-hook separation as fraction of t-SNE diagonal.

    Returns
    -------
    hook_global_idx : (n_hooks,) int array of global indices.
    """
    y_vis  = y_target[vis_idx]
    thresh = np.nanquantile(y_vis, low_quantile)
    cand   = np.where(y_vis <= thresh)[0]           # local indices into vis_idx

    if len(cand) == 0:
        print("[WARN] No hook candidates found; returning empty.")
        return np.array([], dtype=int)

    # Sort by ascending target value (lowest target value first = worst failures)
    cand = cand[np.argsort(y_vis[cand])]

    x_range  = Z[:, 0].ptp()
    y_range  = Z[:, 1].ptp()
    min_dist = min_sep_frac * np.sqrt(x_range**2 + y_range**2)

    selected_local = []
    for loc in cand:
        if len(selected_local) == 0:
            selected_local.append(loc)
        else:
            pts = Z[np.array(selected_local)]
            if np.linalg.norm(pts - Z[loc], axis=1).min() >= min_dist:
                selected_local.append(loc)
        if len(selected_local) >= n_hooks:
            break

    # Relax spatial constraint if we need more hooks
    if len(selected_local) < n_hooks:
        existing = set(selected_local)
        for loc in cand:
            if loc not in existing:
                selected_local.append(loc)
                existing.add(loc)
            if len(selected_local) >= n_hooks:
                break

    hook_local = np.array(selected_local, dtype=int)
    hook_global = vis_idx[hook_local]
    print(
        f"[INFO] {len(hook_global)} hooks selected; "
        f"TTC range [{y_vis[hook_local].min():.4g}, "
        f"{y_vis[hook_local].max():.4g}]"
    )
    return hook_global
