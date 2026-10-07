#!/usr/bin/env python3
"""
utils/baseline/gp_micv_baseline.py
------------------------------------
MI + μ_CV objective with a GP surrogate.

Acquisition rule (mirrors `mi_cv` / `select_micv_from_high_mi_clusters`
from the original notebook):

  1.  Compute MI(x) for all candidates → keep top-`top_mi_frac` as a
      high-MI pool.
  2.  Re-cluster the high-MI pool.
  3.  From each cluster, select the point with highest CV-estimated mean.
  4.  Rank cluster representatives by CV mean; return top-`batch_size`.
  5.  Fill remainder (if any clusters < batch_size) from high-MI pool
      ranked by CV mean.

The lambda schedule determines how MI vs CV is weighted early vs late.
"""

import numpy as np

from utils.toy2D.mi_toy import (
    fit_support_and_compute_mi,
    compute_cv_batch,
    cluster_by_radius,
)


# ---------------------------------------------------------------------------
# Acquisition function
# ---------------------------------------------------------------------------

def gp_micv_acquisition(
    pool_X:         np.ndarray,
    X_support:      np.ndarray,
    gp,
    batch_size:     int,
    iteration:      int   = 0,
    top_mi_frac:    float = 0.12,
    min_top:        int   = 20,
    cluster_eps:    float = 0.35,
    cv_r_local:     float = 0.35,
    cv_n_pair:      int   = 20,
    cv_k_unpaired:  int   = 120,
    cv_n_draws:     int   = 32,
    cv_clip:        float = 1.0,
    # MI hyperparameters — passed explicitly, matching scout.mi conventions
    n_clusters:     int   = 10,
    radius_quantile: float = 0.90,
    pi_new:         float = 0.18,
    eps_exist:      float = 0.04,
    tau_scale:      float = 1.3,
    novelty_radius: float = 0.75,
    gamma:          float = 0.25,
    random_state:   int   = 0,
) -> np.ndarray:
    """
    Select `batch_size` points from `pool_X` using MI + CV(GP) acquisition.

    Parameters
    ----------
    pool_X      : (N, d) candidate pool
    X_support   : (M, d) current training set (used to fit MI support model)
    gp          : fitted BoTorch GP surrogate (used for CV estimation)
    batch_size  : number of points to acquire
    iteration   : current round (narrows top_mi_frac after round 2)
    top_mi_frac : fraction of pool used as high-MI candidates (halved after round 2)
    min_top     : minimum absolute size of the high-MI candidate pool
    cluster_eps : radius for re-clustering the high-MI pool (batch-diversity step)
    cv_*        : parameters forwarded to compute_cv_batch
    n_clusters … gamma : MI hyperparameters forwarded to fit_support_and_compute_mi

    Returns
    -------
    idx : (batch_size,) int array into pool_X
    """
    # Adapt fraction based on iteration (match notebook behaviour)
    if iteration > 6:
        top_mi_frac = min(top_mi_frac, 0.05)
        # novelty_radius=0.1

    # Step 1 — MI values for the entire pool via fit_support_and_compute_mi

    mi_vals, _, _,_,_,_ = fit_support_and_compute_mi(
        X_support=X_support,
        X_query=pool_X,
        n_clusters=n_clusters,
        radius_quantile=radius_quantile,
        pi_new=pi_new,
        eps_exist=eps_exist,
        tau_scale=tau_scale,
        novelty_radius=novelty_radius,
        gamma=gamma,
        random_state=random_state,
    )
    mi_vals = np.nan_to_num(mi_vals, nan=0.0)

    # Step 2 — High-MI subset
    valid   = np.where(np.isfinite(mi_vals))[0]
    n_top   = max(min_top, int(top_mi_frac * len(valid)))
    n_top   = min(n_top, len(valid))
    if iteration<4:
        top_idx = valid[np.argsort(mi_vals[valid])[::-1][:n_top]]
    else:
        top_idx = np.arange(len(pool_X))
    
    # Step 3 — CV values for the high-MI subset

    
    cv_vals_top = compute_cv_batch(
        gp, pool_X[top_idx],
        R_local=cv_r_local,
        n_pair=cv_n_pair,
        k_unpaired=cv_k_unpaired,
        n_f_draws=cv_n_draws,
        clip=cv_clip,
    )
    # Build local cv array indexed by pool position
    cv_full = np.full(len(pool_X), np.nan)
    cv_full[top_idx] = cv_vals_top

    # Step 4 — Re-cluster and pick best CV per cluster
    high_mi_clusters = cluster_by_radius(pool_X[top_idx], eps=cluster_eps)
    representatives  = []

    X_top  = pool_X[top_idx]
    cv_top = cv_vals_top

    for cluster_pts in high_mi_clusters:
        # Find local indices within top_idx for this cluster
        local_members = []
        for pt in cluster_pts:
            hits = np.where(np.all(np.isclose(X_top, pt[None, :]), axis=1))[0]
            local_members.extend(hits.tolist())
        local_members = np.array(sorted(set(local_members)), dtype=int)
        valid_local   = local_members[np.isfinite(cv_top[local_members])]
        if len(valid_local) == 0:
            continue
        best_local = valid_local[np.argmax(cv_top[valid_local])]
        representatives.append(top_idx[best_local])

    if not representatives:
        # Fallback: just return top-CV from high-MI pool
        valid_top = top_idx[np.isfinite(cv_vals_top)]
        order     = valid_top[np.argsort(cv_full[valid_top])[::-1]]
        return order[:batch_size]

    # Step 5 — Rank representatives by CV, fill if needed
    rep_arr = np.array(representatives, dtype=int)
    rep_ord = rep_arr[np.argsort(cv_full[rep_arr])[::-1]]
    selected = rep_ord[:batch_size].tolist()

    if len(selected) < batch_size:
        remaining = np.setdiff1d(top_idx, np.array(selected, dtype=int))
        valid_rem = remaining[np.isfinite(cv_full[remaining])]
        fill = valid_rem[np.argsort(cv_full[valid_rem])[::-1]]
        selected.extend(fill[: batch_size - len(selected)].tolist())

    return np.array(selected, dtype=int)
