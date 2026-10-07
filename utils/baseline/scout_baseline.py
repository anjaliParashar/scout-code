#!/usr/bin/env python3
"""
utils/baseline/scout_baseline.py
----------------------------------
SCOUT baseline — MI + BNN-CV active-learning pipeline.

Changes vs original
--------------------
1. Uses the shared MI estimator where:
     - K is selected automatically via the elbow method (n_clusters=0)
     - Prior p(Z|x) is fully data-driven (Option A — no pi_new):
         p(Z=z_k|x) = w_k / (W+1),   p(Z=z_new|x) = 1/(W+1)
         w_k = exp(-||x-c_k||² / 2l_k²),  W = Σ_k w_k
   pi_new is no longer passed to fit_support_and_compute_mi.

2. Batch selection after CV estimation uses cluster-and-pick:
     a. KMeans the MI shortlist into batch_size clusters.
     b. From each cluster pick the index with the highest CV mean.
   This guarantees spatial diversity — one representative per cluster.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.cluster import KMeans

from scout.mi import fit_support_and_compute_mi
from scout.bnn import HeteroBNNEmbedding, train_bnn
from scout.cv import compute_local_cv_parallel
from scout.al import lambda_schedule, compute_hooks
from scout.plot import plot_acquisition, plot_cumulative_selection


# ---------------------------------------------------------------------------
# Cluster-and-pick batch selection
# ---------------------------------------------------------------------------

def _cluster_and_pick(
    sl_idx:       np.ndarray,   # (S,) global indices of MI shortlist
    X_all:        np.ndarray,   # (N, d) full embedding matrix
    cv_mean:      np.ndarray,   # (S,) CV mean for each shortlist point
    batch_size:   int,
    random_state: int = 0,
) -> tuple:
    """
    Select `batch_size` spatially diverse candidates from the MI shortlist.

    Steps:
      1. Cluster shortlist into `batch_size` groups via KMeans on X_all[sl_idx].
      2. Within each cluster pick the index with the highest CV mean.

    Returns
    -------
    sel_global : (≤batch_size,) global indices
    sel_local  : (≤batch_size,) local indices into sl_idx / cv_mean
    """
    S = len(sl_idx)
    k = min(batch_size, S)
    if k == 0:
        return np.array([], int), np.array([], int)
    if k == S:
        return sl_idx.copy(), np.arange(S)

    km = KMeans(n_clusters=k, random_state=random_state,
                n_init="auto", max_iter=300)
    cluster_labels = km.fit_predict(X_all[sl_idx])   # (S,)

    sel_local = []
    for c in range(k):
        in_cluster = np.where(cluster_labels == c)[0]
        if len(in_cluster) == 0:
            continue
        sel_local.append(int(in_cluster[np.argmin(cv_mean[in_cluster])]))

    sel_local  = np.array(sel_local, int)

    return sl_idx[sel_local], sel_local


# ---------------------------------------------------------------------------
# Main AL loop
# ---------------------------------------------------------------------------

def run_scout(
    # ── data ──────────────────────────────────────────────────────────────
    df:                pd.DataFrame,
    X_all:             np.ndarray,      # (N, d) standardised embedding
    y_target:          np.ndarray,      # (N,) target metric
    y_cv:              np.ndarray,      # (N,) proxy values
    # ── shared t-SNE ──────────────────────────────────────────────────────
    Z_tsne:            np.ndarray,      # (|vis_idx|, 2)
    vis_idx:           np.ndarray,      # global indices for Z_tsne rows
    # ── shared initial sample ─────────────────────────────────────────────
    initial_idx:       np.ndarray,      # (n_init,) global indices
    # ── output ────────────────────────────────────────────────────────────
    output_dir:        Path,
    # ── AL hyperparameters ────────────────────────────────────────────────
    n_acq_iters:       int   = 20,
    batch_acq_size:    int   = 10,
    candidate_pool_size: int = 10_000,
    mi_shortlist_size: int   = 500,
    # ── BNN ───────────────────────────────────────────────────────────────
    epochs_per_iter:   int   = 1200,
    dropout:           float = 0.10,
    lr:                float = 1e-3,
    weight_decay:      float = 1e-6,
    batch_size_bnn:    int   = 4096,
    # ── CV ────────────────────────────────────────────────────────────────
    n_pair_local:      int   = 10,
    k_unpaired:        int   = 20,
    cv_radius:         float = 1.5,
    cv_fallback_k:     int   = 80,
    cv_n_jobs:         int   = -1,
    low_corr_pool:     int   = 50,
    corr_mode:         str   = "abs",
    # ── MI — n_clusters=0 → auto elbow; >0 → fixed K ──────────────────────
    n_clusters:        int   = 0,
    radius_quantile:   float = 0.90,
    eps_exist:         float = 0.04,
    tau_scale:         float = 1.3,
    novelty_radius:    float = 0.75,
    gamma:             float = 0.25,
    # ── pi_new kept as stub for backward compat — not used ─────────────────
    pi_new:            float = None,
    # ── hooks ─────────────────────────────────────────────────────────────
    n_hooks:           int   = 20,
    hook_quantile:     float = 0.05,
    # ── misc ──────────────────────────────────────────────────────────────
    random_state:      int   = 42,
    device:            str   = "cpu",
    make_tsne_plots:   bool  = True,
) -> np.ndarray:
    """
    Run the SCOUT (MI + BNN-CV) active-learning loop.

    Returns
    -------
    train_idx : (T,) final selected global indices in acquisition order
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if make_tsne_plots:
        (output_dir / "tsne_this_iter").mkdir(exist_ok=True)
        (output_dir / "tsne_cumulative").mkdir(exist_ok=True)

    n   = len(X_all)
    rng = np.random.default_rng(random_state)
    g2l = {g: l for l, g in enumerate(vis_idx)}

    # ── Hook anchors ──────────────────────────────────────────────────────
    hook_global = compute_hooks(
        Z=Z_tsne, y_target=y_target, vis_idx=vis_idx,
        n_hooks=n_hooks, low_quantile=hook_quantile,
    )
    hook_local = np.array([g2l[g] for g in hook_global if g in g2l], int)
    Z_hooks    = Z_tsne[hook_local] if len(hook_local) else np.empty((0, 2))
    np.save(output_dir / "hook_global_idx.npy", hook_global)
    print(f"[SCOUT] {len(hook_global)} hooks saved.")

    # ── Initial sample ────────────────────────────────────────────────────
    train_idx     = initial_idx.copy()
    selected_mask = np.zeros(n, bool)
    selected_mask[train_idx] = True
    acq_logs      = []

    # ── AL loop ───────────────────────────────────────────────────────────
    for it in range(n_acq_iters):
        lam = lambda_schedule(it)
        print(f"\n{'='*70}")
        print(f"[SCOUT ITER {it:02d}]  λ={lam:.2f}  train_size={len(train_idx):,}")
        print(f"{'='*70}")

        X_train = X_all[train_idx]
        y_train = y_target[train_idx]

        # ── Train BNN ─────────────────────────────────────────────────────
        model = HeteroBNNEmbedding(
            input_dim=X_all.shape[1], p_drop=dropout
        ).to(device)
        train_bnn(model, X_train, y_train,
                  epochs=epochs_per_iter, lr=lr,
                  weight_decay=weight_decay, device=device)

        # ── Candidate pool ────────────────────────────────────────────────
        unselected = np.where(~selected_mask)[0]
        if len(unselected) == 0:
            print("[STOP] Pool exhausted.")
            break
        candidate_idx = (
            rng.choice(unselected, candidate_pool_size, replace=False)
            if len(unselected) > candidate_pool_size else unselected
        )

        # ── MI (elbow K, data-driven prior, no pi_new) ────────────────────
        mi_cand, _, _, _, _, _ = fit_support_and_compute_mi(
            X_support       = X_train.astype(np.float64),
            X_query         = X_all[candidate_idx].astype(np.float64),
            n_clusters      = n_clusters,
            radius_quantile = radius_quantile,
            eps_exist       = eps_exist,
            tau_scale       = tau_scale,
            novelty_radius  = novelty_radius,
            gamma           = gamma,
            random_state    = random_state + it,
        )
        mi_cand = np.nan_to_num(mi_cand, nan=0.0, posinf=0.0, neginf=0.0)

        # MI on vis points (t-SNE background)
        mi_vis, _, _, _, _, _ = fit_support_and_compute_mi(
            X_support       = X_train.astype(np.float64),
            X_query         = X_all[vis_idx].astype(np.float64),
            n_clusters      = n_clusters,
            radius_quantile = radius_quantile,
            eps_exist       = eps_exist,
            tau_scale       = tau_scale,
            novelty_radius  = novelty_radius,
            gamma           = gamma,
            random_state    = random_state + it,
        )
        mi_vis = np.nan_to_num(mi_vis, nan=0.0, posinf=0.0, neginf=0.0)

        # ── MI shortlist ──────────────────────────────────────────────────
        sl_size  = min(mi_shortlist_size, len(candidate_idx))
        sl_local = np.argsort(mi_cand)[-sl_size:]
        sl_idx   = candidate_idx[sl_local]
        mi_sl    = mi_cand[sl_local]
        print(f"  shortlist={len(sl_idx):,}  "
              f"MI max={mi_sl.max():.4g}  median={np.median(mi_sl):.4g}")

        # ── CV on shortlist ───────────────────────────────────────────────
        cv_mean, cv_var, beta_hat, cv_corr = compute_local_cv_parallel(
            model            = model,
            X_query          = X_all[sl_idx],
            query_global_idx = sl_idx,
            X_target         = X_all,
            y_cv             = y_cv,
            n_pair_local     = n_pair_local,
            k_unpaired       = k_unpaired,
            radius           = cv_radius,
            fallback_k       = cv_fallback_k,
            n_jobs           = cv_n_jobs,
            device           = device,
            batch_size       = batch_size_bnn,
        )
        print(f"  CV min/med/max = "
              f"{cv_mean.min():.4g} / {np.median(cv_mean):.4g} / "
              f"{cv_mean.max():.4g}")

        sl_idx = sl_idx[cv_mean<=0.3]
        mi_sl = mi_sl[cv_mean<=0.3]
        cv_corr = cv_corr[cv_mean<=0.3]
        cv_mean = cv_mean[cv_mean<=0.3]
        

        
        # ── Cluster-and-pick selection ────────────────────────────────────
        sel_global, sel_local = _cluster_and_pick(
            sl_idx       = sl_idx,
            X_all        = X_all,
            cv_mean      = cv_mean,
            batch_size   = batch_acq_size,
            random_state = random_state + it * 1000,
        )
        sel_global = np.array(
            [g for g in sel_global if not selected_mask[g]], int
        )

        # ── t-SNE plot ────────────────────────────────────────────────────
        if make_tsne_plots:
            prev_in_vis = np.array(
                [g2l[g] for g in train_idx  if g in g2l], int)
            sel_in_vis  = np.array(
                [g2l[g] for g in sel_global if g in g2l], int)
            vis_prev = np.zeros(len(vis_idx), bool)
            vis_sel  = np.zeros(len(vis_idx), bool)
            vis_prev[prev_in_vis] = True
            vis_sel[sel_in_vis]   = True
            iter_df = pd.DataFrame({
                "tsne_1":                Z_tsne[:, 0],
                "tsne_2":                Z_tsne[:, 1],
                "mi":                    mi_vis,
                "is_previous_train":     vis_prev,
                "is_selected_this_iter": vis_sel,
            })
            plot_acquisition(
                iter_df, "mi",
                output_dir / "tsne_this_iter" / f"iter_{it:02d}_mi.png",
                f"SCOUT Iter {it:02d} — MI field",
                colorbar_label="MI", cmap="hot", Z_hooks=Z_hooks,
            )

        # ── Log ───────────────────────────────────────────────────────────
        chosen_set = set(sel_global.tolist())
        for j, gj in enumerate(sl_idx):
            acq_logs.append(dict(
                iteration=it, lam=lam, global_idx=int(gj),
                is_selected=int(gj in chosen_set),
                mi=float(mi_sl[j]),
                cv_mean=float(cv_mean[j]),
                cv_var=float(cv_var[j]),
                cv_beta=float(beta_hat[j]),
                cv_corr=float(cv_corr[j]),
                target=float(y_target[gj]),
                cv_col_val=float(y_cv[gj]),
            ))

        if len(sel_global) == 0:
            print("[WARN] No new points selected. Stopping early.")
            break

        selected_mask[sel_global] = True
        train_idx = np.concatenate([train_idx, sel_global])

        sel_mask = np.isin(sl_idx, sel_global)
        print(f"  selected {len(sel_global)}  "
              f"MI/CV/corr = "
              f"{mi_sl[sel_mask].mean():.4g} / "
              f"{cv_mean[sel_mask].mean():.4g} / "
              f"{cv_corr[sel_mask].mean():.4g}")
        print(f"  TTC: {y_target[sel_global].tolist()}")

        if make_tsne_plots:
            plot_cumulative_selection(
                Z                 = Z_tsne,
                vis_idx           = vis_idx,
                train_idx_so_far  = train_idx,
                initial_train_idx = initial_idx,
                y_target          = y_target,
                hook_global_idx   = hook_global,
                output_path       = (output_dir / "tsne_cumulative"
                                     / f"iter_{it:02d}_cumulative.png"),
                iteration         = it,
                target_col_label  = "closed_loop_ttc",
            )
        if len(train_idx)==250:
            break

    # ── Save ──────────────────────────────────────────────────────────────
    np.save(output_dir / "train_indices.npy", train_idx)
    pd.DataFrame(acq_logs).to_csv(
        output_dir / "acquisition_log.csv", index=False)
    sel_df = df.iloc[train_idx].copy()
    sel_df["selected_order"] = np.arange(len(train_idx))
    sel_df.to_csv(output_dir / "selected_points.csv", index=False)

    print(f"\n[SCOUT DONE]  final_train_size={len(train_idx):,}  "
          f"outputs → {output_dir}")
    return train_idx
