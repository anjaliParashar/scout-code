#!/usr/bin/env python3
"""
simpler_utils/bnn_cv_baseline.py
----------------------------------
BNN-CV acquisition for SimplerEnv.

Trains a BNN on current target observations, then for each pool scenario
evaluates the CV-corrected expected performance mu_CV using:
  f       : 4 target BNN posterior draws at x
  g_paired: 4 proxy  BNN posterior draws at x
  g_unpaired: proxy BNN draws at 10 perturbed x's (one draw each)

Selects the batch_size scenarios with the LOWEST mu_CV
(lowest = most likely to fail = highest AL value).

Two BNNs are trained:
  target BNN : on (X_train, y_target_mean)   — M scenarios
  proxy  BNN : on (X_proxy_train, y_proxy_mean) — 5M scenarios
"""

from __future__ import annotations

import os, sys
from typing import List

import numpy as np

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.simpler.al_state import ALState
from utils.simpler.domain   import Scenario, scenarios_to_array
from utils.simpler.cv_utils_simpler import (
    cv_estimate, sample_perturbed_scenarios, N_PERTURB_SCENARIOS,
)
from scout.bnn import HeteroBNNEmbedding, train_bnn, batched_posterior_draws

FEATURE_DIM    = 42
N_TARGET_DRAWS = 4
N_PROXY_DRAWS  = 4   # g_paired


def _train_bnn(X, y, epochs, lr, weight_decay, p_drop, device, label):
    print(f"  [BNN-CV] Training {label} BNN (n={len(X)}, epochs={epochs})…",
          flush=True)
    model = HeteroBNNEmbedding(input_dim=FEATURE_DIM, p_drop=p_drop).to(device)
    train_bnn(model, X, y, epochs=epochs, lr=lr,
              weight_decay=weight_decay, device=device)
    return model


def _draws_at(model, scenarios, n_draws, device):
    X = scenarios_to_array(scenarios).astype(np.float32)
    return batched_posterior_draws(model, X, n_draws=n_draws, device=device)


def bnn_cv_acquisition(
    state:       ALState,
    batch_size:  int  = 2,
    iteration:   int  = 0,
    random_state: int = 0,
    bnn_epochs:  int  = 800,
    bnn_lr:      float = 1e-3,
    bnn_weight_decay: float = 1e-6,
    bnn_p_drop:  float = 0.10,
    device:      str  = "cpu",
) -> List[Scenario]:
    """
    Select batch_size scenarios with lowest CV-estimated target mean.
    Evaluates mu_CV for EVERY pool scenario (no MI shortlisting).
    """
    if not state.train or not state.pool:
        rng = np.random.default_rng(random_state)
        k = min(batch_size, len(state.pool))
        return [state.pool[i] for i in rng.choice(len(state.pool), k, replace=False)]

    # Train both BNNs
    target_model = _train_bnn(
        state.X_train().astype(np.float32), state.y_train_mean(),
        bnn_epochs, bnn_lr, bnn_weight_decay, bnn_p_drop, device, "target")
    proxy_model = _train_bnn(
        state.X_proxy_train().astype(np.float32), state.y_proxy_train_mean(),
        bnn_epochs, bnn_lr, bnn_weight_decay, bnn_p_drop, device, "proxy")

    pool = state.pool
    S    = len(pool)

    # Vectorised draws over entire pool
    f_draws = _draws_at(target_model, pool, N_TARGET_DRAWS, device)  # (4, S)
    g_draws = _draws_at(proxy_model,  pool, N_PROXY_DRAWS,  device)  # (4, S)

    # Perturbed draws for g_unpaired
    perturb_rng = np.random.default_rng(random_state + iteration * 7919)
    all_perturbed = []
    for sc in pool:
        all_perturbed.extend(sample_perturbed_scenarios(sc, rng=perturb_rng))
    g_unp_flat = _draws_at(proxy_model, all_perturbed, n_draws=1, device=device)[0]

    # CV per pool scenario
    cv_vals = np.full(S, np.nan)
    for j in range(S):
        f_j   = f_draws[:, j].astype(np.float64)
        g_p_j = g_draws[:, j].astype(np.float64)
        g_u_j = g_unp_flat[
            j * N_PERTURB_SCENARIOS:(j + 1) * N_PERTURB_SCENARIOS
        ].astype(np.float64)
        cv_vals[j] = cv_estimate(f_j, g_p_j, g_u_j).mu_hat

    fallback  = float(np.mean(state.y_train_mean()))
    cv_vals   = np.where(np.isfinite(cv_vals), cv_vals, fallback)

    # Select lowest mu_CV (most likely to fail)
    order    = np.argsort(cv_vals)
    selected = [pool[i] for i in order[:batch_size]]
    print(f"  [BNN-CV] mu_CV: min={cv_vals.min():.4g}  "
          f"max={cv_vals.max():.4g}  selected={[s.scenario_id for s in selected]}",
          flush=True)
    return selected
