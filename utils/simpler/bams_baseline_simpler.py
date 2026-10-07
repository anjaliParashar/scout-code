#!/usr/bin/env python3
"""
simpler_utils/bams_simpler_baseline.py
----------------------------------------
BAMS acquisition for SimplerEnv using the augmented multi-fidelity GP.

Multi-fidelity BAMS acquisition:
  - Single GP trained on [z_encoded, fidelity_code] augmented inputs
    fidelity_code = 1.0 for target (real), 0.0 for proxy (sim)
  - bams_acquisition selects HF or LF scenarios by gain-per-cost
  - z_encoded is the 12-dim encoder output (train_encoder_simpler.py)

The encoder must be pre-trained and passed in; if no encoder is provided
the raw 42-dim feature vector is used directly.
"""

from __future__ import annotations

import os, sys
from typing import List, Optional, Tuple

import numpy as np

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.simpler.al_state import ALState
from utils.simpler.domain   import Scenario, scenarios_to_array
from scout.gp import train_gp
from utils.baseline.bams_baseline  import bams_acquisition


def _encode(X: np.ndarray, encoder) -> np.ndarray:
    """Apply encoder if provided, else return X as-is."""
    if encoder is None:
        return X.astype(np.float64)
    import torch
    with torch.no_grad():
        Z = encoder(torch.tensor(X, dtype=torch.float32))
    return Z.numpy().astype(np.float64)


def bams_simpler_acquisition(
    state:            ALState,
    encoder,                        # trained ScenarioEncoder or None
    batch_size:       int   = 2,
    iteration:        int   = 0,
    random_state:     int   = 0,
    bams_threshold:   float = 0.5,  # y <= threshold = failure
    lf_cost:          float = 0.15,
    n_clusters:       int   = 6,
    gp_max_train:     int   = 200,
) -> Tuple[List[Scenario], List[Scenario]]:
    """
    Select HF (target) and LF (proxy) scenarios using BAMS.

    Returns
    -------
    hf_selected : scenarios for real (target) evaluation
    lf_selected : scenarios for proxy (sim) evaluation
    """
    if not state.train or not state.pool:
        rng = np.random.default_rng(random_state)
        k   = min(batch_size, len(state.pool))
        sel = [state.pool[i] for i in rng.choice(len(state.pool), k, replace=False)]
        return sel, []

    # Encode all scenarios
    X_train       = state.X_train().astype(np.float32)
    X_proxy_train = state.X_proxy_train().astype(np.float32)
    X_pool        = state.X_pool().astype(np.float32)

    Z_train  = _encode(X_train,       encoder)
    Z_proxy  = _encode(X_proxy_train, encoder)
    Z_pool   = _encode(X_pool,        encoder)

    # Build augmented training set [z, fidelity_code]
    M_hf = len(state.train)
    M_lf = len(state.proxy_only)

    Z_hf_aug = np.concatenate([Z_train[:M_hf],
                                np.ones((M_hf, 1))], axis=1)
    y_hf     = state.y_train_mean().astype(np.float64)

    if M_lf > 0:
        Z_lf_aug = np.concatenate([Z_proxy[M_hf:],
                                    np.zeros((M_lf, 1))], axis=1)
        y_lf     = state.y_proxy_train_mean()[M_hf:].astype(np.float64)
        X_aug    = np.concatenate([Z_hf_aug, Z_lf_aug], axis=0)
        y_aug    = np.concatenate([y_hf, y_lf], axis=0)
    else:
        X_aug = Z_hf_aug
        y_aug = y_hf

    # Subsample for GP tractability
    if len(X_aug) > gp_max_train:
        rng_gp = np.random.default_rng(random_state + iteration)
        sub    = rng_gp.choice(len(X_aug), gp_max_train, replace=False)
        X_aug  = X_aug[sub]
        y_aug  = y_aug[sub]

    try:
        gp_mf = train_gp(X_aug, y_aug)
    except Exception as e:
        print(f"  [BAMS] GP failed ({e}), falling back to random")
        rng = np.random.default_rng(random_state)
        k   = min(batch_size, len(state.pool))
        sel = [state.pool[i] for i in rng.choice(len(state.pool), k, replace=False)]
        return sel, []

    lf_local = np.arange(len(state.pool))
    selected_local, selected_fid, _ = bams_acquisition(
        pool_Z                     = Z_pool,
        gp_mf                      = gp_mf,
        total_budget               = float(batch_size),
        high_threshold             = bams_threshold,
        high_cost                  = 1.0,
        low_cost                   = lf_cost,
        n_clusters                 = min(n_clusters, max(2, len(state.pool) // 5)),
        random_state               = random_state + iteration,
        low_fidelity_available_idx = lf_local,
    )

    hf_local = selected_local[selected_fid == "high"]
    lf_local = selected_local[selected_fid == "low"]

    hf_selected = [state.pool[i] for i in hf_local]
    lf_selected = [state.pool[i] for i in lf_local]
    print(f"  [BAMS] HF={len(hf_selected)}  LF={len(lf_selected)}  "
          f"HF ids={[s.scenario_id for s in hf_selected]}", flush=True)
    return hf_selected, lf_selected
