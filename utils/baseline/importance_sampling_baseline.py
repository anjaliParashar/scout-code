#!/usr/bin/env python3
"""
utils/baseline/importance_sampling_baseline.py
-----------------------------------------------
Importance-sampling baseline using the cheap simulation as a proxy
for the real-world performance metric.

Acquisition rule
----------------
  q(x) ∝ softmax(β · sim_score(x))

Points are sampled proportionally to q.  High-sim-score regions are
up-weighted, exploiting the correlation between sim and real without
ever observing real data beyond the initial seed.

This mirrors the 'is' strategy in the original notebook and the
DS/DS-GP baselines in the BAMS paper.
"""

import numpy as np


def importance_sampling_acquisition(
    pool_X:      np.ndarray,
    sim_score_fn,            # callable: (N, d) → (N,) float array
    batch_size:  int,
    beta:        float = 8.0,
    rng:         np.random.Generator = None,
) -> np.ndarray:
    """
    Select a batch by sampling proportional to soft-max importance weights
    derived from the cheap simulation score.

    Parameters
    ----------
    pool_X       : (N, d) candidate pool
    sim_score_fn : function mapping pool_X → scalar scores (higher = better)
    batch_size   : number of points to select
    beta         : softmax temperature (higher → greedier)
    rng          : seeded Generator; uses global np.random if None

    Returns
    -------
    idx : (batch_size,) int array of selected pool indices
    """
    if rng is None:
        rng = np.random.default_rng()

    scores = np.asarray(sim_score_fn(pool_X), dtype=np.float64)
    # Standardize scores before applying softmax for numerical stability
    scores = (scores - np.mean(scores)) / (np.std(scores) + 1e-8)
    logits = beta * scores
    logits = logits - np.max(logits)          # softmax stability
    q      = np.exp(logits)
    q      = q / q.sum()

    n   = len(pool_X)
    idx = rng.choice(n, size=batch_size, replace=False, p=q)
    return idx


def importance_weights(
    pool_X:      np.ndarray,
    sim_score_fn,
    beta:        float = 8.0,
) -> np.ndarray:
    """
    Return the full importance-weight distribution q(x) for every point
    in pool_X.  Useful for diagnostic plots.
    """
    scores = np.asarray(sim_score_fn(pool_X), dtype=np.float64)
    scores = (scores - np.mean(scores)) / (np.std(scores) + 1e-8)
    logits = beta * scores - np.max(beta * scores)
    q      = np.exp(logits)
    return q / q.sum()
