#!/usr/bin/env python3
"""
utils/baseline/random_baseline.py
----------------------------------
Random sampling — uniform draws from the pool without replacement.
The simplest possible active-learning policy; used as a lower-bound reference.
"""

import numpy as np


def random_acquisition(
    pool_X:     np.ndarray,
    batch_size: int,
    rng:        np.random.Generator,
) -> np.ndarray:
    """
    Return `batch_size` indices into `pool_X` sampled uniformly at random.

    Parameters
    ----------
    pool_X     : (N, d) candidate pool
    batch_size : number of points to select
    rng        : seeded Generator for reproducibility

    Returns
    -------
    idx : (batch_size,) int array of selected pool indices
    """
    n = len(pool_X)
    idx = np.arange(n)
    rng.shuffle(idx)
    return idx[:batch_size]
