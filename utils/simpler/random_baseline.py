#!/usr/bin/env python3
"""
utils.simpler/random_baseline.py
----------------------------------
Random sampling baseline for the SimplerEnv AL experiment.

Acquisition: uniform random draws from the pool without replacement.
No model, no CV — purely a lower-bound reference.
"""

from __future__ import annotations

from typing import List

import numpy as np

from utils.simpler.al_state import ALState
from utils.simpler.domain import Scenario


def random_acquisition(
    state:      ALState,
    batch_size: int,
    rng:        np.random.Generator,
) -> List[Scenario]:
    """
    Select `batch_size` scenarios uniformly at random from state.pool.

    Returns the selected Scenario objects (does NOT remove them from pool —
    the runner script handles state updates).
    """
    n = len(state.pool)
    if n == 0:
        return []
    k   = min(batch_size, n)
    idx = rng.choice(n, size=k, replace=False)
    return [state.pool[i] for i in idx]
