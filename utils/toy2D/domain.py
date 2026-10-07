#!/usr/bin/env python3
"""
utils/toy2D/domain.py
---------------------
Shared 2-D synthetic domain for all toy2D experiments.

Design
------
* Input space  X ⊂ [-3, 3]²  sampled from a 2-D Gaussian.
* Two diamond-shaped failure regions (C1, C2) define where the real-world
  performance metric f_real(x) is high (> 0 inside the diamonds).
* A shifted/scaled copy of the diamond function defines the cheap simulation
  surrogate g_sim(x), correlated with but not identical to f_real.
* Failure = f_real(x) ≤ γ (value INSIDE the diamond is large/positive;
  values outside are negative — we discover the *high* regions).

Conventions
-----------
  diamond_score(x, center, radius) > 0  ↔  x is INSIDE the diamond
  real_mean_fn(x)  ≈  a nonlinear monotone transform of the diamond score
  sim_mean_fn(x)   ≈  a slightly shifted/noisy version of real_mean_fn
"""

import numpy as np

# ---------------------------------------------------------------------------
# Domain parameters
# ---------------------------------------------------------------------------

MU_X    = np.array([0.0, 0.0], dtype=np.float32)
SIGMA_X = np.array([[1.0, 0.0],
                     [0.0, 1.0]], dtype=np.float32)

C1 = np.array([ 1.35,  1.35], dtype=np.float32)   # centre of diamond 1
C2 = np.array([-1.35,  1.35], dtype=np.float32)   # centre of diamond 2
R1 = 0.65                                          # L1 radius of diamond 1
R2 = 0.55                                          # L1 radius of diamond 2

GAMMA = 0.0   # failure threshold: f_real(x) <= GAMMA → failure


# ---------------------------------------------------------------------------
# Input sampling
# ---------------------------------------------------------------------------

def sample_designs(n: int, seed: int = None) -> np.ndarray:
    """Sample n 2-D designs from the Gaussian input distribution."""
    rng = np.random.default_rng(seed)
    x   = rng.uniform(-3.0, 3.0, size=(n, 2)).astype(np.float32)
    # x = rng.multivariate_normal(MU_X, SIGMA_X, size=n).astype(np.float32)
    return np.clip(x, -3.0, 3.0)


# ---------------------------------------------------------------------------
# Ground-truth functions
# ---------------------------------------------------------------------------

def diamond_score(x: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Signed L1-ball distance; positive = inside diamond."""
    return radius - (np.abs(x[..., 0] - center[0]) + np.abs(x[..., 1] - center[1]))


def latent_target_fn(x: np.ndarray) -> np.ndarray:
    """Max of two diamond scores — positive inside either diamond."""
    s1 = diamond_score(x, C1, R1)
    s2 = diamond_score(x, C2, R2)
    return np.maximum(s1, s2)


def real_mean_fn(x: np.ndarray) -> np.ndarray:
    """Ground-truth real-world mean performance metric."""
    base = latent_target_fn(x)
    return (
        1.15 * np.tanh(2.4 * base)
        + 0.10 * np.sin(1.1 * x[..., 0])
        - 0.08 * np.cos(0.9 * x[..., 1])
    )
    


def real_var_fn(x: np.ndarray) -> np.ndarray:
    """Homoscedastic real observation noise."""
    return np.full(x.shape[0], 0.015, dtype=np.float32)


def sim_mean_fn(x: np.ndarray) -> np.ndarray:
    """
    Cheap simulation surrogate: shifted + slightly rescaled diamond function,
    correlated with but not identical to real_mean_fn.
    """
    xs = x.copy()
    xs[..., 0] = xs[..., 0] + 0.18
    xs[..., 1] = xs[..., 1] - 0.12
    # base = np.maximum(
    #     diamond_score(xs, C1, R1 * 1.05),
    #     diamond_score(xs, C2, R2 * 0.95),
    # )
    base = np.maximum(
    diamond_score(xs, C1, R1 * 1.05),
        0.1 * diamond_score(xs, C2, R2 * 0.95),   # 15% strength for C2
    )
    # return (
    #     0.95 * np.tanh(2.0 * base)
    #     + 0.16 * np.sin(0.8 * x[..., 0] + 0.3)
    #     - 0.02
    # )
    return (
        1.15 * np.tanh(2.4 * base)          # ← match real's amplitude and steepness
        + 0.10 * np.sin(1.1 * x[..., 0])    # ← match real's sinusoidal terms exactly
        - 0.08 * np.cos(0.9 * x[..., 1])    # ← so they cancel outside the diamonds
        # no constant offset — real has none
    )
# base = np.maximum(
#         diamond_score(xs, C1, R1 * 1.05),
#         0.05 * diamond_score(xs, C2, R2 * 0.95),  # keep C2 weak if intended
#     )

    

def sim_var_fn(x: np.ndarray) -> np.ndarray:
    """Homoscedastic simulation noise (larger than real)."""
    return np.full(x.shape[0], 0.10, dtype=np.float32)


def sample_real(x: np.ndarray, seed: int = None) -> np.ndarray:
    """Draw noisy real observations at x."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float32)
    return (real_mean_fn(x) + np.sqrt(real_var_fn(x)) * rng.standard_normal(len(x)).astype(np.float32))


def sample_sim(x: np.ndarray, seed: int = None) -> np.ndarray:
    """Draw noisy simulation observations at x."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float32)
    return (sim_mean_fn(x) + np.sqrt(sim_var_fn(x)) * rng.standard_normal(len(x)).astype(np.float32))


# ---------------------------------------------------------------------------
# Membership / evaluation helpers
# ---------------------------------------------------------------------------

def diamond_membership(x: np.ndarray) -> np.ndarray:
    """
    Return integer membership array:
      0 = outside both diamonds
      1 = inside diamond 1
      2 = inside diamond 2
    """
    s1 = diamond_score(x, C1, R1)
    s2 = diamond_score(x, C2, R2)
    out = np.zeros(len(x), dtype=int)
    out[s1 >= 0] = 1
    out[s2 >= 0] = 2
    return out


def is_failure(x: np.ndarray, threshold: float = GAMMA) -> np.ndarray:
    """Binary failure indicator: True if real_mean_fn(x) <= threshold."""
    return real_mean_fn(x) <= threshold


def make_grid(n: int = 120) -> tuple:
    """Return (XX, YY, X_grid) for plotting on [-3, 3]²."""
    g = np.linspace(-3, 3, n, dtype=np.float32)
    XX, YY = np.meshgrid(g, g)
    X_grid = np.stack([XX.ravel(), YY.ravel()], axis=1)
    return XX, YY, X_grid
