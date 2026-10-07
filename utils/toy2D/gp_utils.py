#!/usr/bin/env python3
"""
utils/toy2D/gp_utils.py
-----------------------
BoTorch / GPyTorch GP surrogate shared by all toy2D scripts.
"""

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

_DTYPE  = torch.double
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def _tx(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=_DTYPE, device=_DEVICE)


def _ty(y: np.ndarray) -> torch.Tensor:
    return torch.tensor(y[:, None], dtype=_DTYPE, device=_DEVICE)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_gp(X_np: np.ndarray, y_np: np.ndarray) -> SingleTaskGP:
    """Fit a BoTorch SingleTaskGP with Normalize + Standardize transforms."""
    gp = SingleTaskGP(
        train_X  = _tx(X_np),
        train_Y  = _ty(y_np),
        input_transform   = Normalize(d=X_np.shape[1]),
        outcome_transform = Standardize(m=1),
    ).to(device=_DEVICE, dtype=_DTYPE)

    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    gp.eval()
    gp.likelihood.eval()
    return gp


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def gp_predict(gp: SingleTaskGP, X_np: np.ndarray) -> tuple:
    """
    Return (mean, variance) arrays of shape (N,).
    Variance is the posterior predictive variance (noise-free).
    """
    post = gp.posterior(_tx(X_np))
    mean = post.mean.squeeze(-1).cpu().numpy()
    var  = post.variance.squeeze(-1).clamp_min(1e-12).cpu().numpy()
    return mean, var


@torch.no_grad()
def gp_sample_functions(gp: SingleTaskGP, X_np: np.ndarray, n_draws: int = 1) -> np.ndarray:
    """
    Draw n_draws function samples from the GP posterior at X_np.

    Returns
    -------
    samples : (n_draws, N) float64 array.
    """
    post    = gp.posterior(_tx(X_np))
    samples = post.rsample(sample_shape=torch.Size([n_draws]))
    return samples.squeeze(-1).detach().cpu().numpy()


@torch.no_grad()
def gp_posterior_prob_below(
    gp: SingleTaskGP, X_np: np.ndarray, threshold: float
) -> np.ndarray:
    """
    Compute P(f(x) ≤ threshold) under the GP posterior for each x in X_np.
    Uses the closed-form Gaussian CDF.

    Returns
    -------
    pn : (N,) array in [0, 1].
    """
    from scipy.special import ndtr
    mean, var = gp_predict(gp, X_np)
    std = np.sqrt(np.maximum(var, 1e-12))
    return ndtr((threshold - mean) / std)
