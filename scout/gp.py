#!/usr/bin/env python3
"""
scout/gp.py
--------------------------------
Lightweight BoTorch GP used by the BAMS baseline.
Works on a subsampled version of the training set for tractability.
"""

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.special import ndtr

_DTYPE  = torch.double
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tx(x): return torch.tensor(x, dtype=_DTYPE, device=_DEVICE)
def _ty(y): return torch.tensor(y[:, None], dtype=_DTYPE, device=_DEVICE)


def train_gp(X_np: np.ndarray, y_np: np.ndarray) -> SingleTaskGP:
    gp = SingleTaskGP(
        train_X=_tx(X_np), train_Y=_ty(y_np),
        input_transform=Normalize(d=X_np.shape[1]),
        outcome_transform=Standardize(m=1),
    ).to(device=_DEVICE, dtype=_DTYPE)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    gp.eval(); gp.likelihood.eval()
    return gp


@torch.no_grad()
def gp_predict(gp, X_np: np.ndarray) -> tuple:
    post = gp.posterior(_tx(X_np))
    mean = post.mean.squeeze(-1).cpu().numpy()
    var  = post.variance.squeeze(-1).clamp_min(1e-12).cpu().numpy()
    return mean, var


def gp_posterior_prob_below(
    gp, X_np: np.ndarray, threshold: float
) -> np.ndarray:
    mean, var = gp_predict(gp, X_np)
    std = np.sqrt(np.maximum(var, 1e-12))
    return ndtr((threshold - mean) / std)
