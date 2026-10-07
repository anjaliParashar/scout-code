#!/usr/bin/env python3
"""
scout/bnn.py
------------------
Heteroscedastic MC-dropout BNN surrogate.

Architecture  : input_dim → 96 → 24 → 6 → (mean_head, rawvar_head) → 1
Training loss : Gaussian NLL
Inference     : dropout kept ON for MC forward passes (Gal & Ghahramani 2016).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HeteroBNNEmbedding(nn.Module):
    """Heteroscedastic MC-dropout BNN for scalar regression."""

    def __init__(
        self,
        input_dim: int = 384,
        p_drop: float = 0.10,
        widths: tuple | list | None = None,
    ):
        super().__init__()
        # Default SCOUT widths: input → 96 → 24 → 6 → heads
        w = list(widths) if widths is not None else [96, 24, 6]
        layers: list[nn.Module] = []
        d_in = input_dim
        for d_out in w:
            layers.extend([nn.Linear(d_in, d_out), nn.Tanh(), nn.Dropout(p_drop)])
            d_in = d_out
        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(d_in, 1)
        self.rawvar_head = nn.Linear(d_in, 1)

    def forward(self, x):
        h   = self.net(x)
        mu  = self.mean_head(h)
        var = F.softplus(self.rawvar_head(h)) + 1e-5
        return mu, var


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _gaussian_nll(mu, var, y):
    return (0.5 * torch.log(var) + 0.5 * (y - mu) ** 2 / var).mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_bnn(
    model: HeteroBNNEmbedding,
    X_np: np.ndarray,
    y_np: np.ndarray,
    epochs: int = 2800,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    device: str = "cpu",
) -> None:
    """Train BNN in-place using Gaussian NLL and Adam."""
    X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_np[:, None], dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    log_every = max(1, epochs // 10)
    for ep in range(epochs):
        opt.zero_grad()
        mu, var = model(X_t)
        loss = _gaussian_nll(mu, var, y_t)
        loss.backward()
        opt.step()
        if ep % log_every == 0 or ep == epochs - 1:
            print(f"  [TRAIN] ep={ep:05d}  loss={loss.item():.6f}")


# ---------------------------------------------------------------------------
# Batched MC posterior draws (dropout ON)
# ---------------------------------------------------------------------------

@torch.no_grad()
def batched_posterior_draws(
    model: HeteroBNNEmbedding,
    X_np: np.ndarray,
    n_draws: int,
    device: str = "cpu",
    batch_size: int = 4096,
) -> np.ndarray:
    """
    Draw n_draws MC-dropout mean samples for every row of X_np.

    Returns
    -------
    draws : (n_draws, N) float32 array.
    """
    model.train()           # keep dropout ON
    X_np = np.asarray(X_np, dtype=np.float32)
    N    = len(X_np)
    out  = []
    for _ in range(n_draws):
        chunks = []
        for s in range(0, N, batch_size):
            X_t = torch.tensor(X_np[s : s + batch_size], device=device)
            mu, _ = model(X_t)
            chunks.append(mu.squeeze(-1).cpu().numpy())
        out.append(np.concatenate(chunks))
    return np.asarray(out, dtype=np.float32)   # (n_draws, N)


@torch.no_grad()
def mc_predict(
    model: HeteroBNNEmbedding,
    X_np: np.ndarray,
    n_samples: int = 140,
    device: str = "cpu",
    batch_size: int = 4096,
) -> tuple:
    """
    Full MC-dropout predictive distribution.

    Returns (mean, total_var, ale, epi, mus, vars_)
    where mus / vars_ have shape (n_samples, N).
    """
    mus_list, vars_list = [], []
    for s in range(n_samples):
        draws = batched_posterior_draws(model, X_np, 1, device, batch_size)
        mus_list.append(draws[0])
        # Also get aleatoric var in a separate pass
    # Re-run for aleatoric (need both heads together)
    model.train()
    N = len(X_np)
    all_mus, all_vars = [], []
    for _ in range(n_samples):
        mus_s, vars_s = [], []
        for s in range(0, N, batch_size):
            X_t = torch.tensor(
                np.asarray(X_np[s : s + batch_size], dtype=np.float32), device=device
            )
            mu, var = model(X_t)
            mus_s.append(mu.squeeze(-1).cpu().numpy())
            vars_s.append(var.squeeze(-1).cpu().numpy())
        all_mus.append(np.concatenate(mus_s))
        all_vars.append(np.concatenate(vars_s))
    mus   = np.asarray(all_mus)
    vars_ = np.asarray(all_vars)
    mean  = mus.mean(0)
    ale   = vars_.mean(0)
    epi   = mus.var(0)
    return mean, ale + epi, ale, epi, mus, vars_
