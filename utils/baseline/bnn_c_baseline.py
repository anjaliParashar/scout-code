#!/usr/bin/env python3
"""
utils/baseline/bnn_c_baseline.py
----------------------------------
BNN-C baseline — "Failure Prediction from Few Expert Demonstrations" style.

Algorithm
---------
1. Pre-train a FAILURE CLASSIFIER on 20,000 open-loop (proxy) samples.
   Failure label: proxy values <= failure_threshold (default 0.3).
   Classifier: logistic BNN (same HeteroBNNEmbedding, outputs probability).

2. Initialise a CLOSED-LOOP BNN on the initial n_sample real (HF) observations.
   This BNN predicts target metric.

3. At each acquisition iteration:
   a. Run the classifier on the full candidate pool.
      Keep only points predicted as SUCCESS (p_fail < success_threshold).
      These are scenarios where the simulator does NOT predict failure —
      we want to find the ones that nonetheless fail in closed loop.
   b. Cluster the success-predicted pool into K = batch_acq_size clusters
      (KMeans on X_all).
   c. From each cluster, pick the point with the HIGHEST predicted closed-loop
      failure probability from the closed-loop BNN (lowest predicted TTC).
   d. Evaluate true target metric for the selected batch.
   e. Retrain the closed-loop BNN on the expanded dataset.
   f. Remove selected points from the pool and repeat.

The key insight: we're looking for scenarios that FOOL the sim — they look
safe in open loop but are dangerous in closed loop. The classifier filters
to the "looks safe" region, and the BNN finds the most dangerous ones there.

Returns a (batch_acq_size,) array of local pool indices at each call.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Reuse the existing BNN infrastructure
from scout.bnn import HeteroBNNEmbedding, train_bnn, batched_posterior_draws


# ---------------------------------------------------------------------------
# Failure classifier (logistic BNN on open-loop data)
# ---------------------------------------------------------------------------

class FailureClassifierBNN(nn.Module):
    """
    Standalone MC-dropout BNN failure classifier.

    Mirrors the HeteroBNNEmbedding backbone (input → 96 → 24 → 6) but
    adds a sigmoid classification head instead of a regression head.
    Dropout is applied identically so MC-dropout uncertainty estimates work.

    p_fail(x) = sigmoid(W · tanh(net(x)))
    """

    def __init__(self, input_dim: int, p_drop: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96), nn.Tanh(), nn.Dropout(p_drop),
            nn.Linear(96, 24),        nn.Tanh(), nn.Dropout(p_drop),
            nn.Linear(24, 6),         nn.Tanh(), nn.Dropout(p_drop),
        )
        self.cls_head = nn.Linear(6, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B,) failure probabilities in [0, 1]."""
        h = self.net(x)
        return torch.sigmoid(self.cls_head(h)).squeeze(-1)


def train_failure_classifier(
    X_proxy:          np.ndarray,   # (N_proxy, d) — open-loop scenarios
    y_proxy:          np.ndarray,   # (N_proxy,)   — proxy values
    failure_threshold: float = 0.3,
    epochs:           int   = 500,
    lr:               float = 1e-3,
    weight_decay:     float = 1e-6,
    p_drop:           float = 0.10,
    device:           str   = "cpu",
    batch_size:       int   = 4096,
    random_state:     int   = 0,
) -> FailureClassifierBNN:
    """
    Train a binary BNN failure classifier on open-loop proxy data.

    failure_label[i] = 1 if y_proxy[i] <= failure_threshold, else 0.

    Training uses binary cross-entropy loss.

    Parameters
    ----------
    X_proxy           : standardised embedding matrix for proxy scenarios
    y_proxy           : proxy values for each proxy scenario
    failure_threshold : TTC below this = failure (default 0.3)
    """
    torch.manual_seed(random_state)
    failure_labels = (y_proxy <= failure_threshold).astype(np.float32)
    n_fail = int(failure_labels.sum())
    n_safe = len(failure_labels) - n_fail
    print(f"  [BNN-C] Classifier training data: {len(X_proxy):,}  "
          f"failure={n_fail:,}  safe={n_safe:,}  "
          f"(threshold={failure_threshold})")

    model   = FailureClassifierBNN(input_dim=X_proxy.shape[1], p_drop=p_drop)
    model   = model.to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCELoss()

    X_t = torch.tensor(X_proxy,        dtype=torch.float32)
    y_t = torch.tensor(failure_labels,  dtype=torch.float32)

    model.train()
    for epoch in range(epochs):
        perm  = torch.randperm(len(X_t))
        total = 0.0
        for s in range(0, len(X_t), batch_size):
            idx  = perm[s:s + batch_size]
            xb   = X_t[idx].to(device)
            yb   = y_t[idx].to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"    Classifier epoch {epoch+1}/{epochs}  "
                  f"BCE={total/len(X_t):.4f}", flush=True)

    model.eval()
    return model


@torch.no_grad()
def predict_failure_proba(
    classifier:  FailureClassifierBNN,
    X:           np.ndarray,
    device:      str   = "cpu",
    n_mc:        int   = 20,
    batch_size:  int   = 4096,
) -> np.ndarray:
    """
    Monte-Carlo dropout estimate of failure probability for each row in X.
    Returns (N,) array of p(failure | x).
    """
    classifier.train()   # enable dropout for MC
    all_probs = []
    X_t = torch.tensor(X, dtype=torch.float32)
    for _ in range(n_mc):
        probs_mc = []
        for s in range(0, len(X_t), batch_size):
            xb = X_t[s:s + batch_size].to(device)
            probs_mc.append(classifier(xb).cpu().numpy())
        all_probs.append(np.concatenate(probs_mc))
    classifier.eval()
    return np.stack(all_probs, axis=0).mean(axis=0)   # (N,) MC mean


# ---------------------------------------------------------------------------
# BNN-C acquisition (one iteration)
# ---------------------------------------------------------------------------

def bnn_c_acquisition(
    pool_global_idx:   np.ndarray,   # global indices of candidate pool
    X_all:             np.ndarray,   # (N, d) full embedding matrix
    y_cv:              np.ndarray,   # (N,) proxy values  (for classifier)
    classifier:        FailureClassifierBNN,
    cl_bnn:            HeteroBNNEmbedding,   # closed-loop BNN predictor
    batch_size:        int,
    device:            str   = "cpu",
    success_threshold: float = 0.5,  # p_fail < this → predicted as success
    n_mc_classifier:   int   = 20,
    n_mc_bnn:          int   = 20,
    random_state:      int   = 0,
) -> np.ndarray:
    """
    One iteration of BNN-C acquisition.

    Steps:
      1. Classify pool → keep success-predicted (p_fail < success_threshold).
      2. KMeans into batch_size clusters on X_all[success_pool].
      3. From each cluster, pick the point with highest predicted closed-loop
         failure (lowest predicted TTC = highest BNN-predicted failure).

    Returns
    -------
    selected_local : indices into pool_global_idx of the selected batch
    """
    X_pool      = X_all[pool_global_idx].astype(np.float32)
    S           = len(pool_global_idx)

    # Step 1: filter to success-predicted scenarios
    p_fail_pool = predict_failure_proba(
        classifier, X_pool, device=device, n_mc=n_mc_classifier
    )
    success_mask  = p_fail_pool < success_threshold
    success_local = np.where(success_mask)[0]   # local indices into pool

    print(f"  [BNN-C] Pool={S}  sim-success={success_mask.sum()}  "
          f"sim-failure={S - success_mask.sum()}", flush=True)

    if len(success_local) == 0:
        # All predicted as failure in sim — fall back to random
        rng = np.random.default_rng(random_state)
        return rng.choice(S, min(batch_size, S), replace=False)

    k = min(batch_size, len(success_local))
    if k == len(success_local):
        # Too few success scenarios — use all of them
        return success_local

    # Step 2: KMeans on the success subset
    X_success = X_pool[success_local]
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto", max_iter=300)
    cluster_labels = km.fit_predict(X_success)   # (|success|,)

    # Step 3: BNN posterior draws on success subset
    draws_success = batched_posterior_draws(
        cl_bnn, X_success, n_draws=n_mc_bnn, device=device
    )   # (n_mc_bnn, |success|)
    # Mean predicted TTC — lower = more likely to fail
    mean_ttc = draws_success.mean(axis=0)   # (|success|,)
    # Failure score = negative predicted TTC (higher = worse)
    fail_score = -mean_ttc

    # Step 4: from each cluster pick highest failure score
    selected_in_success = []
    for c in range(k):
        in_c = np.where(cluster_labels == c)[0]
        if len(in_c) == 0:
            continue
        best = in_c[np.argmax(fail_score[in_c])]
        selected_in_success.append(int(best))

    # Map back to pool-local indices
    selected_local = success_local[np.array(selected_in_success, int)]
    return selected_local