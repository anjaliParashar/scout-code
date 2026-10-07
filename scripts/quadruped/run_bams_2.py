#!/usr/bin/env python3
"""
BAMS baseline for Go2 fixed velocity commands.

Uses the augmented multi-fidelity GP (BAMS) to select the next 5 hardware
evaluation candidates by combining:
  - Proxy / sim data: 2000 fixed-twist sim samples (CSV)
  - Target / hardware data: collected hardware pkl files

The input space is 3-D: (vx, vy, wz).  No encoder is needed — we use the
raw 3-dim velocity vector directly as the GP input, augmented with a
fidelity code:
    fidelity_code = 1.0  →  hardware (high-fidelity, expensive)
    fidelity_code = 0.0  →  sim      (low-fidelity, cheap)

Pool: uniform grid over [vx_min, vx_max] × [vy_min, vy_max] × [wz_min, wz_max].

Output: top-5 candidates ranked by BAMS score (gain-per-cost), printed to
stdout in the same format as run_scout.py.

Example
-------
python scripts/quadruped/run_bams.py \
  --hardware_glob "results/go2_hardware/scout/*.pkl" \
  --n_hardware 10 \
  --sim_csv data/quadruped/fixed_twist_combined.csv \
  --proxy_ckpt results/quadruped/sim.pt \
  --target_ckpt results/quadruped/hardware.pt \
  --pool_size 2000 \
  --n_candidates 5 \
  --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from utils.quadruped.train_utils import (
    MLPTrainConfig,
    load_hardware_pkls,
    load_sim_csv,
    train_mlp,
    save_checkpoint,
    load_mlp_checkpoint,
)


# ---------------------------------------------------------------------------
# GP utilities (GPyTorch — same stack used by scout/gp.py)
# ---------------------------------------------------------------------------

def _train_gp_bams(
    X_aug:  np.ndarray,   # (M, 4) — [vx, vy, wz, fidelity_code]
    y:      np.ndarray,   # (M,)
) -> object:
    """
    Train a GPyTorch ExactGP with RBF kernel on the augmented dataset.

    The 4th input dimension is the fidelity code, so the GP automatically
    learns cross-fidelity covariance from the data.
    """
    try:
        import gpytorch
    except ImportError:
        raise ImportError(
            "gpytorch is required for BAMS.  Install with: pip install gpytorch"
        )

    class _ExactGP(gpytorch.models.ExactGP):
        def __init__(self, X_t, y_t, likelihood):
            super().__init__(X_t, y_t, likelihood)
            self.mean_module  = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=X_t.shape[1])
            )

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x),
                self.covar_module(x),
            )

    X_t    = torch.tensor(X_aug, dtype=torch.float64)
    y_t    = torch.tensor(y,     dtype=torch.float64)
    lk     = gpytorch.likelihoods.GaussianLikelihood()
    model  = _ExactGP(X_t, y_t, lk)

    model.train(); lk.train()
    opt    = torch.optim.Adam(model.parameters(), lr=0.1)
    mll    = gpytorch.mlls.ExactMarginalLogLikelihood(lk, model)

    for _ in range(200):
        opt.zero_grad()
        loss = -mll(model(X_t), y_t)
        loss.backward()
        opt.step()

    model.eval(); lk.eval()
    return model, lk, X_t


@torch.no_grad()
def _gp_predict_hf(
    model,
    lk,
    X_train_t: torch.Tensor,
    X_pool:    np.ndarray,   # (N, 3) — velocity only, no fidelity code
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict at pool points using the HF fidelity code (1.0).

    Returns posterior mean, std, and predicted-failure mask.
    """
    import gpytorch
    fid = np.ones((len(X_pool), 1), dtype=np.float64)
    X_aug_pool = np.concatenate([X_pool, fid], axis=1)
    X_t        = torch.tensor(X_aug_pool, dtype=torch.float64)

    with gpytorch.settings.fast_pred_var():
        pred = lk(model(X_t))

    mean = pred.mean.numpy()
    std  = pred.variance.clamp(min=0).sqrt().numpy()
    return mean, std, (mean >= threshold)


# ---------------------------------------------------------------------------
# Uniform velocity grid pool
# ---------------------------------------------------------------------------

def _build_pool(
    vx_min: float, vx_max: float,
    vy_min: float, vy_max: float,
    wz_min: float, wz_max: float,
    pool_size: int,
    seed: int,
) -> np.ndarray:
    """
    Build a uniform random pool of (vx, vy, wz) candidates.
    (Uniform random rather than a regular grid so pool_size is exact.)
    """
    rng = np.random.default_rng(seed)
    vx  = rng.uniform(vx_min, vx_max, pool_size)
    vy  = rng.uniform(vy_min, vy_max, pool_size)
    wz  = rng.uniform(wz_min, wz_max, pool_size)
    return np.stack([vx, vy, wz], axis=1)   # (pool_size, 3)


# ---------------------------------------------------------------------------
# BAMS scoring (gain-per-cost, no external bams_acquisition dependency)
# ---------------------------------------------------------------------------

def _bams_score_pool(
    gp_model,
    gp_lk,
    X_train_t:    torch.Tensor,
    X_pool:       np.ndarray,     # (N, 3)
    X_train_hw:   np.ndarray,     # (M_hf, 3) hardware observations
    threshold:    float,
    hf_cost:      float = 1.0,
    lf_cost:      float = 0.10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Score each pool point at both HF and LF fidelity by:
        score = GP_posterior_std(x, fid) / cost(fid)

    High GP std at HF = uncertain about real performance → evaluate hardware.
    High GP std at LF = uncertain about sim performance → cheaper sim run.

    Returns
    -------
    hf_scores : (N,)  gain-per-cost at fidelity 1.0
    lf_scores : (N,)  gain-per-cost at fidelity 0.0
    """
    import gpytorch

    N = len(X_pool)

    def _std_at_fid(fid_val: float) -> np.ndarray:
        fid   = np.full((N, 1), fid_val, dtype=np.float64)
        X_aug = np.concatenate([X_pool, fid], axis=1)
        X_t   = torch.tensor(X_aug, dtype=torch.float64)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = gp_lk(gp_model(X_t))
        return pred.variance.clamp(min=0).sqrt().numpy()

    hf_scores = _std_at_fid(1.0) / hf_cost
    lf_scores = _std_at_fid(0.0) / lf_cost
    return hf_scores, lf_scores


# ---------------------------------------------------------------------------
# MLP surrogate predictions on pool (for additional context in output)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _mlp_predict(model, X: np.ndarray, device, n_mc: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """MC-dropout mean and std for MLP predictions on X."""
    model.train()   # enable dropout
    preds = []
    X_t   = torch.tensor(X, dtype=torch.float32, device=device)
    for _ in range(n_mc):
        preds.append(model(X_t).cpu().numpy())#.squeeze(-1))
    model.eval()
    stack = np.stack(preds, axis=0)
    return stack.mean(axis=0), stack.std(axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="BAMS baseline: select next hardware evaluation candidates."
    )

    # Data
    p.add_argument("--hardware_glob",   type=str,
                   default="results/go2_hardware/*.pkl")
    p.add_argument("--n_hardware",      type=int, default=None)
    p.add_argument("--sim_csv",         type=str,
                   default="data/quadruped/fixed_twist_combined.csv")
    p.add_argument("--sim_metric",      type=str, default="final_abs_err_sum")

    # Checkpoints
    p.add_argument("--proxy_ckpt",      type=str,
                   default="results/quadruped/sim.pt")
    p.add_argument("--target_ckpt",     type=str,
                   default="results/quadruped/hardware.pt")
    p.add_argument("--force_train_proxy", action="store_true")

    # MLP architecture (must match run_scout.py)
    p.add_argument("--epochs_target",   type=int,   default=1000)
    p.add_argument("--epochs_proxy",    type=int,   default=1200)
    p.add_argument("--hidden_dim",      type=int,   default=8)
    p.add_argument("--depth",           type=int,   default=2)
    p.add_argument("--dropout",         type=float, default=0.05)
    p.add_argument("--lr",              type=float, default=1e-3)

    # Pool
    p.add_argument("--pool_size",       type=int,   default=2000)
    p.add_argument("--n_candidates",    type=int,   default=5,
                   help="Number of hardware candidates to output")
    p.add_argument("--seed",            type=int,   default=0)

    # Velocity bounds
    p.add_argument("--vx_min",  type=float, default=-0.4)
    p.add_argument("--vx_max",  type=float, default=1.0)
    p.add_argument("--vy_min",  type=float, default=-0.8)
    p.add_argument("--vy_max",  type=float, default=0.8)
    p.add_argument("--wz_min",  type=float, default=-0.8)
    p.add_argument("--wz_max",  type=float, default=0.8)

    # BAMS hyperparameters
    p.add_argument("--bams_threshold",  type=float, default=0.9,
                   help="Error above this = failure region (higher GP uncertainty)")
    p.add_argument("--lf_cost",         type=float, default=0.10,
                   help="Cost ratio: sim / hardware.  sim is cheap → small value.")
    p.add_argument("--gp_max_train",    type=int,   default=200,
                   help="Max points used to train the augmented GP")

    return p


def main():
    args    = build_parser().parse_args()
    rng     = np.random.default_rng(args.seed)

    # ── 1. Load hardware data + train target MLP ──────────────────────────
    X_hw, y_hw, hw_paths = load_hardware_pkls(
        args.hardware_glob, n_files=args.n_hardware
    )
    print(f"\n[INFO] Hardware data: N={len(X_hw)}")
    for i, path in enumerate(hw_paths):
        print(f"  {i:03d}: X={X_hw[i]}  y={y_hw[i]:.4f}  {path}")

    target_train_cfg = MLPTrainConfig(
        hidden_dim=args.hidden_dim, depth=args.depth,
        dropout=args.dropout, lr=args.lr,
        epochs=args.epochs_target, seed=args.seed, verbose=True,
    )
    target_ckpt = train_mlp(X_hw, y_hw, target_train_cfg)
    save_checkpoint(target_ckpt, args.target_ckpt)
    target_model, target_ckpt, device = load_mlp_checkpoint(args.target_ckpt)

    # ── 2. Load sim data + train/load proxy MLP ───────────────────────────
    X_sim, y_sim = load_sim_csv(args.sim_csv, metric=args.sim_metric)
    print(f"\n[INFO] Sim data: N={len(X_sim)}  "
          f"y=[{y_sim.min():.4f}, {y_sim.max():.4f}]")

    proxy_path = Path(args.proxy_ckpt)
    if args.force_train_proxy or not proxy_path.exists():
        proxy_train_cfg = MLPTrainConfig(
            hidden_dim=args.hidden_dim, depth=args.depth,
            dropout=args.dropout, lr=args.lr,
            epochs=args.epochs_proxy, seed=args.seed, verbose=True,
        )
        proxy_ckpt = train_mlp(X_sim, y_sim, proxy_train_cfg)
        save_checkpoint(proxy_ckpt, proxy_path)
    proxy_model, proxy_ckpt, _ = load_mlp_checkpoint(proxy_path, device=str(device))

    # ── 3. Build augmented GP dataset [vx, vy, wz, fidelity_code] ─────────
    # HF observations: hardware data, fidelity_code = 1.0
    X_hf_aug = np.concatenate(
        [X_hw, np.ones((len(X_hw), 1))], axis=1
    )                                          # (M_hf, 4)
    y_hf = y_hw.astype(np.float64)

    # LF observations: subsample sim data, fidelity_code = 0.0
    n_lf = min(args.gp_max_train - len(X_hw), len(X_sim))
    n_lf = max(n_lf, 1)
    lf_idx  = rng.choice(len(X_sim), n_lf, replace=False)
    X_lf_aug = np.concatenate(
        [X_sim[lf_idx], np.zeros((n_lf, 1))], axis=1
    )                                          # (n_lf, 4)
    y_lf = y_sim[lf_idx].astype(np.float64)

    X_aug = np.concatenate([X_hf_aug, X_lf_aug], axis=0)
    y_aug = np.concatenate([y_hf, y_lf], axis=0)
    print(f"\n[BAMS] Augmented GP training data: {len(X_aug)} pts  "
          f"(HF={len(X_hf_aug)}, LF={len(X_lf_aug)})")

    # ── 4. Train augmented GP ─────────────────────────────────────────────
    print("[BAMS] Training augmented multi-fidelity GP…")
    gp_model, gp_lk, X_train_t = _train_gp_bams(X_aug, y_aug)
    print("[BAMS] GP trained.")

    # ── 5. Build pool ─────────────────────────────────────────────────────
    X_pool = _build_pool(
        args.vx_min, args.vx_max,
        args.vy_min, args.vy_max,
        args.wz_min, args.wz_max,
        args.pool_size, args.seed,
    )
    # Remove pool points too close to existing hardware observations
    if len(X_hw) > 0:
        dists      = np.min(np.linalg.norm(
            X_pool[:, None, :] - X_hw[None, :, :], axis=2
        ), axis=1)
        X_pool = X_pool[dists > 0.01]   # drop near-duplicates
    print(f"[BAMS] Pool: {len(X_pool)} candidates after dedup")

    # ── 6. Score pool by BAMS gain-per-cost ───────────────────────────────
    print("[BAMS] Scoring pool…")
    hf_scores, lf_scores = _bams_score_pool(
        gp_model, gp_lk, X_train_t,
        X_pool, X_hw,
        threshold = args.bams_threshold,
        hf_cost   = 1.0,
        lf_cost   = args.lf_cost,
    )

    # GP posterior mean at HF fidelity (for display)
    hf_mean, hf_std, _ = _gp_predict_hf(
        gp_model, gp_lk, X_train_t,
        X_pool, args.bams_threshold,
    )

    # MLP predictions for additional context
    target_pred_mean, target_pred_std = _mlp_predict(
        target_model, X_pool.astype(np.float32), device
    )
    proxy_pred_mean, _ = _mlp_predict(
        proxy_model, X_pool.astype(np.float32), device
    )

    # ── 7. Select top n_candidates by HF gain-per-cost ───────────────────
    # We always recommend HF (hardware) evaluations — that is the output.
    # LF scores are printed for reference to show what sim would suggest.
    top_hf_idx = np.argsort(hf_scores)[::-1][:args.n_candidates]
    candidates = X_pool[top_hf_idx]

    # ── 8. Print results ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"NEXT {args.n_candidates} HARDWARE CANDIDATES  (BAMS ranked by GP gain-per-cost)")
    print("=" * 80)

    for rank, (local_i, cand) in enumerate(zip(top_hf_idx, candidates)):
        vx, vy, wz = cand
        print(f"\n--- Candidate {rank + 1} ---")
        print(f"  vx = {vx:.6f}")
        print(f"  vy = {vy:.6f}")
        print(f"  wz = {wz:.6f}")
        print(f"  BAMS HF score (gain/cost)     = {hf_scores[local_i]:.6f}")
        print(f"  BAMS LF score (sim gain/cost) = {lf_scores[local_i]:.6f}")
        print(f"  GP posterior mean  (HF)       = {hf_mean[local_i]:.6f}")
        print(f"  GP posterior std   (HF)       = {hf_std[local_i]:.6f}")
        print(f"  Target MLP pred               = {target_pred_mean[local_i]:.6f} "
              f"± {target_pred_std[local_i]:.6f}")
        print(f"  Proxy  MLP pred               = {proxy_pred_mean[local_i]:.6f}")

    print("\n" + "=" * 80)
    print("Run these candidates on hardware with:")
    print("=" * 80)
    for rank, cand in enumerate(candidates):
        vx, vy, wz = cand
        print(
            f"\n# Candidate {rank + 1}\n"
            f"python scripts/quadruped/run_fixed_cmd_collect_odom.py "
            f"--vx {vx:.6f} "
            f"--vy {vy:.6f} "
            f"--wz {wz:.6f} "
            f"--duration 5.0 "
            f"--publish_hz 10.0 "
            f"--out bams/fixed_vx{11+rank}.pkl"
        )
    print()


if __name__ == "__main__":
    main()
