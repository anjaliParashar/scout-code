#!/usr/bin/env python3
"""
scripts/quadruped/run_bams.py
-------------------------------
Iterative BAMS pipeline for Go2 fixed velocity commands.
All predictions come from the augmented multi-fidelity GP — no MLPs.

Protocol
--------
t=0  (initial):
  GP trained on:
    HF — 10 hardware pkl files  (fidelity_code = 1.0)
    LF — 50 rows sampled from sim CSV  (fidelity_code = 0.0)
  Outputs:
    • n_candidates hardware commands to run on the real robot
    • n_sim_candidates sim CSV rows to collect (nearest neighbours in CSV)
  State saved to:  <state_dir>/state_t0.npz

t=1,2,... (subsequent):
  Load state from previous iteration.
  Load additional hardware pkls from --hardware_glob
    (just add new pkl files to the folder and re-run with --t 1, --t 2, …).
  Auto-fetch previously selected sim rows from state.
  Retrain GP on all accumulated HF + LF data.
  Output next hardware and sim candidates.
  State saved to:  <state_dir>/state_t{t}.npz

State file contents
-------------------
  X_hw_all, y_hw_all     — all hardware observations up to this iteration
  X_sim_all, y_sim_all   — all sim rows used for GP training
  sim_csv_indices        — CSV row indices of all sim samples selected so far
  hw_candidate_cmds      — hardware candidates recommended at this iteration
  sim_candidate_indices  — sim CSV indices recommended at this iteration

Usage
-----
# Iteration 0 — first run
python scripts/quadruped/run_bams.py \
  --t 0 \
  --hardware_glob "results/go2_hardware/bams/*.pkl" \
  --sim_csv data/quadruped/fixed_twist_combined.csv \
  --state_dir results/bams \
  --n_hw_init 15 \
  --n_sim_init 50 \
  --n_candidates 5 \
  --n_sim_candidates 10

# Iteration 1 — after adding new hardware pkl files
python scripts/quadruped/run_bams.py \
  --t 3 \
  --hardware_glob "results/go2_hardware/bams_2/*.pkl" \
  --sim_csv data/quadruped/fixed_twist_combined.csv \
  --state_dir results/bams \
  --n_candidates 5 \
  --n_sim_candidates 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.quadruped.train_utils import (
    load_hardware_pkls,
    load_sim_csv,
)


# ---------------------------------------------------------------------------
# GP — augmented multi-fidelity (no MLP anywhere)
# ---------------------------------------------------------------------------

def _train_gp(
    X_aug: np.ndarray,   # (M, 4)  [vx, vy, wz, fidelity_code]
    y:     np.ndarray,   # (M,)
    n_iter: int = 300,
):
    """
    ExactGP with ARD-RBF kernel trained on the augmented dataset.
    fidelity_code = 1.0 → hardware,  0.0 → sim.
    The GP learns cross-fidelity covariance automatically.
    """
    try:
        import gpytorch
    except ImportError:
        raise ImportError("pip install gpytorch")

    class _GP(gpytorch.models.ExactGP):
        def __init__(self, X_t, y_t, lk):
            super().__init__(X_t, y_t, lk)
            self.mean   = gpytorch.means.ConstantMean()
            self.covar  = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=X_t.shape[1])
            )
        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean(x), self.covar(x)
            )

    X_t  = torch.tensor(X_aug, dtype=torch.float64)
    y_t  = torch.tensor(y,     dtype=torch.float64)
    lk   = gpytorch.likelihoods.GaussianLikelihood()
    gp   = _GP(X_t, y_t, lk)
    gp.train(); lk.train()

    opt = torch.optim.Adam(gp.parameters(), lr=0.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lk, gp)
    for i in range(n_iter):
        opt.zero_grad()
        loss = -mll(gp(X_t), y_t)
        loss.backward()
        opt.step()
        if (i + 1) % 100 == 0:
            print(f"    GP iter {i+1}/{n_iter}  loss={loss.item():.4f}", flush=True)

    gp.eval(); lk.eval()
    return gp, lk, X_t


@torch.no_grad()
def _gp_predict(gp, lk, X: np.ndarray, fidelity: float):
    """
    GP posterior mean and std at X with given fidelity code (1.0 or 0.0).
    Returns (mean, std) arrays of shape (N,).
    """
    import gpytorch
    fid   = np.full((len(X), 1), fidelity, dtype=np.float64)
    X_aug = np.concatenate([X, fid], axis=1)
    X_t   = torch.tensor(X_aug, dtype=torch.float64)
    with gpytorch.settings.fast_pred_var():
        pred = lk(gp(X_t))
    mean = pred.mean.numpy()
    std  = pred.variance.clamp(min=0).sqrt().numpy()
    return mean, std


def _bams_scores(gp, lk, X_pool: np.ndarray,
                 hf_cost: float = 1.0, lf_cost: float = 0.10):
    """
    Gain-per-cost scores at both fidelities for each pool point.
    score = GP_posterior_std(x, fid) / cost(fid)
    """
    _, hf_std = _gp_predict(gp, lk, X_pool, fidelity=1.0)
    _, lf_std = _gp_predict(gp, lk, X_pool, fidelity=0.0)
    return hf_std / hf_cost, lf_std / lf_cost


# ---------------------------------------------------------------------------
# Pool — uniform random velocity candidates
# ---------------------------------------------------------------------------

def _build_pool(vx_min, vx_max, vy_min, vy_max, wz_min, wz_max,
                pool_size, seed):
    rng = np.random.default_rng(seed)
    return np.stack([
        rng.uniform(vx_min, vx_max, pool_size),
        rng.uniform(vy_min, vy_max, pool_size),
        rng.uniform(wz_min, wz_max, pool_size),
    ], axis=1)


def _dedup_pool(X_pool, X_existing, min_dist=0.02):
    if len(X_existing) == 0:
        return X_pool
    dists = np.min(np.linalg.norm(
        X_pool[:, None, :] - X_existing[None, :, :], axis=2
    ), axis=1)
    return X_pool[dists > min_dist]


# ---------------------------------------------------------------------------
# Sim CSV: select nearest CSV rows to pool candidates by LF score
# ---------------------------------------------------------------------------

def _select_sim_rows(
    X_sim:            np.ndarray,   # (N_csv, 3) all sim CSV commands
    X_pool:           np.ndarray,   # (P, 3) pool
    lf_scores:        np.ndarray,   # (P,) LF gain-per-cost
    n_sim_candidates: int,
    already_used_idx: np.ndarray,   # CSV row indices already collected
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select n_sim_candidates sim CSV rows that:
      1. Are not already in already_used_idx.
      2. Are nearest (in velocity space) to the top-LF-score pool points.

    Returns
    -------
    sim_pool_cmds : (n_sim_candidates, 3) velocity commands
    sim_csv_idx   : (n_sim_candidates,) row indices in X_sim
    """
    used_set = set(already_used_idx.tolist())
    avail    = np.array([i for i in range(len(X_sim)) if i not in used_set])

    if len(avail) == 0:
        print("[WARN] All sim CSV rows already used.")
        return np.empty((0, 3)), np.empty(0, int)

    X_sim_avail = X_sim[avail]   # (A, 3)

    # Top LF pool candidates to match against sim CSV
    n_query = min(n_sim_candidates * 5, len(X_pool))
    lf_top  = np.argsort(lf_scores)[::-1][:n_query]
    X_query = X_pool[lf_top]    # (Q, 3)

    # For each query find nearest available sim CSV row
    selected_avail_idx = set()
    result_idx         = []

    for xq in X_query:
        dists = np.linalg.norm(X_sim_avail - xq[None, :], axis=1)
        order = np.argsort(dists)
        for a in order:
            if a not in selected_avail_idx:
                selected_avail_idx.add(a)
                result_idx.append(int(avail[a]))
                break
        if len(result_idx) >= n_sim_candidates:
            break

    result_idx = np.array(result_idx, int)
    return X_sim[result_idx], result_idx


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def _state_path(state_dir: Path, t: int) -> Path:
    return state_dir / f"state_t{t}.npz"


def _save_state(state_dir: Path, t: int, **arrays):
    state_dir.mkdir(parents=True, exist_ok=True)
    path = _state_path(state_dir, t)
    np.savez(str(path), **arrays)
    print(f"[STATE] Saved → {path}")
    return path


def _load_state(state_dir: Path, t: int) -> dict:
    path = _state_path(state_dir, t)
    if not path.exists():
        raise FileNotFoundError(
            f"State file not found: {path}\n"
            f"Run with --t {t} only after --t {t-1} has completed."
        )
    data = np.load(str(path), allow_pickle=True)
    print(f"[STATE] Loaded ← {path}")
    return dict(data)


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print_hw_candidates(candidates, hf_mean, hf_std, hf_scores, lf_scores,
                          top_idx, t, state_dir):
    print("\n" + "=" * 80)
    print(f"HARDWARE CANDIDATES  t={t}  "
          f"(BAMS: ranked by GP HF gain-per-cost)")
    print("=" * 80)
    for rank, (local_i, cand) in enumerate(zip(top_idx, candidates)):
        vx, vy, wz = cand
        print(f"\n--- HW Candidate {rank+1} ---")
        print(f"  vx = {vx:.6f}   vy = {vy:.6f}   wz = {wz:.6f}")
        print(f"  GP mean (HF)  = {hf_mean[local_i]:.6f}")
        print(f"  GP std  (HF)  = {hf_std[local_i]:.6f}")
        print(f"  BAMS HF score = {hf_scores[local_i]:.6f}")
        print(f"  BAMS LF score = {lf_scores[local_i]:.6f}")

    print("\n" + "=" * 80)
    print("Commands to run on hardware:")
    print("=" * 80)
    for rank, cand in enumerate(candidates):
        vx, vy, wz = cand
        out_name = f"fixed_vx{rank+31}.pkl"
        print(
            f"\n# HW Candidate {rank+16}\n"
            f"python run.py --cmd_topic /cmd_vel --odom_topic /utlidar/robot_odom "
            f"--vx {vx:.2f} --vy {vy:.2f} --wz {wz:.2f} "
            f"--duration 5.0 --publish_hz 10.0 "
            f"--out results/quadruped/bams_2/{out_name}"
        )


def _print_sim_candidates(sim_cmds, sim_csv_idx, gp, lk, X_sim, t):
    print("\n" + "=" * 80)
    print(f"SIM CANDIDATES  t={t}  "
          f"(BAMS: nearest CSV rows to top LF pool points)")
    print("=" * 80)
    if len(sim_cmds) == 0:
        print("  [WARN] No sim candidates — all CSV rows already used.")
        return

    lf_mean, lf_std = _gp_predict(gp, lk, sim_cmds, fidelity=0.0)
    hf_mean, hf_std = _gp_predict(gp, lk, sim_cmds, fidelity=1.0)

    print(f"\n{'Rank':>4}  {'CSV_idx':>8}  {'vx':>8}  {'vy':>8}  {'wz':>8}  "
          f"{'GP_LF_mean':>12}  {'GP_LF_std':>11}  {'GP_HF_mean':>12}")
    print("-" * 90)
    for rank, (csv_i, cmd) in enumerate(zip(sim_csv_idx, sim_cmds)):
        vx, vy, wz = cmd
        print(f"  {rank+1:2d}    {csv_i:7d}   {vx:7.4f}   {vy:7.4f}   {wz:7.4f}"
              f"   {lf_mean[rank]:11.4f}   {lf_std[rank]:10.4f}   "
              f"{hf_mean[rank]:11.4f}")

    print("\nTo fetch these rows from the CSV:")
    print(f"  pandas: df.iloc[{sim_csv_idx.tolist()}]")
    print(f"  numpy : X_sim[{sim_csv_idx.tolist()}]")
    print(f"\n  CSV row indices also saved in state file under 'sim_candidate_indices_t{t}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args      = build_parser().parse_args()
    rng       = np.random.default_rng(args.seed)
    state_dir = Path(args.state_dir)
    t         = args.t

    # ── Load sim CSV (always needed) ──────────────────────────────────────
    print(f"\n[INFO] Loading sim CSV: {args.sim_csv}")
    X_sim, y_sim = load_sim_csv(args.sim_csv, metric=args.sim_metric)
    print(f"[INFO] Sim CSV: N={len(X_sim)}  "
          f"y=[{y_sim.min():.4f}, {y_sim.max():.4f}]")

    # ── Load hardware data ─────────────────────────────────────────────────
    print(f"\n[INFO] Loading hardware pkls: {args.hardware_glob}")
    X_hw_all_raw, y_hw_all_raw, hw_paths = load_hardware_pkls(
        args.hardware_glob, n_files=None   # load everything in the folder
    )
    print(f"[INFO] Hardware pkls found: {len(X_hw_all_raw)}")
    for i, p in enumerate(hw_paths):
        print(f"  {i:03d}: X={X_hw_all_raw[i]}  y={y_hw_all_raw[i]:.4f}  {p}")

    # ── Build GP training data depending on t ─────────────────────────────
    if t == 0:
        # ── t=0: use n_hw_init hardware pts + n_sim_init sim pts ──────────
        n_hw  = min(args.n_hw_init,  len(X_hw_all_raw))
        n_sim = min(args.n_sim_init, len(X_sim))

        X_hw_gp  = X_hw_all_raw[:n_hw].astype(np.float64)
        y_hw_gp  = y_hw_all_raw[:n_hw].astype(np.float64)

        sim_init_idx = rng.choice(len(X_sim), n_sim, replace=False)
        X_sim_gp     = X_sim[sim_init_idx].astype(np.float64)
        y_sim_gp     = y_sim[sim_init_idx].astype(np.float64)

        print(f"\n[t=0] GP init: {n_hw} HW pts + {n_sim} sim pts")

    else:
        # ── t>0: load previous state, accumulate ──────────────────────────
        prev = _load_state(state_dir, t - 1)

        # Hardware: all pkls currently in the folder
        X_hw_gp = X_hw_all_raw.astype(np.float64)
        y_hw_gp = y_hw_all_raw.astype(np.float64)

        fig, axes = plt.subplots(3, 1, figsize=(5, 15))
        X_top = X_hw_gp
        axes[0].scatter(X_top[:, 0], X_top[:, 1], c=y_hw_gp); axes[0].set_title('Vx and Vy')
        axes[1].scatter(X_top[:, 0], X_top[:, 2], c=y_hw_gp); axes[1].set_title('Vx and Wz')
        axes[2].scatter(X_top[:, 1], X_top[:, 2], c=y_hw_gp); axes[2].set_title('Vy and Wz')

        print("Y_CUMULATIVE:",len(y_hw_gp[y_hw_gp>=0.6]))
        plt.savefig('results/bams/cv_mean.png')
        plt.close(fig)

        # Sim: previously accumulated rows + auto-fetch from state
        X_sim_prev    = prev["X_sim_all"]
        y_sim_prev    = prev["y_sim_all"]
        used_sim_idx  = prev["sim_csv_indices"]

        # Also load the sim rows that were recommended at t-1 (now collected)
        new_sim_idx   = prev[f"sim_candidate_indices_t{t-1}"]
        new_sim_X     = X_sim[new_sim_idx].astype(np.float64)
        new_sim_y     = y_sim[new_sim_idx].astype(np.float64)

        X_sim_gp   = np.concatenate([X_sim_prev, new_sim_X], axis=0)
        y_sim_gp   = np.concatenate([y_sim_prev, new_sim_y], axis=0)
        used_sim_idx = np.concatenate([used_sim_idx, new_sim_idx])

        print(f"\n[t={t}] GP data: {len(X_hw_gp)} HW pts  "
              f"+ {len(X_sim_gp)} sim pts accumulated")

    # ── Build augmented dataset [vx, vy, wz, fidelity_code] ──────────────
    X_hf_aug = np.concatenate(
        [X_hw_gp, np.ones((len(X_hw_gp), 1))], axis=1)
    X_lf_aug = np.concatenate(
        [X_sim_gp, np.zeros((len(X_sim_gp), 1))], axis=1)
    X_aug    = np.concatenate([X_hf_aug, X_lf_aug], axis=0)
    y_aug    = np.concatenate([y_hw_gp,  y_sim_gp],  axis=0)
    print(f"[BAMS] Augmented GP: {len(X_aug)} pts  "
          f"(HF={len(X_hf_aug)}, LF={len(X_lf_aug)})")

    # ── Train GP ──────────────────────────────────────────────────────────
    print("[BAMS] Training multi-fidelity GP…")
    gp, lk, _ = _train_gp(X_aug, y_aug, n_iter=args.gp_iters)
    print("[BAMS] GP trained.")

    # ── Build pool ────────────────────────────────────────────────────────
    X_pool_raw = _build_pool(
        args.vx_min, args.vx_max,
        args.vy_min, args.vy_max,
        args.wz_min, args.wz_max,
        args.pool_size, args.seed + t,
    )
    X_pool = _dedup_pool(X_pool_raw, X_hw_gp, min_dist=0.02)
    print(f"[BAMS] Pool: {len(X_pool)} candidates after dedup")

    # ── Score pool ────────────────────────────────────────────────────────
    print("[BAMS] Scoring pool…")
    hf_scores, lf_scores = _bams_scores(gp, lk, X_pool,
                                         hf_cost=1.0, lf_cost=args.lf_cost)

    # GP posterior at HF fidelity for output context
    hf_mean, hf_std = _gp_predict(gp, lk, X_pool, fidelity=1.0)

    # ── Select HW candidates ──────────────────────────────────────────────
    top_hf_idx  = np.argsort(hf_scores)[::-1][:args.n_candidates]
    hw_cands    = X_pool[top_hf_idx]

    # ── Select sim candidates from CSV ────────────────────────────────────
    already_used = used_sim_idx if t > 0 else (
        sim_init_idx if t == 0 else np.array([], int)
    )
    sim_cmds, sim_csv_idx = _select_sim_rows(
        X_sim, X_pool, lf_scores,
        n_sim_candidates = args.n_sim_candidates,
        already_used_idx = already_used,
    )

    # ── Print results ─────────────────────────────────────────────────────
    _print_hw_candidates(hw_cands, hf_mean, hf_std,
                          hf_scores, lf_scores, top_hf_idx, t, state_dir)
    _print_sim_candidates(sim_cmds, sim_csv_idx, gp, lk, X_sim, t)

    # ── Save state ────────────────────────────────────────────────────────
    state_kwargs = dict(
        X_hw_all         = X_hw_gp,
        y_hw_all         = y_hw_gp,
        X_sim_all        = X_sim_gp,
        y_sim_all        = y_sim_gp,
        sim_csv_indices  = already_used,
        hw_candidate_cmds = hw_cands,
    )
    state_kwargs[f"sim_candidate_indices_t{t}"] = sim_csv_idx
    if t == 0:
        state_kwargs["sim_csv_indices"] = sim_init_idx

    _save_state(state_dir, t, **state_kwargs)

    print(f"\n[DONE]  t={t}  "
          f"HW cands={len(hw_cands)}  "
          f"Sim cands={len(sim_cmds)}  "
          f"State → {_state_path(state_dir, t)}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Iterative BAMS for Go2 — GP-only, no MLP."
    )

    # Iteration counter
    p.add_argument("--t", type=int, required=True,
                   help="Iteration index. 0 = first run. "
                        "Increment by 1 each time you add new hardware data.")

    # Data paths
    p.add_argument("--hardware_glob", type=str,
                   default="results/go2_hardware/bams/*.pkl",
                   help="Glob for all hardware pkl files collected so far.")
    p.add_argument("--sim_csv",       type=str,
                   default="data/quadruped/fixed_twist_combined.csv")
    p.add_argument("--sim_metric",    type=str, default="final_abs_err_sum")

    # State persistence
    p.add_argument("--state_dir",     type=str, default="results/bams",
                   help="Directory where state_t{t}.npz files are stored.")

    # Initial data sizes (t=0 only)
    p.add_argument("--n_hw_init",     type=int, default=10,
                   help="[t=0] Number of hardware pkl files to use for GP init.")
    p.add_argument("--n_sim_init",    type=int, default=50,
                   help="[t=0] Number of sim CSV rows to sample for GP init.")

    # Candidate counts
    p.add_argument("--n_candidates",      type=int, default=5,
                   help="Number of hardware candidates to recommend.")
    p.add_argument("--n_sim_candidates",  type=int, default=10,
                   help="Number of sim CSV rows to recommend for collection.")

    # Pool
    p.add_argument("--pool_size",     type=int,   default=2000)
    p.add_argument("--seed",          type=int,   default=0)

    # Velocity bounds
    p.add_argument("--vx_min",  type=float, default=-0.4)
    p.add_argument("--vx_max",  type=float, default=1.0)
    p.add_argument("--vy_min",  type=float, default=-0.8)
    p.add_argument("--vy_max",  type=float, default=0.8)
    p.add_argument("--wz_min",  type=float, default=-0.8)
    p.add_argument("--wz_max",  type=float, default=0.8)

    # GP
    p.add_argument("--gp_iters",      type=int,   default=300,
                   help="Adam optimisation steps for GP hyperparameter training.")
    p.add_argument("--lf_cost",       type=float, default=0.05,
                   help="Cost ratio: sim / hardware.  sim CSV lookup ≈ free.")
    p.add_argument("--bams_threshold", type=float, default=0.9)

    return p


if __name__ == "__main__":
    main()
