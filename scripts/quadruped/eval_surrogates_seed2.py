#!/usr/bin/env python3
"""
scripts/quadruped/eval_surrogates_seed2.py
--------------------------------------------
Evaluate surrogate accuracy on the shared failure set, seed 2 only.

Two surrogates trained:

  A) MLP — trained on hardware data from seed 2 of each method.

  B) Multi-fidelity GP — trained on:
       HF: hardware data from seed 2 of each method       (fidelity=1.0)
       LF: sim data extracted from state_t4.npz           (fidelity=0.0)
     The .npz file was produced by run_bams.py and contains X_sim_all /
     y_sim_all — the sim rows that BAMS actually selected and used, which
     is the most faithful LF dataset for this experiment.

Failure eval set: all scenarios with y >= threshold across ALL methods
and BOTH seeds (same as eval_surrogates.py — method-agnostic).

Usage
-----
python scripts/quadruped/eval_surrogates_seed2.py \
    --results_root  results/go2_hardware \
    --bams_state    results/bams/state_t4.npz \
    --threshold     0.8 \
    --output_dir    results/quadruped/eval_seed2
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.quadruped.train_utils import (
    MLPTrainConfig,
    load_sim_csv,
    train_mlp,
    save_checkpoint,
    load_mlp_checkpoint,
    predict_mlp,
)


# ---------------------------------------------------------------------------
# GP helpers
# ---------------------------------------------------------------------------

def _train_gp(X_aug: np.ndarray, y: np.ndarray, n_iter: int = 200):
    try:
        import gpytorch, torch
    except ImportError:
        raise ImportError("pip install gpytorch")

    class _GP(gpytorch.models.ExactGP):
        def __init__(self, Xt, yt, lk):
            super().__init__(Xt, yt, lk)
            self.mean  = gpytorch.means.ConstantMean()
            self.covar = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=Xt.shape[1])
            )
        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean(x), self.covar(x))

    Xt  = torch.tensor(X_aug, dtype=torch.float64)
    yt  = torch.tensor(y,     dtype=torch.float64)
    lk  = gpytorch.likelihoods.GaussianLikelihood()
    gp  = _GP(Xt, yt, lk)
    gp.train(); lk.train()
    opt = torch.optim.Adam(gp.parameters(), lr=0.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lk, gp)
    for i in range(n_iter):
        opt.zero_grad()
        (-mll(gp(Xt), yt)).backward()
        opt.step()
        if (i + 1) % 50 == 0:
            print(f"    GP iter {i+1}/{n_iter}", flush=True)
    gp.eval(); lk.eval()
    return gp, lk


def _gp_predict_hf(gp, lk, X: np.ndarray):
    """Predict at HF fidelity (fidelity_code = 1.0)."""
    import gpytorch, torch
    fid   = np.ones((len(X), 1), dtype=np.float64)
    X_aug = np.concatenate([X.astype(np.float64), fid], axis=1)
    Xt    = torch.tensor(X_aug, dtype=torch.float64)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = lk(gp(Xt))
    return pred.mean.numpy().ravel(), pred.variance.clamp(min=0).sqrt().numpy().ravel()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_pkls(pkl_dir: Path):
    """Load (X, y) from all .pkl files in pkl_dir, sorted."""
    paths = sorted(pkl_dir.glob("*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No .pkl files in {pkl_dir}")
    X_list, y_list = [], []
    for p in paths:
        with open(p, "rb") as f:
            data = pickle.load(f)
        X_list.append([data["command"]["vx"],
                        data["command"]["vy"],
                        data["command"]["wz"]])
        y_list.append(data["error_summary"]["mean_abs_err_sum"])
    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32))


def load_bams_state(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract LF sim data from a BAMS state file (state_tN.npz).

    The .npz was saved by run_bams.py with keys:
        X_sim_all : (M, 3)  sim velocity commands accumulated up to iteration N
        y_sim_all : (M,)    sim tracking errors for those commands

    Returns X_sim (M, 3) and y_sim (M,) as float64.
    """
    state = np.load(npz_path, allow_pickle=True)
    X_sim = state["X_sim_all"].astype(np.float64)
    y_sim = state["y_sim_all"].astype(np.float64)
    print(f"  [STATE] Loaded from {npz_path}")
    print(f"  X_sim_all: {X_sim.shape}  "
          f"y=[{y_sim.min():.4f}, {y_sim.max():.4f}]")
    return X_sim, y_sim


# ---------------------------------------------------------------------------
# Accuracy metric
# ---------------------------------------------------------------------------

def accuracy_vec(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """clip(1 - |pred - true| / (|true| + eps), 0, 1) per sample."""
    eps = 1e-6
    return np.clip(1.0 - np.abs(pred - true) / (np.abs(true) + eps), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng    = np.random.default_rng(args.seed)
    root   = Path(args.results_root)

    methods       = ["scout", "bams", "random"]
    seed2_suffix  = "_2"   # seed 2 only

    # ── Step 1: collect shared failure eval set (all methods, both seeds) ──
    print("[INFO] Collecting shared failure eval set …")
    fail_X, fail_y = [], []

    for method in methods:
        for suf in ("", "_2"):
            d = root / f"{method}{suf}"
            if not d.is_dir():
                continue
            try:
                X, y = load_pkls(d)
                mask = y >= args.threshold
                fail_X.append(X[mask])
                fail_y.append(y[mask])
                print(f"  {d.name}: {mask.sum()} failures / {len(y)}")
            except Exception as e:
                print(f"  [WARN] {d.name}: {e}")

    if not fail_X:
        print("[ERROR] No failure data found.")
        return

    X_all_fail = np.concatenate(fail_X, axis=0)
    y_all_fail = np.concatenate(fail_y, axis=0)
    _, uniq    = np.unique(np.round(X_all_fail, 3), axis=0, return_index=True)
    X_eval     = X_all_fail[uniq].astype(np.float32)
    y_eval     = y_all_fail[uniq].astype(np.float64)
    print(f"\n[INFO] Eval set: {len(X_eval)} unique failure scenarios\n"
          f"  y=[{y_eval.min():.3f}, {y_eval.max():.3f}]")
    np.save(out_dir / "eval_X.npy", X_eval)
    np.save(out_dir / "eval_y.npy", y_eval)

    # ── Step 2: load LF sim data from BAMS state file ─────────────────────
    print(f"\n[INFO] Loading BAMS sim data from {args.bams_state} …")
    X_lf, y_lf = load_bams_state(args.bams_state)

    # ── Step 3: train and evaluate per method (seed 2) ────────────────────
    summary = {}

    for method in methods:
        d = root / f"{method}{seed2_suffix}"
        if not d.is_dir():
            print(f"\n[WARN] Seed-2 dir not found, skipping: {d}")
            continue

        try:
            X_hw, y_hw = load_pkls(d)
        except Exception as e:
            print(f"\n[WARN] Cannot load {d}: {e}")
            continue

        n_hw = len(X_hw)
        print(f"\n[{method.upper()} seed 2]  N_hw={n_hw}")

        # ── A) MLP — hardware data only ───────────────────────────────────
        mlp_cfg = MLPTrainConfig(
            hidden_dim = args.hidden_dim,
            depth      = args.depth,
            dropout    = args.dropout,
            lr         = args.lr,
            epochs     = args.epochs_mlp,
            seed       = args.seed,
            verbose    = False,
        )
        ckpt_path = out_dir / f"mlp_{method}_seed2.pt"
        print(f"  Training MLP ({args.epochs_mlp} epochs) …", flush=True)
        ckpt = train_mlp(X_hw, y_hw, mlp_cfg)
        save_checkpoint(ckpt, str(ckpt_path))
        mlp_model, mlp_ckpt, device = load_mlp_checkpoint(str(ckpt_path))

        mlp_mean, mlp_std = predict_mlp(
            mlp_model, mlp_ckpt,
            X_eval.astype(np.float32),
            device=device, mc_dropout=True, n_mc=args.n_mc,
        )
        mlp_mean = np.asarray(mlp_mean).ravel()
        mlp_std  = np.asarray(mlp_std).ravel()
        mlp_acc  = accuracy_vec(mlp_mean, y_eval)

        np.save(out_dir / f"mlp_mean_{method}_seed2.npy", mlp_mean)
        np.save(out_dir / f"mlp_std_{method}_seed2.npy",  mlp_std)
        print(f"  MLP  — mean_acc={mlp_acc.mean():.4f}  "
              f"std_acc={mlp_acc.std():.4f}  "
              f"min_acc={mlp_acc.min():.4f}")

        # ── B) Multi-fidelity GP — HW (HF) + BAMS sim (LF) ───────────────
        # Augmented input: [vx, vy, wz, fidelity_code]
        X_hf_aug = np.concatenate(
            [X_hw.astype(np.float64), np.ones((n_hw, 1))], axis=1)
        X_lf_aug = np.concatenate(
            [X_lf, np.zeros((len(X_lf), 1))], axis=1)
        X_aug    = np.concatenate([X_hf_aug, X_lf_aug], axis=0)
        y_aug    = np.concatenate([y_hw.astype(np.float64), y_lf], axis=0)

        print(f"  Training GP ({n_hw} HF + {len(X_lf)} LF pts, "
              f"d=4) …", flush=True)
        try:
            gp, lk = _train_gp(X_aug, y_aug, n_iter=args.gp_iters)

            # Predict at HF fidelity on the eval set
            gp_mean, gp_std = _gp_predict_hf(
                gp, lk, X_eval.astype(np.float64))
            gp_acc = accuracy_vec(gp_mean, y_eval)

            np.save(out_dir / f"gp_mean_{method}_seed2.npy", gp_mean)
            np.save(out_dir / f"gp_std_{method}_seed2.npy",  gp_std)
            print(f"  GP   — mean_acc={gp_acc.mean():.4f}  "
                  f"std_acc={gp_acc.std():.4f}  "
                  f"min_acc={gp_acc.min():.4f}")

            summary[method] = dict(
                mlp_acc = mlp_acc,
                gp_acc  = gp_acc,
            )
        except Exception as e:
            print(f"  [WARN] GP failed: {e}")
            summary[method] = dict(mlp_acc=mlp_acc, gp_acc=None)

    # ── Step 4: summary table ─────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"SURROGATE ACCURACY — SEED 2  "
          f"(N_eval={len(X_eval)}, threshold={args.threshold})")
    print(f"LF sim data: {args.bams_state}")
    print(f"{'='*65}")
    print(f"{'Method':<10}  {'Surrogate':<18}  "
          f"{'mean_acc':>9}  {'std_acc':>9}  {'min_acc':>9}")
    print("-" * 65)

    for method, res in summary.items():
        for key, label in [("mlp_acc", "MLP (HW only)"),
                            ("gp_acc",  "GP (HW + sim LF)")]:
            acc = res[key]
            if acc is None:
                print(f"  {method:<10}  {label:<18}  [failed]")
            else:
                print(f"  {method:<10}  {label:<18}  "
                      f"{acc.mean():>9.4f}  {acc.std():>9.4f}  "
                      f"{acc.min():>9.4f}")
        print()

    print(f"{'='*65}")
    print(f"[DONE]  outputs → {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--results_root",  type=str, required=True)
    p.add_argument("--bams_state",    type=str, required=True,
                   help="Path to state_tN.npz from run_bams.py. "
                        "Contains X_sim_all and y_sim_all for LF training.")
    p.add_argument("--threshold",     type=float, default=0.7)
    p.add_argument("--output_dir",    type=str,
                   default="results/quadruped/eval_seed2")
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--epochs_mlp",    type=int,   default=1000)
    p.add_argument("--hidden_dim",    type=int,   default=8)
    p.add_argument("--depth",         type=int,   default=2)
    p.add_argument("--dropout",       type=float, default=0.05)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--n_mc",          type=int,   default=50)
    p.add_argument("--gp_iters",      type=int,   default=200)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())