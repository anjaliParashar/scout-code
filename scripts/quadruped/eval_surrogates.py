#!/usr/bin/env python3
"""
scripts/quadruped/eval_surrogates.py
--------------------------------------
Evaluate surrogate model accuracy on high-failure scenarios.

Pipeline
--------
1. Collect all scenarios with y >= threshold across ALL methods and BOTH seeds.
   This gives a shared failure evaluation set that is method-agnostic.

2. For each seed (1, 2), train two surrogate models:
   a) MLP  — trained on hardware data from that seed only (same as run_scout.py)
   b) BAMS GP — augmented multi-fidelity GP trained on hardware (HF, cost=1.0)
                + sim CSV rows nearest to the hardware commands (LF, cost=0.05)

3. Evaluate both surrogates on the shared failure set:
   Accuracy = 1 - |pred - true| / (|true| + eps)   clamped to [0, 1]

4. Report mean ± std over seeds for each method × surrogate combination,
   printed to stdout as a table.

Directory layout expected
-------------------------
    <results_root>/
        scout/          ← seed 1 hardware pkls
        scout_2/        ← seed 2 hardware pkls
        bams/
        bams_2/
        random/
        random_2/

Example
-------
python scripts/quadruped/eval_surrogates.py \
    --results_root results/go2_hardware \
    --sim_csv      data/quadruped/fixed_twist_combined.csv \
    --threshold    0.9 \
    --output_dir   outputs/quadruped/results
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from glob import glob

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.quadruped.train_utils import (
    MLPTrainConfig,
    load_hardware_pkls,
    load_sim_csv,
    train_mlp,
    save_checkpoint,
    load_mlp_checkpoint,
    predict_mlp,
)


# ---------------------------------------------------------------------------
# GP helpers (copied from run_bams.py so no circular import)
# ---------------------------------------------------------------------------

def _train_gp(X_aug: np.ndarray, y: np.ndarray, n_iter: int = 200):
    """Augmented ExactGP: input = [vx, vy, wz, fidelity_code]."""
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

    import torch
    Xt   = torch.tensor(X_aug, dtype=torch.float64)
    yt   = torch.tensor(y,     dtype=torch.float64)
    lk   = gpytorch.likelihoods.GaussianLikelihood()
    gp   = _GP(Xt, yt, lk)
    gp.train(); lk.train()
    opt  = torch.optim.Adam(gp.parameters(), lr=0.05)
    mll  = gpytorch.mlls.ExactMarginalLogLikelihood(lk, gp)
    for _ in range(n_iter):
        opt.zero_grad()
        loss = -mll(gp(Xt), yt)
        loss.backward()
        opt.step()
    gp.eval(); lk.eval()
    return gp, lk


def _gp_predict(gp, lk, X: np.ndarray, fidelity: float):
    """GP posterior mean and std at X with given fidelity code."""
    import gpytorch, torch
    fid   = np.full((len(X), 1), fidelity, dtype=np.float64)
    X_aug = np.concatenate([X.astype(np.float64), fid], axis=1)
    Xt    = torch.tensor(X_aug, dtype=torch.float64)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = lk(gp(Xt))
    mean = pred.mean.numpy()
    std  = pred.variance.clamp(min=0).sqrt().numpy()
    return mean.ravel(), std.ravel()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_seed_pkls(pkl_dir: Path):
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


def nearest_sim_rows(X_hw, X_sim, n_sim, rng):
    """
    Select n_sim sim CSV rows nearest (in velocity space) to hardware points.
    Falls back to random if not enough distinct rows.
    """
    used = set()
    result = []
    for xq in X_hw:
        dists = np.linalg.norm(X_sim - xq[None, :], axis=1)
        order = np.argsort(dists)
        for idx in order:
            if int(idx) not in used:
                used.add(int(idx))
                result.append(int(idx))
                break
        if len(result) >= n_sim:
            break
    # fill with random if needed
    remaining = n_sim - len(result)
    if remaining > 0:
        avail = list(set(range(len(X_sim))) - used)
        extra = rng.choice(avail, min(remaining, len(avail)), replace=False)
        result.extend(extra.tolist())
    return np.array(result[:n_sim], int)


# ---------------------------------------------------------------------------
# Accuracy metric
# ---------------------------------------------------------------------------

def accuracy_on_set(pred_mean: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Per-sample accuracy = clip(1 - |pred - true| / (|true| + eps), 0, 1).
    Returns (N,) array.
    """
    eps = 1e-6
    abs_err = np.abs(pred_mean - y_true)
    rel_err = abs_err / (np.abs(y_true) + eps)
    return np.clip(1.0 - rel_err, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng     = np.random.default_rng(args.seed)
    root    = Path(args.results_root)

    methods      = ["scout", "bams", "random"]
    seed_suffixes = ["", "_2"]

    # ── Step 1: collect shared failure set across all methods and seeds ───
    print("[INFO] Collecting shared failure set …")
    fail_X_list, fail_y_list = [], []

    for method in methods:
        for suf in seed_suffixes:
            d = root / f"{method}{suf}"
            if not d.is_dir():
                print(f"  [WARN] Not found: {d}")
                continue
            try:
                X, y = load_seed_pkls(d)
                fail_mask = y >= args.threshold
                fail_X_list.append(X[fail_mask])
                fail_y_list.append(y[fail_mask])
                print(f"  {d.name}: {int(fail_mask.sum())} failures / {len(y)} total")
            except Exception as e:
                print(f"  [WARN] {d.name}: {e}")

    if not fail_X_list:
        print("[ERROR] No failure data found.")
        return

    # Deduplicate by rounding to 3 decimal places
    X_fail_all = np.concatenate(fail_X_list, axis=0)
    y_fail_all = np.concatenate(fail_y_list, axis=0)
    _, unique_idx = np.unique(np.round(X_fail_all, 3), axis=0, return_index=True)
    X_eval = X_fail_all[unique_idx].astype(np.float32)
    y_eval = y_fail_all[unique_idx].astype(np.float32)
    print(f"\n[INFO] Shared eval set: {len(X_eval)} unique failure scenarios "
          f"(y >= {args.threshold})\n"
          f"  y range: [{y_eval.min():.3f}, {y_eval.max():.3f}]")
    np.save(out_dir / "eval_X.npy", X_eval)
    np.save(out_dir / "eval_y.npy", y_eval)

    # ── Step 2: load sim CSV (for BAMS GP) ───────────────────────────────
    print(f"\n[INFO] Loading sim CSV …")
    X_sim, y_sim = load_sim_csv(args.sim_csv, metric=args.sim_metric)
    print(f"  sim N={len(X_sim)}  y=[{y_sim.min():.4f}, {y_sim.max():.4f}]")

    # ── Step 3: train surrogates per method × seed, evaluate ─────────────
    # results[method][surrogate] = list of accuracy arrays (one per seed)
    results: dict[str, dict[str, list]] = {
        m: {"mlp": [], "gp": []} for m in methods
    }

    for method in methods:
        for s_idx, suf in enumerate(seed_suffixes):
            d = root / f"{method}{suf}"
            if not d.is_dir():
                continue

            try:
                X_hw, y_hw, _ = load_hardware_pkls(str(d / "*.pkl"))
            except Exception:
                # load_hardware_pkls expects a glob string
                try:
                    X_hw, y_hw = load_seed_pkls(d)
                    X_hw = X_hw.astype(np.float32)
                    y_hw = y_hw.astype(np.float32)
                except Exception as e:
                    print(f"  [WARN] Cannot load {d}: {e}")
                    continue

            n_hw = len(X_hw)
            print(f"\n[{method} seed {s_idx+1}] N_hw={n_hw}")

            # ── Surrogate A: MLP on hardware only ─────────────────────────
            mlp_cfg = MLPTrainConfig(
                hidden_dim = args.hidden_dim,
                depth      = args.depth,
                dropout    = args.dropout,
                lr         = args.lr,
                epochs     = args.epochs_mlp,
                seed       = args.seed + s_idx,
                verbose    = False,
            )
            mlp_ckpt_path = out_dir / f"mlp_{method}_seed{s_idx+1}.pt"
            print(f"  Training MLP ({args.epochs_mlp} epochs) …", flush=True)
            mlp_ckpt = train_mlp(X_hw, y_hw, mlp_cfg)
            save_checkpoint(mlp_ckpt, str(mlp_ckpt_path))
            mlp_model, mlp_ckpt_loaded, device = load_mlp_checkpoint(
                str(mlp_ckpt_path))

            mlp_mean, mlp_std = predict_mlp(
                mlp_model, mlp_ckpt_loaded,
                X_eval.astype(np.float32),
                device=device, mc_dropout=True, n_mc=args.n_mc,
            )
            mlp_mean = np.asarray(mlp_mean).ravel()
            mlp_acc  = accuracy_on_set(mlp_mean, y_eval.astype(np.float64))
            results[method]["mlp"].append(mlp_acc)
            print(f"  MLP  — mean_acc={mlp_acc.mean():.4f}  "
                  f"std_acc={mlp_acc.std():.4f}")

            # ── Surrogate B: BAMS augmented GP ────────────────────────────
            n_sim = min(args.n_sim_per_hw * n_hw, len(X_sim))
            sim_idx = nearest_sim_rows(
                X_hw.astype(np.float64),
                X_sim.astype(np.float64),
                n_sim, rng,
            )
            X_lf = X_sim[sim_idx].astype(np.float64)
            y_lf = y_sim[sim_idx].astype(np.float64)

            X_hf_aug = np.concatenate(
                [X_hw.astype(np.float64),
                 np.ones((n_hw, 1))], axis=1)
            X_lf_aug = np.concatenate(
                [X_lf, np.zeros((len(X_lf), 1))], axis=1)
            X_aug = np.concatenate([X_hf_aug, X_lf_aug], axis=0)
            y_aug = np.concatenate([y_hw.astype(np.float64), y_lf], axis=0)

            print(f"  Training BAMS GP ({n_hw} HF + {len(X_lf)} LF) …",
                  flush=True)
            try:
                gp, lk = _train_gp(X_aug, y_aug, n_iter=args.gp_iters)
                gp_mean, gp_std = _gp_predict(
                    gp, lk, X_eval.astype(np.float64), fidelity=1.0)
                gp_acc = accuracy_on_set(gp_mean, y_eval.astype(np.float64))
                results[method]["gp"].append(gp_acc)
                print(f"  GP   — mean_acc={gp_acc.mean():.4f}  "
                      f"std_acc={gp_acc.std():.4f}")

                # Save GP predictions
                np.save(out_dir / f"gp_mean_{method}_seed{s_idx+1}.npy",
                        gp_mean)
                np.save(out_dir / f"gp_std_{method}_seed{s_idx+1}.npy",
                        gp_std)
            except Exception as e:
                print(f"  [WARN] GP training failed: {e}")

            # Save MLP predictions
            np.save(out_dir / f"mlp_mean_{method}_seed{s_idx+1}.npy", mlp_mean)
            np.save(out_dir / f"mlp_std_{method}_seed{s_idx+1}.npy",
                    np.asarray(mlp_std).ravel())

    # ── Step 4: print summary table ───────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"SURROGATE ACCURACY ON FAILURE SET  "
          f"(N_eval={len(X_eval)}, threshold={args.threshold})")
    print(f"Mean ± Std computed over {len(seed_suffixes)} seeds")
    print(f"{'='*70}")
    print(f"{'Method':<10}  {'Surrogate':<12}  "
          f"{'mean_acc':>10}  {'std_acc':>10}  "
          f"{'min_acc':>9}  {'max_acc':>9}")
    print("-" * 70)

    for method in methods:
        for surrogate in ["mlp", "gp"]:
            accs = results[method][surrogate]
            if not accs:
                print(f"  {method:<10}  {surrogate:<12}  [no data]")
                continue

            # Stack across seeds: (n_seeds, N_eval)
            stacked   = np.stack(accs, axis=0)          # (n_seeds, N_eval)
            seed_means = stacked.mean(axis=1)            # per-seed mean
            mean_acc   = float(seed_means.mean())        # mean over seeds
            std_acc    = float(seed_means.std())         # std over seeds
            min_acc    = float(stacked.min())
            max_acc    = float(stacked.max())

            surr_label = "MLP (HW)" if surrogate == "mlp" else "GP (HW+Sim)"
            print(f"  {method:<10}  {surr_label:<12}  "
                  f"{mean_acc:>10.4f}  {std_acc:>10.4f}  "
                  f"{min_acc:>9.4f}  {max_acc:>9.4f}")
        print()

    print(f"{'='*70}")
    print(f"[DONE]  outputs → {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Evaluate MLP and BAMS GP surrogates on shared failure set."
    )
    p.add_argument("--results_root",  type=str, required=True,
                   help="Root dir containing scout/, scout_2/, bams/, etc.")
    p.add_argument("--sim_csv",       type=str, required=True,
                   help="Path to fixed_twist_combined_2000.csv")
    p.add_argument("--sim_metric",    type=str,
                   default="final_abs_err_sum")
    p.add_argument("--threshold",     type=float, default=0.7,
                   help="y >= threshold = failure")
    p.add_argument("--output_dir",    type=str,
                   default="results/quadruped/eval")
    p.add_argument("--seed",          type=int,   default=0)

    # MLP hyperparameters
    p.add_argument("--epochs_mlp",    type=int,   default=3000)
    p.add_argument("--hidden_dim",    type=int,   default=8)
    p.add_argument("--depth",         type=int,   default=2)
    p.add_argument("--dropout",       type=float, default=0.05)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--n_mc",          type=int,   default=50,
                   help="MC-dropout samples for MLP prediction")

    # BAMS GP hyperparameters
    p.add_argument("--gp_iters",      type=int,   default=200,
                   help="GP optimisation steps")
    p.add_argument("--n_sim_per_hw",  type=int,   default=5,
                   help="Sim rows per hardware point for GP LF training")

    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
