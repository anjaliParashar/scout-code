#!/usr/bin/env python3
"""
scripts/simpler/run_bams_simpler.py
--------------------------------------
BAMS active-learning baseline for SimplerEnv.

Uses the augmented multi-fidelity GP (the multi-fidelity GP used by BAMS):
  HF: target (real robot) evaluations  — fidelity_code = 1.0
  LF: proxy  (sim)        evaluations  — fidelity_code = 0.0

Rollout directory structure matches SCOUT exactly:
    <logging_root>/AL_{t}_{j}/<task>/rt1/<model>/scenario_XXXX/paired_result.json
LF-only evaluations stored as:
    <logging_root>/LF_{t}_{j}/<task>/...
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.simpler.domain         import DesignSpaceLimits, build_scenario_pool, scenarios_to_array
from utils.simpler.al_state       import ALState
from utils.simpler.rollout_runner import run_proxy_rollouts, run_target_rollouts
from utils.simpler.random_baseline import random_acquisition

FEATURE_DIM = 42


# ---------------------------------------------------------------------------
# GP helpers
# ---------------------------------------------------------------------------

def _train_gp(X_aug, y, n_iter=200):
    try:
        import gpytorch
    except ImportError:
        raise ImportError("pip install gpytorch")

    class _GP(gpytorch.models.ExactGP):
        def __init__(self, Xt, yt, lk):
            super().__init__(Xt, yt, lk)
            self.mean  = gpytorch.means.ConstantMean()
            self.covar = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=Xt.shape[1]))
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
    for _ in range(n_iter):
        opt.zero_grad()
        (-mll(gp(Xt), yt)).backward()
        opt.step()
    gp.eval(); lk.eval()
    return gp, lk


@torch.no_grad()
def _gp_score(gp, lk, X_pool, fidelity, cost):
    """GP posterior std / cost at given fidelity."""
    import gpytorch
    fid   = np.full((len(X_pool), 1), fidelity, dtype=np.float64)
    X_aug = np.concatenate([X_pool.astype(np.float64), fid], axis=1)
    Xt    = torch.tensor(X_aug, dtype=torch.float64)
    with gpytorch.settings.fast_pred_var():
        pred = lk(gp(Xt))
    std = pred.variance.clamp(min=0).sqrt().numpy()
    return std / cost


# ---------------------------------------------------------------------------
# BAMS acquisition
# ---------------------------------------------------------------------------

def bams_acquisition(state, gp, lk, batch_size, hf_cost, lf_cost,
                     random_state):
    if not state.pool:
        return [], []

    X_pool = scenarios_to_array(state.pool).astype(np.float64)
    hf_scores = _gp_score(gp, lk, X_pool, fidelity=1.0, cost=hf_cost)
    lf_scores = _gp_score(gp, lk, X_pool, fidelity=0.0, cost=lf_cost)

    # Select HF candidates (top batch_size by HF gain-per-cost)
    top_hf = np.argsort(hf_scores)[::-1][:batch_size]
    hf_selected = [state.pool[i] for i in top_hf]

    # Select LF candidates (top batch_size by LF gain-per-cost,
    # excluding already-selected HF)
    hf_set  = set(top_hf.tolist())
    lf_order = [i for i in np.argsort(lf_scores)[::-1] if i not in hf_set]
    lf_selected = [state.pool[i] for i in lf_order[:batch_size]]

    print(f"  [BAMS] HF={len(hf_selected)}  LF={len(lf_selected)}", flush=True)
    return hf_selected, lf_selected


# ---------------------------------------------------------------------------
# Rollout helpers
# ---------------------------------------------------------------------------

def _collect_seeds(scenario, n_seeds, logging_root, realization, base_seed, dry_run):
    vals = []
    for s in range(n_seeds):
        if realization == "target":
            mean_y, _, _ = run_target_rollouts(
                scenario, logging_root, n_target=1,
                seed=base_seed + s, dry_run=dry_run)
        else:
            mean_y, _, _ = run_proxy_rollouts(
                scenario, logging_root, n_proxy=1,
                seed=base_seed + s, dry_run=dry_run)
        vals.append(mean_y)
    return np.array(vals, dtype=np.float64)


def _eval_both(scenario, n_seeds, logging_root, base_seed, dry_run):
    y_target = _collect_seeds(scenario, n_seeds, logging_root,
                               "target", base_seed, dry_run)
    y_proxy  = _collect_seeds(scenario, n_seeds, logging_root,
                               "proxy",  base_seed + n_seeds, dry_run)
    return y_target, y_proxy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    rng     = np.random.default_rng(args.seed)
    out_dir = Path(args.logging_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    limits = DesignSpaceLimits(n_grid=args.n_grid)
    pool   = build_scenario_pool(limits=limits, n_random=args.pool_size,
                                 seed=args.seed)

    # ── Initial seed ──────────────────────────────────────────────────────
    if args.scout_init_dir:
        scout_init = Path(args.scout_init_dir)
        print(f"[BAMS] Loading init from {scout_init}")
        state     = ALState.load(scout_init)
        pool_ids  = {s.scenario_id for s in pool}
        evaluated = ({s.scenario_id for s in state.train} |
                     {s.scenario_id for s in state.proxy_only})
        unknown   = evaluated - pool_ids
        if unknown:
            print(f"  [WARN] {len(unknown)} ids not in pool.")
        state.pool = [s for s in pool if s.scenario_id not in evaluated]
        state.save(out_dir / "init")
        print(f"[BAMS] Loaded: target={len(state.train)}  "
              f"proxy_only={len(state.proxy_only)}  pool={len(state.pool)}")
    else:
        state          = ALState(pool=pool)
        n_init_proxy   = 5 * args.n_init
        all_init_local = rng.choice(len(state.pool), n_init_proxy, replace=False)
        target_local   = all_init_local[:args.n_init]
        proxy_only_local = all_init_local[args.n_init:]
        orig_pool_map  = {s.scenario_id: s for s in pool}

        print(f"[BAMS] Init: {args.n_init} target + {len(proxy_only_local)} proxy-only")
        for i, li in enumerate(target_local):
            sc   = state.pool[li]
            base = args.seed + i * 100
            sc_dir = str(out_dir / "init_rollouts" / f"{i}")
            Path(sc_dir).mkdir(parents=True, exist_ok=True)
            y_target, y_proxy = _eval_both(sc, args.n_seeds, sc_dir,
                                            base, args.dry_run)
            state.add_observation(sc, y_target, y_proxy)
        for i, li in enumerate(proxy_only_local):
            sc_id = pool[li].scenario_id
            if sc_id not in {s.scenario_id for s in state.pool}:
                continue
            sc   = orig_pool_map[sc_id]
            base = args.seed + args.n_init * 100 + i * 100
            y_p  = _collect_seeds(sc, args.n_seeds,
                                   str(out_dir / "init_rollouts"),
                                   "proxy", base, args.dry_run)
            state.add_proxy_only_observation(sc, y_p)
        state.save(out_dir / "init")
        print(f"[BAMS] Init done. target={len(state.train)}")

    # ── BAMS AL rounds ────────────────────────────────────────────────────
    for t in range(args.n_rounds):
        print(f"\n[BAMS ROUND {t:02d}/{args.n_rounds-1}]  "
              f"target={len(state.train)}  pool={len(state.pool)}")

        if len(state.pool) < args.batch_size:
            print("[STOP] Pool exhausted.")
            break

        X_train = state.X_train().astype(np.float64)
        y_train = state.y_train_mean().astype(np.float64)
        X_proxy = state.X_proxy_train().astype(np.float64)
        y_proxy = state.y_proxy_train_mean().astype(np.float64)

        if len(X_train) < 2:
            print("  [WARN] Not enough data — falling back to random.")
            batch = random_acquisition(state, args.batch_size, rng)
            for j, sc in enumerate(batch):
                base   = args.seed + 100_000 + t * 10_000 + j * 100
                al_dir = str(out_dir / f"AL_{t}_{j}")
                Path(al_dir).mkdir(parents=True, exist_ok=True)
                y_target, y_proxy_out = _eval_both(
                    sc, args.n_seeds, al_dir, base, args.dry_run)
                state.add_observation(sc, y_target, y_proxy_out)
            state.save(out_dir / f"iter_{t:02d}")
            continue

        # Build augmented GP training data
        M_hf = len(X_train)
        M_lf = len(X_proxy)
        X_hf_aug = np.concatenate([X_train, np.ones((M_hf, 1))], axis=1)
        y_hf     = y_train
        if M_lf > 0:
            X_lf_aug = np.concatenate([X_proxy, np.zeros((M_lf, 1))], axis=1)
            X_aug    = np.concatenate([X_hf_aug, X_lf_aug], axis=0)
            y_aug    = np.concatenate([y_hf, y_proxy], axis=0)
        else:
            X_aug = X_hf_aug
            y_aug = y_hf

        if len(X_aug) > args.gp_max_train:
            sub   = rng.choice(len(X_aug), args.gp_max_train, replace=False)
            X_aug = X_aug[sub]
            y_aug = y_aug[sub]

        print(f"  Training GP ({len(X_aug)} pts, d={X_aug.shape[1]})…", flush=True)
        try:
            gp, lk = _train_gp(X_aug, y_aug, n_iter=args.gp_iters)
        except Exception as e:
            print(f"  [WARN] GP failed ({e}) — random fallback.")
            batch = random_acquisition(state, args.batch_size, rng)
            for j, sc in enumerate(batch):
                base   = args.seed + 100_000 + t * 10_000 + j * 100
                al_dir = str(out_dir / f"AL_{t}_{j}")
                Path(al_dir).mkdir(parents=True, exist_ok=True)
                y_target, y_proxy_out = _eval_both(
                    sc, args.n_seeds, al_dir, base, args.dry_run)
                state.add_observation(sc, y_target, y_proxy_out)
            state.save(out_dir / f"iter_{t:02d}")
            continue

        hf_batch, lf_batch = bams_acquisition(
            state, gp, lk,
            batch_size   = args.batch_size,
            hf_cost      = 1.0,
            lf_cost      = args.lf_cost,
            random_state = args.seed + t,
        )

        # Evaluate HF (target + proxy)  — AL_{t}_{j} directories
        for j, sc in enumerate(hf_batch):
            base   = args.seed + 100_000 + t * 10_000 + j * 100
            al_dir = str(out_dir / f"AL_{t}_{j}")
            Path(al_dir).mkdir(parents=True, exist_ok=True)
            y_target, y_proxy_out = _eval_both(sc, args.n_seeds, al_dir,
                                                base, args.dry_run)
            print(f"  [HF] id={sc.scenario_id}  "
                  f"y_target={y_target.mean():.3f}  "
                  f"y_proxy={y_proxy_out.mean():.3f}")
            state.add_observation(sc, y_target, y_proxy_out)

        # Evaluate LF (proxy only) — LF_{t}_{j} directories
        for j, sc in enumerate(lf_batch):
            base   = args.seed + 200_000 + t * 10_000 + j * 100
            lf_dir = str(out_dir / f"LF_{t}_{j}")
            Path(lf_dir).mkdir(parents=True, exist_ok=True)
            y_p = _collect_seeds(sc, args.n_seeds, lf_dir,
                                  "proxy", base, args.dry_run)
            print(f"  [LF] id={sc.scenario_id}  y_proxy={y_p.mean():.3f}")
            state.add_proxy_only_observation(sc, y_p)

        state.save(out_dir / f"iter_{t:02d}")

    state.save(out_dir / "final")
    print(f"\n[BAMS DONE]  target={len(state.train)}  "
          f"proxy_only={len(state.proxy_only)}")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--logging-root",   type=str, default="./results/bams")
    p.add_argument("--pool-size",      type=int, default=500)
    p.add_argument("--n-grid",         type=int, default=3)
    p.add_argument("--n-init",         type=int, default=20)
    p.add_argument("--n-rounds",       type=int, default=20)
    p.add_argument("--batch-size",     type=int, default=2)
    p.add_argument("--n-seeds",        type=int, default=2)
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--device",         type=str, default="cpu")
    p.add_argument("--dry-run",        action="store_true")
    p.add_argument("--scout-init-dir", type=str, default="")
    p.add_argument("--lf-cost",        type=float, default=0.05,
                   help="Cost of proxy eval relative to target (≈ free)")
    p.add_argument("--gp-iters",       type=int, default=200)
    p.add_argument("--gp-max-train",   type=int, default=200)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
