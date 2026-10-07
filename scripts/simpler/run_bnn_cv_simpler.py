#!/usr/bin/env python3
"""
scripts/simpler/run_bnn_cv_simpler.py
---------------------------------------
BNN-CV active-learning baseline for SimplerEnv.

Trains a BNN on current target observations, evaluates the CV-corrected
mean (mu_CV) on the pool, and selects the scenarios with the LOWEST mu_CV
(most likely to fail in the real system).

Rollout directory structure matches SCOUT exactly:
    <logging_root>/AL_{t}_{j}/<task>/rt1/<model>/scenario_XXXX/paired_result.json
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.simpler.domain         import DesignSpaceLimits, build_scenario_pool, scenarios_to_array
from utils.simpler.al_state       import ALState
from utils.simpler.rollout_runner import run_proxy_rollouts, run_target_rollouts
from utils.simpler.random_baseline import random_acquisition

FEATURE_DIM = 19


# ---------------------------------------------------------------------------
# BNN (shared architecture for target and proxy)
# ---------------------------------------------------------------------------

class BNN(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, p_drop=0.01):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96), nn.Tanh(), nn.Dropout(p_drop),
            nn.Linear(96, 24),        nn.Tanh(), nn.Dropout(p_drop),
            nn.Linear(24, 6),         nn.Tanh(), nn.Dropout(p_drop),
        )
        self.head = nn.Linear(6, 1)

    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)


def _train_bnn(X, y, epochs, lr, device, label=""):
    print(f"  [BNN-CV] Training {label} BNN (n={len(X)}, epochs={epochs})…",
          flush=True)
    model   = BNN().to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    loss_fn = nn.MSELoss()
    X_t     = torch.tensor(X, dtype=torch.float32)
    y_t     = torch.tensor(y, dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X_t))
        for s in range(0, len(X_t), 256):
            idx = perm[s:s+256]
            xb  = X_t[idx].to(device)
            yb  = y_t[idx].to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def _mc_draws(model, X, n_mc, device):
    """MC-dropout draws. Returns (n_mc, N)."""
    model.train()
    X_t   = torch.tensor(X, dtype=torch.float32).to(device)
    draws = torch.stack([model(X_t) for _ in range(n_mc)], dim=0)
    model.eval()
    return draws.cpu().numpy()   # (n_mc, N)


# ---------------------------------------------------------------------------
# Control-variate estimator
# ---------------------------------------------------------------------------

def _cv_estimate(f, g, g_unp):
    """
    mu_hat = mean(f - beta*g) + beta * mean(g_unp)
    where beta = cov(g,f) / var(g)   (shrinkage-adjusted)
    """
    f, g, g_unp = [np.asarray(a, np.float64).ravel() for a in (f, g, g_unp)]
    n, k = len(f), len(g_unp)
    if n < 2 or np.isclose(np.var(g), 0):
        return float(np.mean(f))
    cov_gf  = np.cov(g, f)[0, 1]
    var_g   = np.var(g)
    beta    = (k / (k + n)) * cov_gf / var_g
    mu_hat  = np.mean(f - beta * g) + beta * np.mean(g_unp)
    return float(mu_hat)


# ---------------------------------------------------------------------------
# BNN-CV acquisition
# ---------------------------------------------------------------------------

def bnn_cv_acquisition(state, target_bnn, proxy_bnn, batch_size,
                       n_mc, n_pair, n_unp, device):
    if not state.pool or not state.train:
        return None

    X_pool = scenarios_to_array(state.pool).astype(np.float32)

    # MC draws for target and proxy on pool
    f_draws = _mc_draws(target_bnn, X_pool, n_mc, device)   # (n_mc, S)
    g_draws = _mc_draws(proxy_bnn,  X_pool, n_mc, device)   # (n_mc, S)

    S       = len(state.pool)
    cv_vals = np.full(S, np.nan)

    for j in range(S):
        f_j   = f_draws[:n_pair, j]          # paired target draws
        g_p_j = g_draws[:n_pair, j]          # paired proxy draws
        g_u_j = g_draws[n_pair:n_pair+n_unp, j]  # unpaired proxy draws
        if len(g_u_j) < 2:
            g_u_j = g_p_j
        cv_vals[j] = _cv_estimate(f_j, g_p_j, g_u_j)

    # Fill NaN with median
    finite = np.isfinite(cv_vals)
    if finite.any():
        cv_vals[~finite] = float(np.median(cv_vals[finite]))

    # Select lowest mu_CV (most likely to fail)
    order    = np.argsort(cv_vals)
    selected = [state.pool[i] for i in order[:batch_size]]
    print(f"  [BNN-CV] mu_CV min={cv_vals.min():.4f}  "
          f"selected={[s.scenario_id for s in selected]}", flush=True)
    return selected


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
    device  = args.device
    rng     = np.random.default_rng(args.seed)
    out_dir = Path(args.logging_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    limits = DesignSpaceLimits(n_grid=args.n_grid)
    pool   = build_scenario_pool(limits=limits, n_random=args.pool_size,
                                 seed=args.seed)

    # ── Initial seed ──────────────────────────────────────────────────────
    if args.scout_init_dir:
        scout_init = Path(args.scout_init_dir)
        print(f"[BNN-CV] Loading init from {scout_init}")
        state     = ALState.load(scout_init)
        pool_ids  = {s.scenario_id for s in pool}
        evaluated = ({s.scenario_id for s in state.train} |
                     {s.scenario_id for s in state.proxy_only})
        unknown   = evaluated - pool_ids
        if unknown:
            print(f"  [WARN] {len(unknown)} ids not in pool.")
        state.pool = [s for s in pool if s.scenario_id not in evaluated]
        state.save(out_dir / "init")
        print(f"[BNN-CV] Loaded: target={len(state.train)}  "
              f"proxy_only={len(state.proxy_only)}  pool={len(state.pool)}")
    else:
        state          = ALState(pool=pool)
        n_init_proxy   = 5 * args.n_init
        all_init_local = rng.choice(len(state.pool), n_init_proxy, replace=False)
        target_local   = all_init_local[:args.n_init]
        proxy_only_local = all_init_local[args.n_init:]
        orig_pool_map  = {s.scenario_id: s for s in pool}

        print(f"[BNN-CV] Init: {args.n_init} target + "
              f"{len(proxy_only_local)} proxy-only")
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
        print(f"[BNN-CV] Init done. target={len(state.train)}")

    # ── BNN-CV AL rounds ──────────────────────────────────────────────────
    for t in range(args.n_rounds):
        print(f"\n[BNN-CV ROUND {t:02d}/{args.n_rounds-1}]  "
              f"target={len(state.train)}  pool={len(state.pool)}")

        if len(state.pool) < args.batch_size:
            print("[STOP] Pool exhausted.")
            break

        X_train  = state.X_train().astype(np.float32)
        y_train  = state.y_train_mean().astype(np.float32)
        X_proxy  = state.X_proxy_train().astype(np.float32)
        y_proxy  = state.y_proxy_train_mean().astype(np.float32)

        if len(X_train) < 2 or len(X_proxy) < 2:
            print("  [WARN] Not enough data — falling back to random.")
            batch = random_acquisition(state, args.batch_size, rng)
        else:
            target_bnn = _train_bnn(X_train, y_train, args.bnn_epochs,
                                     args.lr, device, label="target")
            proxy_bnn  = _train_bnn(X_proxy, y_proxy, args.bnn_epochs,
                                     args.lr, device, label="proxy")
            batch = bnn_cv_acquisition(
                state      = state,
                target_bnn = target_bnn,
                proxy_bnn  = proxy_bnn,
                batch_size = args.batch_size,
                n_mc       = args.n_mc,
                n_pair     = args.n_pair,
                n_unp      = args.n_unp,
                device     = device,
            )
            if batch is None:
                batch = random_acquisition(state, args.batch_size, rng)

        print(f"  Selected: {[s.scenario_id for s in batch]}")
        for j, sc in enumerate(batch):
            base   = args.seed + 100_000 + t * 10_000 + j * 100
            al_dir = str(out_dir / f"AL_{t}_{j}")
            Path(al_dir).mkdir(parents=True, exist_ok=True)
            y_target, y_proxy_out = _eval_both(sc, args.n_seeds, al_dir,
                                                base, args.dry_run)
            print(f"  [EVAL] id={sc.scenario_id}  "
                  f"y_target={y_target.mean():.3f}  "
                  f"y_proxy={y_proxy_out.mean():.3f}")
            state.add_observation(sc, y_target, y_proxy_out)

        state.save(out_dir / f"iter_{t:02d}")

    state.save(out_dir / "final")
    print(f"\n[BNN-CV DONE]  target={len(state.train)}")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--logging-root",   type=str, default="./results/bnn_cv")
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
    p.add_argument("--bnn-epochs",     type=int, default=500)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--n-mc",           type=int, default=30,
                   help="MC-dropout samples for BNN predictions")
    p.add_argument("--n-pair",         type=int, default=10,
                   help="Paired draws for CV estimator")
    p.add_argument("--n-unp",          type=int, default=20,
                   help="Unpaired draws for CV theta_hat")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
