#!/usr/bin/env python3
"""
scripts/simpler/run_gpc_simpler.py
------------------------------------
GP-C baseline for SimplerEnv AL experiment.

BNN classifier baseline:
  1. Train a failure CLASSIFIER on proxy-only observations
     (label = 1 if y_proxy >= failure_threshold).
  2. Train a target BNN on current target observations.
  3. Filter the pool to scenarios predicted as SUCCESS by the proxy classifier
     (p_fail < success_threshold — sim says it looks safe).
  4. From the success-predicted pool, cluster into batch_size groups and pick
     the scenario with the highest target BNN failure score from each cluster.
  5. Evaluate those scenarios on the real (target) system.

Rollout directory structure matches SCOUT exactly:
    <logging_root>/AL_{t}_{j}/<task>/rt1/<model>/scenario_XXXX/paired_result.json
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.simpler.domain         import DesignSpaceLimits, build_scenario_pool, scenarios_to_array
from utils.simpler.al_state       import ALState
from utils.simpler.rollout_runner import run_proxy_rollouts, run_target_rollouts
from utils.simpler.random_baseline import random_acquisition

FEATURE_DIM = 19


# ---------------------------------------------------------------------------
# Failure classifier (standalone BNN — same architecture as bnn_c_baseline.py)
# ---------------------------------------------------------------------------

class FailureClassifier(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, p_drop=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96), nn.Tanh(), nn.Dropout(p_drop),
            nn.Linear(96, 24),        nn.Tanh(), nn.Dropout(p_drop),
            nn.Linear(24, 6),         nn.Tanh(), nn.Dropout(p_drop),
        )
        self.head = nn.Linear(6, 1)

    def forward(self, x):
        return torch.sigmoid(self.head(self.net(x))).squeeze(-1)


def _train_classifier(X, y_proxy, failure_threshold, epochs, lr, device):
    labels = (y_proxy >= failure_threshold).astype(np.float32)
    n_fail = int(labels.sum())
    print(f"  [GPC] Classifier: {len(X)} pts  "
          f"failure={n_fail}  safe={len(X)-n_fail}", flush=True)
    model  = FailureClassifier().to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    loss_fn = nn.BCELoss()
    X_t    = torch.tensor(X,      dtype=torch.float32)
    y_t    = torch.tensor(labels, dtype=torch.float32)
    model.train()
    for ep in range(epochs):
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


class TargetBNN(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, p_drop=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96), nn.Tanh(), nn.Dropout(p_drop),
            nn.Linear(96, 24),        nn.Tanh(), nn.Dropout(p_drop),
            nn.Linear(24, 6),         nn.Tanh(), nn.Dropout(p_drop),
        )
        self.head = nn.Linear(6, 1)

    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)


def _train_target_bnn(X, y, epochs, lr, device):
    model  = TargetBNN().to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    loss_fn = nn.MSELoss()
    X_t    = torch.tensor(X, dtype=torch.float32)
    y_t    = torch.tensor(y, dtype=torch.float32)
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
def _mc_predict(model, X, n_mc, device):
    model.train()   # enable dropout
    X_t  = torch.tensor(X, dtype=torch.float32).to(device)
    draws = torch.stack([model(X_t) for _ in range(n_mc)], dim=0)
    model.eval()
    return draws.mean(dim=0).cpu().numpy(), draws.std(dim=0).cpu().numpy()


# ---------------------------------------------------------------------------
# GP-C acquisition
# ---------------------------------------------------------------------------

def gpc_acquisition(state, batch_size, classifier, target_bnn,
                    success_threshold, n_mc, device, random_state):
    if not state.pool:
        return []

    X_pool = scenarios_to_array(state.pool).astype(np.float32)
    S      = len(state.pool)

    # Step 1: classify pool — keep success-predicted (sim says safe)
    with torch.no_grad():
        classifier.train()
        X_t    = torch.tensor(X_pool, dtype=torch.float32).to(device)
        p_fail = torch.stack(
            [classifier(X_t) for _ in range(n_mc)], dim=0
        ).mean(dim=0).cpu().numpy()
        classifier.eval()

    success_mask  = p_fail < success_threshold
    success_local = np.where(success_mask)[0]
    print(f"  [GPC] Pool={S}  sim-success={success_mask.sum()}  "
          f"sim-failure={S - success_mask.sum()}", flush=True)

    if len(success_local) == 0:
        rng = np.random.default_rng(random_state)
        return [state.pool[i] for i in
                rng.choice(S, min(batch_size, S), replace=False)]

    k = min(batch_size, len(success_local))
    if k >= len(success_local):
        return [state.pool[i] for i in success_local]

    # Step 2: BNN failure score on success pool
    X_success      = X_pool[success_local]
    target_mean, _ = _mc_predict(target_bnn, X_success, n_mc, device)
    fail_score     = target_mean   # higher predicted y = more likely failure

    # Step 3: cluster success pool, pick highest-score from each cluster
    km     = KMeans(n_clusters=k, random_state=random_state,
                    n_init="auto", max_iter=300)
    labels = km.fit_predict(X_success)

    selected = []
    for c in range(k):
        in_c = np.where(labels == c)[0]
        if len(in_c) == 0:
            continue
        best = in_c[np.argmax(fail_score[in_c])]
        selected.append(state.pool[int(success_local[best])])

    return selected


# ---------------------------------------------------------------------------
# Rollout helpers (identical to run_random_simpler.py)
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
        print(f"[GPC] Loading init from {scout_init}")
        state     = ALState.load(scout_init)
        pool_ids  = {s.scenario_id for s in pool}
        evaluated = ({s.scenario_id for s in state.train} |
                     {s.scenario_id for s in state.proxy_only})
        unknown   = evaluated - pool_ids
        if unknown:
            print(f"  [WARN] {len(unknown)} ids not in pool.")
        state.pool = [s for s in pool if s.scenario_id not in evaluated]
        state.save(out_dir / "init")
        print(f"[GPC] Loaded: target={len(state.train)}  "
              f"proxy_only={len(state.proxy_only)}  pool={len(state.pool)}")
    else:
        state          = ALState(pool=pool)
        n_init_proxy   = 5 * args.n_init
        all_init_local = rng.choice(len(state.pool), n_init_proxy, replace=False)
        target_local   = all_init_local[:args.n_init]
        proxy_only_local = all_init_local[args.n_init:]
        orig_pool_map  = {s.scenario_id: s for s in pool}

        print(f"[GPC] Init: {args.n_init} target + {len(proxy_only_local)} proxy-only")
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
        print(f"[GPC] Init done. target={len(state.train)}")

    # ── GP-C AL rounds ────────────────────────────────────────────────────
    for t in range(args.n_rounds):
        print(f"\n[GPC ROUND {t:02d}/{args.n_rounds-1}]  "
              f"target={len(state.train)}  pool={len(state.pool)}")

        if len(state.pool) < args.batch_size:
            print("[STOP] Pool exhausted.")
            break

        # Train classifier on proxy observations
        X_proxy = state.X_proxy_train().astype(np.float32)
        y_proxy = state.y_proxy_train_mean().astype(np.float32)

        # Train target BNN on target observations
        X_train = state.X_train().astype(np.float32)
        y_train = state.y_train_mean().astype(np.float32)

        if len(X_proxy) < 4 or len(X_train) < 2:
            print("  [WARN] Not enough data — falling back to random.")
            batch = random_acquisition(state, args.batch_size, rng)
        else:
            print(f"  Training classifier ({len(X_proxy)} proxy pts)…", flush=True)
            classifier = _train_classifier(
                X_proxy, y_proxy,
                failure_threshold = args.failure_threshold,
                epochs            = args.cls_epochs,
                lr                = args.lr,
                device            = device,
            )
            print(f"  Training target BNN ({len(X_train)} target pts)…", flush=True)
            target_bnn = _train_target_bnn(
                X_train, y_train,
                epochs = args.bnn_epochs,
                lr     = args.lr,
                device = device,
            )
            batch = gpc_acquisition(
                state             = state,
                batch_size        = args.batch_size,
                classifier        = classifier,
                target_bnn        = target_bnn,
                success_threshold = args.success_threshold,
                n_mc              = args.n_mc,
                device            = device,
                random_state      = args.seed + t,
            )

        print(f"  Selected: {[s.scenario_id for s in batch]}")
        for j, sc in enumerate(batch):
            base   = args.seed + 100_000 + t * 10_000 + j * 100
            al_dir = str(out_dir / f"AL_{t}_{j}")
            Path(al_dir).mkdir(parents=True, exist_ok=True)
            y_target, y_proxy = _eval_both(sc, args.n_seeds, al_dir,
                                            base, args.dry_run)
            print(f"  [EVAL] id={sc.scenario_id}  "
                  f"y_target={y_target.mean():.3f}  y_proxy={y_proxy.mean():.3f}")
            state.add_observation(sc, y_target, y_proxy)

        state.save(out_dir / f"iter_{t:02d}")

    state.save(out_dir / "final")
    print(f"\n[GPC DONE]  target={len(state.train)}")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--logging-root",       type=str, default="./results/gpc")
    p.add_argument("--pool-size",          type=int, default=500)
    p.add_argument("--n-grid",             type=int, default=3)
    p.add_argument("--n-init",             type=int, default=20)
    p.add_argument("--n-rounds",           type=int, default=20)
    p.add_argument("--batch-size",         type=int, default=2)
    p.add_argument("--n-seeds",            type=int, default=2)
    p.add_argument("--seed",               type=int, default=42)
    p.add_argument("--device",             type=str, default="cpu")
    p.add_argument("--dry-run",            action="store_true")
    p.add_argument("--scout-init-dir",     type=str, default="")
    p.add_argument("--failure-threshold",  type=float, default=0.7,
                   help="y >= this on proxy = failure label for classifier")
    p.add_argument("--success-threshold",  type=float, default=0.5,
                   help="p_fail < this = sim-predicted success (kept for clustering)")
    p.add_argument("--cls-epochs",         type=int, default=300)
    p.add_argument("--bnn-epochs",         type=int, default=500)
    p.add_argument("--lr",                 type=float, default=1e-3)
    p.add_argument("--n-mc",               type=int, default=20)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
