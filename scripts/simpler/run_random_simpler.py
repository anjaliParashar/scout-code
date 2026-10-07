#!/usr/bin/env python3
"""
scripts/run_random_simpler.py
------------------------------
Random sampling baseline — same evaluation protocol as run_scout_simpler.py.

Initial seed : n_init target + 4*n_init proxy-only scenarios, 2 seeds each.
Sequential   : batch_size=2 scenarios per round, 2 seeds each (target + proxy).
Rounds       : n_rounds=20.

Reusing SCOUT initial data
---------------------------
Pass --scout-init-dir to point at an existing SCOUT init directory
(e.g. results/scout/39).  The script will load the already-collected
ALState directly — skipping all initial rollouts — and continue with
random acquisition from the remaining pool.

This ensures both methods start from exactly the same initial observations
without running any repeat evaluations.

Example — reuse existing SCOUT seed
-------------------------------------
python scripts/simpler/run_random_simpler.py \
    --logging-root   ./results/random \
    --scout-init-dir ./results/scout/39 \
    --pool-size 500 --n-grid 3 --seed 42 \
    --n-rounds 40 --batch-size 2 --n-seeds 2

Example — collect own seed (original behaviour)
-------------------------------------------------
python scripts/run_random_simpler.py \\
    --logging-root ./results/random \\
    --pool-size 500 --n-grid 3 --seed 42 \\
    --n-rounds 20 --batch-size 2 --n-seeds 2

python scripts/simpler/run_random_simpler.py     --logging-root   ./results/random     --scout-init-dir ./results/scout/init     --pool-size 500 --n-grid 3 --seed 42     --n-rounds 20 --batch-size 2 --n-seeds 2
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.simpler.domain import DesignSpaceLimits, build_scenario_pool
from utils.simpler.al_state import ALState
from utils.simpler.rollout_runner import run_proxy_rollouts, run_target_rollouts
from utils.simpler.random_baseline import random_acquisition


# ---------------------------------------------------------------------------
# Rollout helpers (same as run_scout_simpler.py)
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

    # Always rebuild the pool from the same parameters so scenario_ids align
    limits = DesignSpaceLimits(n_grid=args.n_grid)
    pool   = build_scenario_pool(limits=limits, n_random=args.pool_size,
                                 seed=args.seed)

    # ── Initial seed ──────────────────────────────────────────────────────
    if args.scout_init_dir:
        # ── Load from existing SCOUT init directory ───────────────────────
        scout_init = Path(args.scout_init_dir)
        print(f"[RANDOM] Loading initial seed from SCOUT dir: {scout_init}")
        state = ALState.load(scout_init)
     
        # Verify the loaded state is compatible with the pool we just built
        pool_ids    = {s.scenario_id for s in pool}
        train_ids   = {s.scenario_id for s in state.train}
        proxy_ids   = {s.scenario_id for s in state.proxy_only}
        evaluated   = train_ids | proxy_ids
        unknown_ids = evaluated - pool_ids

        if unknown_ids:
            print(f"  [WARN] {len(unknown_ids)} loaded scenario_ids not found in "
                  f"the rebuilt pool.  Check that --pool-size, --n-grid, and "
                  f"--seed match the original SCOUT run.")

        # Reconstruct the remaining pool: all pool scenarios not yet evaluated
        remaining_pool = [s for s in pool if s.scenario_id not in evaluated]
        state.pool = remaining_pool

        print(f"[RANDOM] Loaded: target={len(state.train)}  "
              f"proxy_only={len(state.proxy_only)}  "
              f"remaining_pool={len(state.pool)}")

        # Save a copy of the loaded init state into this run's output dir
        state.save(out_dir / "init")

    else:
        # ── Collect own initial seed (original behaviour) ─────────────────
        state = ALState(pool=pool)

        n_init_proxy   = 5 * args.n_init
        all_init_local = rng.choice(len(state.pool), n_init_proxy, replace=False)
        target_local   = all_init_local[:args.n_init]
        proxy_only_local = all_init_local[args.n_init:]
        orig_pool_map  = {s.scenario_id: s for s in pool}

        print(f"[RANDOM] Collecting initial seed: {args.n_init} target + "
              f"{len(proxy_only_local)} proxy-only")

        for i, local_idx in enumerate(target_local):
            sc = state.pool[local_idx]
            print(f"  [INIT target {i+1}/{args.n_init}] scenario_id={sc.scenario_id}")
            base = args.seed + i * 100
            scenario_dir = f"{args.logging_root}/{i}"
            scenario_dir_path = Path(scenario_dir)
            scenario_dir_path.mkdir(parents=True, exist_ok=True)
            y_target, y_proxy = _eval_both(sc, args.n_seeds,
                                            scenario_dir, base, args.dry_run)
            state.add_observation(sc, y_target, y_proxy)

        for i, local_idx in enumerate(proxy_only_local):
            sc_id = pool[local_idx].scenario_id
            if sc_id not in {s.scenario_id for s in state.pool}:
                continue
            sc = orig_pool_map[sc_id]
            base    = args.seed + args.n_init * 100 + i * 100
            
            y_proxy = _collect_seeds(sc, args.n_seeds, args.logging_root,
                                     "proxy", base, args.dry_run)
            state.add_proxy_only_observation(sc, y_proxy)

        state.save(out_dir / "init")
        print(f"[RANDOM] Seed done. target={len(state.train)}  "
              f"proxy_only={len(state.proxy_only)}")

    # ── Random AL rounds ──────────────────────────────────────────────────
    for t in range(args.n_rounds):
        print(f"\n[RANDOM ROUND {t:02d}/{args.n_rounds-1}]  "
              f"target={len(state.train)}  pool={len(state.pool)}")

        if len(state.pool) < args.batch_size:
            print("[STOP] Pool exhausted.")
            break

        batch = random_acquisition(state, args.batch_size, rng)
        print(f"  Selected: {[s.scenario_id for s in batch]}")

        for j, sc in enumerate(batch):
        
            base = args.seed + 100_000 + t * 10_000 + j * 100
            scenario_dir = f"{args.logging_root}/random_{t}_{j}"
            scenario_dir_path = Path(scenario_dir)
            scenario_dir_path.mkdir(parents=True, exist_ok=True)
            y_target, y_proxy = _eval_both(sc, args.n_seeds,
                                            scenario_dir, base, args.dry_run)
            print(f"  [EVAL] scenario_id={sc.scenario_id}  "
                  f"y_target mean={np.mean(y_target):.3f}  "
                  f"y_proxy  mean={np.mean(y_proxy):.3f}")
            state.add_observation(sc, y_target, y_proxy)

        state.save(out_dir / f"iter_{t:02d}")

    state.save(out_dir / "final")
    print(f"\n[RANDOM DONE]  target={len(state.train)}  "
          f"proxy_only={len(state.proxy_only)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Random sampling baseline for SimplerEnv AL experiment."
    )
    p.add_argument("--logging-root",    type=str, default="./results/random")
    p.add_argument("--pool-size",       type=int, default=500)
    p.add_argument("--n-grid",          type=int, default=3)
    p.add_argument("--n-init",          type=int, default=20,
                   help="Used only when --scout-init-dir is NOT provided.")
    p.add_argument("--n-rounds",        type=int, default=20)
    p.add_argument("--batch-size",      type=int, default=2)
    p.add_argument("--n-seeds",         type=int, default=2)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--device",          type=str, default="cpu")
    p.add_argument("--dry-run",         action="store_true")

    # Key new argument
    p.add_argument(
        "--scout-init-dir", type=str, default="",
        help=(
            "Path to an existing SCOUT init directory whose ALState will be "
            "loaded and reused as the initial seed for this random run.  "
            "Skips all initial rollouts.  "
            "Must use the same --pool-size, --n-grid, and --seed as the "
            "original SCOUT run so that scenario_ids align correctly.  "
            "Example: --scout-init-dir ./results/scout/39"
        ),
    )
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
