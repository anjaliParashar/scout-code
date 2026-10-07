#!/usr/bin/env python3
"""
scripts/run_baselines_simpler.py
----------------------------------
Three baselines for SimplerEnv, all reusing the 10 initial target
evaluations already collected by SCOUT, then extending with 40 more
target-only rounds using their own strategy.

Baselines
---------
  random   — 40 additional scenarios drawn uniformly from the pool
  is       — 40 additional scenarios drawn via IS weighted by proxy sim score
  bnn_rs   — same 50 scenarios as `random`, plus trains a BNN surrogate on them

Initial data loading
--------------------
The 10 SCOUT init evaluations live at:
  <scout-results-root>/{idx}/<task>/rt1/<ckpt>/scenario_0000/paired_result.json
  for idx in 0..n_init-1

Each JSON has the structure shown in the task description:
  result["target"]["success"], result["target"]["timesteps"]  → y_target
  result["proxy"]["success"],  result["proxy"]["timesteps"]   → y_proxy

Only target evaluations are collected in the 40 additional rounds.
No proxy rollouts are run for these baselines.

Example
-------
# Random baseline
python scripts/simpler/run_baselines_simpler.py \
    --baseline         random \
    --scout-results    ./results/scout \
    --n-scout-init     10 \
    --logging-root     ./results/random \
    --pool-size        1000 --n-grid 3 --seed 42 \
    --n-additional     40 --n-seeds 2

# Importance sampling baseline
python scripts/run_baselines_simpler.py \
    --baseline         is \
    --scout-results    ./results/scout \
    --n-scout-init     10 \
    --logging-root     ./results/is \
    --pool-size        1000 --n-grid 3 --seed 42 \
    --n-additional     40 --n-seeds 2 --is-beta 8.0

# BNN-RS (random + BNN trained on all 50 target observations)
python scripts/run_baselines_simpler.py \
    --baseline         bnn_rs \\
    --scout-results    ./results/scout \\
    --n-scout-init     10 \\
    --logging-root     ./results/bnn_rs \\
    --pool-size        1000 --n-grid 3 --seed 42 \\
    --n-additional     40 --n-seeds 2 \\
    --bnn-epochs       800
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.simpler.domain import (
    DesignSpaceLimits, build_scenario_pool,
    Scenario, scenarios_to_array,
    y_from_rollout_v2, load_rollout_results,
)
from utils.simpler.rollout_runner import (
    run_target_rollouts, CKPT_PATH, MAX_STEPS,
)


# ---------------------------------------------------------------------------
# Load initial target data from SCOUT JSON files
# ---------------------------------------------------------------------------

def load_scout_init_targets(
    scout_results_root: str,
    n_init:             int,
    task:               str,
    ckpt_basename:      str,
    max_steps:          int = MAX_STEPS,
) -> List[Tuple[int, np.ndarray]]:
    """
    Load y_target arrays from the SCOUT initial evaluation JSONs.

    Expected path structure:
      <scout_results_root>/{idx}/<task>/rt1/<ckpt_basename>/scenario_0000/paired_result.json
    where idx runs from 0 to n_init-1.

    Returns
    -------
    List of (scenario_pool_idx, y_target_array) tuples.
    scenario_pool_idx = idx (the SCOUT run index, which maps to the pool
    scenario with scenario_id == idx, assuming pool was built with seed=42).
    y_target_array = (n_seeds,) float64 array of per-seed outcomes.
    """
    root = Path(scout_results_root)
    loaded = []

    for idx in range(n_init):
        scenario_dir = (
            root / str(idx) / task / "rt1" / ckpt_basename / "scenario_0000"
        )
        results = load_rollout_results(scenario_dir)

        if not results:
            print(f"  [WARN] No JSON found at {scenario_dir}  — padding with 1.0")
            y = np.ones(1, dtype=np.float64)
        else:
            y = np.array(
                [y_from_rollout_v2(r, "target", max_steps) for r in results],
                dtype=np.float64,
            )

        loaded.append((idx, y))
        print(f"  [INIT] scout_idx={idx:02d}  "
              f"n_results={len(results)}  "
              f"y_target={y}  mean={y.mean():.3f}")

    return loaded


# ---------------------------------------------------------------------------
# Rollout helpers (target-only, no proxy)
# ---------------------------------------------------------------------------

def _collect_target_seeds(
    scenario:     Scenario,
    n_seeds:      int,
    logging_root: str,
    base_seed:    int,
    dry_run:      bool,
) -> np.ndarray:
    """Run n_seeds target rollouts, return (n_seeds,) y_target array."""
    vals = []
    for s in range(n_seeds):
        mean_y, _, _ = run_target_rollouts(
            scenario, logging_root, n_target=1,
            seed=base_seed + s, dry_run=dry_run,
        )
        vals.append(mean_y)
    return np.array(vals, dtype=np.float64)


# ---------------------------------------------------------------------------
# IS proposal distribution
# ---------------------------------------------------------------------------

def _is_proposal(scenarios: List[Scenario], beta: float) -> np.ndarray:
    """
    IS weights based on sim_mean_fn proxy score.
    Lower proxy score = higher IS weight (more dangerous scenarios first).
    Uses the domain sim_mean_fn evaluated at each scenario's object position.
    """
    from simpler_utils.domain import sim_mean_fn_simpler  # see HOOK below
    # HOOK: sim_mean_fn_simpler should map scenario → scalar proxy danger score.
    # Since we have no sim rollouts here, use a simple heuristic:
    # scenarios with object position farther from centre are harder → higher IS weight.
    # Replace with actual proxy sim score if available.
    scores = np.array([
        # Heuristic: distance from canonical object position (-0.12, 0.20)
        np.sqrt((s.object_x - (-0.12))**2 + (s.object_y - 0.20)**2)
        for s in scenarios
    ], dtype=np.float64)

    scores  = (scores - scores.mean()) / (scores.std() + 1e-8)
    logits  = beta * scores - (beta * scores).max()
    q       = np.exp(logits)
    return q / q.sum()


# ---------------------------------------------------------------------------
# Simple BNN surrogate (target-only, for bnn_rs baseline)
# ---------------------------------------------------------------------------

def train_bnn_on_targets(
    scenarios:   List[Scenario],
    y_targets:   List[np.ndarray],
    epochs:      int = 800,
    lr:          float = 1e-3,
    weight_decay: float = 1e-6,
    p_drop:      float = 0.10,
    device:      str   = "cpu",
):
    """
    Train a HeteroBNNEmbedding on the collected target observations.
    Uses mean y_target per scenario as the training label.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from scout.bnn import HeteroBNNEmbedding, train_bnn

    X = scenarios_to_array(scenarios).astype(np.float32)       # (M, 42)
    y = np.array([np.mean(yt) for yt in y_targets], dtype=np.float32)  # (M,)

    model = HeteroBNNEmbedding(input_dim=42, p_drop=p_drop).to(device)
    train_bnn(model, X, y, epochs=epochs, lr=lr,
              weight_decay=weight_decay, device=device)
    return model


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    out_dir:    Path,
    scenarios:  List[Scenario],
    y_targets:  List[np.ndarray],
    bnn_model   = None,
    device:     str = "cpu",
):
    """Save scenarios, y_target arrays, and optional BNN weights."""
    out_dir.mkdir(parents=True, exist_ok=True)

    from simpler_utils.domain import save_pool
    save_pool(scenarios, out_dir / "train_scenarios.json")

    y_mean = np.array([np.mean(yt) for yt in y_targets], dtype=np.float32)
    np.save(out_dir / "y_targets_mean.npy", y_mean)

    for i, yt in enumerate(y_targets):
        np.save(out_dir / f"y_target_{i:04d}.npy", yt)

    print(f"[SAVE] {len(scenarios)} scenarios → {out_dir}")
    print(f"       y_target: min={y_mean.min():.3f}  "
          f"mean={y_mean.mean():.3f}  max={y_mean.max():.3f}")

    if bnn_model is not None:
        import torch
        torch.save(
            dict(
                model_state_dict = bnn_model.state_dict(),
                input_dim        = 42,
            ),
            out_dir / "bnn_rs.pt",
        )
        print(f"[SAVE] BNN weights → {out_dir / 'bnn_rs.pt'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    rng     = np.random.default_rng(args.seed)
    out_dir = Path(args.logging_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Rebuild pool (same params as SCOUT for scenario_id alignment) ──────
    limits = DesignSpaceLimits(n_grid=args.n_grid)
    pool   = build_scenario_pool(limits=limits, n_random=args.pool_size,
                                 seed=args.seed)
    pool_map = {s.scenario_id: s for s in pool}   # id → Scenario

    # ── Step 1: Load SCOUT init target evaluations (indices 0..n_init-1) ──
    from utils.simpler.rollout_runner import CKPT_PATH
    ckpt_basename = Path(CKPT_PATH).name

    print(f"\n[{args.baseline.upper()}] Loading {args.n_scout_init} SCOUT "
          f"init JSONs from {args.scout_results}…")
    init_data = load_scout_init_targets(
        scout_results_root = args.scout_results,
        n_init             = args.n_scout_init,
        task               = args.task,
        ckpt_basename      = ckpt_basename,
        max_steps          = args.max_steps,
    )

    # Build initial train set — use pool scenario if scenario_id matches,
    # otherwise create a placeholder Scenario with the correct id
    init_scenarios  = []
    init_y_targets  = []
    init_scene_ids  = set()

    for pool_idx, y_target in init_data:
        sc = pool_map.get(pool_idx)
        if sc is None:
            print(f"  [WARN] pool_idx={pool_idx} not found in pool "
                  f"(pool_size={args.pool_size}, seed={args.seed}). "
                  f"Check that --pool-size and --seed match the SCOUT run.")
            continue
        init_scenarios.append(sc)
        init_y_targets.append(y_target)
        init_scene_ids.add(sc.scenario_id)

    print(f"\n[{args.baseline.upper()}] Loaded {len(init_scenarios)} init scenarios")

    # Remaining pool
    remaining_pool = [s for s in pool if s.scenario_id not in init_scene_ids]

    # ── Step 2: Additional target evaluations ─────────────────────────────
    n_add = min(args.n_additional, len(remaining_pool))

    if args.baseline in ("random", "bnn_rs"):
        # Uniform random selection from remaining pool
        add_local = rng.choice(len(remaining_pool), n_add, replace=False)
        selected  = [remaining_pool[i] for i in add_local]
        print(f"[{args.baseline.upper()}] Random: {n_add} additional scenarios")

    elif args.baseline == "is":
        # IS-weighted selection from remaining pool
        q        = _is_proposal(remaining_pool, args.is_beta)
        add_local = rng.choice(len(remaining_pool), n_add, replace=False, p=q)
        selected  = [remaining_pool[i] for i in add_local]
        print(f"[IS] IS-weighted: {n_add} additional scenarios  "
              f"(beta={args.is_beta})")

    else:
        raise ValueError(f"Unknown baseline: {args.baseline!r}")

    # Evaluate selected scenarios at target (real system)
    add_scenarios = []
    add_y_targets = []

    for j, sc in enumerate(selected):
        print(f"  [EVAL {j+1}/{n_add}] scenario_id={sc.scenario_id}")
        base  = args.seed + 100_000 + j * 100
        y_tgt = _collect_target_seeds(
            sc, args.n_seeds, args.logging_root, base, args.dry_run
        )
        print(f"    y_target={y_tgt}  mean={y_tgt.mean():.3f}")
        add_scenarios.append(sc)
        add_y_targets.append(y_tgt)

    # Combine init + additional
    all_scenarios = init_scenarios + add_scenarios
    all_y_targets = init_y_targets + add_y_targets
    print(f"\n[{args.baseline.upper()}] Total: {len(all_scenarios)} scenarios "
          f"({len(init_scenarios)} init + {len(add_scenarios)} additional)")

    # ── Step 3: BNN training (bnn_rs only) ────────────────────────────────
    bnn_model = None
    if args.baseline == "bnn_rs":
        import torch
        device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
        print(f"\n[BNN-RS] Training BNN on {len(all_scenarios)} scenarios "
              f"(epochs={args.bnn_epochs})…")
        bnn_model = train_bnn_on_targets(
            scenarios    = all_scenarios,
            y_targets    = all_y_targets,
            epochs       = args.bnn_epochs,
            lr           = args.lr,
            weight_decay = args.weight_decay,
            p_drop       = args.dropout,
            device       = device,
        )

    # ── Save ──────────────────────────────────────────────────────────────
    save_results(out_dir, all_scenarios, all_y_targets,
                 bnn_model=bnn_model,
                 device="cpu")
    print(f"\n[DONE]  baseline={args.baseline}  outputs → {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="SimplerEnv baselines reusing SCOUT initial target data."
    )

    # Baseline selection
    p.add_argument("--baseline",       type=str, required=True,
                   choices=["random", "is", "bnn_rs"])

    # SCOUT init data
    p.add_argument("--scout-results",  type=str, required=True,
                   help="Root directory of SCOUT results, e.g. ./results/scout")
    p.add_argument("--n-scout-init",   type=int, default=10,
                   help="Number of SCOUT init JSONs to load (indices 0..n-1)")
    p.add_argument("--task",           type=str,
                   default="google_robot_pick_horizontal_coke_can",
                   help="Task name used in the SCOUT logging path")
    p.add_argument("--max-steps",      type=int, default=80)

    # Pool construction (must match SCOUT run exactly)
    p.add_argument("--pool-size",      type=int, default=1000)
    p.add_argument("--n-grid",         type=int, default=3)
    p.add_argument("--seed",           type=int, default=42)

    # Additional evaluations
    p.add_argument("--n-additional",   type=int, default=40,
                   help="Number of additional target evaluations to collect")
    p.add_argument("--n-seeds",        type=int, default=2,
                   help="Target rollout seeds per selected scenario")

    # IS options
    p.add_argument("--is-beta",        type=float, default=8.0)

    # BNN-RS options
    p.add_argument("--bnn-epochs",     type=int,   default=800)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--weight-decay",   type=float, default=1e-6)
    p.add_argument("--dropout",        type=float, default=0.10)
    p.add_argument("--cpu",            action="store_true")

    # Output
    p.add_argument("--logging-root",   type=str, default="./results/baseline")
    p.add_argument("--dry-run",        action="store_true")

    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
