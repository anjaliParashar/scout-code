#!/usr/bin/env python3

"""
End-to-end SCOUT step for Go2 fixed velocity commands.

1. Train target/hardware MLP on collected hardware .pkl files.
2. Train or load proxy/sim MLP from fixed_twist_combined.csv.
3. Use MI + proxy prediction to print next command candidate.

Example:
python scripts/quadruped/run_scout.py \
  --hardware_glob "results/go2_hardware/scout_2/*.pkl" \
  --n_hardware 25 \
  --sim_csv data/quadruped/fixed_twist_combined.csv \
  --proxy_ckpt results/quadruped/sim.pt \
  --target_ckpt results/quadruped/hardware.pt \
  --pool_size 2000 \
  --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils.quadruped.train_utils import (
    MLPTrainConfig,
    load_hardware_pkls,
    load_sim_csv,
    train_mlp,
    save_checkpoint,
    load_mlp_checkpoint,
)
from utils.quadruped.scout_utils import ScoutConfig, select_next_candidate


def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--hardware_glob", type=str, default="results/go2_hardware/*.pkl")
    p.add_argument("--n_hardware", type=int, default=None)

    p.add_argument(
        "--sim_csv",
        type=str,
        default="data/quadruped/fixed_twist_combined.csv",
    )
    p.add_argument("--sim_metric", type=str, default="final_abs_err_sum")

    p.add_argument("--proxy_ckpt", type=str, default="results/scout_models/proxy_sim_mlp.pt")
    p.add_argument("--target_ckpt", type=str, default="results/scout_models/target_hw_mlp.pt")
    p.add_argument("--force_train_proxy", action="store_true")

    p.add_argument("--epochs_target", type=int, default=5000)
    p.add_argument("--epochs_proxy", type=int, default=3000)
    p.add_argument("--hidden_dim", type=int, default=8)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--pool_size", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)


    p.add_argument("--vx_min", type=float, default=-0.4)
    p.add_argument("--vx_max", type=float, default=1.0)
    p.add_argument("--vy_min", type=float, default=-0.8)
    p.add_argument("--vy_max", type=float, default=0.8)
    p.add_argument("--wz_min", type=float, default=-0.8)
    p.add_argument("--wz_max", type=float, default=0.8)

    p.add_argument("--mi_weight", type=float, default=0.5)
    p.add_argument("--proxy_weight", type=float, default=1.0)
    p.add_argument("--target_mean_weight", type=float, default=0.0)
    p.add_argument("--target_uncertainty_weight", type=float, default=0.25)

    return p


def main():
    args = build_parser().parse_args()

    # ------------------------------------------------------------
    # 1. Load hardware data and train target model.
    # ------------------------------------------------------------
    X_hw, y_hw, hw_paths = load_hardware_pkls(
        args.hardware_glob,
        n_files=args.n_hardware,
    )
    print("Y_CUMULATIVE:",len(y_hw[y_hw>=0.6]))
    print("\n[INFO] Hardware training data")
    print(f"  N = {len(X_hw)}")
    for i, path in enumerate(hw_paths):
        print(f"  {i:03d}: X={X_hw[i]} y={y_hw[i]:.4f} path={path}")

    target_train_cfg = MLPTrainConfig(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs_target,
        seed=args.seed,
        verbose=True,
    )

    target_ckpt = train_mlp(X_hw, y_hw, target_train_cfg)
    save_checkpoint(target_ckpt, args.target_ckpt)

    target_model, target_ckpt, device = load_mlp_checkpoint(args.target_ckpt)

    # ------------------------------------------------------------
    # 2. Train or load proxy model.
    # ------------------------------------------------------------
    proxy_path = Path(args.proxy_ckpt)
    X_sim, y_sim = load_sim_csv(
            args.sim_csv,
            metric=args.sim_metric,
        )
    if args.force_train_proxy or not proxy_path.exists():
        print("\n[INFO] Training proxy/sim model")
        print(f"  sim N = {len(X_sim)}")
        print(f"  y range = [{y_sim.min():.4f}, {y_sim.max():.4f}]")

        proxy_train_cfg = MLPTrainConfig(
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            dropout=args.dropout,
            lr=args.lr,
            epochs=args.epochs_proxy,
            seed=args.seed,
            verbose=True,
        )

        proxy_ckpt = train_mlp(X_sim, y_sim, proxy_train_cfg)
        save_checkpoint(proxy_ckpt, proxy_path)

    proxy_model, proxy_ckpt, _ = load_mlp_checkpoint(proxy_path, device=str(device))

    # ------------------------------------------------------------
    # 3. Select next candidate.
    # ------------------------------------------------------------
    scout_cfg = ScoutConfig(
        pool_size=args.pool_size,
        seed=args.seed,

        vx_min=args.vx_min,
        vx_max=args.vx_max,
        vy_min=args.vy_min,
        vy_max=args.vy_max,
        wz_min=args.wz_min,
        wz_max=args.wz_max,

        mi_weight=args.mi_weight,
        proxy_weight=args.proxy_weight,
        target_mean_weight=args.target_mean_weight,
        target_uncertainty_weight=args.target_uncertainty_weight,
    )

    result = select_next_candidate(
        target_model=target_model,
        target_ckpt=target_ckpt,
        proxy_model=proxy_model,
        proxy_ckpt=proxy_ckpt,
        X_target=X_hw,
        X_sim=X_sim,
        cfg=scout_cfg,
        device=device,
        proxy_path='results/quadruped/sim.pt'
    )

    print("\n" + "=" * 80)
    print("NEXT HARDWARE CANDIDATE")
    print("=" * 80)
    print(f"vx = {result['candidate_vx']:.6f}")
    print(f"vy = {result['candidate_vy']:.6f}")
    print(f"wz = {result['candidate_wz']:.6f}")
    print("-" * 80)
    print(f"score              = {result['score']:.6f}")
    print(f"MI                 = {result['mi']:.6f}")
    print(f"dmin               = {result['dmin']:.6f}")
    print(f"proxy_pred_error   = {result['proxy_pred']:.6f}")
    print(f"target_pred_error  = {result['target_pred']:.6f}")
    print(f"target_uncertainty = {result['target_uncertainty']:.6f}")
    print("=" * 80)

    print("\nRun this candidate on hardware with:")
    print(
        "python scripts/quadruped/run_fixed_cmd_collect_odom.py "
        f"--vx {result['candidate_vx']:.6f} "
        f"--vy {result['candidate_vy']:.6f} "
        f"--wz {result['candidate_wz']:.6f} "
        "--duration 5.0 "
        "--publish_hz 10.0 "
        f"--out results/go2_hardware/scout_vx{result['candidate_vx']:.3f}_"
        f"vy{result['candidate_vy']:.3f}_wz{result['candidate_wz']:.3f}.pkl"
    )


if __name__ == "__main__":
    main()