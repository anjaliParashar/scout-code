#!/usr/bin/env python3
"""
scripts/toy2D/baseline_importance_sampling.py
---------------------------------------------
Baseline 2 — Importance sampling using sim-only data.

Acquisition: q(x) ∝ softmax(β · sim_score(x)).
No real-data feedback beyond the initial seed.  Relies entirely on the
correlation between the cheap simulator and the real system.

Example
-------
python scripts/toy2D/baseline_importance_sampling.py \
    --n-rounds 6 --batch-size 5 --n-init 10 --pool-size 1500 \
    --beta 8.0 --output-dir outputs/toy2D/is --seed 7
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from utils.toy2D.domain import sample_designs, sample_real, sim_mean_fn
from utils.toy2D.gp_utils import train_gp
from utils.toy2D.plot_toy import (
    plot_gp_mean_heatmap, plot_best_seen_curve, print_discovery_summary,
)
from utils.baseline.importance_sampling_baseline import importance_sampling_acquisition


def main(args):
    rng = np.random.default_rng(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X_init      = sample_designs(args.n_init, seed=args.seed)
    y_real_init = sample_real(X_init)
    pool_X      = sample_designs(args.pool_size, seed=args.seed + 1)

    X_pair  = X_init.copy()
    y_real  = y_real_init.copy()
    pool    = pool_X.copy()
    best_seen = []

    for t in range(args.n_rounds):
        print(f"[ROUND {t:02d}] train_size={len(X_pair)}")
        idx    = importance_sampling_acquisition(
            pool, sim_mean_fn, args.batch_size,
            beta=args.beta, rng=rng,
        )
        X_next = pool[idx]
        y_next = sample_real(X_next)

        X_pair = np.concatenate([X_pair, X_next])
        y_real = np.concatenate([y_real, y_next])
        mask   = np.ones(len(pool), bool); mask[idx] = False
        pool   = pool[mask]
        best_seen.append(float(np.max(y_real)))

    gp_final = train_gp(X_pair, y_real)
    plot_gp_mean_heatmap(
        gp_final, X_init, X_pair[args.n_init:],
        title=f"IS (sim only) | GP fit after {args.n_rounds} rounds",
        output_path=out / "gp_heatmap_final.png",
        n_rounds=args.n_rounds,
    )
    results = {"is": {"gp_final": gp_final, "X_init": X_init,
                       "X_final": X_pair, "best_real_seen": np.array(best_seen),
                       "n_init": args.n_init}}
    plot_best_seen_curve(results, ["is"], args.n_rounds,
                         output_path=out / "best_seen.png")
    print_discovery_summary(results, ["is"])
    print(f"[DONE]  Outputs → {out}")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--n-rounds",   type=int,   default=6)
    p.add_argument("--batch-size", type=int,   default=5)
    p.add_argument("--n-init",     type=int,   default=10)
    p.add_argument("--pool-size",  type=int,   default=1500)
    p.add_argument("--beta",       type=float, default=8.0)
    p.add_argument("--seed",       type=int,   default=7)
    p.add_argument("--output-dir", type=str,   default="outputs/toy2D/is")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
