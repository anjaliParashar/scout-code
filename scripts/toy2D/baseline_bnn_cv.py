#!/usr/bin/env python3
"""
scripts/toy2D/baseline_bnn_cv.py
----------------------------------
Baseline 4 — μ_CV objective with a BNN surrogate (no MI term).

Each round:
  1. Train a HeteroBNNEmbedding on the current labelled set.
  2. Compute CV-estimated mean for all pool points using local-ball sim samples
     and BNN posterior draws.
  3. Select the `batch_size` points with the HIGHEST CV-estimated mean.

This isolates the contribution of the MI term by removing it and keeping
only the CV-based exploitation.

Example
-------
python scripts/toy2D/baseline_bnn_cv.py \
    --n-rounds 6 --batch-size 5 --n-init 10 --pool-size 1500 \
    --epochs 800 --output-dir outputs/toy2D/bnn_cv --seed 7
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from utils.toy2D.domain import sample_designs, sample_real
from utils.toy2D.gp_utils import train_gp
from utils.toy2D.plot_toy import (
    plot_gp_mean_heatmap, plot_best_seen_curve, print_discovery_summary,
)
from utils.baseline.bnn_cv_baseline import (
    train_bnn_surrogate, bnn_cv_acquisition,
)


def main(args):
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X_init      = sample_designs(args.n_init, seed=args.seed)
    y_real_init = sample_real(X_init)
    pool_X      = sample_designs(args.pool_size, seed=args.seed + 1)

    X_pair  = X_init.copy()
    y_real  = y_real_init.copy()
    pool    = pool_X.copy()
    best_seen = []

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    for t in range(args.n_rounds):
        print(f"[ROUND {t:02d}] train_size={len(X_pair)}")
        model = train_bnn_surrogate(
            X_pair, y_real,
            input_dim=X_pair.shape[1],
            epochs=args.epochs,
            device=device,
        )
        idx = bnn_cv_acquisition(
            pool_X=pool,
            model=model,
            batch_size=args.batch_size,
            device=device,
            maximise=True,
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
        title=f"BNN μ_CV | GP fit after {args.n_rounds} rounds",
        output_path=out / "gp_heatmap_final.png",
        n_rounds=args.n_rounds,
    )
    results = {"bnn_cv": {"gp_final": gp_final, "X_init": X_init,
                           "X_final": X_pair, "best_real_seen": np.array(best_seen),
                           "n_init": args.n_init}}
    plot_best_seen_curve(results, ["bnn_cv"], args.n_rounds,
                         output_path=out / "best_seen.png")
    print_discovery_summary(results, ["bnn_cv"])
    print(f"[DONE]  Outputs → {out}")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--n-rounds",   type=int, default=6)
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--n-init",     type=int, default=10)
    p.add_argument("--pool-size",  type=int, default=1500)
    p.add_argument("--epochs",     type=int, default=800,
                   help="BNN training epochs per round")
    p.add_argument("--seed",       type=int, default=7)
    p.add_argument("--output-dir", type=str, default="outputs/toy2D/bnn_cv")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
