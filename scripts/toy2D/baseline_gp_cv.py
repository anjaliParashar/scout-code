#!/usr/bin/env python3
"""
scripts/toy2D/baseline_gp_cv.py
---------------------------------
Baseline — μ_CV objective with a GP surrogate, no MI term.

Each round:
  1. Fit a GP on the current labelled set (X_pair, y_real).
  2. Compute the CV-estimated mean for every pool point using local-ball
     sim samples and GP posterior draws (gp_sample_functions).
  3. Select the `batch_size` points with the HIGHEST CV-estimated mean.

This is the direct GP analogue of baseline_bnn_cv.py.  Comparing the two
isolates the effect of the surrogate choice (GP vs BNN) inside the CV
estimator, holding the acquisition objective (μ_CV only, no MI) fixed.

Comparing this to run_mi_cv.py (which uses GP + MI) isolates the
contribution of the MI exploration term, holding the surrogate fixed.

Example
-------
python scripts/toy2D/baseline_gp_cv.py \\
    --n-rounds 6 --batch-size 5 --n-init 10 --pool-size 1500 \\
    --output-dir outputs/toy2D/gp_cv --seed 7
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from utils.toy2D.domain import sample_designs, sample_real
from utils.toy2D.gp_utils import train_gp
from utils.toy2D.mi_toy import compute_cv_batch      # GP posterior draws + CV estimator
from utils.toy2D.plot_toy import (
    plot_gp_mean_heatmap, plot_best_seen_curve, print_discovery_summary,
)


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------

def _gp_cv_acquisition(
    pool:       np.ndarray,
    gp,
    batch_size: int,
    R_local:    float = 0.35,
    n_pair:     int   = 20,
    k_unpaired: int   = 120,
    n_f_draws:  int   = 32,
    clip:       float = 1.0,
) -> np.ndarray:
    """
    Select `batch_size` pool points with the highest GP-CV estimated mean.

    Uses compute_cv_batch from mi_toy.py, which draws GP posterior samples
    as the real-side estimate f and local-ball sim values as the control g.

    Parameters
    ----------
    pool       : (N, 2) candidate pool
    gp         : fitted BoTorch GP
    batch_size : number of points to select
    R_local    : radius of the local ball for paired sim samples
    n_pair     : number of paired (f, g) samples per point
    k_unpaired : size of the unpaired sim pool per point
    n_f_draws  : number of GP posterior draws used as f samples
    clip       : upper clip on CV mean estimate

    Returns
    -------
    idx : (batch_size,) int array into pool
    """
    cv_vals = compute_cv_batch(
        gp, pool,
        R_local=R_local,
        n_pair=n_pair,
        k_unpaired=k_unpaired,
        n_f_draws=n_f_draws,
        clip=clip,
    )
    valid = np.where(np.isfinite(cv_vals))[0]
    if len(valid) == 0:
        # Fallback: random selection
        return np.random.choice(len(pool), batch_size, replace=False)
    order = valid[np.argsort(cv_vals[valid])[::-1]]
    return order[:batch_size]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X_init      = sample_designs(args.n_init, seed=args.seed)
    y_real_init = sample_real(X_init)
    pool_X      = sample_designs(args.pool_size, seed=args.seed + 1)

    X_pair    = X_init.copy()
    y_real    = y_real_init.copy()
    pool      = pool_X.copy()
    best_seen = []

    for t in range(args.n_rounds):
        print(f"[ROUND {t:02d}] train_size={len(X_pair)}  pool_size={len(pool)}")
        gp = train_gp(X_pair, y_real)

        idx = _gp_cv_acquisition(
            pool=pool,
            gp=gp,
            batch_size=args.batch_size,
            R_local=args.r_local,
            n_pair=args.n_pair,
            k_unpaired=args.k_unpaired,
            n_f_draws=args.n_f_draws,
        )

        X_next = pool[idx]
        y_next = sample_real(X_next)

        X_pair = np.concatenate([X_pair, X_next])
        y_real = np.concatenate([y_real, y_next])
        mask   = np.ones(len(pool), bool)
        mask[idx] = False
        pool   = pool[mask]

        best_seen.append(float(np.max(y_real)))
        print(f"  best_real_seen = {best_seen[-1]:.4f}")

    gp_final = train_gp(X_pair, y_real)

    plot_gp_mean_heatmap(
        gp_final, X_init, X_pair[args.n_init:],
        title=f"GP μ_CV (no MI) | GP fit after {args.n_rounds} rounds",
        output_path=out / "gp_heatmap_final.png",
        n_rounds=args.n_rounds,
    )
    results = {
        "gp_cv": {
            "gp_final":       gp_final,
            "X_init":         X_init,
            "X_final":        X_pair,
            "best_real_seen": np.array(best_seen),
            "n_init":         args.n_init,
        }
    }
    plot_best_seen_curve(results, ["gp_cv"], args.n_rounds,
                         output_path=out / "best_seen.png")
    print_discovery_summary(results, ["gp_cv"])

    np.save(out / "X_final.npy",      X_pair)
    np.save(out / "y_real_final.npy", y_real)
    print(f"\n[DONE]  Outputs → {out}")


def build_parser():
    p = argparse.ArgumentParser(
        description="GP μ_CV baseline (no MI) on toy2D domain."
    )
    p.add_argument("--n-rounds",    type=int,   default=6)
    p.add_argument("--batch-size",  type=int,   default=5)
    p.add_argument("--n-init",      type=int,   default=10)
    p.add_argument("--pool-size",   type=int,   default=1500)
    p.add_argument("--r-local",     type=float, default=0.35,
                   help="Radius of local ball for paired sim samples")
    p.add_argument("--n-pair",      type=int,   default=20,
                   help="Number of paired (f,g) samples per candidate")
    p.add_argument("--k-unpaired",  type=int,   default=120,
                   help="Size of the unpaired sim pool per candidate")
    p.add_argument("--n-f-draws",   type=int,   default=32,
                   help="Number of GP posterior draws used as f samples")
    p.add_argument("--seed",        type=int,   default=7)
    p.add_argument("--output-dir",  type=str,   default="outputs/toy2D/gp_cv")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
