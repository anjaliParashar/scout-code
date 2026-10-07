#!/usr/bin/env python3
"""
scripts/toy2D/baseline_bams.py
--------------------------------
Baseline 5 — BAMS (Bayesian Adaptive Multifidelity Sampling).
Implements Algorithm 1 from Sinha et al. arXiv:2411.17826.

The GP models the real-valued performance metric f(x) rather than the
binary failure indicator.  The acquisition function minimises the expected
forward-looking point variance of the GP-based failure-rate estimator:

    J({xj}) = E_P[βn(X; {xj})]

where βn(x; Xm) = Φ₂(ŝ(x), −ŝ(x); [[1, t̂−1], [t̂−1, 1]])
is the expected Bernoulli variance at x after observing Xm
(closed form from Chevalier et al. 2014 via bivariate normal CDF).

Single-fidelity variant (BAS) is used here by default.  Pass
--low-fidelity-cost to enable the full multifidelity BAMS with sim as the
cheap fidelity.

Example — BAS (single fidelity)
--------------------------------
python scripts/toy2D/baseline_bams.py \
    --n-rounds 6 --batch-size 5 --n-init 10 --pool-size 500 \
    --n-clusters 6 --output-dir outputs/toy2D/bams --seed 7

Example — full multifidelity BAMS
-----------------------------------
python scripts/toy2D/baseline_bams.py \
    --n-rounds 6 --batch-size 5 --n-init 10 --pool-size 500 \
    --n-clusters 6 --low-fidelity-cost 0.15 \
    --output-dir outputs/toy2D/bams_mf --seed 7
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from utils.toy2D.domain import (
    sample_designs, sample_real, sample_sim, GAMMA,
)
from utils.toy2D.gp_utils import train_gp
from utils.toy2D.plot_toy import (
    plot_gp_mean_heatmap, plot_best_seen_curve, print_discovery_summary,
)
from utils.baseline.bams_baseline import bams_acquisition


def main(args):
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Initialization ────────────────────────────────────────────────────
    X_init      = sample_designs(args.n_init, seed=args.seed)
    y_real_init = sample_real(X_init)
    pool_X      = sample_designs(args.pool_size, seed=args.seed + 1)

    X_pair  = X_init.copy()
    y_real  = y_real_init.copy()
    pool    = pool_X.copy()
    best_seen     = []
    selected_fids = []   # track how many low-fidelity evaluations were used

    # Multifidelity: treat all pool points as available at low fidelity
    use_mf     = args.low_fidelity_cost is not None
    lf_cost    = args.low_fidelity_cost
    lf_idx_all = np.arange(len(pool_X)) if use_mf else None

    # ── Active learning loop ──────────────────────────────────────────────
    for t in range(args.n_rounds):
        print(f"\n[ROUND {t:02d}] train_size={len(X_pair)}  pool_size={len(pool)}")

        # Fit GP on current real observations
        gp = train_gp(X_pair, y_real)

        # Current low-fidelity availability (indices into current pool)
        lf_idx = np.arange(len(pool)) if use_mf else None

        # BAMS acquisition
        idx, fids = bams_acquisition(
            pool_X=pool,
            gp=gp,
            batch_size=args.batch_size,
            threshold=args.threshold,
            n_clusters=min(args.n_clusters, len(pool) // 2 + 1),
            low_fidelity_idx=lf_idx,
            low_fidelity_cost=lf_cost,
            random_state=args.seed + t,
        )

        if len(idx) == 0:
            print("[WARN] No points selected; stopping early.")
            break

        X_next = pool[idx]

        # Evaluate: high-fidelity → real sample; low-fidelity → sim sample
        y_next = np.zeros(len(idx), dtype=np.float32)
        for j, (pool_idx, fid) in enumerate(zip(idx, fids)):
            if fid == 0:
                y_next[j] = float(sample_real(X_next[j : j + 1]))
            else:
                y_next[j] = float(sample_sim(X_next[j : j + 1]))

        n_hf = int(np.sum(fids == 0))
        n_lf = int(np.sum(fids == 1))
        print(f"  Selected {len(idx)} pts: {n_hf} high-fidelity, {n_lf} low-fidelity")

        # Only add high-fidelity observations to the real training set
        hf_mask = fids == 0
        if hf_mask.any():
            X_pair = np.concatenate([X_pair, X_next[hf_mask]])
            y_real = np.concatenate([y_real, y_next[hf_mask]])

        # Remove evaluated points from pool
        mask = np.ones(len(pool), bool)
        mask[idx] = False
        pool = pool[mask]

        best_seen.append(float(np.max(y_real)))
        selected_fids.extend(fids.tolist())
        print(f"  best_real_seen = {best_seen[-1]:.4f}")

    # ── Final GP and plots ────────────────────────────────────────────────
    gp_final = train_gp(X_pair, y_real)
    X_new    = X_pair[args.n_init:]

    strategy = "bams" if use_mf else "bas"
    plot_gp_mean_heatmap(
        gp=gp_final,
        X_init=X_init,
        X_new=X_new,
        title=f"{strategy.upper()} | GP fit after {args.n_rounds} rounds",
        output_path=out / "gp_heatmap_final.png",
        n_rounds=args.n_rounds,
    )
    results = {
        strategy: {
            "gp_final":      gp_final,
            "X_init":        X_init,
            "X_final":       X_pair,
            "best_real_seen": np.array(best_seen),
            "n_init":        args.n_init,
        }
    }
    plot_best_seen_curve(results, [strategy], args.n_rounds,
                         output_path=out / "best_seen.png")
    print_discovery_summary(results, [strategy])

    np.save(out / "X_final.npy",    X_pair)
    np.save(out / "y_real_final.npy", y_real)
    np.save(out / "selected_fids.npy", np.array(selected_fids))
    print(f"\n[DONE]  Outputs → {out}")
    fid_arr = np.array(selected_fids)
    if len(fid_arr):
        print(f"  Fidelity breakdown: {int((fid_arr==0).sum())} HF, "
              f"{int((fid_arr==1).sum())} LF")


def build_parser():
    p = argparse.ArgumentParser(
        description="BAMS / BAS baseline on toy2D domain (arXiv:2411.17826)."
    )
    p.add_argument("--n-rounds",          type=int,   default=6)
    p.add_argument("--batch-size",        type=int,   default=5)
    p.add_argument("--n-init",            type=int,   default=10)
    p.add_argument("--pool-size",         type=int,   default=500,
                   help="Smaller pool than other baselines — BAMS is O(mN²/S²)")
    p.add_argument("--threshold",         type=float, default=0.0,
                   help="γ: failure threshold on f_real(x).  "
                        "Points with f_real ≤ threshold are failures.")
    p.add_argument("--n-clusters",        type=int,   default=6,
                   help="S clusters for BAMS speedup (Algorithm 1)")
    p.add_argument("--low-fidelity-cost", type=float, default=None,
                   help="Cost of sim (low-fidelity) evaluation relative to "
                        "real (high-fidelity) = 1.0.  "
                        "If not set, runs single-fidelity BAS.")
    p.add_argument("--seed",              type=int,   default=7)
    p.add_argument("--output-dir",        type=str,   default="outputs/toy2D/bams")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
