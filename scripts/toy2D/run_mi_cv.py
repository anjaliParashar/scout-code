#!/usr/bin/env python3
"""
scripts/toy2D/run_mi_cv.py
--------------------------
MI + μ_CV active-learning pipeline on the 2-D toy domain.
Closed-loop MI + control variates on the 2-D toy domain, with a GP surrogate.

Algorithm
---------
Each round t:
  1. Fit a GP on the current labelled set (X_pair, y_real).
  2. Compute MI(x) for every pool point using the support-state estimator.
  3. Compute CV-estimated mean for the high-MI shortlist using local-ball sim samples.
  4. Select the batch using `select_micv_from_high_mi_clusters`.
  5. Query real + sim observations for the selected points.
  6. Update the labelled set and remove from pool.

At the end:
  - Save the final GP heatmap and best-seen curve.
  - Print a diamond-discovery summary.

Example
-------
python scripts/toy2D/run_mi_cv.py \
    --n-rounds 6 --batch-size 5 --n-init 10 --pool-size 1500 \
    --output-dir outputs/toy2D/mi_cv --seed 7
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
from utils.toy2D.gp_utils import train_gp, gp_predict
from utils.toy2D.mi_toy import (
    fit_support_and_compute_mi, compute_cv_batch, cluster_by_radius,
)
from utils.toy2D.plot_toy import (
    plot_gp_mean_heatmap, plot_best_seen_curve, print_discovery_summary,
    plot_discovery_bar,
)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _select_micv(
    pool_X: np.ndarray,
    X_support: np.ndarray,
    gp,
    batch_size: int,
    iteration: int,
    top_mi_frac: float = 0.12,
    min_top: int = 20,
    cluster_eps: float = 0.35,
    # MI hyperparameters — passed explicitly, matching scout.mi conventions
    n_clusters: int = 20,
    radius_quantile: float = 0.90,
    pi_new: float = 0.18,
    eps_exist: float = 0.04,
    tau_scale: float = 1.3,
    novelty_radius: float = 0.75,
    gamma: float = 0.25,
    random_state: int = 0,
) -> np.ndarray:
    """MI + CV batch selection (matches notebook's select_micv_from_high_mi_clusters)."""
    mi_vals, _, _ = fit_support_and_compute_mi(
        X_support=X_support,
        X_query=pool_X,
        n_clusters=n_clusters,
        radius_quantile=radius_quantile,
        pi_new=pi_new,
        eps_exist=eps_exist,
        tau_scale=tau_scale,
        novelty_radius=novelty_radius,
        gamma=gamma,
        random_state=random_state,
    )
    mi_vals = np.nan_to_num(mi_vals)

    # Narrow the MI fraction after round 2
    frac = 0.05 if iteration > 2 else top_mi_frac
    valid = np.where(np.isfinite(mi_vals))[0]
    n_top = max(min_top, int(frac * len(valid)))
    top_idx = valid[np.argsort(mi_vals[valid])[::-1][:n_top]]

    cv_top = compute_cv_batch(gp, pool_X[top_idx])
    cv_full = np.full(len(pool_X), np.nan)
    cv_full[top_idx] = cv_top

    # Cluster high-MI candidates and pick best-CV per cluster
    high_mi_clusters = cluster_by_radius(pool_X[top_idx], eps=cluster_eps)
    X_top = pool_X[top_idx]
    reps  = []
    for pts in high_mi_clusters:
        local = []
        for pt in pts:
            hits = np.where(np.all(np.isclose(X_top, pt[None, :]), axis=1))[0]
            local.extend(hits.tolist())
        local = np.array(sorted(set(local)), dtype=int)
        valid_loc = local[np.isfinite(cv_top[local])]
        if len(valid_loc) == 0:
            continue
        best = valid_loc[np.argmax(cv_top[valid_loc])]
        reps.append(top_idx[best])

    if not reps:
        # Fallback: just return top-CV from high-MI pool
        vt = top_idx[np.isfinite(cv_top)]
        return vt[np.argsort(cv_full[vt])[::-1]][:batch_size]

    rep_arr = np.array(reps, dtype=int)
    rep_ord = rep_arr[np.argsort(cv_full[rep_arr])[::-1]]
    selected = rep_ord[:batch_size].tolist()

    if len(selected) < batch_size:
        remaining = np.setdiff1d(top_idx, np.array(selected, dtype=int))
        vr = remaining[np.isfinite(cv_full[remaining])]
        fill = vr[np.argsort(cv_full[vr])[::-1]]
        selected.extend(fill[: batch_size - len(selected)].tolist())

    return np.array(selected, dtype=int)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Initial dataset
    X_init     = sample_designs(args.n_init, seed=args.seed)
    y_real_init = sample_real(X_init)
    pool_X     = sample_designs(args.pool_size, seed=args.seed + 1)

    X_pair     = X_init.copy()
    y_real     = y_real_init.copy()
    pool       = pool_X.copy()
    best_seen  = []

    for t in range(args.n_rounds):
        print(f"\n[ROUND {t:02d}] train_size={len(X_pair)}  pool_size={len(pool)}")
        gp = train_gp(X_pair, y_real)

        idx = _select_micv(
            pool_X=pool,
            X_support=X_pair,
            gp=gp,
            batch_size=args.batch_size,
            iteration=t,
        )

        X_next     = pool[idx]
        y_real_next = sample_real(X_next)

        X_pair  = np.concatenate([X_pair,  X_next],      axis=0)
        y_real  = np.concatenate([y_real,  y_real_next], axis=0)

        mask  = np.ones(len(pool), dtype=bool)
        mask[idx] = False
        pool  = pool[mask]

        best_seen.append(float(np.max(y_real)))
        print(f"  best_real_seen = {best_seen[-1]:.4f}")

    gp_final = train_gp(X_pair, y_real)

    # Plots
    X_new = X_pair[args.n_init:]
    plot_gp_mean_heatmap(
        gp=gp_final,
        X_init=X_init,
        X_new=X_new,
        title=f"MI+CV | GP fit after {args.n_rounds} rounds",
        output_path=out / "gp_heatmap_final.png",
        n_rounds=args.n_rounds,
    )
    results = {
        "mi_cv": {
            "gp_final":      gp_final,
            "X_init":        X_init,
            "X_final":       X_pair,
            "best_real_seen": np.array(best_seen),
            "n_init":        args.n_init,
        }
    }
    plot_best_seen_curve(results, ["mi_cv"], args.n_rounds,
                         output_path=out / "best_seen.png")
    print_discovery_summary(results, ["mi_cv"])

    np.save(out / "X_final.npy",    X_pair)
    np.save(out / "y_real_final.npy", y_real)
    print(f"\n[DONE]  Outputs → {out}")


def build_parser():
    p = argparse.ArgumentParser(description="MI+CV AL on toy2D domain.")
    p.add_argument("--n-rounds",    type=int,   default=6)
    p.add_argument("--batch-size",  type=int,   default=5)
    p.add_argument("--n-init",      type=int,   default=10)
    p.add_argument("--pool-size",   type=int,   default=1500)
    p.add_argument("--seed",        type=int,   default=7)
    p.add_argument("--output-dir",  type=str,   default="outputs/toy2D/mi_cv")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
