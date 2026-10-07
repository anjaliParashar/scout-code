#!/usr/bin/env python3
"""
scripts/toy2D/compare_all.py
------------------------------
Run all 6 strategies from a shared initial dataset and produce comparison plots.

Strategies
----------
  mi_cv  — MI + μ_CV with GP (our method)
  random — Uniform random sampling
  is     — Importance sampling via sim
  gp_cv  — μ_CV with GP surrogate, no MI term
  gp_micv— MI + μ_CV with GP surrogate
  bnn_cv — μ_CV with BNN surrogate, no MI term
  bams   — BAMS / BAS (Sinha et al. 2024)

All strategies start from the *same* initial dataset and pool, mirroring
the notebook's experimental protocol.

Example
-------
python scripts/toy2D/compare_all.py \
    --n-rounds 6 --batch-size 5 --n-init 10 --pool-size 500 \
    --output-dir outputs/toy2D/comparison --seed 7
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from utils.toy2D.domain import (
    sample_designs, sample_real, sample_sim, sim_mean_fn, GAMMA,
)
from utils.toy2D.gp_utils import train_gp
from utils.toy2D.mi_toy import (
    fit_support_and_compute_mi, compute_cv_batch, cluster_by_radius,
)
from utils.toy2D.plot_toy import (
    plot_comparison_grid, plot_best_seen_curve,
    print_discovery_summary, plot_discovery_bar,
)
from utils.baseline.random_baseline import random_acquisition
from utils.baseline.importance_sampling_baseline import importance_sampling_acquisition
from utils.baseline.gp_micv_baseline import gp_micv_acquisition
from utils.baseline.bnn_cv_baseline import train_bnn_surrogate, bnn_cv_acquisition
from utils.baseline.bams_baseline import bas_acquisition
import matplotlib.pyplot as plt
import matplotlib
fontsize = 40
parameters = {
    'font.family': 'Times New Roman',
    'axes.labelsize': fontsize,
    'axes.titlesize': fontsize,
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize,
    'legend.fontsize': fontsize
}
plt.rcParams.update(parameters)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# ---------------------------------------------------------------------------
# Strategy runner
# ---------------------------------------------------------------------------

def _pool_remove(pool: np.ndarray, idx: np.ndarray) -> np.ndarray:
    mask = np.ones(len(pool), bool)
    mask[idx] = False
    return pool[mask]


def _micv_select(pool, X_support, gp, batch_size, iteration,
                 n_clusters=10, radius_quantile=0.90, pi_new=0.18,
                 eps_exist=0.04, tau_scale=1.3, novelty_radius=0.5,
                 gamma=0.25, random_state=0):
    """Inline MI+CV selection — uses fit_support_and_compute_mi, matching run_mi_cv.py."""
    mi_vals, _, _,_,_,_ = fit_support_and_compute_mi(
        X_support=X_support,
        X_query=pool,
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
    frac     = 0.05 if iteration > 2 else 0.12
    valid    = np.where(np.isfinite(mi_vals))[0]
    n_top    = max(500, int(frac * len(valid)))
    print(n_top)
    top_idx  = valid[np.argsort(mi_vals[valid])[::-1][:n_top]]
    cv_top   = compute_cv_batch(gp, pool[top_idx])
    cv_full  = np.full(len(pool), np.nan)
    cv_full[top_idx] = cv_top
    clusters = cluster_by_radius(pool[top_idx], eps=0.35)
    X_top    = pool[top_idx]
    reps = []
    for pts in clusters:
        local = []
        for pt in pts:
            hits = np.where(np.all(np.isclose(X_top, pt[None, :]), axis=1))[0]
            local.extend(hits.tolist())
        local = np.array(sorted(set(local)), dtype=int)
        vloc  = local[np.isfinite(cv_top[local])]
        if len(vloc) == 0:
            continue
        best = vloc[np.argmax(cv_top[vloc])]
        reps.append(top_idx[best])
    if not reps:
        vt = top_idx[np.isfinite(cv_top)]
        return vt[np.argsort(cv_full[vt])[::-1]][:batch_size]
    rep_arr = np.array(reps, dtype=int)
    ord_r   = rep_arr[np.argsort(cv_full[rep_arr])[::-1]]
    sel     = ord_r[:batch_size].tolist()
    if len(sel) < batch_size:
        rem  = np.setdiff1d(top_idx, np.array(sel, dtype=int))
        vr   = rem[np.isfinite(cv_full[rem])]
        fill = vr[np.argsort(cv_full[vr])[::-1]]
        sel.extend(fill[: batch_size - len(sel)].tolist())
    return np.array(sel, dtype=int)


def run_strategy(name, X_init, y_real_init, pool_X, args, seed_offset=0):
    rng     = np.random.default_rng(args.seed + seed_offset)
    np.random.seed(args.seed + seed_offset)
    device  = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    X_pair  = X_init.copy()
    y_real  = y_real_init.copy()
    pool    = pool_X.copy()
    best_seen = []

    for t in range(args.n_rounds):
        gp = train_gp(X_pair, y_real)

        if name == "mi_cv":
            idx = _micv_select(pool, X_pair, gp, args.batch_size, t)

        elif name == "random":
            idx = random_acquisition(pool, args.batch_size, rng)

        elif name == "is":
            idx = importance_sampling_acquisition(
                pool, sim_mean_fn, args.batch_size, rng=rng)

        elif name == "gp_cv":
            cv_vals = compute_cv_batch(gp, pool)
            valid   = np.where(np.isfinite(cv_vals))[0]
            order   = valid[np.argsort(cv_vals[valid])[::-1]]
            idx     = order[:args.batch_size]

        elif name == "gp_micv":
            idx = gp_micv_acquisition(
                pool, X_pair, gp, args.batch_size, iteration=t)

        elif name == "bnn_cv":
            model = train_bnn_surrogate(
                X_pair, y_real, epochs=args.bnn_epochs, device=device)
            idx = bnn_cv_acquisition(pool, model, args.batch_size,
                                     device=device, maximise=True)

        elif name == "bams":
            idx = bas_acquisition(
                pool, gp, args.batch_size, threshold=GAMMA,
                n_clusters=min(args.n_clusters, max(2, len(pool) // 5)),
                random_state=args.seed + seed_offset + t,
            )

        else:
            raise ValueError(f"Unknown strategy: {name}")

        X_next = pool[idx]
        y_next = sample_real(X_next)
        X_pair = np.concatenate([X_pair, X_next])
        y_real = np.concatenate([y_real, y_next])
        pool   = _pool_remove(pool, idx)
        best_seen.append(float(np.max(y_real)))

    gp_final = train_gp(X_pair, y_real)
    return {
        "gp_final":       gp_final,
        "X_init":         X_init,
        "X_final":        X_pair,
        "best_real_seen": np.array(best_seen),
        "n_init":         args.n_init,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# STRATEGIES = ["random", "is", "gp_cv", "gp_micv", "bams"]
STRATEGIES = [ "gp_micv"]


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Shared initial data and pool
    np.random.seed(args.seed)
    X_init      = sample_designs(args.n_init, seed=args.seed)
    y_real_init = sample_real(X_init)
    pool_X      = sample_designs(args.pool_size, seed=args.seed + 999)

    results = {}
    for i, s in enumerate(STRATEGIES):
        print(f"\n{'='*60}")
        print(f"[STRATEGY {s.upper()}]")
        print(f"{'='*60}")
        try:
            results[s] = run_strategy(s, X_init, y_real_init, pool_X,
                                       args, seed_offset=i)
            print(f"  best_seen history: {results[s]['best_real_seen']}")
        except Exception as e:
            print(f"[WARN] {s} failed: {e}")
            import traceback; traceback.print_exc()

    if not results:
        print("[ERROR] No strategies succeeded.")
        return

    ran = list(results.keys())

    plot_comparison_grid(
        results=results,
        strategies=ran,
        n_init=args.n_init,
        n_rounds=args.n_rounds,
        output_path=out / "comparison_heatmaps.png",
    )
    plot_best_seen_curve(
        results=results,
        strategies=ran,
        n_rounds=args.n_rounds,
        output_path=out / "best_seen_comparison.png",
    )
    plot_discovery_bar(
        results=results,
        strategies=ran,
        output_path=out / "discovery_bar.png",
    )
    print_discovery_summary(results, ran)
    print(f"\n[DONE]  Outputs → {out}")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--n-rounds",    type=int, default=6)
    p.add_argument("--batch-size",  type=int, default=5)
    p.add_argument("--n-init",      type=int, default=10)
    p.add_argument("--pool-size",   type=int, default=500)
    p.add_argument("--seed",        type=int, default=7)
    p.add_argument("--bnn-epochs",  type=int, default=600)
    p.add_argument("--n-clusters",  type=int, default=6)
    p.add_argument("--output-dir",  type=str, default="outputs/toy2D/comparison")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
