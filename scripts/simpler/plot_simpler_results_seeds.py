#!/usr/bin/env python3
"""
scripts/simpler/plot_simpler_results_seeds.py
----------------------------------------------
Same three plots as plot_simpler_results.py but with mean ± std across
seeds instead of a single-seed curve.

Seed directories expected at:
    <seeds-root>_0/<method>/AL_0_0/...
    <seeds-root>_1/<method>/AL_0_0/...
    ...

i.e. the seeds root is a PREFIX — the seed index is appended as a suffix.

Example
-------
python scripts/simpler/plot_simpler_results_seeds.py \
    --seeds-root   results \
    --seeds        0 1 2\
    --methods      scout bnn_cv random gpc\
    --threshold    1.0 \
    --n-init       0 \
    --n-show       60 \
    --output-dir   ./outputs/simpler/paper_plots_seeds
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.simpler.al_state import ALState
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
# Style — unchanged from plot_simpler_results.py
# ---------------------------------------------------------------------------

_STYLE = {
    "scout":  dict(color="#e41a1c", lw=2.2, ls="-",  label="SCOUT (ours)"),
    "random": dict(color="#999999", lw=1.5, ls="--", label="Random"),
    "bnn_cv": dict(color="#4daf4a", lw=1.5, ls="-.", label="BNN-CV"),
    "bams":   dict(color="#377eb8", lw=1.8, ls=":",  label="BAMS"),
    "gpc":    dict(color="#ff7f00", lw=1.5, ls="--", label="GP-C"),
}

def _style(name: str) -> dict:
    return _STYLE.get(name, dict(color="black", lw=1.5, ls="-", label=name))

def _save(fig, path: Path, dpi: int = 200):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Data loading — identical to plot_simpler_results.py
# ---------------------------------------------------------------------------

def get_task_name(iter_dir: Path) -> str | None:
    matches = list(iter_dir.glob("*/rt1/*/scenario_*/paired_result.json"))
    if not matches:
        return None
    return matches[0].relative_to(iter_dir).parts[0]


def _iter_dirs(method_dir: Path, method: str) -> list:
    if method in ("scout", "bams", "bnn_cv", "gpc"):
        al_dirs: dict = {}
        for p in method_dir.glob("AL_*"):
            parts = p.name.split("_")
            if len(parts) >= 3:
                try:
                    t = int(parts[1])
                    al_dirs.setdefault(t, []).append(p)
                except ValueError:
                    pass
        if al_dirs:
            result = [sorted(al_dirs[t]) for t in sorted(al_dirs)]
            return [p for sublist in result for p in sublist]

    elif method == "random":
        al_dirs: dict = {}
        for p in method_dir.glob("random_*"):
            parts = p.name.split("_")
            if len(parts) >= 3:
                try:
                    t = int(parts[1])
                    al_dirs.setdefault(t, []).append(p)
                except ValueError:
                    pass
        if al_dirs:
            result = [sorted(al_dirs[t]) for t in sorted(al_dirs)]
            return [p for sublist in result for p in sublist]

    return []


def _load_y_sequence(method_dir: Path, method: str,
                     n_show: int = 0) -> np.ndarray | None:
    """
    Reconstruct y values in acquisition order from AL_{t}_{j} directories.
    Returns None if no data found (seed silently skipped).
    """
    dirs = _iter_dirs(method_dir, method)
    if n_show > 0:
        dirs = dirs[:n_show]

    y_seq = []
    for d in dirs:
        task_name = get_task_name(d)
        if task_name is None:
            continue
        filename = (f"{d}/{task_name}/rt1/"
                    f"rt_1_x_tf_trained_for_002272480_step/"
                    f"scenario_0000/paired_result.json")
        try:
            with open(filename) as f:
                data = json.load(f)
            y_seq.append(0.0 if data["target"]["success"] else 1.0)
        except Exception as e:
            print(f"  [WARN] Cannot read {filename}: {e}")

    if not y_seq:
        return None
    return np.array(y_seq, dtype=np.float64)


def _load_X_sequence(method_dir: Path) -> np.ndarray | None:
    for candidate in [method_dir / "final", method_dir]:
        if (candidate / "train_scenarios.json").exists():
            state = ALState.load(candidate)
            return state.X_train().astype(np.float64)
    return None


def _compute_curves(y_seq: np.ndarray, X_all,
                    threshold: float, n_init: int,
                    step: int, n_show: int) -> dict:
    """Compute all three curve arrays for one seed of one method."""
    T     = len(y_seq)
    start = max(n_init, 0)

    # Plot 1: cumulative running average
    full_avg      = np.cumsum(y_seq) / np.arange(1, T + 1)
    running_avg   = full_avg[start:]
    n_samples_ttc = np.arange(start + 1, T + 1)

    # Plot 3: cumulative failures
    fail_mask           = y_seq >= threshold
    cumulative_failures = np.cumsum(fail_mask.astype(float))[start:]
    n_samples_fail      = n_samples_ttc.copy()

    # Plot 2: pairwise diversity among failures
    checkpoints = np.arange(max(n_init, 2), T + 1, step)
    if len(checkpoints) == 0 or checkpoints[-1] != T:
        checkpoints = np.append(checkpoints, T)

    n_samples_div      = []
    mean_pairwise_dist = []

    for k in checkpoints:
        y_k    = y_seq[:k]
        fail_k = np.where(y_k >= threshold)[0]
        if X_all is not None and len(fail_k) >= 2:
            X_k  = X_all[fail_k[:min(len(fail_k), 500)]]
            D    = cdist(X_k, X_k, metric="euclidean")
            upper_sum = D[np.triu_indices_from(D, k=1)].sum()
            mpd = upper_sum / (110*k)
        else:
            mpd = 0.0
        n_samples_div.append(k)
        mean_pairwise_dist.append(mpd)

    return dict(
        y_seq               = y_seq,
        n_samples_ttc       = n_samples_ttc,
        running_avg         = running_avg,
        n_samples_fail      = n_samples_fail,
        cumulative_failures = cumulative_failures,
        n_samples_div       = np.array(n_samples_div),
        mean_pairwise_dist  = np.array(mean_pairwise_dist),
    )


# ---------------------------------------------------------------------------
# Seed aggregation
# ---------------------------------------------------------------------------

def _aggregate(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Truncate all seed curves to the shortest, return (mean, std)."""
    T_min   = min(len(c) for c in curves)
    stacked = np.stack([c[:T_min] for c in curves], axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)


def load_method_seeds(
    seeds_root: str,
    seeds:      list[int],
    method:     str,
    threshold:  float,
    n_init:     int,
    step:       int,
    n_show:     int,
) -> dict | None:
    """
    Load all seeds for one method and return aggregated mean ± std curves.

    seeds_root is a PREFIX: results from seed s are at {seeds_root}_{s}/{method}.
    """
    avg_curves  = []
    fail_curves = []
    div_curves  = []
    n_ttc_ref   = None
    n_div_ref   = None

    for s in seeds:
        method_dir = Path(f"{seeds_root}_{s}") / method
        if not method_dir.is_dir():
            print(f"  [WARN] Not found: {method_dir}")
            continue
        print(f"  Seed {s}: {method_dir}")

        y_seq = _load_y_sequence(method_dir, method, n_show=n_show)
        if y_seq is None or len(y_seq) == 0:
            print(f"  [WARN] No y data for seed {s}, skipping.")
            continue

        X_all = _load_X_sequence(method_dir)
        d     = _compute_curves(y_seq, X_all, threshold, n_init, step, n_show)

        avg_curves.append(d["running_avg"])
        fail_curves.append(d["cumulative_failures"])
        div_curves.append(d["mean_pairwise_dist"])

        if n_ttc_ref is None:
            n_ttc_ref = d["n_samples_ttc"]
        if n_div_ref is None:
            n_div_ref = d["n_samples_div"]

        print(f"    T={len(y_seq)}  "
              f"failures={int((y_seq >= threshold).sum())}  "
              f"final_avg={y_seq.mean():.3f}")

    if not avg_curves:
        return None

    avg_mean,  avg_std  = _aggregate(avg_curves)
    fail_mean, fail_std = _aggregate(fail_curves)
    div_mean,  div_std  = _aggregate(div_curves)
    T = len(avg_mean)

    return dict(
        n_seeds    = len(avg_curves),
        n_ttc      = n_ttc_ref[:T] if n_ttc_ref is not None else np.arange(1, T + 1),
        avg_mean   = avg_mean,   avg_std   = avg_std,
        n_div      = n_div_ref[:len(div_mean)] if n_div_ref is not None
                     else np.arange(1, len(div_mean) + 1),
        div_mean   = div_mean,   div_std   = div_std,
        fail_mean  = fail_mean,  fail_std  = fail_std,
    )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _band(ax, x, mean, std, st, alpha=0.18):
    ax.plot(x, mean,
            color=st["color"], lw=st["lw"], ls=st["ls"], label=st["label"])
    ax.fill_between(x, mean - std, mean + std,
                    color=st["color"], alpha=alpha)


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_cumulative_avg(stats, n_init, threshold, output_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, d in stats.items():
        _band(ax, d["n_ttc"], d["avg_mean"], d["avg_std"], _style(name))
    if n_init > 0:
        ax.axvline(n_init, color="k", lw=0.8, ls=":", alpha=0.5,
                   label=f"End of seed (n={n_init})")
    ax.axhline(threshold, color="gray", lw=0.8, ls="--", alpha=0.6,
               label=f"γ={threshold}")
    ax.set_xlabel("Number of samples collected", fontsize=11)
    ax.set_ylabel("Cumulative average metric (y)", fontsize=11)
    ax.set_title(f"Cumulative average metric vs samples\n"
                 f"(higher = more failures, γ={threshold})", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    # fig.tight_layout()

    _save(fig, output_path)


def plot_diversity(stats, n_init, threshold, output_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, d in stats.items():
        _band(ax, d["n_div"], d["div_mean"], d["div_std"], _style(name))
    if n_init > 0:
        ax.axvline(n_init, color="k", lw=0.8, ls=":", alpha=0.5,
                   label=f"End of seed (n={n_init})")
    ax.set_xlabel("Number of samples collected", fontsize=11)
    ax.set_ylabel("Σ upper-tri cdist / k  (failure scenarios)", fontsize=11)
    ax.set_title(f"Failure diversity (y >= {threshold})\n"
                 "(higher = broader coverage)", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    # fig.tight_layout()
    _save(fig, output_path)


def plot_cumulative_failures(stats, n_init, threshold, output_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, d in stats.items():
        x = np.arange(1, len(d["fail_mean"]) + 1)
        _band(ax, x, d["fail_mean"], d["fail_std"], _style(name))
    if n_init > 0:
        ax.axvline(n_init, color="k", lw=0.8, ls=":", alpha=0.5,
                   label=f"End of seed (n={n_init})")
    ax.set_xlabel("Number of samples collected", fontsize=11)
    ax.set_ylabel(f"Cumulative failures (y >= {threshold})", fontsize=11)
    ax.set_title(f"Cumulative failure discovery\n"
                 f"(steeper = faster, γ={threshold})", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    # fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Combined 1×3 panel
# ---------------------------------------------------------------------------

def plot_all(stats, n_init, threshold, output_dir, n_show=0):
    plot_cumulative_avg(
        stats, n_init, threshold,
        output_dir / "01_cumulative_avg_metric.pdf")
    plot_diversity(
        stats, n_init, threshold,
        output_dir / "02_failure_diversity.pdf")
    plot_cumulative_failures(
        stats, n_init, threshold,
        output_dir / "03_cumulative_failures.pdf")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    def _trim(arr):
        return arr[:n_show] if n_show > 0 else arr

    for name, d in stats.items():
        st = _style(name)
        kw = dict(color=st["color"], lw=st["lw"], ls=st["ls"], label=st["label"])

        for ax, mean_key, std_key, x_key in [
            (axes[0], "avg_mean",  "avg_std",  "n_ttc"),
            (axes[1], "div_mean",  "div_std",  "n_div"),
        ]:
            x = _trim(d[x_key])
            m = _trim(d[mean_key])
            s = _trim(d[std_key])
            ax.plot(x, m, **kw)
            ax.fill_between(x, m - s, m + s,
                            color=st["color"], alpha=0.18)

        # Plot 3 — x is implicit
        x3 = _trim(np.arange(1, len(d["fail_mean"]) + 1))
        m3 = _trim(d["fail_mean"])
        s3 = _trim(d["fail_std"])
        axes[2].plot(x3, m3, **kw)
        axes[2].fill_between(x3, m3 - s3, m3 + s3,
                              color=st["color"], alpha=0.18)

    # axes[0].axhline(threshold, color="gray", lw=0.8, ls="--", alpha=0.6)
    for ax in axes:
        if n_init > 0:
            ax.axvline(n_init, color="k", lw=0.8, ls=":", alpha=0.5)
        ax.grid(alpha=0.3)

    axes[0].set_xlabel("No. of samples", fontsize=30)
    axes[0].set_ylabel("Cumulative avg. fail", fontsize=30)
    axes[0].tick_params(axis="x", labelsize=25)
    axes[0].tick_params(axis="y", labelsize=25)
    
    # axes[0].set_title(f"Cumulative avg. metric\n(higher = more failures, γ={threshold})",
                    #   fontsize=12)
    axes[1].set_xlabel("No. of samples", fontsize=30)
    axes[1].set_ylabel("Cumulative coverage", fontsize=30)
    axes[1].tick_params(axis="x", labelsize=25)
    axes[1].tick_params(axis="y", labelsize=25)
    # axes[1].set_title(f"Failure diversity (y >= {threshold})\n"
    #                   "(higher = broader coverage)", fontsize=30)
    axes[2].set_xlabel("No. of samples", fontsize=30)
    axes[2].set_ylabel("Positive samples", fontsize=30)
    axes[2].tick_params(axis="x", labelsize=25)
    axes[2].tick_params(axis="y", labelsize=25)
    # axes[2].set_title("Cumulative failure discovery\n(steeper = faster)", fontsize=12)

    axes[0].legend(
        fontsize=20, ncols=len(stats), framealpha=0.9,
        loc="lower center", bbox_to_anchor=(0.5, 1.08),
    )
    # fig.tight_layout()
    _save(fig, output_dir / "00_combined_panel.pdf")
    _save(fig, output_dir / "00_combined_panel.png")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(stats, threshold):
    print(f"\n{'='*70}")
    print(f"SUMMARY  (γ={threshold})")
    print(f"{'Method':<12}  {'Seeds':>6}  {'final_avg':>10}±{'std':>6}  "
          f"{'final_fail':>11}±{'std':>6}")
    print("-" * 70)
    for name, d in stats.items():
        print(f"  {name:<12}  {d['n_seeds']:>6}  "
              f"{d['avg_mean'][-1]:>10.4f}±{d['avg_std'][-1]:<6.4f}  "
              f"{d['fail_mean'][-1]:>10.1f}±{d['fail_std'][-1]:<6.2f}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    for method in args.methods:
        print(f"\n[INFO] Loading {method} across seeds {args.seeds} …")
        d = load_method_seeds(
            seeds_root = args.seeds_root,
            seeds      = args.seeds,
            method     = method,
            threshold  = args.threshold,
            n_init     = args.n_init,
            step       = args.step,
            n_show     = args.n_show,
        )
        if d is None:
            print(f"  [WARN] No data for {method}, skipping.")
        else:
            stats[method] = d
            print(f"  → {d['n_seeds']} seeds  T={len(d['avg_mean'])}")

    if not stats:
        print("[ERROR] No data loaded.")
        return

    print_summary(stats, args.threshold)
    print(f"[INFO] Plotting → {out_dir}")
    plot_all(stats, args.n_init, args.threshold, out_dir, args.n_show)
    print("[DONE]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="SimplerEnv comparison plots — mean ± std across seeds."
    )
    p.add_argument("--seeds-root", type=str, required=True,
                   help="Prefix for seed directories. "
                        "Seed s is read from {seeds-root}_{s}/{method}/. "
                        "E.g. results  →  "
                        "results_0/scout, results_1/scout, …")
    p.add_argument("--seeds",      type=int, nargs="+", default=[0, 1, 2],
                   help="Seed indices to include")
    p.add_argument("--methods",    type=str, nargs="+",
                   default=["random", "scout", "bnn_cv", "gpc"])
    p.add_argument("--threshold",  type=float, default=1.0,
                   help="y >= threshold = failure")
    p.add_argument("--n-init",     type=int,   default=0)
    p.add_argument("--step",       type=int,   default=1)
    p.add_argument("--n-show",     type=int,   default=60,
                   help="Use only first N AL directories per seed (0 = all)")
    p.add_argument("--output-dir", type=str,   default="./outputs/simpler/paper_plots_seeds")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())