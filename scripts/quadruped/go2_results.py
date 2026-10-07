#!/usr/bin/env python3
"""
scripts/quadruped/plot_go2_results.py
---------------------------------------
Plot comparison of SCOUT, BAMS, and Random baselines on Go2 quadruped
hardware experiments.  Two seeds per method are averaged with std bands.

Data format
-----------
Each .pkl file contains a single hardware trial:
    data['command']['vx']                    → float  (x velocity)
    data['command']['vy']                    → float  (y velocity)
    data['command']['wz']                    → float  (yaw rate)
    data['error_summary']['mean_abs_err_sum'] → float  (tracking error, y)

Files are loaded in sorted order — the sort order is the acquisition order
within each seed.

Three plots
-----------
1. Cumulative running average of y (error) vs number of samples
   Higher y = worse tracking (failure region).

2. Mean pairwise distance among failure scenarios (y >= threshold)
   Computed as: sum(upper triangle of cdist) / n_samples
   (not divided by number of pairs — as specified).

3. Cumulative number of failures discovered (y >= threshold) vs samples.

For each plot, the mean and ±1 std across seeds is shown as a shaded band.

Example
-------
python scripts/quadruped/go2_results.py \
    --results-root results_0/go2_hardware \
    --threshold 0.7 \
    --output-dir outputs/quadruped/paper_plots
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
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
# Style
# ---------------------------------------------------------------------------

_STYLE = {
    "scout":  dict(color="#e41a1c", lw=2.2, ls="-",  label="Our"),
    "bams":   dict(color="#377eb8", lw=1.8, ls=":",  label="BAMS"),
    "random": dict(color="#999999", lw=1.5, ls="--", label="Random"),
}

def _style(name: str) -> dict:
    base = name.rstrip("_0123456789")   # strip seed suffix
    return _STYLE.get(base, dict(color="black", lw=1.5, ls="-", label=name))

def _save(fig, path: Path, dpi: int = 200):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_seed(pkl_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all .pkl files from pkl_dir in sorted order.

    Returns
    -------
    X : (N, 3)  command velocities [vx, vy, wz]
    y : (N,)    tracking errors (mean_abs_err_sum)
    """
    pkl_files = sorted(pkl_dir.glob("*.pkl"))
    pkl_files = pkl_files[10:30]
  
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {pkl_dir}")

    X_list, y_list = [], []
    for f in pkl_files:
        with open(f, "rb") as fh:
            data = pickle.load(fh)
        X_list.append([
            data["command"]["vx"],
            data["command"]["vy"],
            data["command"]["wz"],
        ])
        y_list.append(data["error_summary"]["mean_abs_err_sum"])

    X = np.array(X_list, dtype=np.float64)   # (N, 3)
    y = np.array(y_list, dtype=np.float64)   # (N,)
    print(f"  {pkl_dir.name}: {len(y)} trials  "
          f"y=[{y.min():.3f}, {y.max():.3f}]  "
          f"failures(>={pkl_dir.parent.parent.name})=N/A")
    return X, y


def load_method(
    results_root: Path,
    method_base:  str,          # "scout", "bams", "random"
    seed_suffixes: list[str],   # ["", "_2"]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Load all seeds for a method.  Returns list of (X, y) per seed.
    Missing seed directories are silently skipped.
    """
    seeds = []
    for suf in seed_suffixes:
        d = results_root / f"{method_base}{suf}"
        if not d.is_dir():
            print(f"  [WARN] Not found, skipping: {d}")
            continue
        print(f"  Loading {d.name} …")
        try:
            X, y = load_seed(d)
            seeds.append((X, y))
        except Exception as e:
            print(f"  [WARN] {d.name}: {e}")
    return seeds


# ---------------------------------------------------------------------------
# Curve computation per seed
# ---------------------------------------------------------------------------

def cumulative_avg_curve(y: np.ndarray, threshold) -> np.ndarray:
    """Running mean of y vs sample index."""
    # y_keep = []
    # for y_ in y:
    #     if y_>=threshold:
    #         y_keep.append(y_)
    #     else:
    #         y_keep.append(0)
    # y_keep = np.array(y_keep)
    return np.cumsum(y) / np.arange(1, len(y) + 1)


def pairwise_diversity_curve(
    X: np.ndarray,
    y: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    At each sample index k, compute:
        sum(upper triangle of cdist(X_fail[:k], X_fail[:k])) / k

    where X_fail[:k] = X[:k][y[:k] >= threshold].

    Denominator is n_samples (k), not number of pairs, as specified.
    """
    N   = len(y)
    out = np.zeros(N, dtype=np.float64)
    for k in range(1, N + 1):
        y_k    = y[:k]
        X_k    = X[:k]
        fail_k = X_k[y_k >= threshold]
        if len(fail_k) >= 2:
            D    = cdist(fail_k, fail_k, metric="euclidean")
            triu = D[np.triu_indices(len(fail_k), k=1)]
            out[k - 1] = float(triu.sum()) / (4*k)
        else:
            out[k - 1] = 0.0
    return out


def cumulative_failures_curve(y: np.ndarray, threshold: float) -> np.ndarray:
    """Cumulative count of y >= threshold vs sample index."""
    return np.cumsum((y >= threshold).astype(float))/len(y)


# ---------------------------------------------------------------------------
# Per-method: stack seeds → mean + std arrays
# ---------------------------------------------------------------------------

def compute_curves(
    seeds:     list[tuple[np.ndarray, np.ndarray]],
    threshold: float,
) -> dict:
    """
    Compute all three curves for each seed, then align to the shortest
    seed length and compute mean ± std across seeds.

    Returns dict with:
        n_samples          : (T,) x-axis
        avg_mean, avg_std  : (T,)
        div_mean, div_std  : (T,)
        fail_mean, fail_std: (T,)
    """
    T_min = min(len(y) for _, y in seeds)

    avg_curves  = []
    div_curves  = []
    fail_curves = []

    for X, y in seeds:
        X, y = X[:T_min], y[:T_min]
        avg_curves.append(cumulative_avg_curve(y, threshold))
        div_curves.append(pairwise_diversity_curve(X, y, threshold))
        fail_curves.append(cumulative_failures_curve(y, threshold))

    def _ms(curves):
        arr = np.stack(curves, axis=0)   # (n_seeds, T)
        return arr.mean(axis=0), arr.std(axis=0)

    avg_mean,  avg_std  = _ms(avg_curves)
    div_mean,  div_std  = _ms(div_curves)
    fail_mean, fail_std = _ms(fail_curves)

    return dict(
        n_samples  = np.arange(1, T_min + 1),
        avg_mean   = avg_mean,  avg_std  = avg_std,
        div_mean   = div_mean,  div_std  = div_std,
        fail_mean  = fail_mean, fail_std = fail_std,
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_band(ax, x, mean, std, st):
    ax.plot(x, mean, color=st["color"], lw=st["lw"], ls=st["ls"],
            label=st["label"])
    ax.fill_between(x, mean - std, mean + std,
                    color=st["color"], alpha=0.15)


# ---------------------------------------------------------------------------
# Three individual plot functions
# ---------------------------------------------------------------------------

def plot_cumulative_avg(method_curves, threshold, output_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, d in method_curves.items():
        st = _style(name)
        _plot_band(ax, d["n_samples"], d["avg_mean"], d["avg_std"], st)
    ax.axhline(threshold, color="gray", lw=0.8, ls="--", alpha=0.7,
               label=f"Failure threshold γ={threshold}")
    ax.set_xlabel("Number of hardware trials", fontsize=11)
    ax.set_ylabel("Cumulative avg. tracking error (y)", fontsize=11)
    ax.set_title(f"Cumulative average error vs samples\n"
                 f"(higher = worse tracking, γ={threshold})", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, output_path)


def plot_diversity(method_curves, threshold, output_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, d in method_curves.items():
        st = _style(name)
        _plot_band(ax, d["n_samples"], d["div_mean"], d["div_std"], st)
    ax.set_xlabel("Number of hardware trials", fontsize=11)
    ax.set_ylabel("Σ upper-tri cdist / n_samples  (failure scenarios)", fontsize=11)
    ax.set_title(f"Failure diversity (y >= {threshold})\n"
                 "(higher = broader failure coverage)", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, output_path)


def plot_cumulative_failures(method_curves, threshold, output_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, d in method_curves.items():
        st = _style(name)
        _plot_band(ax, d["n_samples"], d["fail_mean"], d["fail_std"], st)
    ax.set_xlabel("Number of hardware trials", fontsize=11)
    ax.set_ylabel(f"Cumulative failures (y >= {threshold})", fontsize=11)
    ax.set_title(f"Cumulative failure discovery vs samples\n"
                 f"(steeper = faster failure discovery, γ={threshold})", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Combined 1×3 panel
# ---------------------------------------------------------------------------

def plot_all(method_curves, threshold, output_dir):
    plot_cumulative_avg(
        method_curves, threshold,
        output_dir / "01_cumulative_avg_error.pdf")
    plot_diversity(
        method_curves, threshold,
        output_dir / "02_failure_diversity.pdf")
    plot_cumulative_failures(
        method_curves, threshold,
        output_dir / "03_cumulative_failures.pdf")

    # fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for name, d in method_curves.items():
        st = _style(name)
        x  = d["n_samples"]

        for ax, mean_key, std_key in [
            (axes[0], "avg_mean",  "avg_std"),
            (axes[1], "div_mean",  "div_std"),
            (axes[2], "fail_mean", "fail_std"),
        ]:
            m, s = d[mean_key], d[std_key]
            ax.plot(x, m, color=st["color"], lw=st["lw"],
                    ls=st["ls"], label=st["label"])
            ax.fill_between(x, m - s, m + s,
                            color=st["color"], alpha=0.15)

    # axes[0].axhline(threshold, color="gray", lw=0., ls="--", alpha=0.7)
    for ax in axes:
        ax.grid(alpha=0.3)

    axes[0].set_xlabel("No. of samples", fontsize=30)
    axes[0].set_ylabel("Cumulative avg. MSE", fontsize=30)
    # axes[0].set_xticks([0,5,10,15,20],fontsize=15)
    axes[0].tick_params(axis='x', labelsize=25)
    axes[0].tick_params(axis='y', labelsize=25)

    # axes[0].set_title(f"Avg. tracking error\n(higher = more failures, γ={threshold})",
    #                   fontsize=12)

    axes[1].set_xlabel("No. of samples", fontsize=30)
    axes[1].set_ylabel("Cumulative coverage", fontsize=30)
    axes[1].tick_params(axis='x', labelsize=25)
    axes[1].tick_params(axis='y', labelsize=25)
    # axes[1].set_title(f"Failure diversity (y >= {threshold})\n"
    #                   "(higher = broader coverage)", fontsize=12)

    axes[2].set_xlabel("No. of samples", fontsize=30)
    axes[2].set_ylabel(f"Positive samples", fontsize=30)
    axes[2].tick_params(axis='x', labelsize=25)
    axes[2].tick_params(axis='y', labelsize=25)
    # axes[2].set_title("Cumulative failures\n(steeper = faster discovery)",
    #                   fontsize=12)

    # Shared legend above panel 1
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles, labels,
        fontsize=20, ncols=len(method_curves),
        framealpha=0.9, loc="lower center",
        bbox_to_anchor=(1.0, 1.00),
    )

    # fig.tight_layout()
    _save(fig, output_dir / "00_combined_panel.pdf")
    _save(fig, output_dir / "00_combined_panel.png")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(method_curves, threshold):
    print(f"\n{'='*68}")
    print(f"SUMMARY  (γ={threshold})")
    print(f"{'Method':<12}  {'T':>5}  {'final_avg':>10}  "
          f"{'final_fail':>11}  {'final_div':>10}")
    print("-" * 68)
    for name, d in method_curves.items():
        T = len(d["n_samples"])
        print(f"  {name:<12}  {T:>5}  "
              f"{d['avg_mean'][-1]:>10.4f} ±{d['avg_std'][-1]:.3f}  "
              f"{d['fail_mean'][-1]:>10.1f} ±{d['fail_std'][-1]:.1f}  "
              f"{d['div_mean'][-1]:>10.4f} ±{d['div_std'][-1]:.3f}")
    print(f"{'='*68}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_root = Path(args.results_root)

    # Each method: base name + seed suffixes
    methods = {
        "scout":  ["", "_2"],
        "bams":   ["", "_2"],
        "random": ["", "_2"],
    }

    method_curves = {}
    for method_base, suffixes in methods.items():
        print(f"\n[INFO] Loading {method_base} …")
        seeds = load_method(results_root, method_base, suffixes)
        if not seeds:
            print(f"  [WARN] No data for {method_base}, skipping.")
            continue
        method_curves[method_base] = compute_curves(seeds, args.threshold)
        print(f"  → {len(seeds)} seed(s)  T_min={len(method_curves[method_base]['n_samples'])}")

    if not method_curves:
        print("[ERROR] No data loaded.")
        return

    print_summary(method_curves, args.threshold)
    print(f"\n[INFO] Plotting → {out_dir}")
    plot_all(method_curves, args.threshold, out_dir)
    print("[DONE]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Plot Go2 hardware baseline comparisons (SCOUT, BAMS, Random)."
    )
    p.add_argument("--results-root", type=str,
                   default="results/go2_hardware",
                   help="Directory containing bams/, bams_2/, scout/, scout_2/, "
                        "random/, random_2/ sub-directories of .pkl files.")
    p.add_argument("--threshold", type=float, default=0.7,
                   help="Failure threshold: y >= threshold = failure.")
    p.add_argument("--output-dir", type=str,
                   default="results/quadruped/plots")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())