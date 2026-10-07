#!/usr/bin/env python3
"""
scripts/simpler/plot_simpler_results.py
-----------------------------------------
Comparison plots for SimplerEnv active-learning baselines.

THREE PLOTS
-----------
1. Cumulative average metric vs number of samples
   y-axis: running mean of y_target_mean collected so far.
   Higher y = failure in SimplerEnv (higher means a more severe failure).
   A good AL method should find high-y (failure) scenarios → curve trends UP.

2. Sample diversity of failure scenarios
   Mean pairwise L2 distance in 42-dim feature space among collected
   scenarios with y >= threshold.  Higher = more diverse failure coverage.

3. Cumulative number of failure scenarios vs number of samples
   Count of scenarios with y >= threshold seen up to each sample index.
   A good method accumulates failures faster → steeper rise.

DIRECTORY STRUCTURE
-------------------
Each method saves an ALState at each iteration under:
    <results_root>/<method>/init/
    <results_root>/<method>/iter_00/
    <results_root>/<method>/iter_01/
    ...
    <results_root>/<method>/final/

Each iter_XX/ contains y_targets_mean.npy with the CUMULATIVE mean y values
for all target scenarios collected up to that iteration, in acquisition order.

Example
-------
python scripts/simpler/plot_simpler_results.py \\
    --results-root ./results \\
    --methods random scout bnn_cv bams \\
    --threshold 0.7 \\
    --n-init 20 \\
    --output-dir ./results/plots

python scripts/simpler/plot_simpler_results.py \
    --results-root ./results_0 \
    --methods scout bnn_cv random gpc\
    --threshold 1.0 \
    --n-init 0 \
    --n-show 60 \
    --output-dir ./outputs/simpler/paper_plots 
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.simpler.al_state import ALState
from utils.simpler.domain   import scenarios_to_array
import json

# ---------------------------------------------------------------------------
# Style — shared paper-plot style
# ---------------------------------------------------------------------------

_STYLE = {
    "scout":   dict(color="#e41a1c", lw=2.2, ls="-",  label="SCOUT (ours)"),
    "random":  dict(color="#999999", lw=1.5, ls="--", label="Random"),
    "bnn_cv":  dict(color="#4daf4a", lw=1.5, ls="-.", label="BNN-CV"),
    "bams":    dict(color="#377eb8", lw=1.8, ls=":",  label="BAMS"),
    "gpc":  dict(color="#ff7f00", lw=1.5, ls="--", label="GP-C"),
}

def _style(name: str) -> dict:
    return _STYLE.get(name, dict(color="black", lw=1.5, ls="-", label=name))

def _save(fig, path: Path, dpi: int = 200):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def get_task_name(iter_dir: str | Path) -> str | None:
    """
    Given an iteration directory like results/scout/AL_0_0,
    find the task name by locating paired_result.json and reading
    the task folder name from the path.
    """
    iter_dir = Path(iter_dir)
    matches  = list(iter_dir.glob("*/rt1/*/scenario_*/paired_result.json"))
    if not matches:
        return None
    # The task name is the first component after iter_dir
    return matches[0].relative_to(iter_dir).parts[0]

def _iter_dirs(method_dir: Path, method) -> list:
    """
    Return sorted list of iter_XX directories under method_dir.
    Also handles AL_{t}_{j} naming for backward compat.
    """
    # iters = sorted(
    #     method_dir.glob("iter_*"),
    #     key=lambda p: int(p.name.split("_")[-1])
    # )
    # if iters:
    #     return iters

    # Fallback: AL_{t}_{j} — group by t, keep last j per t
    if method=='scout' or method=='bams' or method=='bnn_cv' or method=='gpc':
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
            result = []
            
            for t in sorted(al_dirs):

                result.append(
                    sorted(al_dirs[t])#, key=lambda p: int(p.name.split("_")[-1]))[-1]
                )
            flat_list = [p for sublist in result for p in sublist]
            return flat_list
    elif method=='random':
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
            result = []
            
            for t in sorted(al_dirs):

                result.append(
                    sorted(al_dirs[t])#, key=lambda p: int(p.name.split("_")[-1]))[-1]
                )
            flat_list = [p for sublist in result for p in sublist]
            return flat_list
    
    return []


def _load_y_sequence(method_dir: Path,method) -> np.ndarray:
    """
    Reconstruct y_target values in acquisition order.

    Reads y_targets_mean.npy from init/ then each iter_XX/ directory.
    Each file is CUMULATIVE — new points at iter t are file_t[len(file_{t-1}):].

    Falls back to final/y_targets_mean.npy or ALState.load() if no iter dirs.
    """
    y_seq    = []
    prev_len = 0

    # Initial seed
    # init_dir = method_dir / "init"
    # if init_dir.exists() and (init_dir / "y_targets_mean.npy").exists():
    #     y_init   = np.load(init_dir / "y_targets_mean.npy").astype(np.float64)
    #     breakpoint()
    #     y_seq.extend(y_init.tolist())
    #     prev_len = len(y_init)

    # Iteration dirs
    iter_dir = _iter_dirs(method_dir,method)[0:18]
    for d in iter_dir:
 
        task_name = get_task_name(d)
      
        # results/scout/AL_0_0/google_robot_pick_standing_coke_can/rt1/rt_1_x_tf_trained_for_002272480_step
        filename = f"{str(d)}/{task_name}/rt1/rt_1_x_tf_trained_for_002272480_step/scenario_0000/paired_result.json"
        print(filename)
        with open(filename) as f:
            data = json.load(f)
        if data['target']['success']:
            y =0.0
        else:
            y=1.0
   
        # timesteps = data["target"]["timesteps"] 
        # y = timesteps/80
        y_seq.append(y)
        # f = d / "y_targets_mean.npy"
        # if not f.exists():
        #     continue
        # y_cum = np.load(y).astype(np.float64)
        # y_seq.extend(y_cum[prev_len:].tolist())
        # prev_len = len(y_cum)
    if y_seq:
        return np.array(y_seq, dtype=np.float64)
    

    # # Fallback: final state
    # for candidate in [method_dir / "final", method_dir]:
    #     f = candidate / "y_targets_mean.npy"
    #     if f.exists():
    #         print(f"  [INFO] Falling back to {f}")
    #         return np.load(f).astype(np.float64)
    #     if (candidate / "train_scenarios.json").exists():
    #         state = ALState.load(candidate)
    #         return state.y_train_mean().astype(np.float64)

    raise FileNotFoundError(
        f"Cannot reconstruct y sequence from {method_dir}. "
        f"Expected iter_XX/y_targets_mean.npy or final/y_targets_mean.npy."
    )


def _load_X_sequence(method_dir: Path) -> np.ndarray | None:
    """
    Load (T, 42) feature matrix in acquisition order from the final ALState.
    Returns None if not available (diversity plot will be skipped).
    """
    for candidate in [method_dir / "final", method_dir]:
        if (candidate / "train_scenarios.json").exists():
            state = ALState.load(candidate)
            return state.X_train().astype(np.float64)
    return None


def load_method_data(
    method_dir: Path,
    threshold:  float,
    n_init:     int,
    step:       int = 1,
    method='scout'
) -> dict:
    """
    Compute all three curve arrays for one method.
    """

    y_seq = _load_y_sequence(method_dir,method)
    T     = len(y_seq)
    print(f"  [INFO] {method_dir.name}: T={T}  "
          f"y=[{y_seq.min():.3f}, {y_seq.max():.3f}]  "
          f"failures(>={threshold})={int((y_seq >= threshold).sum())}")

    # ── Plot 1: cumulative running average ────────────────────────────────
    full_avg      = np.cumsum(y_seq) / np.arange(1, T + 1)
    start         = max(n_init, 0)
    running_avg   = full_avg[start:]
    n_samples_ttc = np.arange(start + 1, T + 1)

    # ── Plot 3: cumulative failure count ──────────────────────────────────
    fail_mask           = y_seq >= threshold
    cumulative_failures = np.cumsum(fail_mask.astype(float))[start:]
    n_samples_fail      = n_samples_ttc.copy()

    # ── Plot 2: diversity among failure scenarios ─────────────────────────
    X_all = _load_X_sequence(method_dir)

    checkpoints = np.arange(max(n_init, 2), T + 1, step)
    if len(checkpoints) == 0 or checkpoints[-1] != T:
        checkpoints = np.append(checkpoints, T)

    n_samples_div      = []
    mean_pairwise_dist = []

    for k in checkpoints:
        y_k    = y_seq[:k]
        fail_k = np.where(y_k >= threshold)[0]
        X_k  = X_all[fail_k[:min(len(fail_k), 500)]]
        D    = cdist(X_k, X_k, metric="euclidean")
        upper_sum = D[np.triu_indices_from(D, k=1)].sum()
        mean_pairwise_dist.append(upper_sum/k)

        # if X_all is not None and len(fail_k) >= 2:
           
        #     D    = cdist(X_k, X_k, metric="euclidean")
        #     triu = D[np.triu_indices_from(D, k=1)]
        #     mpd  = float(triu.mean()) if len(triu) > 0 else 0.0
        # else:
        #     mpd = 0.0

        n_samples_div.append(k)
    
        # mean_pairwise_dist.append(mpd)
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
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_cumulative_avg(method_data, n_init, threshold, output_path):
    """Plot 1 — cumulative average y vs samples. Higher = more failures found."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, d in method_data.items():
        st = _style(name)
        ax.plot(d["n_samples_ttc"], d["running_avg"],
                color=st["color"], lw=st["lw"], ls=st["ls"], label=st["label"])
    if n_init > 0:
        ax.axvline(n_init, color="k", lw=0.8, ls=":", alpha=0.5,
                   label=f"End of seed (n={n_init})")
    ax.axhline(threshold, color="gray", lw=0.8, ls="--", alpha=0.6,
               label=f"Failure threshold γ={threshold}")
    ax.set_xlabel("Number of samples collected", fontsize=11)
    ax.set_ylabel("Cumulative average metric (y)", fontsize=11)
    ax.set_title(f"Cumulative average metric vs samples\n"
                 f"(higher = method finds more failures, γ={threshold})", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, output_path)


def plot_diversity(method_data, n_init, threshold, output_path):
    """Plot 2 — mean pairwise L2 distance among failure scenarios."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, d in method_data.items():
        st = _style(name)
        ax.plot(d["n_samples_div"], d["mean_pairwise_dist"],
                color=st["color"], lw=st["lw"], ls=st["ls"], label=st["label"])
    if n_init > 0:
        ax.axvline(n_init, color="k", lw=0.8, ls=":", alpha=0.5,
                   label=f"End of seed (n={n_init})")
    ax.set_xlabel("Number of samples collected", fontsize=11)
    ax.set_ylabel("Mean pairwise distance (42-dim features)", fontsize=11)
    ax.set_title(f"Failure diversity (among y >= {threshold})\n"
                 "(higher = broader failure coverage)", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, output_path)


def plot_cumulative_failures(method_data, n_init, threshold, output_path):
    """Plot 3 — cumulative failure count vs samples."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, d in method_data.items():
        st = _style(name)
        ax.plot(d["n_samples_fail"], d["cumulative_failures"],
                color=st["color"], lw=st["lw"], ls=st["ls"], label=st["label"])
    if n_init > 0:
        ax.axvline(n_init, color="k", lw=0.8, ls=":", alpha=0.5,
                   label=f"End of seed (n={n_init})")
    ax.set_xlabel("Number of samples collected", fontsize=11)
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

def plot_all(method_data, n_init, threshold, output_dir, n_show=0):
    # Individual PDFs
    plot_cumulative_avg(
        method_data, n_init, threshold,
        output_dir / "01_cumulative_avg_metric.pdf")
    plot_diversity(
        method_data, n_init, threshold,
        output_dir / "02_failure_diversity.pdf")
    plot_cumulative_failures(
        method_data, n_init, threshold,
        output_dir / "03_cumulative_failures.pdf")

    # Combined 1×3
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    def _trim(arr,n_show=60):
        return arr[:n_show] if n_show > 0 else arr

    for name, d in method_data.items():
        st = _style(name)
        kw = dict(color=st["color"], lw=st["lw"], ls=st["ls"], label=st["label"])

        axes[0].plot(_trim(d["n_samples_ttc"]),  _trim(d["running_avg"]),        **kw)
        axes[1].plot(_trim(d["n_samples_div"]),  _trim(d["mean_pairwise_dist"]), **kw)
        axes[2].plot(_trim(d["n_samples_fail"]), _trim(d["cumulative_failures"]), **kw)

    axes[0].axhline(threshold, color="gray", lw=0.8, ls="--", alpha=0.6)
    for ax in axes:
        if n_init > 0:
            ax.axvline(n_init, color="k", lw=0.8, ls=":", alpha=0.5)
        ax.grid(alpha=0.3)

    axes[0].set_xlabel("# samples", fontsize=14)
    axes[0].set_ylabel("Cumulative avg. metric", fontsize=14)
    axes[0].set_title(f"Cumulative avg. metric\n(higher = more failures, γ={threshold})",
                      fontsize=12)

    axes[1].set_xlabel("# samples", fontsize=14)
    axes[1].set_ylabel("Mean pairwise dist. (42-dim)", fontsize=14)
    axes[1].set_title(f"Failure diversity (y >= {threshold})\n"
                      "(higher = broader coverage)", fontsize=12)

    axes[2].set_xlabel("# samples", fontsize=14)
    axes[2].set_ylabel(f"Cumulative failures (y >= {threshold})", fontsize=14)
    axes[2].set_title("Cumulative failure discovery\n(steeper = faster)", fontsize=12)

    # Shared legend above panel 1
    axes[0].legend(
        fontsize=10, ncols=len(method_data), framealpha=0.9,
        loc="lower center", bbox_to_anchor=(0.5, 1.08),
    )

    fig.tight_layout()
    _save(fig, output_dir / "00_combined_panel.pdf")
    _save(fig, output_dir / "00_combined_panel.png")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(method_data, threshold, n_init):
    print(f"\n{'='*72}")
    print(f"SUMMARY  (γ={threshold},  n_init={n_init})")
    print(f"{'Method':<20}  {'T':>5}  {'final_avg':>10}  "
          f"{'n_fail':>7}  {'fail_rate':>10}  {'final_div':>10}")
    print("-" * 72)
    for name, d in method_data.items():
        y        = d["y_seq"]
        T        = len(y)
        n_fail   = int((y >= threshold).sum())
        final_div = float(d["mean_pairwise_dist"][-1]) \
                    if len(d["mean_pairwise_dist"]) else 0.0
        print(f"  {name:<20}  {T:>5}  {float(y.mean()):>10.4f}  "
              f"{n_fail:>7}  {n_fail/T:>10.4f}  {final_div:>10.4f}")
    print(f"{'='*72}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    out_dir      = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_root = Path(args.results_root)
    method_data  = {}

    for method in args.methods:
        method_dir = results_root / method
        if not method_dir.is_dir():
            print(f"[WARN] Not found, skipping: {method_dir}")
            continue
        print(f"[INFO] Loading {method} …")
        try:
            method_data[method] = load_method_data(
                method_dir = method_dir,
                threshold  = args.threshold,
                n_init     = args.n_init,
                step       = args.step,
                method=method
            )
            
        except Exception as e:
            print(f"[WARN] {method}: {e}")

    if not method_data:
        print("[ERROR] No method data loaded.")
        return

    print_summary(method_data, args.threshold, args.n_init)
    print(f"[INFO] Plotting → {out_dir}")
    plot_all(
        method_data = method_data,
        n_init      = args.n_init,
        threshold   = args.threshold,
        output_dir  = out_dir,
        n_show      = args.n_show,
    )
    print("[DONE]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Comparison plots for SimplerEnv AL baselines."
    )
    p.add_argument("--results-root", type=str, required=True,
                   help="Root dir containing one sub-dir per method, "
                        "e.g. ./results  (contains ./results/random/, ./results/scout/)")
    p.add_argument("--methods",      type=str, nargs="+",
                   default=["random", "scout", "bnn_cv", "bams"])
    p.add_argument("--threshold",    type=float, default=0.7,
                   help="y >= threshold = failure  (higher y = failure in SimplerEnv)")
    p.add_argument("--n-init",       type=int,   default=20,
                   help="Initial seed size (vertical reference line on plots)")
    p.add_argument("--step",         type=int,   default=1,
                   help="Evaluate diversity every N samples")
    p.add_argument("--n-show",       type=int,   default=0,
                   help="Truncate plots to first N samples (0 = all)")
    p.add_argument("--output-dir",   type=str,   default="./results/plots")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
