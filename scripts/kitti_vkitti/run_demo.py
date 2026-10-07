#!/usr/bin/env python3
"""
Short KITTI–Virtual KITTI demo: SCOUT, two ablations, and two baselines.

Methods
-------
scout          MI shortlist + local control variates that use the proxy
real_only      same loop with β = 0 (no proxy in the estimator)
mi_only        MI shortlist only, no surrogate  (ablation of the CV term)
random         uniform draws from the unlabeled pool
is             importance sampling with weights from the proxy failure

All five share one initial real-labeled set. The plot is cumulative mean
target failure versus number of real labels (higher means failures are found sooner).

The defaults are sized for a notebook. Paper-scale sweeps live in
``ablate_local_cv.py``, ``ablate_scenario_space.py``, and ``compare_beta_ablation.py``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.kitti_vkitti.run_scout import _load_task, _shared_init  # noqa: E402
from utils.baseline.importance_sampling_baseline import (  # noqa: E402
    importance_sampling_acquisition,
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

_COLORS = {
    "scout": "#0072B2",
    "real_only": "#D55E00",
    "mi_only": "#009E73",
    "random": "#999999",
    "is": "#E69F00",
}
_LABELS = {
    "scout": "SCOUT (MI + CV)",
    "real_only": "Ablation: β = 0",
    "mi_only": "Ablation: MI only",
    "random": "Random",
    "is": "Importance sampling",
}


def _cummean(y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    vals = y[np.asarray(idx, dtype=int)]
    return np.cumsum(vals) / np.arange(1, len(vals) + 1)


def _run_scout_cli(mode: str, extra: list[str], args, out_dir: Path) -> None:
    cmd = [
        sys.executable, str(_ROOT / "scripts" / "kitti_vkitti" / "run_scout.py"),
        "--mode", mode,
        "--task-csv", str(args.task_csv),
        "--output-dir", str(out_dir / mode),
        "--init-cache-dir", str(Path(args.output_dir) / "init"),
        "--seed", str(args.seed),
        "--scenario-space", "full",
        "--n-init", str(args.n_init),
        "--n-proxy-only", str(args.n_proxy_only),
        "--n-acq-iters", str(args.n_acq_iters),
        "--batch-acq-size", str(args.batch_size),
        "--mi-shortlist", str(args.mi_shortlist),
        "--bnn-epochs", str(args.bnn_epochs),
        "--cv-n-jobs", str(args.cv_n_jobs),
        "--cpu",
        *extra,
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=_ROOT)


def _acquire_baseline(name: str, y_target, y_proxy, initial_idx, args, out_dir: Path) -> np.ndarray:
    rng = np.random.default_rng(args.seed)
    chosen = [int(i) for i in initial_idx]
    selected = set(chosen)
    n = len(y_target)
    for _ in range(args.n_acq_iters):
        pool = np.array([i for i in range(n) if i not in selected], dtype=int)
        if name == "random":
            pick = rng.choice(pool, size=min(args.batch_size, len(pool)), replace=False)
        else:
            local = importance_sampling_acquisition(
                np.zeros((len(pool), 1)),
                lambda _x, pool=pool: y_proxy[pool],
                batch_size=min(args.batch_size, len(pool)),
                beta=4.0,
                rng=rng,
            )
            pick = pool[np.asarray(local, dtype=int)]
        for i in np.asarray(pick, dtype=int):
            selected.add(int(i))
            chosen.append(int(i))
    idx = np.asarray(chosen, dtype=int)
    dest = out_dir / name / f"seed_{args.seed}" / name
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / "train_indices.npy", idx)
    return idx


def _load_indices(out_dir: Path, mode: str, seed: int) -> np.ndarray:
    path = out_dir / mode / f"seed_{seed}" / mode / "train_indices.npy"
    return np.load(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task-csv", type=str,
        default=str(_ROOT / "data" / "kitti_vkitti" / "paired_detection_task_linked.csv"),
    )
    p.add_argument("--output-dir", type=str, default="outputs/kitti_vkitti/demo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-init", type=int, default=8)
    p.add_argument("--n-proxy-only", type=int, default=24)
    p.add_argument("--n-acq-iters", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--mi-shortlist", type=int, default=40)
    p.add_argument("--bnn-epochs", type=int, default=40)
    p.add_argument("--cv-n-jobs", type=int, default=2)
    args = p.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _df, _x, y_target, y_proxy, _cols = _load_task(Path(args.task_csv), scenario_space="full")
    initial_idx, _proxy_only = _shared_init(
        len(y_target), args.n_init, args.n_proxy_only, args.seed,
        out_dir / "init" / f"seed_{args.seed}",
    )

    scout_dir = out_dir / "runs"
    _run_scout_cli("with_beta", [], args, scout_dir)
    _run_scout_cli("real_only", [], args, scout_dir)
    # MI-only must not share with_beta's output directory.
    mi_args_dir = scout_dir / "mi_only_stage"
    cmd_mode_dir = mi_args_dir
    cmd = [
        sys.executable, str(_ROOT / "scripts" / "kitti_vkitti" / "run_scout.py"),
        "--mode", "with_beta",
        "--acq-score", "mi",
        "--no-surrogate",
        "--task-csv", str(args.task_csv),
        "--output-dir", str(cmd_mode_dir),
        "--init-cache-dir", str(out_dir / "init"),
        "--seed", str(args.seed),
        "--n-init", str(args.n_init),
        "--n-proxy-only", str(args.n_proxy_only),
        "--n-acq-iters", str(args.n_acq_iters),
        "--batch-acq-size", str(args.batch_size),
        "--mi-shortlist", str(args.mi_shortlist),
        "--bnn-epochs", str(args.bnn_epochs),
        "--cv-n-jobs", str(args.cv_n_jobs),
        "--cpu",
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=_ROOT)

    curves = {
        "scout": _cummean(y_target, _load_indices(scout_dir, "with_beta", args.seed)),
        "real_only": _cummean(y_target, _load_indices(scout_dir, "real_only", args.seed)),
        "mi_only": _cummean(y_target, np.load(cmd_mode_dir / f"seed_{args.seed}" / "with_beta" / "train_indices.npy")),
        "random": _cummean(y_target, _acquire_baseline("random", y_target, y_proxy, initial_idx, args, scout_dir)),
        "is": _cummean(y_target, _acquire_baseline("is", y_target, y_proxy, initial_idx, args, scout_dir)),
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for key, curve in curves.items():
        ax.plot(np.arange(1, len(curve) + 1), curve, color=_COLORS[key], lw=2.0, label=_LABELS[key])
    ax.axvline(args.n_init, color="k", ls=":", lw=1, label="end of shared init")
    ax.set_xlabel("Real labels")
    ax.set_ylabel("Cumulative mean target failure")
    ax.set_title("KITTI failure discovery")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_dir / "kitti_demo.png", dpi=140, bbox_inches="tight")
    fig.savefig(out_dir / "kitti_demo.pdf", bbox_inches="tight")
    summary = {k: float(v[-1]) for k, v in curves.items()}
    print("final cumulative mean target failure:")
    for k, v in summary.items():
        print(f"  {k:10s}  {v:.4f}")
    print(out_dir / "kitti_demo.png")
    return fig, summary


if __name__ == "__main__":
    main()
