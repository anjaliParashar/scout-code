#!/usr/bin/env python3
"""
Ablate proxy channel: sim y_proxy vs target-BNN self-draws vs β=0.

For each scenario space, compare:
  with_beta  — nominal (sim proxy)
  bnn_proxy  — same CV/shortlist/blend machinery, g = target-BNN mean
  real_only  — β=0

Reuses existing with_beta / real_only runs under ablation_scenario_space/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from scripts.kitti_vkitti.run_scout import SCENARIO_SPACES  # noqa: E402

PY = sys.executable
TASK = "data/kitti_vkitti/paired_detection_task_linked.csv"
# Reuse prior space ablation for with_beta / real_only
BASE = Path("outputs/kitti_vkitti/ablation_scenario_space")
OUT = Path("outputs/kitti_vkitti/ablation_bnn_proxy")
INIT_CACHE = BASE / "shared_init"
MODES = ("with_beta", "bnn_proxy", "real_only")
MODE_LABEL = {
    "with_beta": "nominal (sim proxy)",
    "bnn_proxy": "BNN self-proxy",
    "real_only": "β=0 (real-only)",
}


def _cum(run_dir: Path, y: np.ndarray) -> float:
    return float(y[np.load(run_dir / "train_indices.npy")].mean())


def _run_dir(space: str, seed: int, mode: str) -> Path:
    if mode == "bnn_proxy":
        return OUT / space / f"seed_{seed}" / mode
    return BASE / space / f"seed_{seed}" / mode


def run_all(spaces, seeds, dry: bool = False):
    y = pd.read_csv(TASK)["target_failure"].to_numpy(float)
    rows = []
    for space in spaces:
        for seed in seeds:
            for mode in MODES:
                done = _run_dir(space, seed, mode) / "train_indices.npy"
                if mode == "bnn_proxy" and not done.exists() and not dry:
                    out = OUT / space
                    cmd = [
                        PY, "scripts/kitti_vkitti/run_scout.py",
                        "--mode", "bnn_proxy", "--seed", str(seed),
                        "--task-csv", TASK,
                        "--output-dir", str(out),
                        "--scenario-space", space,
                        "--init-cache-dir", str(INIT_CACHE),
                        "--n-init", "10", "--n-proxy-only", "40",
                        "--n-acq-iters", "15", "--batch-acq-size", "5",
                        "--mi-shortlist", "50", "--proxy-shortlist", "500",
                        "--proxy-blend", "0.8", "--acq-pick", "cluster",
                    ]
                    print(f"[run] space={space} seed={seed} mode=bnn_proxy", flush=True)
                    subprocess.check_call(cmd, cwd=str(_ROOT))
                elif not done.exists():
                    raise FileNotFoundError(done)
                cum = _cum(_run_dir(space, seed, mode), y)
                rows.append(dict(space=space, seed=seed, mode=mode, cum_failure=cum))
                print(f"  → {space}/{seed}/{mode}: cum={cum:.4f}", flush=True)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame):
    g = df.groupby(["space", "mode"], as_index=False)["cum_failure"].mean()
    pivot = g.pivot(index="space", columns="mode", values="cum_failure")
    pivot = pivot.reindex([s for s in SCENARIO_SPACES if s in pivot.index])
    pivot = pivot[[m for m in MODES if m in pivot.columns]]

    stats = {}
    for mode in pivot.columns:
        vals = pivot[mode].to_numpy(float)
        stats[mode] = dict(
            mean=float(vals.mean()),
            std_across_spaces=float(vals.std(ddof=1)),
            range=float(vals.max() - vals.min()),
            min=float(vals.min()),
            max=float(vals.max()),
            by_space={s: float(pivot.loc[s, mode]) for s in pivot.index},
        )
    # Gaps vs real_only / vs nominal
    stats["gaps"] = {
        "nominal_minus_real_only": float((pivot["with_beta"] - pivot["real_only"]).mean()),
        "bnn_proxy_minus_real_only": float((pivot["bnn_proxy"] - pivot["real_only"]).mean()),
        "nominal_minus_bnn_proxy": float((pivot["with_beta"] - pivot["bnn_proxy"]).mean()),
    }
    return stats, pivot


def plot(pivot: pd.DataFrame, stats: dict, out_pdf: Path):
    spaces = list(pivot.index)
    x = np.arange(len(spaces))
    w = 0.25
    colors = {"with_beta": "#2c7fb8", "bnn_proxy": "#fdae61", "real_only": "#f03b20"}
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    for i, mode in enumerate(MODES):
        ax.bar(x + (i - 1) * w, pivot[mode], w, label=MODE_LABEL[mode], color=colors[mode])
    ax.set_xticks(x)
    ax.set_xticklabels(spaces, rotation=20, ha="right")
    ax.set_ylabel("Cum. mean target failure ↑")
    ax.set_xlabel("Scenario-space definition")
    ax.set_title(
        "Sim proxy vs BNN self-proxy vs β=0\n"
        f"mean gap vs β=0: nominal "
        f"+{stats['gaps']['nominal_minus_real_only']:.3f}, "
        f"BNN-proxy {stats['gaps']['bnn_proxy_minus_real_only']:+.3f}"
    )
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spaces", nargs="+", default=list(SCENARIO_SPACES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--dry-summarize", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df = run_all(args.spaces, args.seeds, dry=args.dry_summarize)
    df.to_csv(OUT / "per_run.csv", index=False)
    stats, pivot = summarize(df)
    pivot.to_csv(OUT / "by_space.csv")
    (OUT / "summary.json").write_text(json.dumps(stats, indent=2))
    plot(pivot, stats, OUT / "bnn_proxy_vs_nominal.pdf")

    print("\n=== Sim proxy vs BNN self-proxy vs β=0 ===")
    print(pivot.rename(columns=MODE_LABEL).round(4).to_string())
    print("\nstd across spaces:")
    for m in MODES:
        print(f"  {MODE_LABEL[m]:24s}  {stats[m]['std_across_spaces']:.4f}")
    print("\nmean gaps (cum failure):")
    for k, v in stats["gaps"].items():
        print(f"  {k}: {v:+.4f}")
    print(f"Wrote {OUT}/summary.json and bnn_proxy_vs_nominal.pdf")


if __name__ == "__main__":
    main()
