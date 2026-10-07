#!/usr/bin/env python3
"""
Ablate proxy channel across scenario spaces:

  with_beta      — true sim y_proxy (nominal)
  offline_proxy  — frozen BNN fit offline on (X, y_proxy), then g=ŷ_proxy
  bnn_proxy      — g = online target-BNN mean
  real_only      — β=0

Reuses prior with_beta / bnn_proxy / real_only artifacts; runs offline_proxy only.
Also reports high-severity (y_target >= 0.5) discovery rates.
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
BASE = Path("outputs/kitti_vkitti/ablation_scenario_space")
BNN = Path("outputs/kitti_vkitti/ablation_bnn_proxy")
OUT = Path("outputs/kitti_vkitti/ablation_offline_proxy")
INIT_CACHE = BASE / "shared_init"
MODES = ("with_beta", "offline_proxy", "bnn_proxy", "real_only")
MODE_LABEL = {
    "with_beta": "nominal (sim)",
    "offline_proxy": "offline proxy ŷ",
    "bnn_proxy": "BNN self-proxy",
    "real_only": "β=0",
}
HIGH_THR = 0.5


def _run_dir(space: str, seed: int, mode: str) -> Path:
    if mode == "offline_proxy":
        return OUT / space / f"seed_{seed}" / mode
    if mode == "bnn_proxy":
        return BNN / space / f"seed_{seed}" / mode
    return BASE / space / f"seed_{seed}" / mode


def run_all(spaces, seeds, offline_frac: float, dry: bool = False):
    y = pd.read_csv(TASK)["target_failure"].to_numpy(float)
    rows = []
    for space in spaces:
        for seed in seeds:
            for mode in MODES:
                done = _run_dir(space, seed, mode) / "train_indices.npy"
                if mode == "offline_proxy" and not done.exists() and not dry:
                    out = OUT / space
                    cmd = [
                        PY, "scripts/kitti_vkitti/run_scout.py",
                        "--mode", "offline_proxy", "--seed", str(seed),
                        "--task-csv", TASK,
                        "--output-dir", str(out),
                        "--scenario-space", space,
                        "--init-cache-dir", str(INIT_CACHE),
                        "--offline-proxy-frac", str(offline_frac),
                        "--n-init", "10", "--n-proxy-only", "40",
                        "--n-acq-iters", "15", "--batch-acq-size", "5",
                        "--mi-shortlist", "50", "--proxy-shortlist", "500",
                        "--proxy-blend", "0.8", "--acq-pick", "cluster",
                    ]
                    print(f"[run] space={space} seed={seed} mode=offline_proxy", flush=True)
                    subprocess.check_call(cmd, cwd=str(_ROOT))
                elif not done.exists():
                    raise FileNotFoundError(done)
                train = np.load(_run_dir(space, seed, mode) / "train_indices.npy")
                yh = y[train]
                rows.append(dict(
                    space=space, seed=seed, mode=mode,
                    cum_failure=float(yh.mean()),
                    high_sev_rate=float((yh >= HIGH_THR).mean()),
                    n_high=int((yh >= HIGH_THR).sum()),
                ))
                print(
                    f"  → {space}/{seed}/{mode}: cum={yh.mean():.4f} "
                    f"high≥{HIGH_THR}={(yh >= HIGH_THR).mean():.3f}",
                    flush=True,
                )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame):
    stats = {}
    pivots = {}
    for metric in ("cum_failure", "high_sev_rate"):
        g = df.groupby(["space", "mode"], as_index=False)[metric].mean()
        pivot = g.pivot(index="space", columns="mode", values=metric)
        pivot = pivot.reindex([s for s in SCENARIO_SPACES if s in pivot.index])
        pivot = pivot[[m for m in MODES if m in pivot.columns]]
        pivots[metric] = pivot
        mode_stats = {}
        for mode in pivot.columns:
            vals = pivot[mode].to_numpy(float)
            mode_stats[mode] = dict(
                mean=float(vals.mean()),
                std_across_spaces=float(vals.std(ddof=1)),
                range=float(vals.max() - vals.min()),
                by_space={s: float(pivot.loc[s, mode]) for s in pivot.index},
            )
        mode_stats["gaps_vs_real_only"] = {
            m: float((pivot[m] - pivot["real_only"]).mean())
            for m in pivot.columns if m != "real_only"
        }
        mode_stats["gaps_vs_nominal"] = {
            m: float((pivot["with_beta"] - pivot[m]).mean())
            for m in pivot.columns if m != "with_beta"
        }
        stats[metric] = mode_stats
    return stats, pivots


def plot(pivots, stats, out_pdf: Path):
    spaces = list(pivots["cum_failure"].index)
    x = np.arange(len(spaces))
    w = 0.2
    colors = {
        "with_beta": "#2c7fb8",
        "offline_proxy": "#31a354",
        "bnn_proxy": "#fdae61",
        "real_only": "#f03b20",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for ax, metric, ylab, title in [
        (axes[0], "cum_failure", "Cum. mean target failure ↑", "All severities"),
        (axes[1], "high_sev_rate", f"Frac. high-sev (y≥{HIGH_THR}) ↑", "High severity only"),
    ]:
        pivot = pivots[metric]
        for i, mode in enumerate(MODES):
            ax.bar(x + (i - 1.5) * w, pivot[mode], w, label=MODE_LABEL[mode], color=colors[mode])
        ax.set_xticks(x)
        ax.set_xticklabels(spaces, rotation=20, ha="right")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Proxy channel ablation across scenario spaces", y=1.02)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spaces", nargs="+", default=list(SCENARIO_SPACES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--offline-proxy-frac", type=float, default=1.0)
    ap.add_argument("--dry-summarize", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df = run_all(args.spaces, args.seeds, args.offline_proxy_frac, dry=args.dry_summarize)
    df.to_csv(OUT / "per_run.csv", index=False)
    stats, pivots = summarize(df)
    for metric, pivot in pivots.items():
        pivot.to_csv(OUT / f"by_space_{metric}.csv")
    (OUT / "summary.json").write_text(json.dumps(dict(
        offline_proxy_frac=args.offline_proxy_frac,
        high_severity_threshold=HIGH_THR,
        **stats,
    ), indent=2))
    plot(pivots, stats, OUT / "offline_proxy_ablation.pdf")

    print("\n=== Cum. mean target failure ===")
    print(pivots["cum_failure"].rename(columns=MODE_LABEL).round(4).to_string())
    print("\n=== High-severity rate (y≥0.5) ===")
    print(pivots["high_sev_rate"].rename(columns=MODE_LABEL).round(4).to_string())
    print("\nMean gaps vs β=0 (cum failure):")
    for m, v in stats["cum_failure"]["gaps_vs_real_only"].items():
        print(f"  {MODE_LABEL[m]:20s}  {v:+.4f}")
    print("Mean gaps vs β=0 (high-sev rate):")
    for m, v in stats["high_sev_rate"]["gaps_vs_real_only"].items():
        print(f"  {MODE_LABEL[m]:20s}  {v:+.4f}")
    print(f"Wrote {OUT}/summary.json")


if __name__ == "__main__":
    main()
