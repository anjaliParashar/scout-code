#!/usr/bin/env python3
"""
Ablate scenario-space definitions: show real_only is more sensitive than with_beta.

For each embedding (thin / thin_det / geom / detector / rich / full), run the
locked SCOUT protocol on seeds 0–2. Shared init across spaces for fairness.

Reports std / range of cum target-failure across spaces (mean over seeds).
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
OUT = Path("outputs/kitti_vkitti/ablation_scenario_space")
INIT_CACHE = OUT / "shared_init"


def _cum(run_dir: Path, y: np.ndarray) -> float:
    train = np.load(run_dir / "train_indices.npy")
    return float(y[train].mean())


def run_all(spaces, seeds, dry: bool = False):
    df = pd.read_csv(TASK)
    y = df["target_failure"].to_numpy(float)
    rows = []
    for space in spaces:
        for seed in seeds:
            for mode in ("with_beta", "real_only"):
                out = OUT / space
                done = out / f"seed_{seed}" / mode / "train_indices.npy"
                if not done.exists() and not dry:
                    cmd = [
                        PY, "scripts/kitti_vkitti/run_scout.py",
                        "--mode", mode, "--seed", str(seed),
                        "--task-csv", TASK,
                        "--output-dir", str(out),
                        "--scenario-space", space,
                        "--init-cache-dir", str(INIT_CACHE),
                        "--n-init", "10", "--n-proxy-only", "40",
                        "--n-acq-iters", "15", "--batch-acq-size", "5",
                        "--mi-shortlist", "50", "--proxy-shortlist", "500",
                        "--proxy-blend", "0.8", "--acq-pick", "cluster",
                    ]
                    print(f"[run] space={space} seed={seed} mode={mode}", flush=True)
                    subprocess.check_call(cmd, cwd=str(_ROOT))
                elif not done.exists():
                    raise FileNotFoundError(done)
                cum = _cum(out / f"seed_{seed}" / mode, y)
                rows.append(dict(space=space, seed=seed, mode=mode, cum_failure=cum))
                print(f"  → {space}/{seed}/{mode}: cum={cum:.4f}", flush=True)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    # Mean over seeds for each (space, mode)
    g = df.groupby(["space", "mode"], as_index=False)["cum_failure"].mean()
    pivot = g.pivot(index="space", columns="mode", values="cum_failure")
    # Preserve space order
    pivot = pivot.reindex([s for s in SCENARIO_SPACES if s in pivot.index])

    stats = {}
    for mode in ("with_beta", "real_only"):
        vals = pivot[mode].to_numpy(float)
        stats[mode] = dict(
            mean=float(vals.mean()),
            std_across_spaces=float(vals.std(ddof=1)),
            range=float(vals.max() - vals.min()),
            min=float(vals.min()),
            max=float(vals.max()),
            by_space={s: float(pivot.loc[s, mode]) for s in pivot.index},
        )
    stats["variance_ratio_real_only_over_with_beta"] = (
        stats["real_only"]["std_across_spaces"] / max(stats["with_beta"]["std_across_spaces"], 1e-12)
    )
    stats["range_ratio_real_only_over_with_beta"] = (
        stats["real_only"]["range"] / max(stats["with_beta"]["range"], 1e-12)
    )
    return stats, pivot


def plot(pivot: pd.DataFrame, stats: dict, out_pdf: Path):
    spaces = list(pivot.index)
    x = np.arange(len(spaces))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(x - w / 2, pivot["with_beta"], w, label="with_β", color="#2c7fb8")
    ax.bar(x + w / 2, pivot["real_only"], w, label="real-only (β=0)", color="#f03b20")
    ax.set_xticks(x)
    ax.set_xticklabels(spaces, rotation=20, ha="right")
    ax.set_ylabel("Cum. mean target failure ↑")
    ax.set_xlabel("Scenario-space definition")
    ax.set_title(
        "Sensitivity to scenario space\n"
        f"std across spaces: real-only={stats['real_only']['std_across_spaces']:.3f}, "
        f"with_β={stats['with_beta']['std_across_spaces']:.3f} "
        f"(ratio {stats['variance_ratio_real_only_over_with_beta']:.1f}×)"
    )
    ax.legend(frameon=False)
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
    INIT_CACHE.mkdir(parents=True, exist_ok=True)

    df = run_all(args.spaces, args.seeds, dry=args.dry_summarize)
    df.to_csv(OUT / "per_run.csv", index=False)
    stats, pivot = summarize(df)
    pivot.to_csv(OUT / "by_space.csv")
    (OUT / "summary.json").write_text(json.dumps(stats, indent=2))
    plot(pivot, stats, OUT / "scenario_space_sensitivity.pdf")

    print("\n=== Scenario-space sensitivity ===")
    print(pivot.round(4).to_string())
    print(
        f"\nstd across spaces — with_β: {stats['with_beta']['std_across_spaces']:.4f}  "
        f"real_only: {stats['real_only']['std_across_spaces']:.4f}  "
        f"ratio: {stats['variance_ratio_real_only_over_with_beta']:.2f}×"
    )
    print(
        f"range across spaces — with_β: {stats['with_beta']['range']:.4f}  "
        f"real_only: {stats['real_only']['range']:.4f}  "
        f"ratio: {stats['range_ratio_real_only_over_with_beta']:.2f}×"
    )
    print(f"Wrote {OUT}/summary.json and scenario_space_sensitivity.pdf")


if __name__ == "__main__":
    main()
