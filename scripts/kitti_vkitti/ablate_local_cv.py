#!/usr/bin/env python3
"""
Local-CV SCOUT ablations on KITTI–VKITTI across scenario spaces.

Modes: with_beta (sim y_proxy), offline_proxy, bnn_proxy, real_only.
Writes cum-failure and high-severity tables under ablation_local_cv/.
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
DEFAULT_OUT = Path("outputs/kitti_vkitti/ablation_local_cv")
MODES = ("with_beta", "offline_proxy", "bnn_proxy", "real_only")
MODE_LABEL = {
    "with_beta": "nominal (sim)",
    "offline_proxy": "offline proxy ŷ",
    "bnn_proxy": "BNN self-proxy",
    "real_only": "β=0",
}
HIGH_THR = 0.3


def run_all(
    spaces,
    seeds,
    out: Path,
    dry: bool = False,
    bnn_epochs: int = 800,
    dropout: float = 0.05,
    bnn_widths: list[int] | None = None,
    n_init: int = 10,
    n_proxy_only: int | None = None,
    mi_shortlist: int = 1000,
    sev_cutoff: float = 0.3,
    high_thr: float = HIGH_THR,
    acq_pick: str = "cluster",
    acq_score: str = "cv",
    no_mi: bool = False,
):
    y = pd.read_csv(TASK)["target_failure"].to_numpy(float)
    if n_proxy_only is None:
        n_proxy_only = 4 * n_init
    init_cache = out / "shared_init"
    init_cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for space in spaces:
        for seed in seeds:
            for mode in MODES:
                done = out / space / f"seed_{seed}" / mode / "train_indices.npy"
                if not done.exists() and not dry:
                    cmd = [
                        PY, "scripts/kitti_vkitti/run_scout.py",
                        "--mode", mode, "--seed", str(seed),
                        "--task-csv", TASK,
                        "--output-dir", str(out / space),
                        "--scenario-space", space,
                        "--init-cache-dir", str(init_cache),
                        "--n-init", str(n_init),
                        "--n-proxy-only", str(n_proxy_only),
                        "--n-acq-iters", "15", "--batch-acq-size", "5",
                        "--mi-shortlist", str(mi_shortlist),
                        "--sev-cutoff", str(sev_cutoff),
                        "--acq-pick", acq_pick,
                        "--acq-score", acq_score,
                        "--n-pair-local", "10", "--k-unpaired", "20",
                        "--cv-radius", "1.5", "--cv-fallback-k", "80",
                        "--offline-proxy-frac", "1.0",
                        "--bnn-epochs", str(bnn_epochs),
                        "--dropout", str(dropout),
                    ]
                    if no_mi:
                        cmd.append("--no-mi")
                    if bnn_widths:
                        cmd += ["--bnn-widths", *[str(w) for w in bnn_widths]]
                    print(f"[run] {space}/{seed}/{mode} n_init={n_init}", flush=True)
                    subprocess.check_call(cmd, cwd=str(_ROOT))
                elif not done.exists():
                    raise FileNotFoundError(done)
                train = np.load(done)
                yh = y[train]
                rows.append(dict(
                    space=space, seed=seed, mode=mode,
                    n_init=n_init, n_labelled=int(len(train)),
                    cum_failure=float(yh.mean()),
                    high_sev_rate=float((yh >= high_thr).mean()),
                    n_high=int((yh >= high_thr).sum()),
                ))
                print(
                    f"  → cum={yh.mean():.4f} high={(yh >= high_thr).mean():.3f} "
                    f"n={len(train)}",
                    flush=True,
                )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame):
    stats, pivots = {}, {}
    for metric in ("cum_failure", "high_sev_rate"):
        g = df.groupby(["space", "mode"], as_index=False)[metric].mean()
        pivot = g.pivot(index="space", columns="mode", values=metric)
        pivot = pivot.reindex([s for s in SCENARIO_SPACES if s in pivot.index])
        pivot = pivot[[m for m in MODES if m in pivot.columns]]
        pivots[metric] = pivot
        ms = {}
        for mode in pivot.columns:
            vals = pivot[mode].to_numpy(float)
            ms[mode] = dict(
                mean=float(vals.mean()),
                std_across_spaces=float(vals.std(ddof=1)),
                range=float(vals.max() - vals.min()),
                by_space={s: float(pivot.loc[s, mode]) for s in pivot.index},
            )
        ms["gaps_vs_real_only"] = {
            m: float((pivot[m] - pivot["real_only"]).mean())
            for m in pivot.columns if m != "real_only"
        }
        stats[metric] = ms
    return stats, pivots


def plot(pivots, out_pdf: Path):
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
        (axes[1], "high_sev_rate", f"Frac. high-sev (y≥{HIGH_THR}) ↑", "High severity"),
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
    fig.suptitle("Local-CV SCOUT proxy-channel ablation", y=1.02)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spaces", nargs="+", default=list(SCENARIO_SPACES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--bnn-epochs", type=int, default=800)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--bnn-widths", type=int, nargs="+", default=None)
    ap.add_argument("--n-init", type=int, default=10)
    ap.add_argument("--n-proxy-only", type=int, default=None,
                    help="Default 4× n_init")
    ap.add_argument("--mi-shortlist", type=int, default=1000)
    ap.add_argument("--sev-cutoff", type=float, default=0.3)
    ap.add_argument("--high-thr", type=float, default=HIGH_THR,
                    help="Eval threshold for high-severity rate")
    ap.add_argument("--acq-pick", choices=["cluster", "greedy"], default="cluster")
    ap.add_argument("--acq-score", choices=["cv", "mi", "micv"], default="cv")
    ap.add_argument("--no-mi", action="store_true")
    ap.add_argument("--dry-summarize", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = run_all(
        args.spaces, args.seeds, out=out, dry=args.dry_summarize,
        bnn_epochs=args.bnn_epochs, dropout=args.dropout,
        bnn_widths=args.bnn_widths,
        n_init=args.n_init, n_proxy_only=args.n_proxy_only,
        mi_shortlist=args.mi_shortlist, sev_cutoff=args.sev_cutoff,
        high_thr=args.high_thr, acq_pick=args.acq_pick,
        acq_score=args.acq_score, no_mi=args.no_mi,
    )
    df.to_csv(out / "per_run.csv", index=False)
    stats, pivots = summarize(df)
    for metric, pivot in pivots.items():
        pivot.to_csv(out / f"by_space_{metric}.csv")
    (out / "summary.json").write_text(json.dumps(dict(
        cv="local_scout", high_severity_threshold=args.high_thr,
        sev_cutoff=args.sev_cutoff, mi_shortlist=args.mi_shortlist,
        acq_pick=args.acq_pick, acq_score=args.acq_score, no_mi=args.no_mi,
        bnn_epochs=args.bnn_epochs, dropout=args.dropout,
        bnn_widths=args.bnn_widths,
        n_init=args.n_init,
        n_proxy_only=args.n_proxy_only if args.n_proxy_only is not None else 4 * args.n_init,
        **stats,
    ), indent=2))
    plot(pivots, out / "local_cv_ablation.pdf")

    print("\n=== Cum. mean target failure (local CV) ===")
    print(pivots["cum_failure"].rename(columns=MODE_LABEL).round(4).to_string())
    print(f"\n=== High-severity rate (y≥{args.high_thr}) ===")
    print(pivots["high_sev_rate"].rename(columns=MODE_LABEL).round(4).to_string())
    print("\nMean gaps vs β=0 (cum):")
    for m, v in stats["cum_failure"]["gaps_vs_real_only"].items():
        print(f"  {MODE_LABEL[m]:20s}  {v:+.4f}")
    print("Mean gaps vs β=0 (high-sev):")
    for m, v in stats["high_sev_rate"]["gaps_vs_real_only"].items():
        print(f"  {MODE_LABEL[m]:20s}  {v:+.4f}")
    print(f"Wrote {out}/summary.json")


if __name__ == "__main__":
    main()
