#!/usr/bin/env python3
"""
Compare SCOUT with_beta vs real_only with richer metrics:
  - cumulative mean target failure
  - oracle regret (vs top-b true failures)
  - top-k failure recall
  - held-out surrogate MAE of a BNN fit on acquired labels

Example
-------
python scripts/kitti_vkitti/compare_beta_ablation.py \
    --scout-root outputs/kitti_vkitti/scout_v2 \
    --seeds 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from utils.kitti_vkitti.config import OUTPUT_ROOT
from utils.kitti_vkitti.io_utils import read_task_csv
from scout.bnn import HeteroBNNEmbedding, train_bnn


def _cummean(y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    vals = y[idx.astype(int)]
    return np.cumsum(vals) / np.arange(1, len(vals) + 1)


def _oracle_cummean(y: np.ndarray, budget: int) -> np.ndarray:
    order = np.argsort(-y)  # highest failure first
    return _cummean(y, order[:budget])


def _topk_recall(y: np.ndarray, idx: np.ndarray, frac: float = 0.1) -> np.ndarray:
    """At each prefix of idx, fraction of true top-frac failures acquired."""
    n = len(y)
    k = max(1, int(round(frac * n)))
    top_set = set(np.argsort(-y)[:k].tolist())
    out = np.zeros(len(idx))
    seen = set()
    for t, g in enumerate(idx.astype(int)):
        seen.add(int(g))
        out[t] = len(seen & top_set) / k
    return out


def _surrogate_mae(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    holdout_idx: np.ndarray,
    epochs: int = 600,
    device: str = "cpu",
) -> float:
    if len(train_idx) < 5 or len(holdout_idx) < 5:
        return float("nan")
    model = HeteroBNNEmbedding(input_dim=X.shape[1], p_drop=0.05).to(device)
    train_bnn(
        model,
        X[train_idx], y[train_idx],
        epochs=epochs, lr=1e-3, device=device,
    )
    model.eval()
    with torch.no_grad():
        mu, _ = model(torch.tensor(X[holdout_idx], dtype=torch.float32, device=device))
        pred = mu.squeeze(-1).cpu().numpy()
    return float(np.mean(np.abs(pred - y[holdout_idx])))


def main(args):
    df = read_task_csv(args.task_csv)
    y = df["target_failure"].to_numpy(float)
    rich = Path(args.rich_npy)
    if rich.exists() and np.load(rich).shape[0] == len(df):
        Xr = np.load(rich)
    else:
        from utils.kitti_vkitti.rich_features import write_rich_features
        Xr, _ = write_rich_features(task_csv=Path(args.task_csv), output_npy=rich)
    seqs = sorted(df["sequence_id"].astype(str).str.zfill(4).unique())
    onehot = np.stack(
        [(df["sequence_id"].astype(str).str.zfill(4) == s).to_numpy(float) for s in seqs],
        axis=1,
    )
    X = StandardScaler().fit_transform(
        np.nan_to_num(np.concatenate([Xr, onehot], 1), nan=0.0, posinf=0.0, neginf=0.0)
    )
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    root = Path(args.scout_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    curves = {m: [] for m in ("with_beta", "real_only")}
    regret = {m: [] for m in ("with_beta", "real_only")}
    recall = {m: [] for m in ("with_beta", "real_only")}
    mae_rows = []

    for seed in args.seeds:
        for mode in curves:
            p = root / f"seed_{seed}" / mode / "train_indices.npy"
            if not p.exists():
                print(f"[WARN] missing {p}")
                continue
            idx = np.load(p)
            cm = _cummean(y, idx)
            curves[mode].append(cm)
            budget = len(idx)
            oracle = _oracle_cummean(y, budget)
            # regret: oracle_cummean - method_cummean (positive => method worse at finding failures)
            # For failure discovery higher cummean is better, so regret = oracle - method
            regret[mode].append(oracle - cm)
            recall[mode].append(_topk_recall(y, idx, frac=args.topk_frac))

            # Held-out: everything not in train
            hold = np.setdiff1d(np.arange(len(y)), idx)
            rng = np.random.default_rng(seed)
            if len(hold) > 500:
                hold = rng.choice(hold, 500, replace=False)
            mae = _surrogate_mae(X, y, idx, hold, epochs=args.mae_epochs, device=device)
            mae_rows.append(dict(seed=seed, mode=mode, surrogate_mae=mae, budget=budget))
            print(f"seed={seed} {mode}: final_cum={cm[-1]:.4f}  "
                  f"final_regret={oracle[-1]-cm[-1]:.4f}  "
                  f"topk_recall={recall[mode][-1][-1]:.4f}  mae={mae:.4f}")

    summary = {}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    metrics_plot = [
        (curves, "Cumulative mean target failure", "cum_failure", axes[0], True),
        (regret, "Oracle regret (↓ better)", "oracle_regret", axes[1], False),
        (recall, f"Top-{args.topk_frac:.0%} failure recall (↑)", "topk_recall", axes[2], True),
    ]
    colors = {"with_beta": "#e41a1c", "real_only": "#377eb8"}
    labels = {"with_beta": "SCOUT (with β / proxy BNN)", "real_only": "SCOUT real-only (β=0)"}

    for store, title, key, ax, higher_better in metrics_plot:
        summary[key] = {}
        for mode in ("with_beta", "real_only"):
            arrs = store[mode]
            if not arrs:
                continue
            mlen = min(len(a) for a in arrs)
            M = np.stack([a[:mlen] for a in arrs], 0)
            mean, std = M.mean(0), M.std(0)
            xs = np.arange(1, mlen + 1)
            ax.plot(xs, mean, color=colors[mode], lw=2, label=labels[mode])
            ax.fill_between(xs, mean - std, mean + std, color=colors[mode], alpha=0.2)
            summary[key][mode] = dict(
                final_mean=float(mean[-1]), final_std=float(std[-1]),
                n_seeds=int(len(arrs)), budget=int(mlen),
            )
        if "with_beta" in summary[key] and "real_only" in summary[key]:
            gain = (
                summary[key]["with_beta"]["final_mean"]
                - summary[key]["real_only"]["final_mean"]
            )
            # For regret, lower is better → report real_only - with_beta as "gain"
            if key == "oracle_regret":
                gain = (
                    summary[key]["real_only"]["final_mean"]
                    - summary[key]["with_beta"]["final_mean"]
                )
            summary[key]["gain_with_beta_vs_real_only"] = float(gain)
        ax.set_xlabel("Real labels used")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "with_beta_vs_real_only.pdf")
    fig.savefig(out / "with_beta_vs_real_only.png", dpi=150)
    plt.close(fig)

    mae_df = pd.DataFrame(mae_rows)
    if len(mae_df):
        mae_df.to_csv(out / "surrogate_mae.csv", index=False)
        summary["surrogate_mae"] = (
            mae_df.groupby("mode")["surrogate_mae"].agg(["mean", "std", "count"]).to_dict()
        )

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task-csv", type=str,
                   default=str(OUTPUT_ROOT / "paired_detection_task.csv"))
    p.add_argument("--scout-root", type=str,
                   default=str(_ROOT / "outputs" / "kitti_vkitti" / "scout_v2"))
    p.add_argument("--output-dir", type=str,
                   default=str(_ROOT / "outputs" / "kitti_vkitti" / "ablation_beta_v2"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--topk-frac", type=float, default=0.10)
    p.add_argument("--mae-epochs", type=int, default=600)
    p.add_argument("--rich-npy", type=str,
                   default=str(OUTPUT_ROOT / "X_rich.npy"))
    p.add_argument("--cpu", action="store_true")
    main(p.parse_args())
