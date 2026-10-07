#!/usr/bin/env python3
"""Seed-0 hyperparameter sweep to maximize with_beta − real_only cum-failure gap."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
TASK = "data/kitti_vkitti/paired_detection_task_linked.csv"
OUT_ROOT = Path("outputs/kitti_vkitti/sweep_gap_s0")

# (name, kwargs)
CFGS = [
    ("baseline", dict(proxy_blend=0.8, mi_shortlist=50, proxy_shortlist=500,
                      acq_pick="cluster", beta_scale=1.0, n_acq_iters=20)),
    ("greedy_b08", dict(proxy_blend=0.8, mi_shortlist=50, proxy_shortlist=500,
                        acq_pick="greedy", beta_scale=1.0, n_acq_iters=20)),
    ("greedy_b10", dict(proxy_blend=1.0, mi_shortlist=50, proxy_shortlist=500,
                        acq_pick="greedy", beta_scale=1.0, n_acq_iters=20)),
    ("greedy_proxy_only", dict(proxy_blend=1.0, mi_shortlist=0, proxy_shortlist=500,
                               acq_pick="greedy", beta_scale=1.0, n_acq_iters=20)),
    ("greedy_b10_mi20", dict(proxy_blend=1.0, mi_shortlist=20, proxy_shortlist=800,
                             acq_pick="greedy", beta_scale=1.0, n_acq_iters=20)),
    ("cluster_b10_bs15", dict(proxy_blend=1.0, mi_shortlist=0, proxy_shortlist=500,
                              acq_pick="cluster", beta_scale=1.5, n_acq_iters=15)),
    ("greedy_b10_i15", dict(proxy_blend=1.0, mi_shortlist=0, proxy_shortlist=500,
                            acq_pick="greedy", beta_scale=1.0, n_acq_iters=15)),
    ("greedy_b09_scale2", dict(proxy_blend=0.9, mi_shortlist=20, proxy_shortlist=500,
                               acq_pick="greedy", beta_scale=2.0, n_acq_iters=20)),
]


def cum_mean(log_csv: Path, n_init: int) -> float:
    log = pd.read_csv(log_csv)
    sel = log[log.is_selected == 1].sort_values("iteration")
    # train set ≈ init (not in log as selected) + selected; use selected mean
    # Match compare script: load train_indices
    return float("nan")


def metrics(run_dir: Path, task_csv: Path) -> dict:
    df = pd.read_csv(task_csv)
    # preserve string ids
    y = df["target_failure"].to_numpy(float)
    train = np.load(run_dir / "train_indices.npy")
    cum = float(y[train].mean())
    log = pd.read_csv(run_dir / "acquisition_log.csv")
    sel = log[log.is_selected == 1]
    thr = np.quantile(y, 0.9)
    top = set(np.where(y >= thr)[0].tolist())
    recall = len(set(sel.global_idx.astype(int)) & top) / max(1, len(top))
    return dict(
        cum=cum,
        sel_tgt=float(sel.target.mean()) if len(sel) else float("nan"),
        sel_proxy=float(sel.proxy.mean()) if len(sel) else float("nan"),
        recall=float(recall),
        n_train=int(len(train)),
    )


def run_one(name: str, cfg: dict) -> dict:
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    # skip if both modes done
    wb = out / "seed_0" / "with_beta" / "train_indices.npy"
    ro = out / "seed_0" / "real_only" / "train_indices.npy"
    common = [
        PY, "scripts/kitti_vkitti/run_scout.py",
        "--seed", "0",
        "--task-csv", TASK,
        "--output-dir", str(out),
        "--n-init", "10",
        "--n-proxy-only", "40",
        "--batch-acq-size", "5",
        "--n-acq-iters", str(cfg["n_acq_iters"]),
        "--mi-shortlist", str(cfg["mi_shortlist"]),
        "--proxy-shortlist", str(cfg["proxy_shortlist"]),
        "--proxy-blend", str(cfg["proxy_blend"]),
        "--beta-scale", str(cfg["beta_scale"]),
        "--acq-pick", cfg["acq_pick"],
    ]
    for mode in ("with_beta", "real_only"):
        done = out / "seed_0" / mode / "train_indices.npy"
        if done.exists():
            print(f"[skip] {name}/{mode}", flush=True)
            continue
        cmd = common + ["--mode", mode]
        print(f"[run] {name}/{mode}: {cfg}", flush=True)
        subprocess.check_call(cmd, cwd=str(_ROOT))
    m_wb = metrics(out / "seed_0" / "with_beta", Path(TASK))
    m_ro = metrics(out / "seed_0" / "real_only", Path(TASK))
    row = dict(name=name, **{f"wb_{k}": v for k, v in m_wb.items()},
               **{f"ro_{k}": v for k, v in m_ro.items()},
               gain=m_wb["cum"] - m_ro["cum"],
               recall_gain=m_wb["recall"] - m_ro["recall"],
               **cfg)
    print(f"[done] {name} gain={row['gain']:.4f}  "
          f"wb={m_wb['cum']:.4f} ro={m_ro['cum']:.4f}  "
          f"recallΔ={row['recall_gain']:.3f}", flush=True)
    return row


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, cfg in CFGS:
        rows.append(run_one(name, cfg))
        pd.DataFrame(rows).to_csv(OUT_ROOT / "summary.csv", index=False)
    rows = sorted(rows, key=lambda r: -r["gain"])
    best = rows[0]
    (OUT_ROOT / "best.json").write_text(json.dumps(best, indent=2))
    print("\n=== RANKED BY GAIN ===", flush=True)
    for r in rows:
        print(f"  {r['name']:20s}  gain={r['gain']:+.4f}  "
              f"wb={r['wb_cum']:.4f} ro={r['ro_cum']:.4f}  "
              f"recallΔ={r['recall_gain']:+.3f}", flush=True)
    print("BEST →", best["name"], flush=True)


if __name__ == "__main__":
    main()
