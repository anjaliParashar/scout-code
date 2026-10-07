#!/usr/bin/env python3
"""
Sweep KITTI hyperparams under FIXED local-CV SCOUT algorithm.
Maximize with_beta − real_only cum-failure gap on seed 0 / full space.
"""

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
OUT = Path("outputs/kitti_vkitti/sweep_localcv_gap")
INIT = OUT / "shared_init"

# name -> CLI kwargs (beyond shared)
CFGS = [
    ("baseline", dict()),
    # CV neighbourhood / pairing size
    ("cv_r3_k40", dict(cv_radius=3.0, k_unpaired=40, n_pair_local=20, cv_fallback_k=120)),
    ("cv_r5_k60", dict(cv_radius=5.0, k_unpaired=60, n_pair_local=30, cv_fallback_k=160)),
    ("cv_r0.8_k20", dict(cv_radius=0.8, k_unpaired=20, n_pair_local=10)),
    # BNN capacity / training
    ("bnn_ep2000_drop01", dict(bnn_epochs=2000, dropout=0.10)),
    ("bnn_ep2000_drop02", dict(bnn_epochs=2000, dropout=0.20)),
    ("bnn_wide", dict(bnn_widths=[192, 64, 16], bnn_epochs=1200, dropout=0.10)),
    ("bnn_deep", dict(bnn_widths=[128, 64, 32, 8], bnn_epochs=1200, dropout=0.10)),
    ("bnn_narrow_ep2800", dict(bnn_widths=[64, 16, 4], bnn_epochs=2800, dropout=0.05)),
    # AL / MI shortlist
    ("mi_500", dict(mi_shortlist=500)),
    ("mi_1000_r3", dict(mi_shortlist=1000, cv_radius=3.0, k_unpaired=40, n_pair_local=20)),
    # longer budget
    ("iters25_r3", dict(n_acq_iters=25, cv_radius=3.0, k_unpaired=40, n_pair_local=20, bnn_epochs=1200)),
    # scenario spaces (same local-CV algorithm)
    ("space_geom", dict(scenario_space="geom")),
    ("space_rich", dict(scenario_space="rich")),
    ("space_thin_det", dict(scenario_space="thin_det")),
    ("space_geom_r5", dict(scenario_space="geom", cv_radius=5.0, k_unpaired=60, n_pair_local=30, cv_fallback_k=160)),
    # combo: wide BNN + big neighbourhood + more draws
    ("combo_wide_r5", dict(
        bnn_widths=[192, 64, 16], bnn_epochs=1500, dropout=0.10,
        cv_radius=5.0, k_unpaired=60, n_pair_local=30, cv_fallback_k=160,
        mi_shortlist=500,
    )),
]


def cum(run_dir: Path, y: np.ndarray) -> float:
    return float(y[np.load(run_dir / "train_indices.npy")].mean())


def run_cfg(name: str, cfg: dict, y: np.ndarray) -> dict:
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    shared = dict(
        n_init=10, n_proxy_only=40, n_acq_iters=15, batch_acq_size=5,
        mi_shortlist=200, acq_pick="cluster",
        n_pair_local=10, k_unpaired=20, cv_radius=1.5, cv_fallback_k=80,
        bnn_epochs=800, dropout=0.05, lr=1e-3,
        scenario_space="full",
    )
    shared.update(cfg)
    for mode in ("with_beta", "real_only"):
        done = out / f"seed_0" / mode / "train_indices.npy"
        if done.exists():
            print(f"[skip] {name}/{mode}", flush=True)
            continue
        cmd = [
            PY, "scripts/kitti_vkitti/run_scout.py",
            "--mode", mode, "--seed", "0",
            "--task-csv", TASK,
            "--output-dir", str(out),
            "--scenario-space", shared["scenario_space"],
            "--init-cache-dir", str(INIT),
            "--n-init", str(shared["n_init"]),
            "--n-proxy-only", str(shared["n_proxy_only"]),
            "--n-acq-iters", str(shared["n_acq_iters"]),
            "--batch-acq-size", str(shared["batch_acq_size"]),
            "--mi-shortlist", str(shared["mi_shortlist"]),
            "--acq-pick", shared["acq_pick"],
            "--n-pair-local", str(shared["n_pair_local"]),
            "--k-unpaired", str(shared["k_unpaired"]),
            "--cv-radius", str(shared["cv_radius"]),
            "--cv-fallback-k", str(shared["cv_fallback_k"]),
            "--bnn-epochs", str(shared["bnn_epochs"]),
            "--dropout", str(shared["dropout"]),
            "--lr", str(shared["lr"]),
        ]
        if shared.get("bnn_widths"):
            cmd += ["--bnn-widths", *[str(w) for w in shared["bnn_widths"]]]
        print(f"[run] {name}/{mode} {cfg}", flush=True)
        subprocess.check_call(cmd, cwd=str(_ROOT))
    wb = cum(out / "seed_0" / "with_beta", y)
    ro = cum(out / "seed_0" / "real_only", y)
    log = pd.read_csv(out / "seed_0" / "with_beta" / "acquisition_log.csv")
    sel = log[log.is_selected == 1]
    yt = pd.read_csv(_ROOT / TASK)["target_failure"].to_numpy(float)
    row = dict(
        name=name, cum_wb=wb, cum_ro=ro, gain=wb - ro,
        mean_abs_beta=float(sel.cv_beta.abs().mean()),
        mean_corr=float(sel.cv_corr.mean()),
        high_wb=float((yt[np.load(out / "seed_0" / "with_beta" / "train_indices.npy")] >= 0.5).mean()),
        high_ro=float((yt[np.load(out / "seed_0" / "real_only" / "train_indices.npy")] >= 0.5).mean()),
        **{k: (str(v) if isinstance(v, list) else v) for k, v in cfg.items()},
    )
    row["high_gain"] = row["high_wb"] - row["high_ro"]
    print(
        f"[done] {name}: gain={row['gain']:+.4f}  wb={wb:.4f} ro={ro:.4f}  "
        f"|β|={row['mean_abs_beta']:.4f} corr={row['mean_corr']:.4f}  "
        f"highΔ={row['high_gain']:+.3f}",
        flush=True,
    )
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    INIT.mkdir(parents=True, exist_ok=True)
    y = pd.read_csv(TASK)["target_failure"].to_numpy(float)
    rows = []
    for name, cfg in CFGS:
        rows.append(run_cfg(name, cfg, y))
        pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    rows = sorted(rows, key=lambda r: -r["gain"])
    (OUT / "best.json").write_text(json.dumps(rows[0], indent=2, default=str))
    print("\n=== RANKED BY CUM GAIN ===")
    for r in rows:
        print(
            f"  {r['name']:22s}  gain={r['gain']:+.4f}  "
            f"wb={r['cum_wb']:.4f} ro={r['cum_ro']:.4f}  "
            f"|β|={r['mean_abs_beta']:.3f}  highΔ={r['high_gain']:+.3f}"
        )
    print("BEST →", rows[0]["name"])


if __name__ == "__main__":
    main()
