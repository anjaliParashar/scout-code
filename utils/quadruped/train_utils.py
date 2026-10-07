#!/usr/bin/env python3

"""
Utilities for training MLP surrogates for Go2 velocity-command experiments.

Target / hardware data:
  - Loads a collection of .pkl files such as:
      results/go2_hardware/fixed_vx01.pkl
  - Extracts:
      X = [cmd_vx, cmd_vy, cmd_wz]
      y = final abs tracking error sum

Proxy / sim data:
  - Loads:
      data/quadruped/fixed_twist_combined.csv
  - Groups by fixed_command_id
  - Extracts:
      X = [ref_vx, ref_vy, ref_wz]
      y = final abs tracking error sum

The trained checkpoint contains:
  - model state_dict
  - input/output normalization stats
  - config metadata
"""

from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
from torch import nn


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        depth: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()

        layers = []
        d = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim

        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


@dataclass
class MLPTrainConfig:
    hidden_dim: int = 64
    depth: int = 3
    dropout: float = 0.05
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 1000
    batch_size: int = 32
    seed: int = 0
    device: str | None = None
    verbose: bool = True


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def load_hardware_pkls(
    paths_or_glob: str | list[str],
    n_files: int | None = None,
    sort_paths: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load hardware .pkl result files.

    Expected pkl structure from your hardware recorder:
      payload["command"] = {"vx": ..., "vy": ..., "wz": ...}
      payload["records"][-1]["abs_err_sum"] OR
      payload["error_summary"]["mean_abs_err_sum"]

    Returns:
      X: (N, 3) [vx, vy, wz]
      y: (N,) error score
      used_paths: list of loaded files
    """
    if isinstance(paths_or_glob, str):
        paths = sorted(glob.glob(paths_or_glob)) if sort_paths else glob.glob(paths_or_glob)
    else:
        paths = list(paths_or_glob)
        if sort_paths:
            paths = sorted(paths)

    if n_files is not None:
        paths = paths[:n_files]

    X_rows = []
    y_rows = []
    used = []

    for path in paths:
        path = str(path)
        with open(path, "rb") as f:
            payload = pickle.load(f)

        cmd = payload.get("command", {})
        vx = _safe_float(cmd.get("vx"))
        vy = _safe_float(cmd.get("vy"))
        wz = _safe_float(cmd.get("wz"))

        records = payload.get("records", [])

        # Prefer final per-timestep abs_err_sum if present.
        # y = np.nan
        # if len(records) > 0 and "abs_err_sum" in records[-1]:
        #     y = _safe_float(records[-1]["abs_err_sum"])

        # Fallback to aggregate summary.
        # if not np.isfinite(y):
        summary = payload.get("error_summary", {})
        y = _safe_float(summary.get("mean_abs_err_sum"))

        if not (np.isfinite(vx) and np.isfinite(vy) and np.isfinite(wz) and np.isfinite(y)):
            print(f"[WARN] Skipping invalid hardware file: {path}")
            continue

        X_rows.append([vx, vy, wz])
        y_rows.append(y)
        used.append(path)

    if len(X_rows) == 0:
        raise RuntimeError("No valid hardware records loaded.")

    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.float32)

    return X, y, used


def load_sim_csv(
    csv_path: str,
    metric: str = "final_abs_err_sum",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load sim fixed-command CSV.

    Expected columns:
      fixed_command_id
      ref_vx, ref_vy, ref_wz
      tracking_err_vx, tracking_err_vy, tracking_err_wz

    metric:
      final_abs_err_sum: use final row per command
      mean_abs_err_sum:  mean over all rows per command
      max_abs_err_sum:   max over all rows per command
    """
    df = pd.read_csv(csv_path)

    required = [
        "fixed_command_id",
        "ref_vx",
        "ref_vy",
        "ref_wz",
        "tracking_err_vx",
        "tracking_err_vy",
        "tracking_err_wz",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in sim CSV: {missing}")

    df = df.copy()
    df["abs_err_sum"] = (
        df["tracking_err_vx"].abs()
        + df["tracking_err_vy"].abs()
        + df["tracking_err_wz"].abs()
    )

    X_rows = []
    y_rows = []

    for cid, g in df.groupby("fixed_command_id"):
        g = g.sort_values("t") if "t" in g.columns else g

        ref_vx = float(g["ref_vx"].iloc[0])
        ref_vy = float(g["ref_vy"].iloc[0])
        ref_wz = float(g["ref_wz"].iloc[0])

        if metric == "final_abs_err_sum":
            y = float(g["abs_err_sum"].iloc[-1])
        elif metric == "mean_abs_err_sum":
            y = float(g["abs_err_sum"].mean())
        elif metric == "max_abs_err_sum":
            y = float(g["abs_err_sum"].max())
        else:
            raise ValueError(f"Unknown metric: {metric}")

        X_rows.append([ref_vx, ref_vy, ref_wz])
        y_rows.append(y)

    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.float32)

    return X, y


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------

def fit_standardizer(x: np.ndarray, eps: float = 1e-8):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.maximum(std, eps)
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(x, mean, std):
    return (x - mean) / std


def unstandardize_y(y_norm, y_mean, y_std):
    return y_norm * y_std + y_mean


# ---------------------------------------------------------------------
# Training / prediction
# ---------------------------------------------------------------------

def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    cfg: MLPTrainConfig,
) -> dict:
    """
    Train an MLP on X -> y.

    Returns a checkpoint-style dict.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    x_mean, x_std = fit_standardizer(X)
    y_mean, y_std = fit_standardizer(y[:, None])
    y_mean = float(y_mean[0])
    y_std = float(y_std[0])

    Xn = standardize(X, x_mean, x_std)
    yn = ((y - y_mean) / y_std).astype(np.float32)

    model = MLPRegressor(
        input_dim=X.shape[1],
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        dropout=cfg.dropout,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(Xn, dtype=torch.float32, device=device)
    y_t = torch.tensor(yn, dtype=torch.float32, device=device)

    n = len(X)
    batch_size = min(cfg.batch_size, n)

    model.train()
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n, device=device)
        losses = []

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            pred = model(X_t[idx])
            loss = loss_fn(pred, y_t[idx])

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(float(loss.detach().cpu()))

        if cfg.verbose and (epoch % max(1, cfg.epochs // 10) == 0 or epoch == cfg.epochs - 1):
            print(f"[TRAIN] epoch={epoch:04d}, loss={np.mean(losses):.6f}")

    model.eval()

    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": X.shape[1],
            "hidden_dim": cfg.hidden_dim,
            "depth": cfg.depth,
            "dropout": cfg.dropout,
        },
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "train_config": asdict(cfg),
        "X_train": X,
        "y_train": y,
    }

    return ckpt


def save_checkpoint(ckpt: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    print(f"[INFO] Saved checkpoint: {path}")


def load_mlp_checkpoint(path: str | Path, device: str | None = None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # ckpt = torch.load(path, map_location=device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    model = MLPRegressor(**cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt, device


@torch.no_grad()
def predict_mlp(
    model: nn.Module,
    ckpt: dict,
    X: np.ndarray,
    device: str | torch.device | None = None,
    mc_dropout: bool = False,
    n_mc: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict mean and uncertainty.

    If mc_dropout=True, returns MC-dropout mean/std.
    Otherwise std is zeros.
    """
    device = torch.device(device or next(model.parameters()).device)

    X = np.asarray(X, dtype=np.float32)
    Xn = standardize(X, ckpt["x_mean"], ckpt["x_std"])
    X_t = torch.tensor(Xn, dtype=torch.float32, device=device)

    if mc_dropout:
        model.train()
        preds = []
        for _ in range(n_mc):
            p = model(X_t).detach().cpu().numpy()
            preds.append(p)
        preds = np.stack(preds, axis=0)
        mean_n = preds.mean(axis=0)
        std_n = preds.std(axis=0)
        model.eval()
    else:
        model.eval()
        mean_n = model(X_t).detach().cpu().numpy()
        std_n = np.zeros_like(mean_n)

    mean = unstandardize_y(mean_n, ckpt["y_mean"], ckpt["y_std"])
    std = std_n * ckpt["y_std"]

    return mean.astype(np.float32), std.astype(np.float32)


# ---------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--mode", choices=["hardware", "sim"], required=True)

    # Hardware
    p.add_argument("--hardware_glob", type=str, default="results/go2_hardware/random/*.pkl")
    p.add_argument("--n_hardware", type=int, default=None)

    # Sim
    p.add_argument(
        "--sim_csv",
        type=str,
        default="data/quadruped/fixed_twist_combined.csv",
    )
    p.add_argument(
        "--metric",
        choices=["final_abs_err_sum", "mean_abs_err_sum", "max_abs_err_sum"],
        default="mean_abs_err_sum",
    )

    p.add_argument("--out", type=str, required=True)

    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)

    return p


def main():
    args = build_parser().parse_args()

    cfg = MLPTrainConfig(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
    )

    if args.mode == "hardware":
        X, y, paths = load_hardware_pkls(args.hardware_glob, n_files=args.n_hardware)
        print(f"[INFO] Loaded hardware samples: {len(X)}")
        for i, path in enumerate(paths):
            print(f"  {i:03d}: {path}  X={X[i]}  y={y[i]:.4f}")

    else:
        X, y = load_sim_csv(args.sim_csv, metric=args.metric)
        print(f"[INFO] Loaded sim samples: {len(X)}")

    ckpt = train_mlp(X, y, cfg)
    save_checkpoint(ckpt, args.out)


if __name__ == "__main__":
    main()