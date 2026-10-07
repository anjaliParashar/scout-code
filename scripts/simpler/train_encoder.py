#!/usr/bin/env python3
"""
scripts/simpler/train_encoder_simpler.py
------------------------------------------
Train a 12-dimensional scenario encoder for the SimplerEnv BAMS baseline.

Architecture: MLP encoder  42 → 64 → 32 → 12  with ReLU activations.
Trained as an autoencoder on the pool + collected scenarios using MSE
reconstruction loss.  The 12-dim bottleneck is used as the BAMS input space.

The encoder is saved to --output-dir/encoder_weights.pt for use by
run_bams_simpler.py via --encoder-path.

Example
-------
python scripts/simpler/train_encoder.py \
    --scout-init-dir ./results/scout/init \
    --pool-size 1000 --n-grid 3 --seed 42 \
    --output-dir ./outputs/simpler/encoder \
    --epochs 2000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.simpler.domain import DesignSpaceLimits, build_scenario_pool, scenarios_to_array
from utils.simpler.al_state import ALState


# ---------------------------------------------------------------------------
# Encoder architecture
# ---------------------------------------------------------------------------

class ScenarioEncoder(nn.Module):
    """
    42 → 64 → 32 → 12  encoder (bottleneck).
    Decoder: 12 → 32 → 64 → 42  for autoencoder training.
    """
    def __init__(self, input_dim: int = 19, latent_dim: int = 12):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),        nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64),         nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device  = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect feature vectors ───────────────────────────────────────────
    # Use the pool + any already-collected scenarios from the scout init dir
    limits = DesignSpaceLimits(n_grid=args.n_grid)
    pool   = build_scenario_pool(limits=limits, n_random=args.pool_size,
                                 seed=args.seed)
    X_pool = scenarios_to_array(pool).astype(np.float32)
    print(f"[ENCODER] Pool: {len(pool)} scenarios  X_pool={X_pool.shape}")

    extra_X = []
    if args.scout_init_dir:
        init_path = Path(args.scout_init_dir)
        if init_path.exists():
            state = ALState.load(init_path)
            if len(state.train) > 0:
                extra_X.append(state.X_train().astype(np.float32))
            if len(state.proxy_only) > 0:
                extra_X.append(state.X_proxy_train().astype(np.float32))
            print(f"[ENCODER] Loaded {len(state.train)} target + "
                  f"{len(state.proxy_only)} proxy-only from {init_path}")

    X_all = np.concatenate([X_pool] + extra_X, axis=0) if extra_X else X_pool

    # Normalise
    X_mean = X_all.mean(axis=0, keepdims=True)
    X_std  = X_all.std(axis=0, keepdims=True) + 1e-6
    X_norm = (X_all - X_mean) / X_std

    np.save(out_dir / "X_mean.npy", X_mean)
    np.save(out_dir / "X_std.npy",  X_std)
    print(f"[ENCODER] Training on {len(X_norm)} scenarios (d=42 → 12)")

    # ── Train autoencoder ─────────────────────────────────────────────────
    X_t      = torch.tensor(X_norm, dtype=torch.float32)
    dataset  = TensorDataset(X_t)
    loader   = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model    = ScenarioEncoder(input_dim=X_t.shape[1], latent_dim=12).to(device)
    opt      = torch.optim.Adam(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    loss_fn  = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            loss = loss_fn(model.reconstruct(xb), xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{args.epochs}  "
                  f"MSE={total/len(X_norm):.6f}", flush=True)

    # ── Save ──────────────────────────────────────────────────────────────
    weights_path = out_dir / "encoder_weights.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim":  42,
        "latent_dim": 12,
        "X_mean":     X_mean,
        "X_std":      X_std,
    }, weights_path)
    print(f"[ENCODER] Saved → {weights_path}")

    # Verify reconstruction
    model.eval()
    with torch.no_grad():
        Z = model(X_t.to(device)).cpu().numpy()
    print(f"[ENCODER] Latent space: shape={Z.shape}  "
          f"mean={Z.mean():.4f}  std={Z.std():.4f}")
    np.save(out_dir / "Z_encoded_pool.npy", Z)
    print(f"[DONE]  outputs → {out_dir}")


def build_parser():
    p = argparse.ArgumentParser(
        description="Train 12-dim scenario encoder for SimplerEnv BAMS baseline."
    )
    p.add_argument("--pool-size",      type=int,   default=1000)
    p.add_argument("--n-grid",         type=int,   default=3)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--scout-init-dir", type=str,   default="",
                   help="Load already-collected scenarios for richer training data")
    p.add_argument("--output-dir",     type=str,   default="./results/encoder")
    p.add_argument("--epochs",         type=int,   default=2000)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--weight-decay",   type=float, default=1e-5)
    p.add_argument("--batch-size",     type=int,   default=256)
    p.add_argument("--cpu",            action="store_true")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
