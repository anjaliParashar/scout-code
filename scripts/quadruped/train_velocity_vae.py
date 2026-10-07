import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

"""
python scripts/quadruped/train_velocity_vae.py \
    --num_trajs 10000 \
    --duration 10.0 \
    --dt 0.1 \
    --latent_dim 8 \
    --epochs 100
"""
# -----------------------------
# Synthetic spline trajectory data
# -----------------------------

def cubic_bezier(p0, p1, p2, p3, t):
    """
    Cubic Bezier curve.
    p0, p1, p2, p3: (2,)
    t: (T,)
    returns: (T, 2)
    """
    t = t[:, None]
    return (
        (1 - t) ** 3 * p0
        + 3 * (1 - t) ** 2 * t * p1
        + 3 * (1 - t) * t ** 2 * p2
        + t ** 3 * p3
    )


def generate_reference_trajectory(T=100, dt=0.1):
    """
    Generate one 10s velocity trajectory using a smooth 2D spline path.
    Output shape: (T, 3), columns [vx_body, vy_body, yaw_rate].
    """

    # Random smooth path endpoints
    p0 = np.array([0.0, 0.0])
    p3 = np.array([
        np.random.uniform(1.0, 4.0),
        np.random.uniform(-2.0, 2.0),
    ])

    # Random control points
    p1 = p0 + np.array([
        np.random.uniform(0.3, 1.5),
        np.random.uniform(-1.5, 1.5),
    ])
    p2 = p3 + np.array([
        np.random.uniform(-1.5, -0.3),
        np.random.uniform(-1.5, 1.5),
    ])

    t = np.linspace(0.0, 1.0, T)

    # Smooth speed profile: slow start and stop
    tau = 3 * t**2 - 2 * t**3

    pos = cubic_bezier(p0, p1, p2, p3, tau)

    # World-frame velocity
    vel_world = np.gradient(pos, dt, axis=0)
    dx = vel_world[:, 0]
    dy = vel_world[:, 1]

    # Heading from path tangent
    yaw = np.unwrap(np.arctan2(dy, dx + 1e-8))
    yaw_rate = np.gradient(yaw, dt)

    # Convert world velocity into body frame
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    vx_body = cos_yaw * dx + sin_yaw * dy
    vy_body = -sin_yaw * dx + cos_yaw * dy

    traj = np.stack([vx_body, vy_body, yaw_rate], axis=-1)

    # Add small smooth command noise
    noise = np.random.normal(scale=0.03, size=traj.shape)
    traj = traj + noise

    # Clip to realistic-ish quadruped command ranges
    # traj[:, 0] = np.clip(traj[:, 0], -0.3, 0.8)
    # traj[:, 1] = np.clip(traj[:, 1], -0.4, 0.4)
    # traj[:, 2] = np.clip(traj[:, 2], -1.2, 1.2)

    traj[:, 0] = np.clip(traj[:, 0], -1.0, 1.0)
    traj[:, 1] = np.clip(traj[:, 1], -0.5,0.5)
    traj[:, 2] = np.clip(traj[:, 2], -0.8, 0.8)

    return traj.astype(np.float32)


def generate_dataset(num_trajs=10000, T=100, dt=0.1):
    data = [generate_reference_trajectory(T=T, dt=dt) for _ in range(num_trajs)]
    return np.stack(data, axis=0)  # (N, T, 3)


# -----------------------------
# Dataset
# -----------------------------

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories):
        """
        trajectories: normalized array of shape (N, T, 3)
        """
        self.x = torch.from_numpy(trajectories).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]


# -----------------------------
# VAE
# -----------------------------

class VelocityVAE(nn.Module):
    def __init__(self, T=100, input_dim=3, latent_dim=8, hidden_dim=256):
        super().__init__()

        self.T = T
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.flat_dim = T * input_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.flat_dim),
        )

    def encode(self, x):
        x = x.reshape(x.shape[0], -1)
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        out = self.decoder(z)
        return out.reshape(z.shape[0], self.T, self.input_dim)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, beta=1e-3):
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl


# -----------------------------
# Training
# -----------------------------

def train(args):
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    T = int(args.duration / args.dt)

    print("Generating synthetic trajectories...")
    trajectories = generate_dataset(
        num_trajs=args.num_trajs,
        T=T,
        dt=args.dt,
    )

    mean = trajectories.mean(axis=(0, 1), keepdims=True)
    std = trajectories.std(axis=(0, 1), keepdims=True) + 1e-6
    trajectories_norm = (trajectories - mean) / std

    dataset = TrajectoryDataset(trajectories_norm)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = VelocityVAE(
        T=T,
        input_dim=3,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()

        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        for x in loader:
            x = x.to(device)

            recon, mu, logvar = model(x)
            loss, recon_loss, kl = vae_loss(
                recon,
                x,
                mu,
                logvar,
                beta=args.beta,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl.item()

        n = len(loader)
        print(
            f"Epoch {epoch + 1:04d} | "
            f"loss={total_loss / n:.6f} | "
            f"recon={total_recon / n:.6f} | "
            f"kl={total_kl / n:.6f}"
        )

    ckpt = {
        "model_state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
        "T": T,
        "dt": args.dt,
        "duration": args.duration,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
    }

    save_path = os.path.join(args.out_dir, "velocity_vae.pt")
    torch.save(ckpt, save_path)
    print(f"Saved model to: {save_path}")

    data_path = os.path.join(args.out_dir, "reference_trajectories.npy")
    np.save(data_path, trajectories)
    print(f"Saved reference trajectories to: {data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--out_dir", type=str, default="velocity_vae_runs")
    parser.add_argument("--num_trajs", type=int, default=10000)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.1)

    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=256)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1e-3)

    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    train(args)