import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# -----------------------------
# Same VAE architecture as training
# -----------------------------
"""
python scripts/quadruped/visualize_velocity_vae.py \
    --model_path velocity_vae_runs/velocity_vae.pt \
    --num_samples 1 \
    --latent_scale 1.0
"""
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

    def decode(self, z):
        out = self.decoder(z)
        return out.reshape(z.shape[0], self.T, self.input_dim)


# -----------------------------
# Utilities
# -----------------------------

def integrate_body_velocity(traj, dt):
    """
    Integrate body-frame velocity commands into approximate world-frame path.

    traj: (T, 3), columns [vx_body, vy_body, yaw_rate]
    returns:
        positions: (T, 2)
        yaws: (T,)
    """
    T = traj.shape[0]

    pos = np.zeros((T, 2), dtype=np.float32)
    yaw = np.zeros(T, dtype=np.float32)

    for t in range(1, T):
        vx, vy, wz = traj[t - 1]

        yaw[t] = yaw[t - 1] + wz * dt

        c = np.cos(yaw[t - 1])
        s = np.sin(yaw[t - 1])

        vx_world = c * vx - s * vy
        vy_world = s * vx + c * vy

        pos[t, 0] = pos[t - 1, 0] + vx_world * dt
        pos[t, 1] = pos[t - 1, 1] + vy_world * dt

    return pos, yaw

def load_reference_trajectories(reference_path, num_reference):
    if reference_path == "" or reference_path is None:
        return None

    ref = np.load(reference_path)  # (N, T, 3)

    if num_reference < ref.shape[0]:
        idx = np.random.choice(ref.shape[0], size=num_reference, replace=False)
        ref = ref[idx]

    return ref

def sample_trajectories(model, mean, std, num_samples, latent_scale, device):
    model.eval()

    with torch.no_grad():
        z = latent_scale * torch.randn(num_samples, model.latent_dim).to(device)
        traj_norm = model.decode(z).cpu().numpy()

    traj = traj_norm * std + mean
    traj = traj.squeeze()

    # Safety clipping after decoding
    # traj[:, :, 0] = np.clip(traj[:, :, 0], -0.3, 0.8)
    # traj[:, :, 1] = np.clip(traj[:, :, 1], -0.4, 0.4)
    # traj[:, :, 2] = np.clip(traj[:, :, 2], -1.2, 1.2)

    traj[:, 0] = np.clip(traj[:, 0], -1.0, 1.0)
    traj[:, 1] = np.clip(traj[:, 1], -0.5,0.5)
    traj[:, 2] = np.clip(traj[:, 2], -0.8, 0.8)
    return traj


# -----------------------------
# Visualization
# -----------------------------

def visualize(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)

    T = ckpt["T"]
    dt = ckpt["dt"]
    latent_dim = ckpt["latent_dim"]
    hidden_dim = ckpt["hidden_dim"]

    mean = ckpt["mean"]
    std = ckpt["std"]

    model = VelocityVAE(
        T=T,
        input_dim=3,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])

    sampled = sample_trajectories(
        model=model,
        mean=mean,
        std=std,
        num_samples=args.num_samples,
        latent_scale=args.latent_scale,
        device=device,
    )

    time = np.arange(T) * dt

    # Plot velocity profiles
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    labels = [r"$v_x$", r"$v_y$", r"$v_{\mathrm{yaw}}$"]

    for i in range(3):
        ax = axes[i]
        for n in range(args.num_samples):
            ax.plot(time, sampled[n, :, i], alpha=0.75)
        ax.set_ylabel(labels[i])
        ax.grid(True)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Sampled VAE Velocity Trajectories")
    plt.tight_layout()
    plt.savefig('outputs/quadruped/visualize_velocity.png')

    # Plot integrated 2D paths
    plt.figure(figsize=(7, 7))

    for n in range(args.num_samples):
        pos, yaw = integrate_body_velocity(sampled[n], dt)
        plt.plot(pos[:, 0], pos[:, 1], alpha=0.85)
        plt.scatter(pos[0, 0], pos[0, 1], marker="o", s=20)
        plt.scatter(pos[-1, 0], pos[-1, 1], marker="x", s=40)

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Integrated 2D Paths from Sampled Velocities")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/quadruped/visualize_xy.png')

    if args.save_samples:
        np.save(args.save_samples, sampled)
        print(f"Saved sampled trajectories to: {args.save_samples}")

def generate_trajectory(model_path,num_samples, latent_scale,idx, savefig_dir = None, save_samples=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    T = ckpt["T"]
    dt = ckpt["dt"]
    latent_dim = ckpt["latent_dim"]
    hidden_dim = ckpt["hidden_dim"]

    mean = ckpt["mean"]
    std = ckpt["std"]
    model = VelocityVAE(
        T=T,
        input_dim=3,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])

    sampled = sample_trajectories(
        model=model,
        mean=mean,
        std=std,
        num_samples=num_samples,
        latent_scale=latent_scale,
        device=device,
    )

    time = np.arange(T) * dt
    sampled = sampled.reshape((num_samples,T,3))
    # Plot velocity profiles
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    labels = [r"$v_x$", r"$v_y$", r"$v_{\mathrm{yaw}}$"]
    
    for i in range(3):
        ax = axes[i]
        for n in range(num_samples):
            ax.plot(time, sampled[n, :, i], alpha=0.75)
        ax.set_ylabel(labels[i])
        ax.grid(True)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Sampled VAE Velocity Trajectories")
    plt.tight_layout()
    if savefig_dir:
       plt.savefig(f"{savefig_dir}/visualize_velocity_{idx}.png")
    else:
        plt.savefig(f'outputs/quadruped/random/visualize_velocity_{idx}.png')


    # Plot integrated 2D paths
    plt.figure(figsize=(7, 7))

    for n in range(num_samples):
        pos, yaw = integrate_body_velocity(sampled[n], dt)
        plt.plot(pos[:, 0], pos[:, 1], alpha=0.85)
        plt.scatter(pos[0, 0], pos[0, 1], marker="o", s=20)
        plt.scatter(pos[-1, 0], pos[-1, 1], marker="x", s=40)

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Integrated 2D Paths from Sampled Velocities")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    if savefig_dir:
        plt.savefig(f"{savefig_dir}/visualize_xy_{idx}.png")
    else:
        plt.savefig(f'outputs/quadruped/random/visualize_xy_{idx}.png')
   

    if save_samples:
        np.save(save_samples, sampled)
        print(f"Saved sampled trajectories to: {save_samples}")

    return sampled.squeeze()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default="velocity_vae_runs/velocity_vae.pt",
    )
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--idx", type=int, default=1)
    parser.add_argument("--latent_scale", type=float, default=1.0)
    parser.add_argument("--save_samples", type=str, default="")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--reference_path",
        type=str,
        default="velocity_vae_runs/reference_trajectories.npy",
        )
    parser.add_argument("--num_reference", type=int, default=12)
    args = parser.parse_args()
    # visualize(args)
    model_path = "velocity_vae_runs/velocity_vae.pt"
    num_samples = 1
    latent_scale = 1
    idx= 1
    velocity_trajectory = generate_trajectory(model_path=model_path,num_samples=1, latent_scale=1,idx=1)

