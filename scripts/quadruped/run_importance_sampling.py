#!/usr/bin/env python3

"""
Importance-sampling-style candidate selection from fixed_twist_combined.csv.

Usage:
python scripts/quadruped/run_importance_sampling.py \
  --sim_csv data/quadruped/fixed_twist_combined.csv \
  --pool_size 2000 \
  --n_init 10 \
  --n_select 30 \
  --metric final_abs_err_sum \
  --out_dir results/go2_importance_sampling \
  --out_name is_fixed_twist
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RANDOM_BASELINE = np.array([
    [0.491746,  0.150757, -0.152717],
    [-0.022299, -0.088863, -0.482379],
    [-0.342637, -0.291923, -0.654795],
    [-0.376861,  0.177191,  0.128532],
    [0.738578,  0.020283, -0.322086],
    [0.877858, -0.151807,  0.275192],
    [0.449290, -0.011332, -0.480775],
    [0.621295,  0.311590,  0.707381],
    [0.361075,  0.347235, -0.215824],
    [0.909101, -0.113764, -0.631208],
    [0.742195,  0.057224,  0.206573],
    [-0.396166, -0.142504,  0.683447],
    [0.800366,  0.075440, -0.095397],
    [-0.352980, -0.129671,  0.727345],
    [0.621518, -0.086705, -0.000167],
    [-0.154082,  0.312219, -0.119634],
    [0.808451, -0.218274,  0.192342],
    [0.358046,  0.098550,  0.792154],
    [0.019597, -0.332788,  0.718310],
    [0.191762,  0.266115, -0.063928],
    [-0.360352,  0.229679,  0.412366],
    [-0.226003, -0.208504, -0.004124],
    [0.538874,  0.301187,  0.046899],
    [0.506065, -0.353146,  0.457257],
    [0.461539, -0.131106, -0.136551],
    [0.137149, -0.279776,  0.375174],
    [0.996094, -0.039729,  0.337829],
    [0.973169,  0.237059,  0.691296],
    [0.559759, -0.215486, -0.616108],
    [0.510643, -0.358383,  0.366424],
], dtype=np.float32)


def compute_abs_err_sum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "abs_err_sum" in df.columns:
        return df

    required = ["tracking_err_vx", "tracking_err_vy", "tracking_err_wz"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Cannot compute abs_err_sum. Missing columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df["abs_err_sum"] = (
        df["tracking_err_vx"].abs()
        + df["tracking_err_vy"].abs()
        + df["tracking_err_wz"].abs()
    )
    return df


def collapse_sim_rollouts(csv_path: str, metric: str) -> pd.DataFrame:
    """
    Convert rollout-level CSV into one row per fixed command.

    Expected columns:
      fixed_command_id, ref_vx, ref_vy, ref_wz, tracking_err_*
    """
    df = pd.read_csv(csv_path)
    df = compute_abs_err_sum(df)

    required = ["fixed_command_id", "ref_vx", "ref_vy", "ref_wz", "abs_err_sum"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns from sim CSV: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    rows = []

    for cid, g in df.groupby("fixed_command_id"):
        if "t" in g.columns:
            g = g.sort_values("t")

        if metric == "final_abs_err_sum":
            y = float(g["abs_err_sum"].iloc[-1])
        elif metric == "mean_abs_err_sum":
            y = float(g["abs_err_sum"].mean())
        elif metric == "max_abs_err_sum":
            y = float(g["abs_err_sum"].max())
        else:
            raise ValueError(f"Unknown metric: {metric}")

        rows.append({
            "fixed_command_id": int(cid),
            "vx": float(g["ref_vx"].iloc[0]),
            "vy": float(g["ref_vy"].iloc[0]),
            "wz": float(g["ref_wz"].iloc[0]),
            "sim_error": y,
            "num_rollout_rows": len(g),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values("fixed_command_id").reset_index(drop=True)
    return out


def sample_pool(sim_df: pd.DataFrame, pool_size: int, seed: int) -> pd.DataFrame:
    if len(sim_df) <= pool_size:
        return sim_df.copy().reset_index(drop=True)

    return sim_df.sample(n=pool_size, random_state=seed).reset_index(drop=True)


def min_dist_to_set(X: np.ndarray, S: np.ndarray) -> np.ndarray:
    if len(S) == 0:
        return np.full(len(X), np.inf)

    dmin = np.zeros(len(X), dtype=np.float32)
    for i, x in enumerate(X):
        dmin[i] = np.min(np.linalg.norm(S - x[None, :], axis=1))
    return dmin


def select_is_candidates(
    pool_df: pd.DataFrame,
    initial_X: np.ndarray,
    n_select: int,
    min_dist: float,
) -> pd.DataFrame:
    """
    Select high-error candidates from sim pool, excluding points too close
    to the initial random hardware set.

    This is the practical IS acquisition used here:
      score(x) = simulated error magnitude
    and the candidate pool is the large proxy/sim dataset.
    """
    pool_df = pool_df.copy()

    X_pool = pool_df[["vx", "vy", "wz"]].to_numpy(dtype=np.float32)
    dmin = min_dist_to_set(X_pool, initial_X)

    pool_df["dist_to_initial"] = dmin
    pool_df["is_score"] = pool_df["sim_error"].to_numpy()

    valid = pool_df["dist_to_initial"] > min_dist
    cand = pool_df[valid].copy()

    cand = cand.sort_values("is_score", ascending=False).reset_index(drop=True)

    if len(cand) < n_select:
        print(
            f"[WARN] Only {len(cand)} valid candidates after min_dist filtering. "
            f"Requested {n_select}."
        )

    selected = cand.head(n_select).copy()
    selected["selection_rank"] = np.arange(len(selected))
    return selected


def print_candidates(title: str, X: np.ndarray):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for i, (vx, vy, wz) in enumerate(X):
        print(f"{i:02d}: vx={vx:.6f}, vy={vy:.6f}, wz={wz:.6f}")
    print("=" * 80)


def print_selected_df(selected: pd.DataFrame):
    print("\n" + "=" * 100)
    print("IMPORTANCE SAMPLING SELECTED CANDIDATES")
    print("=" * 100)

    for _, r in selected.iterrows():
        print(
            f"{int(r['selection_rank']):02d}: "
            f"vx={r['vx']:.6f}, "
            f"vy={r['vy']:.6f}, "
            f"wz={r['wz']:.6f}, "
            f"sim_error={r['sim_error']:.6f}, "
            f"dist_to_initial={r['dist_to_initial']:.6f}, "
            f"source_id={int(r['fixed_command_id'])}"
        )

    print("=" * 100)


def plot_pairwise_spread(
    pool_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    initial_X: np.ndarray,
    random_X: np.ndarray,
    out_path: Path,
):
    X_pool = pool_df[["vx", "vy", "wz"]].to_numpy()
    y_pool = pool_df["sim_error"].to_numpy()

    X_sel = selected_df[["vx", "vy", "wz"]].to_numpy()
    X_rand = random_X

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    pairs = [
        (0, 1, "vx", "vy"),
        (0, 2, "vx", "wz"),
        (1, 2, "vy", "wz"),
    ]

    for ax, (i, j, xlabel, ylabel) in zip(axes, pairs):
        sc = ax.scatter(
            X_pool[:, i],
            X_pool[:, j],
            c=y_pool,
            s=12,
            alpha=0.35,
            label="sim pool",
        )
        ax.scatter(
            X_rand[:, i],
            X_rand[:, j],
            marker="x",
            s=90,
            label="random baseline 30",
        )
        ax.scatter(
            initial_X[:, i],
            initial_X[:, j],
            marker="o",
            s=100,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            label="initial 10",
        )
        ax.scatter(
            X_sel[:, i],
            X_sel[:, j],
            marker="*",
            s=140,
            label="IS selected 30",
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend(fontsize=8)

    fig.colorbar(sc, ax=axes, label="sim error magnitude")
    # plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"[INFO] Saved spread plot: {out_path}")
    plt.show()


def plot_error_histogram(
    pool_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    random_X: np.ndarray,
    out_path: Path,
):
    random_df = match_random_to_pool(pool_df, random_X)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(pool_df["sim_error"], bins=40, alpha=0.45, label="sim pool")
    ax.hist(selected_df["sim_error"], bins=20, alpha=0.75, label="IS selected")
    if len(random_df) > 0:
        ax.hist(random_df["sim_error"], bins=20, alpha=0.75, label="random baseline matched")

    ax.set_xlabel("sim error magnitude")
    ax.set_ylabel("count")
    ax.set_title("Error distribution: IS vs random baseline")
    ax.grid(True)
    ax.legend()

    # plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"[INFO] Saved error histogram: {out_path}")
    plt.show()


def match_random_to_pool(pool_df: pd.DataFrame, random_X: np.ndarray) -> pd.DataFrame:
    """
    For visualization, assign each provided random baseline point the nearest
    sim-pool error value.
    """
    X_pool = pool_df[["vx", "vy", "wz"]].to_numpy(dtype=np.float32)
    rows = []

    for i, x in enumerate(random_X):
        d = np.linalg.norm(X_pool - x[None, :], axis=1)
        j = int(np.argmin(d))
        row = pool_df.iloc[j].to_dict()
        row["random_index"] = i
        row["random_vx"] = float(x[0])
        row["random_vy"] = float(x[1])
        row["random_wz"] = float(x[2])
        row["nearest_pool_dist"] = float(d[j])
        rows.append(row)

    return pd.DataFrame(rows)


def save_outputs(
    selected_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    initial_X: np.ndarray,
    random_X: np.ndarray,
    out_dir: Path,
    out_name: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_csv = out_dir / f"{out_name}_selected_is.csv"
    selected_npz = out_dir / f"{out_name}_selected_is.npz"
    pool_csv = out_dir / f"{out_name}_pool.csv"
    initial_csv = out_dir / f"{out_name}_initial10.csv"
    random_csv = out_dir / f"{out_name}_random_baseline30.csv"

    selected_df.to_csv(selected_csv, index=False)
    pool_df.to_csv(pool_csv, index=False)

    pd.DataFrame(initial_X, columns=["vx", "vy", "wz"]).to_csv(initial_csv, index=False)
    pd.DataFrame(random_X, columns=["vx", "vy", "wz"]).to_csv(random_csv, index=False)

    np.savez(
        selected_npz,
        selected_X=selected_df[["vx", "vy", "wz"]].to_numpy(dtype=np.float32),
        selected_error=selected_df["sim_error"].to_numpy(dtype=np.float32),
        selected_source_id=selected_df["fixed_command_id"].to_numpy(dtype=np.int64),
        initial_X=initial_X.astype(np.float32),
        random_X=random_X.astype(np.float32),
    )

    print(f"[INFO] Saved selected CSV: {selected_csv}")
    print(f"[INFO] Saved selected NPZ: {selected_npz}")
    print(f"[INFO] Saved pool CSV: {pool_csv}")


def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--sim_csv",
        type=str,
        default="data/quadruped/fixed_twist_combined.csv",
    )
    p.add_argument("--pool_size", type=int, default=2000)
    p.add_argument("--n_init", type=int, default=10)
    p.add_argument("--n_select", type=int, default=30)
    p.add_argument(
        "--metric",
        choices=["final_abs_err_sum", "mean_abs_err_sum", "max_abs_err_sum"],
        default="final_abs_err_sum",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--min_dist",
        type=float,
        default=1e-4,
        help="Exclude sim candidates closer than this to initial random commands.",
    )

    p.add_argument("--out_dir", type=str, default="results/go2_importance_sampling")
    p.add_argument("--out_name", type=str, default="is_fixed_twist")

    return p


def main():
    args = build_parser().parse_args()

    out_dir = Path(args.out_dir)

    sim_df = collapse_sim_rollouts(args.sim_csv, metric=args.metric)
    pool_df = sample_pool(sim_df, pool_size=args.pool_size, seed=args.seed)

    initial_X = RANDOM_BASELINE[:args.n_init]
    random_X = RANDOM_BASELINE

    selected_df = select_is_candidates(
        pool_df=pool_df,
        initial_X=initial_X,
        n_select=args.n_select,
        min_dist=args.min_dist,
    )

    print(f"\n[INFO] Loaded sim commands: {len(sim_df)}")
    print(f"[INFO] Using pool size: {len(pool_df)}")
    print(f"[INFO] Initial random samples: {args.n_init}")
    print(f"[INFO] Selecting IS candidates: {args.n_select}")
    print(f"[INFO] Metric: {args.metric}")

    print_candidates("INITIAL 10 RANDOM SAMPLES USED FOR IS", initial_X)
    print_candidates("RANDOM SAMPLING BASELINE 30", random_X)
    print_selected_df(selected_df)

    save_outputs(
        selected_df=selected_df,
        pool_df=pool_df,
        initial_X=initial_X,
        random_X=random_X,
        out_dir=out_dir,
        out_name=args.out_name,
    )

    plot_pairwise_spread(
        pool_df=pool_df,
        selected_df=selected_df,
        initial_X=initial_X,
        random_X=random_X,
        out_path=out_dir / f"{args.out_name}_spread.png",
    )

    plot_error_histogram(
        pool_df=pool_df,
        selected_df=selected_df,
        random_X=random_X,
        out_path=out_dir / f"{args.out_name}_error_hist.png",
    )


if __name__ == "__main__":
    main()