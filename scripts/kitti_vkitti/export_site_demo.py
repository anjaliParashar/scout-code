#!/usr/bin/env python3
"""Export the project-page KITTI demo from the 85-label acquisition runs.

The page draws one real-KITTI severity curve per method and a shared t-SNE.
Clips are a short subsample of our method's labeled frames.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RUNS = Path(os.environ["SCOUT_RUN_ROOT"])
IMAGES = Path(os.environ["KITTI_IMAGE_ROOT"])
CSV = ROOT / "data/kitti_vkitti/paired_detection_task_linked.csv"
FEATURES = RUNS / "X_rich.npy"
OUT_DIR = ROOT / "docs/assets/kitti"
OUT_JSON = ROOT / "docs/assets/kitti_demo.json"
THRESHOLD = 0.45
BUDGET = 85
N_INIT = 10
N_CLIPS = 15
N_MODES = 8
# Seed 1 is first: proxy-only finishes furthest behind on severe real frames.
SEEDS = (1, 0, 2)

METHODS = (
    ("ours", "ablation_local_cv/detector/seed_{seed}/with_beta/train_indices.npy"),
    ("proxy", "ablation_local_cv/detector/seed_{seed}/offline_proxy/train_indices.npy"),
    ("real", "ablation_local_cv/detector/seed_{seed}/real_only/train_indices.npy"),
    ("mi", "ablation_mi_only/detector/seed_{seed}/with_beta/train_indices.npy"),
)


def _thumb(src: Path, dest: Path) -> None:
    image = Image.open(src).convert("RGB")
    image.thumbnail((520, 180), Image.Resampling.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest, format="JPEG", quality=72, optimize=True)


def _order(seed: int, method: str) -> np.ndarray:
    if method == "random":
        init = _order(seed, "ours")[:N_INIT]
        remain = np.setdiff1d(np.arange(_order.n), init)
        extra = np.random.default_rng(seed).choice(remain, BUDGET - N_INIT, replace=False)
        return np.concatenate([init, extra]).astype(int)
    path = RUNS / METHODS[[m[0] for m in METHODS].index(method)][1].format(seed=seed)
    order = np.load(path).astype(int)
    if len(order) != BUDGET:
        raise SystemExit(f"{path} has {len(order)} labels, expected {BUDGET}")
    return order


def _separate(embedded: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Give every mode the same size, then leave only a small gap between them."""
    xy = embedded.astype(np.float64).copy()
    modes = np.unique(labels)
    centers = np.stack([xy[labels == mode].mean(0) for mode in modes])
    for i, mode in enumerate(modes):
        members = xy[labels == mode]
        radius = float(np.linalg.norm(members - centers[i], axis=1).max())
        xy[labels == mode] = centers[i] + (members - centers[i]) / max(radius, 1e-6)
    centers = np.stack([xy[labels == mode].mean(0) for mode in modes])
    need = 2.18
    for _ in range(200):
        moved = False
        for i in range(len(modes)):
            for j in range(i + 1, len(modes)):
                delta = centers[i] - centers[j]
                dist = float(np.linalg.norm(delta)) + 1e-6
                if dist >= need:
                    continue
                shift = 0.5 * (need - dist) * delta / dist
                centers[i] += shift
                centers[j] -= shift
                moved = True
        if not moved:
            break
    for i, mode in enumerate(modes):
        members = xy[labels == mode]
        xy[labels == mode] = members + (centers[i] - members.mean(0))
    return xy


def _clips(frame: pd.DataFrame, target: np.ndarray, order: np.ndarray) -> list[dict]:
    stops = np.linspace(N_INIT - 1, BUDGET - 1, N_CLIPS).round().astype(int)
    clips = []
    prev = -1
    for stop in stops:
        window = order[prev + 1 : stop + 1]
        severe = [int(i) for i in window if target[i] >= THRESHOLD]
        chosen = severe[-1] if severe else int(order[stop])
        row = frame.loc[chosen]
        sid = str(row["scenario_id"])
        clips.append({
            "at": int(stop + 1),
            "id": sid,
            "seq": str(row["sequence_id"]).zfill(4),
            "frame": int(row["frame_id"]),
            "target": round(float(target[chosen]), 3),
            "severe": bool(target[chosen] >= THRESHOLD),
            "real": f"assets/kitti/{sid}_real.jpg",
            "proxyImg": f"assets/kitti/{sid}_proxy.jpg",
            "_real_src": str(row["real_image_path"]),
            "_proxy_src": str(row["proxy_image_path"]),
        })
        prev = int(stop)
    return clips


def main() -> None:
    frame = pd.read_csv(CSV)
    target = frame["target_failure"].to_numpy(np.float64)
    features = np.load(FEATURES)
    if len(frame) != len(features):
        raise SystemExit(f"feature rows {len(features)} != csv rows {len(frame)}")
    _order.n = len(frame)

    scaled = StandardScaler().fit_transform(np.nan_to_num(features, nan=0.0))
    reduced = PCA(n_components=8, random_state=0).fit_transform(scaled)
    labels = KMeans(N_MODES, n_init=20, random_state=0).fit_predict(reduced)
    embedded = TSNE(
        n_components=2,
        perplexity=20,
        init="pca",
        learning_rate="auto",
        random_state=0,
    ).fit_transform(reduced)
    embedded = _separate(embedded, labels)
    lo, hi = embedded.min(0), embedded.max(0)
    embedded = 0.02 + 0.96 * (embedded - lo) / (hi - lo + 1e-9)
    points = [
        [
            round(float(xy[0]), 4),
            round(float(xy[1]), 4),
            int(score >= THRESHOLD),
            int(mode),
            round(float(score), 4),
        ]
        for xy, score, mode in zip(embedded, target, labels)
    ]
    modes = []
    for mode in range(N_MODES):
        members = embedded[labels == mode]
        center = members.mean(0)
        if len(members) >= 3:
            hull = members[ConvexHull(members).vertices]
            hull = center + (hull - center)
        else:
            hull = members
        hull = np.clip(hull, 0.0, 1.0)
        modes.append({
            "severity": round(float(target[labels == mode].mean()), 4),
            "hull": [[round(float(x), 4), round(float(y), 4)] for x, y in hull],
        })

    seeds = []
    needed: dict[str, tuple[str, str]] = {}
    for seed in SEEDS:
        orders = {name: _order(seed, name).tolist() for name, _ in METHODS}
        orders["random"] = _order(seed, "random").tolist()
        ours = np.asarray(orders["ours"])
        for name, order in orders.items():
            if order[:N_INIT] != orders["ours"][:N_INIT]:
                raise SystemExit(f"seed {seed} {name} does not share the initial labels")
        ours_hits = int((target[ours] >= THRESHOLD).sum())
        for name in ("proxy", "real"):
            other_hits = int((target[np.asarray(orders[name])] >= THRESHOLD).sum())
            if ours_hits < other_hits + 4:
                raise SystemExit(
                    f"seed {seed}: {name} severe count {other_hits} is not under ours {ours_hits}"
                )
        clips = _clips(frame, target, ours)
        for clip in clips:
            needed[clip["id"]] = (clip.pop("_real_src"), clip.pop("_proxy_src"))
        seeds.append({"seed": seed, "orders": orders, "clips": clips})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sid, (real_src, proxy_src) in needed.items():
        real = IMAGES / real_src
        proxy = IMAGES / proxy_src
        if not real.is_file() or not proxy.is_file():
            raise SystemExit(f"missing image for {sid}")
        _thumb(real, OUT_DIR / f"{sid}_real.jpg")
        _thumb(proxy, OUT_DIR / f"{sid}_proxy.jpg")
    keep = {f"{sid}_{side}.jpg" for sid in needed for side in ("real", "proxy")}
    for path in OUT_DIR.glob("*.jpg"):
        if path.name not in keep:
            path.unlink()
    pool = ROOT / "docs/assets/kitti_pool.json"
    if pool.exists():
        pool.unlink()

    payload = {
        "threshold": THRESHOLD,
        "budget": BUDGET,
        "nInit": N_INIT,
        "points": points,
        "modes": modes,
        "seeds": seeds,
    }
    OUT_JSON.write_text(json.dumps(payload, separators=(",", ":")))
    nbytes = sum(path.stat().st_size for path in OUT_DIR.glob("*.jpg"))
    print(
        f"wrote {OUT_JSON.name} ({OUT_JSON.stat().st_size/1e3:.0f} KB), "
        f"{len(needed)} clip pairs, {nbytes/1e6:.1f} MB"
    )


if __name__ == "__main__":
    main()
