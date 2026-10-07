#!/usr/bin/env python3
"""
SCOUT on KITTI–VKITTI with SIMPLER-style proxy BNN + enriched features.

Modes
-----
  with_beta     — SCOUT: MI shortlist + local control-variate CV with sim y_s
  offline_proxy — same, but y_cv = frozen offline proxy-BNN ŷ_s
  bnn_proxy     — same, but y_cv = current target-BNN mean (self-proxy)
  real_only     — MI + target-BNN mean only (β=0 / no local CV)

Local CV matches utils/baseline/scout_baseline.py (radius/kNN neighbourhoods
in embedding space via compute_local_cv_parallel).

--acq-score
----------
  cv   — default: rank MI shortlist by μ_CV (or f_mean if β=0)
  mi   — CV off: rank MI shortlist by MI only (proxy channel unused for scoring)
  micv — vanilla blend: rank by minmax(MI) + minmax(μ_CV) (or f_mean if β=0)

--no-surrogate
--------------
Skip all BNN / offline-proxy training. Requires --acq-score mi. MI support can
use proxy-labelled indices via --mi-support proxy (paired init + proxy-only).

Example
-------
python scripts/kitti_vkitti/run_scout.py --mode with_beta --seed 0
python scripts/kitti_vkitti/run_scout.py --mode real_only --seed 0
python scripts/kitti_vkitti/run_scout.py --mode with_beta --acq-score mi --seed 0
python scripts/kitti_vkitti/run_scout.py --mode with_beta --acq-score mi \
    --no-surrogate --mi-support proxy --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from scout.seed import set_seed
from scout.tsne import get_or_compute_tsne, choose_vis_indices
from scout.mi import fit_support_and_compute_mi
from scout.bnn import HeteroBNNEmbedding, train_bnn, batched_posterior_draws
from scout.cv import compute_local_cv_parallel
from utils.baseline.scout_baseline import _cluster_and_pick
from utils.kitti_vkitti.config import OUTPUT_ROOT, SEED
from utils.kitti_vkitti.enrich_features import FEATURE_COLS as THIN_FEATURE_COLS
from utils.kitti_vkitti.io_utils import read_task_csv
from utils.kitti_vkitti.rich_features import RICH_FEATURE_NAMES, write_rich_features

N_TARGET_DRAWS = 10   # match SCOUT local paired draws
N_UNPAIRED = 20
CV_RADIUS = 1.5
CV_FALLBACK_K = 80

# Named scenario-space definitions (subsets of the feature embedding).
# Real-only AL lives entirely in this space (MI + target BNN); with_beta can
# also lean on pool-wide y_proxy, so it is less sensitive to the embedding.
SCENARIO_SPACES = (
    "thin",       # original CSV geometry (+ brightness)
    "thin_det",   # thin + detector diagnostics from CSV
    "geom",       # rich GT geometry only
    "detector",   # rich detector-side + image cues
    "rich",       # full rich features without PLS
    "full",       # rich + PLS toward proxy (locked default)
)

_GEOM_NAMES = set(RICH_FEATURE_NAMES[:17])          # n_cars … occ_mean
_DET_NAMES = set(RICH_FEATURE_NAMES[17:33])         # n_pred … frame_norm
_PLS_NAMES = set(RICH_FEATURE_NAMES[33:37])         # pls0…3


def _load_task(
    task_csv: Path,
    rich_npy: Path | None = None,
    scenario_space: str = "full",
):
    if scenario_space not in SCENARIO_SPACES:
        raise ValueError(f"unknown scenario_space={scenario_space!r}; "
                         f"choose from {SCENARIO_SPACES}")

    df = read_task_csv(task_csv)
    y_target = df["target_failure"].to_numpy(np.float64)
    y_proxy = df["proxy_failure"].to_numpy(np.float64)

    seqs = sorted(df["sequence_id"].astype(str).str.zfill(4).unique())
    onehot = np.stack(
        [(df["sequence_id"].astype(str).str.zfill(4) == s).to_numpy(np.float64) for s in seqs],
        axis=1,
    )
    seq_cols = [f"seq_{s}" for s in seqs]

    if scenario_space in ("thin", "thin_det"):
        cols = [c for c in THIN_FEATURE_COLS if c in df.columns]
        if scenario_space == "thin":
            cols = [c for c in cols if not c.startswith("proxy_")]
        X_base = df[cols].to_numpy(np.float64)
        names = list(cols)
    else:
        bundled_rich = _ROOT / "data" / "kitti_vkitti" / "X_rich.npy"
        if rich_npy:
            rich_npy = Path(rich_npy)
        elif bundled_rich.exists():
            rich_npy = bundled_rich
        else:
            rich_npy = OUTPUT_ROOT / "X_rich.npy"
        if not rich_npy.exists() or np.load(rich_npy).shape[0] != len(df):
            X_rich, names = write_rich_features(task_csv=task_csv, output_npy=rich_npy)
        else:
            X_rich = np.load(rich_npy)
            names_path = rich_npy.with_name("X_rich_names.txt")
            names = names_path.read_text().splitlines() if names_path.exists() else [
                f"f{i}" for i in range(X_rich.shape[1])
            ]
        keep = []
        for i, name in enumerate(names):
            if scenario_space == "geom" and name not in _GEOM_NAMES:
                continue
            if scenario_space == "detector" and name not in _DET_NAMES:
                continue
            if scenario_space == "rich" and name in _PLS_NAMES:
                continue
            # full: keep everything
            keep.append(i)
        X_base = X_rich[:, keep]
        names = [names[i] for i in keep]

    X = np.concatenate([X_base, onehot], axis=1)
    X = StandardScaler().fit_transform(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))
    cols = list(names) + seq_cols
    return df, X, y_target, y_proxy, cols


def _shared_init(n: int, n_init: int, n_proxy_only: int, seed: int, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    init_p = cache_dir / "shared_initial_idx.npy"
    proxy_p = cache_dir / "shared_proxy_only_idx.npy"
    rng = np.random.default_rng(seed)
    if init_p.exists():
        initial_idx = np.load(init_p)
    else:
        initial_idx = np.sort(rng.choice(n, size=min(n_init, n), replace=False))
        np.save(init_p, initial_idx)
    if proxy_p.exists():
        proxy_only = np.load(proxy_p)
    else:
        remain = np.setdiff1d(np.arange(n), initial_idx)
        k = min(n_proxy_only, len(remain))
        proxy_only = np.sort(rng.choice(remain, size=k, replace=False)) if k else np.array([], int)
        np.save(proxy_p, proxy_only)
    return initial_idx.astype(int), proxy_only.astype(int)


def _train(X, y, epochs, dropout, lr, device, label, widths=None):
    print(f"  [BNN-{label}] n={len(X)} epochs={epochs} widths={widths or 'default'}", flush=True)
    model = HeteroBNNEmbedding(
        input_dim=X.shape[1], p_drop=dropout, widths=widths,
    ).to(device)
    train_bnn(model, X, y, epochs=epochs, lr=lr, device=device)
    return model


def _paired_cv_scores(
    target_model,
    X_all,
    y_target,
    y_proxy,
    train_idx,
    sl_idx,
    n_draws,
    device,
    beta_scale: float = 1.0,
):
    """
    Paired proxy-corrected failure score at the query (proxy known for all x):

        beta  = cov(y_t, y_p) / var(y_p)   on real-labelled train points
        theta = mean(y_proxy) over the pool
        mu(x) = mean(f(x)) + beta_scale * beta * (y_proxy(x) - theta)

    Returns mu_CV, beta, corr, f_mean, f_var (MC-dropout over n_draws).
    """
    yt = y_target[train_idx]
    yp = y_proxy[train_idx]
    if len(train_idx) >= 3 and np.var(yp) > 1e-8:
        beta = float(np.cov(yt, yp)[0, 1] / np.var(yp))
        # Shrink slightly for small-n stability; keep sign (positive link)
        beta *= len(train_idx) / (len(train_idx) + 5.0)
        beta = float(np.clip(beta, -1.5, 1.5))
    else:
        beta = 0.0
    beta *= float(beta_scale)
    corr = float(np.corrcoef(yt, yp)[0, 1]) if len(train_idx) >= 3 and np.std(yp) > 1e-8 and np.std(yt) > 1e-8 else 0.0
    theta = float(np.mean(y_proxy))

    f_draws = batched_posterior_draws(
        target_model, X_all[sl_idx], n_draws=n_draws, device=device,
    )
    f_mean = f_draws.mean(axis=0)
    f_var = f_draws.var(axis=0)  # sample variance over MC-dropout draws
    g = y_proxy[sl_idx]
    mu = f_mean + beta * (g - theta)
    mu = np.clip(mu, -0.5, 1.5)
    betas = np.full(len(sl_idx), beta, dtype=np.float64)
    corrs = np.full(len(sl_idx), corr, dtype=np.float64)
    return mu, betas, corrs, f_mean, f_var


def _minmax01(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Map finite values to [0, 1]; non-finite → median then scale."""
    x = np.asarray(x, dtype=float)
    fin = np.isfinite(x)
    if not fin.any():
        return np.zeros_like(x)
    med = np.nanmedian(x[fin])
    x = np.where(fin, x, med)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < eps:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo + eps)


def _greedy_pick(sl_idx: np.ndarray, score: np.ndarray, batch_size: int):
    """Pick top-`batch_size` by score (high failure first)."""
    k = min(int(batch_size), len(sl_idx))
    if k <= 0:
        return np.array([], int), np.array([], int)
    order = np.argsort(-np.asarray(score, dtype=np.float64))[:k]
    return sl_idx[order], order


def _run_al(args, mode, X_all, y_target, y_proxy, initial_idx, proxy_only_idx, out_dir, device):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(X_all)
    rng = np.random.default_rng(args.seed)
    train_idx = initial_idx.copy()
    # Proxy training pool: paired init + proxy-only (+ later acquired paired)
    proxy_train_idx = np.unique(np.concatenate([initial_idx, proxy_only_idx]))
    selected = np.zeros(n, bool)
    selected[train_idx] = True
    # Proxy-only stay in the pool (can later receive a real label).
    logs = []
    use_sim_proxy = mode == "with_beta"
    use_bnn_proxy = mode == "bnn_proxy"
    use_offline_proxy = mode == "offline_proxy"
    use_proxy_channel = use_sim_proxy or use_bnn_proxy or use_offline_proxy
    no_surrogate = bool(getattr(args, "no_surrogate", False))
    mi_support = str(getattr(args, "mi_support", "target"))
    if no_surrogate and args.acq_score != "mi":
        raise ValueError("--no-surrogate requires --acq-score mi")

    # Offline proxy surrogate: fit once on proxy labels, freeze, cache g(x)
    g_offline = None
    g_offline_var = None
    if use_offline_proxy and not no_surrogate:
        frac = float(np.clip(args.offline_proxy_frac, 0.05, 1.0))
        n_off = max(10, int(round(frac * n)))
        off_idx = np.sort(rng.choice(n, size=n_off, replace=False))
        print(
            f"[offline_proxy] fitting surrogate on {n_off}/{n} "
            f"proxy labels (frac={frac:.2f})",
            flush=True,
        )
        proxy_model = _train(
            X_all[off_idx], y_proxy[off_idx],
            args.bnn_epochs, args.dropout, args.lr, device, "offline_proxy",
            widths=args.bnn_widths,
        )
        g_draws = batched_posterior_draws(
            proxy_model, X_all, n_draws=args.n_pair_local, device=device,
        )
        g_offline = g_draws.mean(axis=0)
        g_offline_var = g_draws.var(axis=0)
        # Diagnostic: how well surrogate recovers true proxy
        r = float(np.corrcoef(g_offline, y_proxy)[0, 1])
        mae = float(np.mean(np.abs(g_offline - y_proxy)))
        print(f"[offline_proxy] ŷ vs y_proxy: pearson={r:.3f} mae={mae:.4f}", flush=True)
        np.save(out_dir / "g_offline.npy", g_offline)
        np.save(out_dir / "g_offline_var.npy", g_offline_var)
        (out_dir / "offline_proxy_meta.json").write_text(json.dumps(dict(
            n_offline=int(n_off), frac=frac, pearson_vs_proxy=r, mae_vs_proxy=mae,
            n_draws=int(args.n_pair_local),
        ), indent=2))

    print(
        f"[{mode}] n_init_real={len(train_idx)}  "
        f"n_proxy_train={len(proxy_train_idx)}  "
        f"(proxy_only={len(proxy_only_idx)})  "
        f"no_surrogate={no_surrogate}  mi_support={mi_support}",
        flush=True,
    )

    for it in range(args.n_acq_iters):
        print(f"\n[{mode.upper()} ITER {it:02d}] real_train={len(train_idx)}  "
              f"proxy_train={len(proxy_train_idx)}", flush=True)

        target_model = None
        if not no_surrogate:
            target_model = _train(
                X_all[train_idx], y_target[train_idx],
                args.bnn_epochs, args.dropout, args.lr, device, "target",
                widths=args.bnn_widths,
            )
            # Proxy labels enter via local CV (y_cv); optional proxy BNN if requested.
            if use_sim_proxy and args.train_proxy_bnn:
                _train(
                    X_all[proxy_train_idx], y_proxy[proxy_train_idx],
                    args.bnn_epochs, args.dropout, args.lr, device, "proxy",
                    widths=args.bnn_widths,
                )
        else:
            print("  [no-surrogate] skip BNN train; score by MI only", flush=True)

        # Candidates: not yet real-labelled
        unselected = np.where(~selected)[0]
        if len(unselected) == 0:
            break
        cand = (
            rng.choice(unselected, args.candidate_pool_size, replace=False)
            if len(unselected) > args.candidate_pool_size else unselected
        )

        if args.no_mi:
            # No MI gate: score the full candidate pool with CV / f_mean
            sl_idx = cand.copy()
            mi_sl = np.full(len(sl_idx), np.nan)
        else:
            support_idx = proxy_train_idx if mi_support == "proxy" else train_idx
            if len(support_idx) < 2:
                raise RuntimeError(
                    f"MI support too small (n={len(support_idx)}); "
                    "need >=2 points (use --mi-support proxy with proxy-only init)."
                )
            mi_cand, *_ = fit_support_and_compute_mi(
                X_support=X_all[support_idx].astype(np.float64),
                X_query=X_all[cand].astype(np.float64),
                n_clusters=0,
                random_state=args.seed + it,
            )
            mi_cand = np.nan_to_num(mi_cand, nan=0.0)
            # SCOUT: MI shortlist only (no proxy-shortlist arm)
            sl_n = min(max(1, int(args.mi_shortlist)), len(cand))
            sl_local = np.argsort(mi_cand)[-sl_n:]
            sl_idx = cand[sl_local]
            mi_map = {int(c): float(m) for c, m in zip(cand, mi_cand)}
            mi_sl = np.array([mi_map[int(g)] for g in sl_idx], float)

        # Build y_cv channel for local CV (pool-wide proxy / surrogate)
        g_var_pool = None
        if use_sim_proxy:
            y_cv = y_proxy
            g_var_pool = np.zeros(n, dtype=np.float64)
        elif use_offline_proxy and g_offline is not None:
            y_cv = g_offline
            g_var_pool = g_offline_var
        elif use_bnn_proxy and target_model is not None:
            g_draws = batched_posterior_draws(
                target_model, X_all, n_draws=args.n_pair_local, device=device,
            )
            y_cv = g_draws.mean(axis=0)
            g_var_pool = g_draws.var(axis=0)
        else:
            y_cv = y_proxy if no_surrogate else None
            if no_surrogate:
                g_var_pool = np.zeros(n, dtype=np.float64)

        # Target BNN posterior at shortlist (for logging); skipped w/ no-surrogate
        if target_model is not None:
            f_draws = batched_posterior_draws(
                target_model, X_all[sl_idx], n_draws=args.n_pair_local, device=device,
            )
            f_mean = f_draws.mean(axis=0)
            f_var = f_draws.var(axis=0)
        else:
            f_mean = np.full(len(sl_idx), np.nan)
            f_var = np.full(len(sl_idx), np.nan)

        if args.acq_score == "mi":
            # CV off: acquire by MI only (failure / proxy unused for ranking)
            mu_cv = f_mean
            cv_var = f_var
            score = mi_sl
            beta_hat = np.zeros(len(sl_idx))
            corr_hat = np.zeros(len(sl_idx))
            if y_cv is not None:
                g_mean_sl = y_cv[sl_idx]
                g_var_sl = (
                    g_var_pool[sl_idx] if g_var_pool is not None
                    else np.full(len(sl_idx), np.nan)
                )
            else:
                g_mean_sl = np.full(len(sl_idx), np.nan)
                g_var_sl = np.full(len(sl_idx), np.nan)
        elif use_proxy_channel:
            # Classical SCOUT local CV over embedding neighbourhoods
            mu_cv, cv_var, beta_hat, corr_hat = compute_local_cv_parallel(
                model=target_model,
                X_query=X_all[sl_idx],
                query_global_idx=sl_idx,
                X_target=X_all,
                y_cv=y_cv,
                n_pair_local=args.n_pair_local,
                k_unpaired=args.k_unpaired,
                radius=args.cv_radius,
                fallback_k=args.cv_fallback_k,
                n_jobs=args.cv_n_jobs,
                device=device,
            )
            score = mu_cv  # estimate of E[y_target | x]; pick high failure
            g_mean_sl = y_cv[sl_idx]
            g_var_sl = (
                g_var_pool[sl_idx] if g_var_pool is not None
                else np.full(len(sl_idx), np.nan)
            )
        else:
            mu_cv = f_mean
            cv_var = f_var
            score = f_mean
            beta_hat = np.zeros(len(sl_idx))
            corr_hat = np.zeros(len(sl_idx))
            g_mean_sl = np.full(len(sl_idx), np.nan)
            g_var_sl = np.full(len(sl_idx), np.nan)

        if args.acq_pick == "greedy":
            sel_global, _ = _greedy_pick(sl_idx, score, args.batch_acq_size)
        else:
            # scout_baseline argmins cv_mean (TTC↓); we want failure↑ → pass -score
            sel_global, _ = _cluster_and_pick(
                sl_idx=sl_idx,
                X_all=X_all,
                cv_mean=-score,
                batch_size=args.batch_acq_size,
                random_state=args.seed + it * 1000,
            )
        sel_global = np.array([g for g in sel_global if not selected[g]], int)
        chosen = set(sel_global.tolist())
        for j, gj in enumerate(sl_idx):
            logs.append(dict(
                iteration=it,
                global_idx=int(gj),
                is_selected=int(gj in chosen),
                mi=float(mi_sl[j]),
                score=float(score[j]),
                f_mean=float(f_mean[j]),
                f_var=float(f_var[j]),
                mu_cv=float(mu_cv[j]),
                cv_var=float(cv_var[j]),
                g_mean=float(g_mean_sl[j]),
                g_var=float(g_var_sl[j]),
                cv_beta=float(beta_hat[j]),
                cv_corr=float(corr_hat[j]),
                target=float(y_target[gj]),
                proxy=float(y_proxy[gj]),
                n_draws=int(args.n_pair_local),
            ))
        if len(sel_global) == 0:
            continue
        selected[sel_global] = True
        train_idx = np.concatenate([train_idx, sel_global])
        proxy_train_idx = np.unique(np.concatenate([proxy_train_idx, sel_global]))
        print(
            f"  selected {len(sel_global)}  "
            f"mean_score={score[np.isin(sl_idx, sel_global)].mean():.4g}  "
            f"mean_f={f_mean[np.isin(sl_idx, sel_global)].mean():.4g}  "
            f"mean_muCV={mu_cv[np.isin(sl_idx, sel_global)].mean():.4g}  "
            f"mean_beta={beta_hat[np.isin(sl_idx, sel_global)].mean():.4g}  "
            f"mean_corr={corr_hat[np.isin(sl_idx, sel_global)].mean():.4g}",
            flush=True,
        )

    np.save(out_dir / "train_indices.npy", train_idx)
    np.save(out_dir / "proxy_train_indices.npy", proxy_train_idx)
    pd.DataFrame(logs).to_csv(out_dir / "acquisition_log.csv", index=False)
    return train_idx


def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    task_csv = Path(args.task_csv)
    df, X_all, y_target, y_proxy, feat_cols = _load_task(
        task_csv, scenario_space=args.scenario_space,
    )
    n = len(df)
    print(f"[kitti_vkitti] N={n} mode={args.mode} seed={args.seed} device={device}")
    print(f"  scenario_space={args.scenario_space}  features={len(feat_cols)}  "
          f"pearson(proxy,target)={np.corrcoef(y_proxy, y_target)[0,1]:.3f}")

    out_root = Path(args.output_dir) / f"seed_{args.seed}" / args.mode
    # Shared init across modes (and optionally across scenario-space ablations)
    cache_dir = (
        Path(args.init_cache_dir) / f"seed_{args.seed}"
        if args.init_cache_dir
        else Path(args.output_dir) / f"seed_{args.seed}"
    )
    n_proxy_only = args.n_proxy_only if args.mode == "with_beta" else 0
    # Shared paired init across modes; proxy-only only for with_beta (still saved shared for fairness)
    n_proxy_only_shared = args.n_proxy_only
    initial_idx, proxy_only_idx = _shared_init(
        n, args.n_init, n_proxy_only_shared, args.seed, cache_dir,
    )
    if args.mode != "with_beta":
        proxy_only_idx = np.array([], int)

    vis_idx = choose_vis_indices(n, np.array([], int), min(n, args.tsne_max), args.seed)
    _ = get_or_compute_tsne(
        X_all[vis_idx],
        cache_path=Path(args.output_dir) / "tsne_cache_v2.npy",
        perplexity=min(30, max(5, len(vis_idx) // 4)),
        random_state=args.seed,
        force_recompute=False,
    )

    meta = dict(
        mode=args.mode, seed=args.seed, n=n,
        n_init=args.n_init, n_proxy_only=int(len(proxy_only_idx)),
        n_acq_iters=args.n_acq_iters, batch_acq_size=args.batch_acq_size,
        corr_gate=args.corr_gate, device=device,
        task_csv=str(task_csv), feature_cols=feat_cols,
        scenario_space=args.scenario_space,
        acq_score=args.acq_score,
        no_mi=bool(args.no_mi),
        no_surrogate=bool(args.no_surrogate),
        mi_support=args.mi_support,
        sev_cutoff=args.sev_cutoff,
        mi_shortlist=args.mi_shortlist,
        acq_pick=args.acq_pick,
        experiment="v2_proxy_bnn_enriched",
    )
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "run_meta.json").write_text(json.dumps(meta, indent=2))

    _run_al(
        args, args.mode, X_all, y_target, y_proxy,
        initial_idx, proxy_only_idx, out_root, device,
    )
    print(f"[done] → {out_root}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["with_beta", "offline_proxy", "bnn_proxy", "real_only"],
                   required=True)
    p.add_argument("--task-csv", type=str,
                   default=str(_ROOT / "data" / "kitti_vkitti" / "paired_detection_task_linked.csv"))
    p.add_argument("--output-dir", type=str,
                   default=str(_ROOT / "outputs" / "kitti_vkitti" / "scout_v2"))
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--scenario-space", choices=list(SCENARIO_SPACES), default="full",
                   help="Feature embedding / scenario-space definition")
    p.add_argument("--init-cache-dir", type=str, default="",
                   help="Optional shared init cache (for fair space ablations)")
    # Defaults: cluster batch pick (pipeline feature) + shorter budget (15 iters)
    p.add_argument("--n-init", type=int, default=10)
    p.add_argument("--n-proxy-only", type=int, default=40,
                   help="Extra proxy-only scenarios (default 4× n_init)")
    p.add_argument("--n-acq-iters", type=int, default=15)
    p.add_argument("--batch-acq-size", type=int, default=5)
    p.add_argument("--candidate-pool-size", type=int, default=5000)
    p.add_argument("--mi-shortlist", type=int, default=1000,
                   help="MI shortlist size (SCOUT-style)")
    p.add_argument("--proxy-shortlist", type=int, default=0,
                   help="Unused with local CV (kept for CLI compat)")
    p.add_argument("--proxy-blend", type=float, default=0.0,
                   help="Unused with local CV (kept for CLI compat)")
    p.add_argument("--offline-proxy-frac", type=float, default=1.0,
                   help="Fraction of pool proxy labels used to fit offline surrogate")
    p.add_argument("--beta-scale", type=float, default=1.0,
                   help="Unused with local CV (kept for CLI compat)")
    p.add_argument("--acq-pick", choices=["cluster", "greedy"], default="cluster",
                   help="Batch selection: diverse clusters (default) or greedy top-score")
    p.add_argument("--acq-score", choices=["cv", "mi", "micv"], default="cv",
                   help="cv: rank by μ_CV / f_mean; mi: MI only; "
                        "micv: minmax(MI)+minmax(μ_CV) vanilla blend")
    p.add_argument("--sev-cutoff", type=float, default=0.3,
                   help="Keep score>=cutoff before cluster/greedy (KITTI failure↑). "
                        "Use <0 to disable. Skipped when --acq-score mi.")
    p.add_argument("--no-mi", action="store_true",
                   help="Disable MI shortlist; score full candidate pool with CV/f_mean")
    p.add_argument("--no-surrogate", action="store_true",
                   help="Skip all BNN/offline-proxy training (requires --acq-score mi)")
    p.add_argument("--mi-support", choices=["target", "proxy"], default="target",
                   help="Embedding support for MI: target-labelled train set, or "
                        "proxy set (paired init + proxy-only + acquired)")
    p.add_argument("--cv-radius", type=float, default=CV_RADIUS,
                   help="Local CV neighbourhood radius in embedding space")
    p.add_argument("--cv-fallback-k", type=int, default=CV_FALLBACK_K,
                   help="kNN fallback size for local CV")
    p.add_argument("--cv-n-jobs", type=int, default=-1,
                   help="joblib workers for local CV")
    p.add_argument("--bnn-epochs", type=int, default=800)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bnn-widths", type=int, nargs="+", default=None,
                   help="BNN hidden widths (default 96 24 6)")
    p.add_argument("--n-pair-local", type=int, default=N_TARGET_DRAWS)
    p.add_argument("--k-unpaired", type=int, default=N_UNPAIRED)
    p.add_argument("--corr-gate", type=float, default=0.0)
    p.add_argument("--train-proxy-bnn", action="store_true",
                   help="Also train a proxy BNN (not required for local CV)")
    p.add_argument("--tsne-max", type=int, default=2000)
    p.add_argument("--cpu", action="store_true")
    main(p.parse_args())
