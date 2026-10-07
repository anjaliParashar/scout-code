#!/usr/bin/env python3
"""
utils.simpler/al_state.py
--------------------------
Shared AL state container.

Dataset structure
-----------------
Target train set (M scenarios, 2 seeds each):
    train[i]       : Scenario
    y_targets[i]   : (2,) real outcomes — mean used as y_target[i]

Proxy train set (5*M scenarios, 2 seeds each):
    Composed of:
      (a) the same M target scenarios  →  y_proxies[i]    : (2,) sim outcomes
      (b) 4*M proxy-only scenarios     →  y_proxy_only[i] : (2,) sim outcomes
    X_proxy_train() and y_proxy_train_mean() concatenate (a) + (b).

Sequential acquisition (each round):
    batch_size=2 scenarios selected.
    Each gets 2 target seeds + 2 proxy seeds.
    y = mean over the 2 seeds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from utils.simpler.domain import Scenario, scenarios_to_array, save_pool, load_pool
from utils.simpler.cv_utils_simpler import CVResult


class ALState:

    def __init__(self, pool: List[Scenario]):
        self.pool: List[Scenario] = pool

        # --- Target train set (real, M scenarios × 2 seeds) ---
        self.train:        List[Scenario]   = []
        self.y_targets:    List[np.ndarray] = []  # each (2,)
        self.y_proxies:    List[np.ndarray] = []  # each (2,) paired proxy at same scenario

        # --- Proxy-only scenarios (sim only, 4*M scenarios × 2 seeds) ---
        self.proxy_only:       List[Scenario]   = []
        self.y_proxy_only:     List[np.ndarray] = []  # each (2,)

        self.cv_results: List[CVResult] = []

    # ── Feature matrices ──────────────────────────────────────────────────

    def X_train(self) -> np.ndarray:
        """(M, 42) target training features."""
        if not self.train:
            return np.empty((0, 42), dtype=np.float32)
        return scenarios_to_array(self.train)

    def X_proxy_train(self) -> np.ndarray:
        """(5*M, 42) proxy training features = train + proxy_only."""
        combined = self.train + self.proxy_only
        if not combined:
            return np.empty((0, 42), dtype=np.float32)
        return scenarios_to_array(combined)

    def X_pool(self) -> np.ndarray:
        """(N, 42) remaining pool features."""
        if not self.pool:
            return np.empty((0, 42), dtype=np.float32)
        return scenarios_to_array(self.pool)

    # ── y vectors ─────────────────────────────────────────────────────────

    def y_train_mean(self) -> np.ndarray:
        """(M,) mean y_target per target scenario (mean over 2 seeds)."""
        return np.array([np.mean(yt) for yt in self.y_targets], dtype=np.float32)

    def y_proxy_train_mean(self) -> np.ndarray:
        """(5*M,) mean y_proxy — paired then proxy-only, matching X_proxy_train()."""
        paired     = [np.mean(yp) for yp in self.y_proxies]
        proxy_only = [np.mean(yp) for yp in self.y_proxy_only]
        return np.array(paired + proxy_only, dtype=np.float32)

    def cv_mean_train(self) -> np.ndarray:
        """(M,) CV-estimated mean per target training scenario."""
        return np.array([r.mu_hat for r in self.cv_results], dtype=np.float64)

    # ── Adding observations ───────────────────────────────────────────────

    def add_observation(
        self,
        scenario: Scenario,
        y_target: np.ndarray,   # (2,) real outcomes, 2 seeds
        y_proxy:  np.ndarray,   # (2,) sim  outcomes, 2 seeds
    ):
        """
        Record a fully evaluated scenario (target + proxy, 2 seeds each).
        Removes the scenario from the pool.
        y values are the raw per-seed outcomes; mean is computed on demand.
        """
        self.train.append(scenario)
        self.y_targets.append(np.asarray(y_target, dtype=np.float64))
        self.y_proxies.append(np.asarray(y_proxy,  dtype=np.float64))
        self.pool = [s for s in self.pool if s.scenario_id != scenario.scenario_id]

    def add_proxy_only_observation(
        self,
        scenario: Scenario,
        y_proxy:  np.ndarray,   # (2,) sim outcomes, 2 seeds
    ):
        """
        Record a proxy-only evaluation (no real rollout, 2 seeds).
        Used during init to expand proxy training set to 5×M.
        Removes the scenario from the pool.
        """
        self.proxy_only.append(scenario)
        self.y_proxy_only.append(np.asarray(y_proxy, dtype=np.float64))
        self.pool = [s for s in self.pool if s.scenario_id != scenario.scenario_id]

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, output_dir: Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_pool(self.train,      output_dir / "train_scenarios.json")
        save_pool(self.proxy_only, output_dir / "proxy_only_scenarios.json")
        save_pool(self.pool,       output_dir / "pool_scenarios.json")

        for i, yt in enumerate(self.y_targets):
            np.save(output_dir / f"y_target_{i:04d}.npy", yt)
        for i, yp in enumerate(self.y_proxies):
            np.save(output_dir / f"y_proxy_{i:04d}.npy", yp)
        for i, yp in enumerate(self.y_proxy_only):
            np.save(output_dir / f"y_proxy_only_{i:04d}.npy", yp)

        np.save(output_dir / "y_targets_mean.npy",     self.y_train_mean())
        np.save(output_dir / "y_proxy_train_mean.npy", self.y_proxy_train_mean())

        print(f"[ALState] Saved  target={len(self.train)}  "
              f"proxy_only={len(self.proxy_only)}  pool={len(self.pool)}  "
              f"→ {output_dir}")

    @classmethod
    def load(cls, output_dir: Path) -> "ALState":
        output_dir  = Path(output_dir)
        train       = load_pool(output_dir / "train_scenarios.json")
        proxy_only  = (load_pool(output_dir / "proxy_only_scenarios.json")
                       if (output_dir / "proxy_only_scenarios.json").exists() else [])
        pool        = load_pool(output_dir / "pool_scenarios.json")

        state = cls(pool=pool)
        state.train      = train
        state.proxy_only = proxy_only

        for i in range(len(train)):
            p = output_dir / f"y_target_{i:04d}.npy"
            state.y_targets.append(np.load(p) if p.exists() else np.ones(2))
            p = output_dir / f"y_proxy_{i:04d}.npy"
            state.y_proxies.append(np.load(p) if p.exists() else np.ones(2))
        for i in range(len(proxy_only)):
            p = output_dir / f"y_proxy_only_{i:04d}.npy"
            state.y_proxy_only.append(np.load(p) if p.exists() else np.ones(2))

        print(f"[ALState] Loaded target={len(train)}  "
              f"proxy_only={len(proxy_only)}  pool={len(pool)}  from {output_dir}")
        return state
