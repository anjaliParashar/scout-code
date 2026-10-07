#!/usr/bin/env python3
"""
utils.simpler/cv_utils_simpler.py
-----------------------------------
Control-variates estimator — all quantities are BNN draws, no subprocess calls.

mu_CV at shortlist scenario x
-------------------------------
  f  (real-side, paired, n=4):
      4 draws from the TARGET BNN posterior at x.

  g_paired  (proxy, paired, n=4):
      4 draws from the PROXY BNN posterior at x.

  g_unpaired  (proxy, unpaired, k=10):
      10 draws from the PROXY BNN — one draw each at 10 perturbed copies of x.
      Perturbed copies vary ONLY object_x, object_y, cam_dx, cam_dy.
      All other fields (task, distractors, brightness, contrast) are fixed.

CV estimator:
    beta   = (k/(k+n)) * cov(f, g_paired) / var(g_paired)
    mu_hat = mean(f - beta * g_paired) + beta * mean(g_unpaired)
    var_hat = (var(f) + beta² var(g) - 2β cov(f,g)) / n
              + beta² var(g_unpaired) / k
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Perturbation parameters
# ---------------------------------------------------------------------------

PERTURB_OBJ_RADIUS:  float = 0.03   # ±3 cm object position
PERTURB_CAM_RADIUS:  float = 0.01   # ±1 cm camera shift
N_PERTURB_SCENARIOS: int   = 10     # 10 perturbed scenarios, 1 BNN draw each


# ---------------------------------------------------------------------------
# Scenario perturbation — pure numpy, no rollouts
# ---------------------------------------------------------------------------

def sample_perturbed_scenarios(
    scenario,
    n_scenarios: int   = N_PERTURB_SCENARIOS,
    obj_radius:  float = PERTURB_OBJ_RADIUS,
    cam_radius:  float = PERTURB_CAM_RADIUS,
    rng: np.random.Generator = None,
    seed: int = 0,
) -> list:
    """
    Return `n_scenarios` perturbed copies of `scenario`.
    Only object_x, object_y, cam_dx, cam_dy are varied within L2 balls.
    All other fields (task, distractors, brightness, contrast) are fixed.
    """
    from utils.simpler.domain import Scenario, DesignSpaceLimits
    if rng is None:
        rng = np.random.default_rng(seed)
    limits = DesignSpaceLimits()
    out    = []
    for i in range(n_scenarios):
        angle_obj = rng.uniform(0, 2 * np.pi)
        r_obj     = obj_radius * np.sqrt(rng.uniform(0, 1))
        angle_cam = rng.uniform(0, 2 * np.pi)
        r_cam     = cam_radius * np.sqrt(rng.uniform(0, 1))
        out.append(Scenario(
            scenario_id = -(i + 1),
            object_x    = float(np.clip(
                scenario.object_x + r_obj * np.cos(angle_obj),
                limits.object_x_min, limits.object_x_max)),
            object_y    = float(np.clip(
                scenario.object_y + r_obj * np.sin(angle_obj),
                limits.object_y_min, limits.object_y_max)),
            cam_dx      = float(np.clip(
                scenario.cam_dx + r_cam * np.cos(angle_cam),
                limits.cam_dx_min, limits.cam_dx_max)),
            cam_dy      = float(np.clip(
                scenario.cam_dy + r_cam * np.sin(angle_cam),
                limits.cam_dy_min, limits.cam_dy_max)),
            brightness  = scenario.brightness,
            contrast    = scenario.contrast,
            task        = scenario.task,
            distractors = scenario.distractors,
        ))
    return out


# ---------------------------------------------------------------------------
# CV estimator
# ---------------------------------------------------------------------------

@dataclass
class CVResult:
    mu_hat:     float
    var_hat:    float
    beta:       float
    n_paired:   int
    n_unpaired: int


def cv_estimate(
    f:          np.ndarray,   # (4,) target BNN draws at x
    g_paired:   np.ndarray,   # (4,) proxy  BNN draws at x
    g_unpaired: np.ndarray,   # (10,) proxy BNN draws at 10 perturbed x's
) -> CVResult:
    """
    Standard control-variates estimator.

    f          : n=4 target BNN posterior draws  (real-side, paired)
    g_paired   : n=4 proxy  BNN posterior draws  (proxy-side, paired, same x)
    g_unpaired : k=10 proxy BNN draws at 10 perturbed x's, one each (unpaired)

    Falls back to raw mean(f) if variance conditions fail.
    """
    n    = len(f)
    k    = len(g_unpaired)
    mu_f = float(np.mean(f))

    if n <= 1 or len(g_paired) <= 1 or k == 0:
        return CVResult(mu_hat=mu_f, var_hat=float(np.var(f)) + 1e-8,
                        beta=0.0, n_paired=n, n_unpaired=k)

    n_pair = min(n, len(g_paired))
    f_p    = np.asarray(f[:n_pair],        dtype=np.float64)
    g_p    = np.asarray(g_paired[:n_pair], dtype=np.float64)
    g_u    = np.asarray(g_unpaired,        dtype=np.float64)

    var_g  = float(np.var(g_p, ddof=1))
    var_f  = float(np.var(f_p, ddof=1))
    var_gu = float(np.var(g_u, ddof=1)) if k > 1 else 0.0
    cov_fg = float(np.cov(f_p, g_p, ddof=1)[0, 1]) if n_pair > 1 else 0.0
    mu_gu  = float(np.mean(g_u))

    if var_g < 1e-10:
        return CVResult(mu_hat=mu_f, var_hat=var_f / max(n_pair, 1) + 1e-8,
                        beta=0.0, n_paired=n_pair, n_unpaired=k)

    beta    = (k / (k + n_pair)) * (cov_fg / var_g)
    mu_hat  = float(np.mean(f_p - beta * g_p)) + beta * mu_gu
    var_hat = (
        max(var_f + beta**2 * var_g - 2 * beta * cov_fg, 0.0) / n_pair
        + beta**2 * var_gu / k
    ) + 1e-10

    return CVResult(
        mu_hat     = float(np.clip(mu_hat, 0.0, 1.0)),
        var_hat    = float(var_hat),
        beta       = float(beta),
        n_paired   = n_pair,
        n_unpaired = k,
    )
