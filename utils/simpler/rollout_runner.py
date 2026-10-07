#!/usr/bin/env python3
"""
utils.simpler/rollout_runner.py
---------------------------------
Launch paired_simpler_scenario_rollouts_gap.py for a given scenario,
wait for it to finish, and return the outcome y values.

Two evaluation modes
--------------------
  run_proxy_rollouts(scenario, n=10)   → y_proxy  (cheap, sim)
  run_target_rollouts(scenario, n=2)   → y_target (expensive, real)

Both modes call the same script with --n-trajs and --gap-applies-to.

The fixed gap parameters are always passed unchanged (they represent the
constant sim-real rendering gap, not a design variable).

HOOK: Set SCRIPT_PATH to the absolute path of
      paired_simpler_scenario_rollouts_gap.py in your environment.
HOOK: Set CKPT_PATH to the RT-1 checkpoint path.
HOOK: Set FIXED_GAP_* to the gap parameters you want to hold constant.
HOOK: Add any additional fixed CLI flags in FIXED_EXTRA_FLAGS.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.simpler.domain import Scenario, aggregate_y

# ---------------------------------------------------------------------------
# HOOKS — configure these for your environment
# ---------------------------------------------------------------------------

# HOOK: Set to the absolute path of the rollout script
SCRIPT_PATH: str = os.environ.get("SIMPLER_ROLLOUT_SCRIPT", "")

# HOOK: Set to the RT-1 checkpoint directory
CKPT_PATH: str = os.environ.get("SIMPLER_CKPT_PATH", "")

# HOOK: Fixed gap parameters (held constant across all experiments)
FIXED_GAP_BLUR:          float = 0.5
FIXED_GAP_NOISE_STD:     float = 1.5
FIXED_GAP_OCCLUSION_FRAC: float = 0.03
FIXED_GAP_APPLIES_TO:    str   = "target"

# HOOK: Target camera mode
TARGET_CAMERA_MODE: str = "image"

# HOOK: Episode max steps (for normalising y)
MAX_STEPS: int = 80

# HOOK: Additional fixed flags to pass to every rollout call
FIXED_EXTRA_FLAGS: List[str] = [
    "--distractor-config", "less",
    "--fixed-episode-id", "0",
]


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------


def _require_rollout_config() -> None:
    missing = []
    if not SCRIPT_PATH:
        missing.append("SIMPLER_ROLLOUT_SCRIPT")
    if not CKPT_PATH:
        missing.append("SIMPLER_CKPT_PATH")
    if missing:
        raise RuntimeError(
            "SIMPLER rollouts need " + " and ".join(missing) + " set to paths inside your SimplerEnv checkout."
        )

def _build_cmd(
    scenario:      Scenario,
    logging_root:  str,
    n_trajs:       int,
    seed:          int = 0,
    verbose:       bool = False,
) -> List[str]:
    """Build the full CLI command for one scenario evaluation."""
    _require_rollout_config()
    cli = scenario.to_cli_args()
    cmd = [
        sys.executable, SCRIPT_PATH,
        "--ckpt-path",          CKPT_PATH,
        "--task",               cli["task"],
        "--logging-root",       logging_root,
        "--n-trajs",            str(n_trajs),
        "--seed",               str(seed),
        "--fixed-episode-id",   "0",
        "--object-x",           str(cli["object_x"]),
        "--object-y",           str(cli["object_y"]),
        "--cam-dx",             str(cli["cam_dx"]),
        "--cam-dy",             str(cli["cam_dy"]),
        "--brightness",         str(cli["brightness"]),
        "--contrast",           str(cli["contrast"]),
        "--target-camera-mode", TARGET_CAMERA_MODE,
        "--gap-applies-to",     FIXED_GAP_APPLIES_TO,
        "--gap-blur",           str(FIXED_GAP_BLUR),
        "--gap-noise-std",      str(FIXED_GAP_NOISE_STD),
        "--gap-occlusion-frac", str(FIXED_GAP_OCCLUSION_FRAC),
        # "--distractor-config",  "less",
        "--distractor-json",    cli["distractor_json"],
    ]
    if verbose:
        cmd.append("--verbose")
    return cmd


def _scenario_logging_dir(logging_root: str, scenario: Scenario) -> Path:
    """
    Returns the directory where paired_result.json files will be written.

    HOOK: This must match the logging convention in
    paired_simpler_scenario_rollouts_gap.py.  Currently the script writes:
      <logging_root>/<task>/rt1/<ckpt_basename>/scenario_XXXX/
    We write results to scenario_<scenario_id:04d>/.
    """
    from pathlib import Path as _Path
    ckpt_basename = _Path(CKPT_PATH).name

    return (
        _Path(logging_root)/ scenario.task/ "rt1"/ ckpt_basename/ "scenario_0000"
    )


def run_rollouts(
    scenario:     Scenario,
    logging_root: str,
    n_trajs:      int,
    realization:  str = "target",
    seed:         int = 0,
    verbose:      bool = False,
    dry_run:      bool = False,
) -> Tuple[float, float, int]:
    """
    Run n_trajs rollouts for `scenario`, return (mean_y, std_y, n_found).

    Parameters
    ----------
    scenario     : scenario to evaluate
    logging_root : root output directory
    n_trajs      : number of rollouts to run
    realization  : "target" (real) or "proxy" (sim)
    seed         : random seed for rollout script
    verbose      : pass --verbose to the script
    dry_run      : if True, skip subprocess and return dummy values

    Returns
    -------
    mean_y : float — mean performance metric
    std_y  : float — std across rollouts
    n      : int   — number of rollouts found on disk
    """
    if dry_run:
        print(f"[DRY RUN] scenario={scenario.scenario_id} realization={realization} n={n_trajs}")
        return float(np.random.default_rng(scenario.scenario_id).uniform(0.5, 1.0)), 0.1, n_trajs

    cmd = _build_cmd(scenario, logging_root, n_trajs, seed, verbose)
    print(f"[RUN] scenario={scenario.scenario_id} {realization} n={n_trajs}")
    print(f"      {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=not verbose, text=True)
    if result.returncode != 0:
        print(f"[WARN] rollout failed (rc={result.returncode})")
        if not verbose and result.stderr:
            print(result.stderr[-2000:])

    scenario_dir = _scenario_logging_dir(logging_root, scenario)
    print("[SCENARIO DIR]",scenario_dir)
    return aggregate_y(scenario_dir, realization=realization, max_steps=MAX_STEPS)


def run_proxy_rollouts(
    scenario:     Scenario,
    logging_root: str,
    n_proxy:      int = 10,
    seed:         int = 0,
    verbose:      bool = False,
    dry_run:      bool = False,
) -> Tuple[float, float, int]:
    """Run n_proxy sim (proxy) rollouts."""
    return run_rollouts(scenario, logging_root, n_proxy,
                        realization="proxy", seed=seed,
                        verbose=verbose, dry_run=dry_run)


def run_target_rollouts(
    scenario:     Scenario,
    logging_root: str,
    n_target:     int = 2,
    seed:         int = 0,
    verbose:      bool = False,
    dry_run:      bool = False,
) -> Tuple[float, float, int]:
    """Run n_target real (target) rollouts."""
    return run_rollouts(scenario, logging_root, n_target,
                        realization="target", seed=seed,
                        verbose=verbose, dry_run=dry_run)
