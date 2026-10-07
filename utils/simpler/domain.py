#!/usr/bin/env python3
"""
utils.simpler/domain.py
------------------------
Design space, scenario grid, and outcome extraction for the SimplerEnv
active-learning experiment.

Design variables
----------------
  object_x, object_y   — target coke-can position on the table
  cam_dx, cam_dy       — camera image-space shift
  brightness           — shared scene brightness control
  contrast             — shared scene contrast control
  task                 — encodes coke-can orientation (3 values)
  d{0,1,2}_name        — distractor identity (8 categories each)
  d{0,1,2}_{x,y,yaw}   — distractor poses (continuous)

Feature vector: 42 dims (6 continuous + 3-task one-hot + 3×(8 name one-hot + 3 pose))

Outcome metric  y(scenario)
----------------------------
  y = step_at_success / total_steps   if success else 1.0

y in (0,1].  Lower = better.  Failure = 1.0.
  y_target = mean over n_target=2 real rollouts
  y_proxy  = mean over n_proxy=10 sim rollouts
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Catalogues
# ---------------------------------------------------------------------------

# HOOK: extend when new objects are added to ManiSkill2_real2sim
DISTRACTOR_NAMES: List[str] = [
    "opened_pepsi_can",
    # "opened_7up_can",
    # "opened_sprite_can",
    # "opened_fanta_can",
    "apple",
    "sponge",
    "orange",
    # "bridge_spoon_generated_modified",
]


TASK_NAMES: List[str] = [
    "google_robot_pick_horizontal_coke_can",
    "google_robot_pick_vertical_coke_can",
    "google_robot_pick_standing_coke_can",
]

FEATURE_DIM: int = 42   # 6 + 3 + 3*(8+3)

# ---------------------------------------------------------------------------
# Design space limits
# ---------------------------------------------------------------------------

@dataclass
class DesignSpaceLimits:
    """
    Min/max for all continuous design variables.
    HOOK: Adjust for your table setup and camera rig.
    """
    # object_x_min:   float = -0.20
    # object_x_max:   float = -0.04
    # object_y_min:   float =  0.10
    # object_y_max:   float =  0.30
    # cam_dx_min:     float = -0.04
    # cam_dx_max:     float =  0.04
    # cam_dy_min:     float = -0.04
    # cam_dy_max:     float =  0.04
    # brightness_min: float =  0.60
    # brightness_max: float =  1.20
    # contrast_min:   float =  0.80
    # contrast_max:   float =  1.40
    # dist_x_min:     float = -0.36
    # dist_x_max:     float = -0.10
    # dist_y_min:     float =  0.08
    # dist_y_max:     float =  0.32
    # dist_yaw_min:   float = -1.57
    # dist_yaw_max:   float =  1.57
    object_x_min:   float = -0.5
    object_x_max:   float = -0.1
    object_y_min:   float =  0.0
    object_y_max:   float =  0.4

    cam_dx_min:     float = -0.025
    cam_dx_max:     float =  0.025
    cam_dy_min:     float = -0.025
    cam_dy_max:     float =  0.025

    brightness_min: float =  0.70
    brightness_max: float =  1.10
    contrast_min:   float =  0.90
    contrast_max:   float =  1.25

    dist_x_min:     float = -0.5
    dist_x_max:     float = -0.1
    dist_y_min:     float =  0.0
    dist_y_max:     float =  0.4
    dist_yaw_min:   float = -3.14
    dist_yaw_max:   float =  3.14
    n_grid:         int   =  3


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    scenario_id:  int
    object_x:     float
    object_y:     float
    cam_dx:       float
    cam_dy:       float
    brightness:   float
    contrast:     float
    task:         str
    distractors:  List[Dict[str, Any]] = field(default_factory=list)

    def distractor_json(self) -> str:
        return json.dumps(self.distractors)

    def to_feature_vector(self) -> np.ndarray:
        vec: List[float] = [
            self.object_x, self.object_y,
            self.cam_dx,   self.cam_dy,
            self.brightness, self.contrast,
        ]

        # task_oh = len(TASK_NAMES)
        task_id = TASK_NAMES.index(self.task)
        task_oh = task_id/len(TASK_NAMES)
        # if self.task in TASK_NAMES:
        #     task_oh[TASK_NAMES.index(self.task)] = 1.0
        
        vec.extend([task_oh])
        dists = (self.distractors + [{} for _ in range(3)])[:3]
        for d in dists:
            # name_oh = [0.0] * len(DISTRACTOR_NAMES)
            
            # if d.get("name") in DISTRACTOR_NAMES:
            #     name_oh[DISTRACTOR_NAMES.index(d["name"])] = 1.0
            name_oh = DISTRACTOR_NAMES.index(d.get("name"))/len(DISTRACTOR_NAMES)
            vec.extend([name_oh])
            vec.extend([float(d.get("x", 0.0)),
                        float(d.get("y", 0.0)),
                        float(d.get("yaw", 0.0))])
        return np.array(vec, dtype=np.float32)

    def to_cli_args(self) -> Dict[str, Any]:
        return dict(
            task           = self.task,
            object_x       = self.object_x,
            object_y       = self.object_y,
            cam_dx         = self.cam_dx,
            cam_dy         = self.cam_dy,
            brightness     = self.brightness,
            contrast       = self.contrast,
            distractor_json= self.distractor_json(),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = self.to_cli_args()
        d["scenario_id"]  = self.scenario_id
        d["distractors"]  = self.distractors
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Scenario":
        return cls(
            scenario_id = int(d["scenario_id"]),
            object_x    = float(d["object_x"]),
            object_y    = float(d["object_y"]),
            cam_dx      = float(d["cam_dx"]),
            cam_dy      = float(d["cam_dy"]),
            brightness  = float(d["brightness"]),
            contrast    = float(d["contrast"]),
            task        = str(d["task"]),
            distractors = d.get("distractors", []),
        )


# ---------------------------------------------------------------------------
# Outcome extraction
# ---------------------------------------------------------------------------

def y_from_rollout(result: Dict[str, Any], realization: str = "target") -> float:
    """
    y = step_at_success / total_steps   if success else 1.0.

    y in (0, 1].  Lower = better.  Failure = 1.0.

    HOOK: If the env continues running after success, extract step_success
    from result[realization]["final_info"]["elapsed_steps"] at success=True
    rather than using timesteps directly.
    """
    r = result[realization]
    if r.get("success", False):
        steps = int(r.get("timesteps", 1))
        return float(steps) / float(max(steps, 1))   # = 1.0 / 1.0 = 1.0... see note
        # HOOK: actual step_at_success is ambiguous from the JSON alone.
        # The timesteps field = total steps run.  If success terminates the episode
        # then timesteps == step_at_success.  Adjust if needed.
        # A better metric: return 1.0 - float(steps) / float(max_steps)
        # where max_steps is the episode time limit.  Use: step_ratio = steps / MAX_STEPS.
    return 1.0


def y_from_rollout_v2(
    result: Dict[str, Any],
    realization: str = "target",
    max_steps: int = 80,
) -> float:
    """
    Normalised failure metric: y = steps / max_steps.

    y = 0 means instant success; y = 1 means failure / truncated at max_steps.
    This is probably more useful than the v1 metric.

    HOOK: Set max_steps to match your episode time limit.
    """
    r = result[realization]
    steps = int(r.get("timesteps", max_steps))
  
    if r['success']:
        # return float(steps) / float(max_steps)
        return 0.0
    else:
        return 1.0


def load_rollout_results(scenario_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all paired_result*.json files from a scenario directory.
    Supports both legacy single-file (paired_result.json) and numbered files.
    HOOK: Adjust glob if your logging convention differs.
    """
    results = []

    # for p in sorted(scenario_dir.glob("paired_result_*.json")):
    #     with open(p) as f:
    #         results.append(json.load(f))

    # if not results:
    p = scenario_dir / "paired_result.json"
    if p.exists():
        with open(p) as f:
            results.append(json.load(f))
    return results


def aggregate_y(
    scenario_dir: Path,
    realization: str = "target",
    max_steps: int = 80,
    n_expected: int = None,
) -> Tuple[float, float, int]:
    """
    Mean and std of y_from_rollout_v2 across all rollouts in scenario_dir.

    Returns (mean_y, std_y, n_found).
    Pads with 1.0 (failure) if n_found < n_expected.
    """
    results = load_rollout_results(scenario_dir)
    
    if not results:
        return 1.0, 0.0, 0
    ys = [y_from_rollout_v2(r, realization, max_steps) for r in results]

    if n_expected is not None and len(ys) < n_expected:
        ys.extend([1.0] * (n_expected - len(ys)))
    
    return float(np.mean(ys)), float(np.std(ys)), len(results)


# ---------------------------------------------------------------------------
# Scenario grid / pool construction
# ---------------------------------------------------------------------------

def _linspace(lo: float, hi: float, n: int) -> List[float]:
    if n == 1:
        return [float((lo + hi) / 2.0)]
    return [float(v) for v in np.linspace(lo, hi, n)]


def build_scenario_pool(
    limits:   DesignSpaceLimits = None,
    n_random: int = 500,
    seed:     int = 0,
    rng:      np.random.Generator = None,
) -> List[Scenario]:
    """
    Sample n_random scenarios uniformly from the grid cells of each variable.

    For the full factorial grid call build_scenario_grid_full() — only feasible
    for very small n_grid (e.g. 2).

    Parameters
    ----------
    limits   : DesignSpaceLimits (uses defaults if None)
    n_random : pool size
    seed     : used when rng is None
    rng      : seeded Generator

    Returns
    -------
    List[Scenario] of length n_random, each with a unique scenario_id.
    """
    if limits is None:
        limits = DesignSpaceLimits()
    if rng is None:
        rng = np.random.default_rng(seed)

    n = limits.n_grid
    obj_xs    = _linspace(limits.object_x_min,   limits.object_x_max,   n)
    obj_ys    = _linspace(limits.object_y_min,    limits.object_y_max,   n)
    cam_dxs   = _linspace(limits.cam_dx_min,      limits.cam_dx_max,     n)
    cam_dys   = _linspace(limits.cam_dy_min,      limits.cam_dy_max,     n)
    brights   = _linspace(limits.brightness_min,  limits.brightness_max, n)
    contrasts = _linspace(limits.contrast_min,    limits.contrast_max,   n)
    dist_xs   = _linspace(limits.dist_x_min,      limits.dist_x_max,     n)
    dist_ys   = _linspace(limits.dist_y_min,      limits.dist_y_max,     n)
    dist_yaws = _linspace(limits.dist_yaw_min,    limits.dist_yaw_max,   n)

    scenarios = []
    for sid in range(n_random):
        dists = []
        for _ in range(3):
            dists.append({
                "name": str(rng.choice(DISTRACTOR_NAMES)),
                "x":   float(rng.choice(dist_xs)),
                "y":   float(rng.choice(dist_ys)),
                "yaw": float(rng.choice(dist_yaws)),
            })
        scenarios.append(Scenario(
            scenario_id = sid,
            object_x    = float(rng.choice(obj_xs)),
            object_y    = float(rng.choice(obj_ys)),
            cam_dx      = float(rng.choice(cam_dxs)),
            cam_dy      = float(rng.choice(cam_dys)),
            brightness  = float(rng.choice(brights)),
            contrast    = float(rng.choice(contrasts)),
            task        = str(rng.choice(TASK_NAMES)),
            distractors = dists,
        ))
    return scenarios


def scenarios_to_array(scenarios: List[Scenario]) -> np.ndarray:
    """Stack feature vectors into (N, FEATURE_DIM) float32."""
    return np.stack([s.to_feature_vector() for s in scenarios], axis=0)


def save_pool(scenarios: List[Scenario], path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([s.to_dict() for s in scenarios], f, indent=2)


def load_pool(path: Path) -> List[Scenario]:
    with open(path) as f:
        dicts = json.load(f)
    return [Scenario.from_dict(d) for d in dicts]
