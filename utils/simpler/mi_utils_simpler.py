#!/usr/bin/env python3
"""
utils.simpler/mi_utils_simpler.py
-----------------------------------
Thin wrapper that imports fit_support_and_compute_mi from scout/mi.py
and applies it to the SimplerEnv feature vectors.

The API is identical — the feature vector (42-dim) plays the role of the
scenario embedding. No changes to the MI estimator itself are needed.

HOOK: If the al_project root is not on sys.path, set SIMPLER_AL_ROOT below.
"""

import os
import sys

# HOOK: Set to absolute path of the al_project root if not on sys.path
# _AL_PROJECT_ROOT = os.environ.get(
#     "SIMPLER_AL_ROOT",
#     os.path.join(os.path.dirname(__file__), "..", "..","..",".."),
# )
# breakpoint()
# if _AL_PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, _AL_PROJECT_ROOT)

from scout.mi import fit_support_and_compute_mi  # noqa: F401

# Default MI hyperparameters tuned for the 42-dim feature space.
# HOOK: Tune these after examining MI values on the initial seed.
MI_DEFAULTS = dict(
    n_clusters      = 0,    # suited to this smaller feature space
    radius_quantile = 0.90,
    pi_new          = 0.18,
    eps_exist       = 0.04,
    tau_scale       = 1.3,
    novelty_radius  = 0.75,
    gamma           = 0.25,
    random_state    = 0,
)
