"""Shared SCOUT estimators.

Domain tasks (KITTI, SIMPLER, quadruped, toy2D) import this package.
It does not import any task module.
"""

from scout.al import compute_hooks, lambda_schedule, minmax01, select_batch
from scout.bnn import HeteroBNNEmbedding, batched_posterior_draws, mc_predict, train_bnn
from scout.cv import compute_local_cv_parallel, control_variates_estimator
from scout.mi import fit_support_and_compute_mi
from scout.seed import set_seed

__all__ = [
    "HeteroBNNEmbedding",
    "batched_posterior_draws",
    "compute_hooks",
    "compute_local_cv_parallel",
    "control_variates_estimator",
    "fit_support_and_compute_mi",
    "lambda_schedule",
    "mc_predict",
    "minmax01",
    "select_batch",
    "set_seed",
    "train_bnn",
]
