# utils/toy2D/__init__.py
from utils.toy2D.domain import (
    sample_designs, real_mean_fn, real_var_fn,
    sim_mean_fn, sim_var_fn, sample_real, sample_sim,
    diamond_membership, latent_target_fn, make_grid, GAMMA,
)
from utils.toy2D.gp_utils import train_gp, gp_predict, gp_sample_functions, gp_posterior_prob_below
from utils.toy2D.mi_toy import (
    # MI — same entry points as scout.mi
    build_cluster_info,
    compute_mi_batch,
    fit_support_and_compute_mi,
    # toy2D-only
    cluster_by_radius,
    compute_cv_batch,
    control_variates_estimator,
)
from utils.toy2D.plot_toy import (
    plot_gp_mean_heatmap, plot_comparison_grid,
    plot_best_seen_curve, print_discovery_summary, plot_discovery_bar,
)
