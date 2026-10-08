# SCOUT

SCOUT looks for failures of a real system by choosing which scenarios to label next. A mutual-information shortlist proposes scenarios the current labels do not explain. A control variate then re-ranks that shortlist with a cheap proxy, so the next real labels land where the proxy-corrected failure score is high.

Shared estimators live in `scout/`. KITTI, SIMPLER, and the quadruped task import that package. The 2-D notebook does not: it runs on its own.

## 1. Install

From a checkout of this repository:

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=.
```

## 2. 2-D toy

Open [notebooks/toy2d.ipynb](notebooks/toy2d.ipynb).

[Open in Colab](https://colab.research.google.com/github/anjaliParashar/scout-code/blob/main/notebooks/toy2d.ipynb)

That notebook defines the domain, the mutual-information shortlist, and the control variate in the cells themselves. It does not import this repository.

1. Install `numpy`, `scikit-learn`, and `matplotlib` (the first code cell does this if they are missing).
2. Define the two-diamond target and the shifted proxy.
3. Plot the real field next to the proxy.
4. Draw a handful of noisy real labels.
5. Score random designs by mutual information and keep a shortlist.
6. Re-rank the shortlist with the control variate and mark the next batch.

To write the same four figures from the command line, from the repository root:

```bash
python scripts/toy2D/stepwise_viz.py --output-dir outputs/toy2D/stepwise
```

A longer comparison against the baselines is:

```bash
python scripts/toy2D/compare_all.py --n-rounds 4 --batch-size 5 --n-init 10 \
    --pool-size 300 --output-dir outputs/toy2D/comparison --seed 7
```

## 3. KITTI

Open [notebooks/kitti.ipynb](notebooks/kitti.ipynb).

[Open in Colab](https://colab.research.google.com/github/anjaliParashar/scout-code/blob/main/notebooks/kitti.ipynb)

Each row is one paired frame. `target_failure` is the missed-car score of a frozen detector on real KITTI. `proxy_failure` is the same score on the Virtual KITTI clone. The notebook uses the bundled table and feature matrix, so the image archives are not required.

1. Clone this repository (the first cell does this in Colab) and install the Python dependencies.
2. Load `data/kitti_vkitti/paired_detection_task_linked.csv` and `data/kitti_vkitti/X_rich.npy`, and plot proxy failure against target failure.
3. Run five methods from one shared initial set: SCOUT, the β = 0 ablation, the mutual-information-only ablation, uniform random, and importance sampling.
4. Read the cumulative-mean curve. A higher curve after the dotted line means the new real labels hit worse detector failures.

From the repository root, the same short run is:

```bash
python scripts/kitti_vkitti/run_demo.py
```

Paper-scale sweeps, with more seeds and a longer training budget:

```bash
python scripts/kitti_vkitti/ablate_local_cv.py
python scripts/kitti_vkitti/ablate_scenario_space.py
python scripts/kitti_vkitti/compare_beta_ablation.py --seeds 0 1 2
```

To rebuild the pool from images instead of the bundled table, set `DATA_ROOT` and follow `scripts/kitti_vkitti/README.md`.

## 4. SIMPLER

Rollouts call an external SimplerEnv script. Point these two variables at that checkout before any command below:

```bash
export SIMPLER_ROLLOUT_SCRIPT=/path/to/paired_simpler_scenario_rollouts_gap.py
export SIMPLER_CKPT_PATH=/path/to/policy_checkpoint
```

Then, from the repository root:

1. Build or load a scenario pool with `utils/simpler/domain.py`.
2. Run one method, for example random, BNN control variates, or BAMS:

```bash
python scripts/simpler/run_random_simpler.py --logging-root results/simpler/random --pool-size 1000 --n-rounds 20 --batch-size 3
python scripts/simpler/run_bnn_cv_simpler.py --logging-root results/simpler/bnn_cv --pool-size 1000 --n-rounds 20 --batch-size 3
python scripts/simpler/run_bams_simpler.py --logging-root results/simpler/bams --pool-size 1000 --n-rounds 20 --batch-size 3
```

3. Plot a finished logging directory with `scripts/simpler/plot_simpler_results.py`.

## 5. Real hardware demo with Unitree Quadruped

The quadruped task searches over 3-D velocity commands \((v_x, v_y, \omega_z)\).

- The proxy is 3-D velocity data from a MuJoCo simulation: each fixed command is rolled out in simulation, and the score is the velocity-tracking error.
- The target is real-world velocity-control data from a Unitree quadruped. Each hardware trial records the commanded twist and the realized body velocity, and the score is again the tracking error.

Sample hardware trials ship in `experiments_quadruped/sample_trials/` (`success/` and `failure/`). The MuJoCo table is not in the repository. Place it at `data/quadruped/fixed_twist_combined.csv` with the columns listed in `data/quadruped/README.md`.

1. Export the MuJoCo rollouts to that CSV.
2. Point `--hardware_glob` at the real trial pickles you want to condition on.
3. Ask SCOUT for the next command:

```bash
python scripts/quadruped/run_scout.py \
    --hardware_glob "experiments_quadruped/sample_trials/failure/*.pkl" \
    --sim_csv data/quadruped/fixed_twist_combined.csv
```

Importance sampling and BAMS on the same command space are `scripts/quadruped/run_importance_sampling.py` and `scripts/quadruped/run_bams.py`.

## 6. Tests

```bash
python -m pytest -q tests
```

The tests build tiny synthetic KITTI and Virtual KITTI files. They do not download data.

## Layout

```
scout/                  mutual information, heteroscedastic BNN, local control variates, GP
notebooks/toy2d.ipynb   self-contained 2-D walkthrough
notebooks/kitti.ipynb   KITTI walkthrough
utils/kitti_vkitti/     KITTI ↔ Virtual KITTI 2 detection task
utils/simpler/          SimplerEnv manipulation scenarios
utils/quadruped/        Unitree velocity-command task
utils/baseline/         random, importance sampling, GP-MI+CV, BNN-CV, BAMS
data/kitti_vkitti/      bundled paired labels and rich features (no images)
experiments_quadruped/  sample Unitree hardware trials
```
