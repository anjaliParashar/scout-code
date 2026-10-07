# SCOUT

SCOUT finds failures of a real system by actively choosing which scenarios to label, using a cheap proxy (simulation, or a paired synthetic domain) inside a control-variate estimator. A mutual-information shortlist keeps the search from collapsing onto one region. The next batch is then ranked by the proxy-corrected failure score.

This repository is self-contained. Shared estimators live in `scout/`. The three evaluation tasks and the 2-D toy problem import that package and do not import each other.

## Layout

```
scout/                  MI, heteroscedastic BNN, local control variates, GP
utils/toy2D/            2-D diamond domain
utils/kitti_vkitti/     KITTI ↔ Virtual KITTI 2 detection task
utils/simpler/          SimplerEnv manipulation scenarios
utils/quadruped/        Go2 velocity-command task
utils/baseline/         random, importance sampling, GP-MI+CV, BNN-CV, BAMS
scripts/                one driver directory per task
data/kitti_vkitti/      bundled paired labels and rich features (no images)
experiments_quadruped/  sample hardware trials
notebooks/scout_demo.ipynb
```

## Install

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=.
```

## Notebook

`notebooks/scout_demo.ipynb` is the main demonstration. It draws the 2-D method step by step, then runs SCOUT, the β = 0 and MI-only ablations, and the random and importance-sampling baselines on the bundled KITTI pool.

From a checkout:

```bash
jupyter notebook notebooks/scout_demo.ipynb
```

In Colab, open that file. The first cell installs dependencies and, if the package is not already on the path, clones this repository.

## 2-D toy

Two diamonds in \([-3, 3]^2\) are the real failure regions. The proxy is the same field with a small shift, so it is correlated with the target and still wrong in places.

```bash
python scripts/toy2D/stepwise_viz.py --output-dir outputs/toy2D/stepwise
python scripts/toy2D/compare_all.py --n-rounds 4 --batch-size 5 --n-init 10 \
    --pool-size 300 --output-dir outputs/toy2D/comparison --seed 7
```

## KITTI ↔ Virtual KITTI

The bundled pool has 1,049 paired frames. `target_failure` is the real KITTI missed-car score and `proxy_failure` is the same score on the Virtual KITTI clone. Image archives are not required for the demo.

```bash
python scripts/kitti_vkitti/run_demo.py
```

Paper-scale ablations (several seeds, full training budgets) are:

```bash
python scripts/kitti_vkitti/ablate_local_cv.py
python scripts/kitti_vkitti/ablate_scenario_space.py
python scripts/kitti_vkitti/compare_beta_ablation.py --seeds 0 1 2
```

To rebuild the pool from images, set `DATA_ROOT` and follow `scripts/kitti_vkitti/README.md`.

## SIMPLER

Scenario construction, acquisition, and result plots are in `scripts/simpler/` and `utils/simpler/`. Rollouts call an external SimplerEnv script. Set both of these before a rollout:

- `SIMPLER_ROLLOUT_SCRIPT` — path to the paired rollout script
- `SIMPLER_CKPT_PATH` — path to the policy checkpoint

## Quadruped

Go2 commands are scored by velocity-tracking error. Sample hardware trials are in `experiments_quadruped/sample_trials/`. The simulator CSV is not bundled; put it at `data/quadruped/fixed_twist_combined.csv` (column list in `data/quadruped/README.md`).

```bash
python scripts/quadruped/run_scout.py \
    --hardware_glob "experiments_quadruped/sample_trials/failure/*.pkl" \
    --sim_csv data/quadruped/fixed_twist_combined.csv
```

## Tests

```bash
python -m pytest -q tests
```

The tests build tiny synthetic KITTI and Virtual KITTI files. They do not download data.
