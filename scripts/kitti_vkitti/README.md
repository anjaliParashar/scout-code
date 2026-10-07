# KITTI ↔ Virtual KITTI 2 paired detection task (SCOUT)

Minimal headless task for failure-discovery experiments:

- **Target** `y_r`: missed-car rate of a frozen YOLO detector on real KITTI
- **Proxy** `y_s`: same detector / metric on Virtual KITTI 2 `clone` / `Camera_0`

Paired by sequence map + identical zero-based frame index.

| KITTI | Virtual KITTI |
|-------|---------------|
| 0001  | Scene01       |
| 0002  | Scene02       |
| 0006  | Scene06       |
| 0018  | Scene18       |
| 0020  | Scene20       |

## Sim data available

| Source | Size / count | Notes |
|--------|--------------|-------|
| VKITTI RGB archive | **7.53 GB** | full tar (all weathers/cameras); we extract only 5×`clone`×`Camera_0` |
| VKITTI text GT | **23 MB** | `bbox.txt` + `info.txt` |
| Clone frames with GT (5 scenes) | **≈ 2066** | Scene01:426, 02:233, 06:270, 18:314 (starts @25), 20:837 |
| Every-10th candidates | **≈ 207** | `FRAME_STRIDE=10` (default task setting) |
| Expected paired scenarios (Car GT both sides) | **≈ 150–200** | after height≥25 + image existence |
| Dense pool for SCOUT (`FRAME_STRIDE=1`) | **≈ 1.5k–2k** | recommended for AL / β ablation |
| Car tracks in clone (proxy-only signal) | **245** | 88+15+11+18+113 across scenes |

KITTI tracking (public AWS `s3://avg-kitti/`):

- `data_tracking_image_2.zip` ≈ **15.8 GB**
- `data_tracking_label_2.zip` ≈ **2.2 MB**
- We keep only sequences 0001/0002/0006/0018/0020.

A precomputed pool is already in the repo, so the active-learning demo does not need the image archives:

- `data/kitti_vkitti/paired_detection_task_linked.csv` — 1,049 paired scenarios, with `target_failure` and `proxy_failure`
- `data/kitti_vkitti/X_rich.npy` — rich features aligned to that table

```bash
export PYTHONPATH=.
python scripts/kitti_vkitti/run_demo.py
```

Regenerating labels from images is optional and described below.

## Quick start

```bash
export DATA_ROOT=${DATA_ROOT:-./data/kitti_vkitti}
export PYTHONPATH=.

# deps (prefer an env that already has torch)
pip install -r scripts/kitti_vkitti/requirements.txt

# unit + smoke tests (no downloads)
python -m pytest -q tests/test_parsers.py tests/test_box_matching.py tests/test_smoke_pipeline.py

# data
bash scripts/kitti_vkitti/download_vkitti.sh
bash scripts/kitti_vkitti/download_kitti.sh
bash scripts/kitti_vkitti/prepare_data.sh

# detector + metrics + diagnostics
bash scripts/kitti_vkitti/run_pipeline.sh
```

Resume individual stages:

```bash
python -m utils.kitti_vkitti.build_manifest --frame-stride 10
python -m utils.kitti_vkitti.compute_metrics
python -m utils.kitti_vkitti.diagnostics --examples
```

## Linked labels (sim–real failure coupling)

Hard FN rate only correlates at Pearson **r≈0.31**. We therefore use:

```bash
python -m utils.kitti_vkitti.relabel   # → paired_detection_task_linked.csv
python -m utils.kitti_vkitti.enrich_features --task-csv ... --output ...
```

- **Failure** = `1 − mean assignment IoU` (unmatched GT → 0)
- **Filter** = `|Δ GT cars| ≤ 1` and `mean_bbox_height_ratio ≥ 0.13`
- Result: **N≈1049**, Pearson **r≈0.58**, Spearman **ρ≈0.56**

SCOUT should point at the linked CSV + rich features:

```bash
python -m utils.kitti_vkitti.rich_features \
  --task-csv outputs/kitti_vkitti/paired_detection_task_linked.csv

python scripts/kitti_vkitti/run_scout.py --mode with_beta --seed 0 \
  --task-csv outputs/kitti_vkitti/paired_detection_task_linked.csv \
  --output-dir outputs/kitti_vkitti/scout_v5 \
  --proxy-shortlist 250 --proxy-blend 0.6 --mi-shortlist 250
```

Rich features (`X_rich.npy`, 37-D + sequence one-hot): size bins, trunc/occ,
spatial layout, detector score histograms, image edge energy, PLS→proxy.

## SCOUT experiment plan (proxy CV vs β=0)

Goal: show that **including proxy (sim) in the CV estimator** finds more real failures per real label than **real-only (β=0)**, matching the SIMPLER ablations.

### Pool construction (important)

Default task CSV uses `FRAME_STRIDE=10` (~150–200 rows). For SCOUT, rebuild a denser pool:

```bash
FRAME_STRIDE=1 bash scripts/kitti_vkitti/prepare_data.sh
OVERWRITE=1 FRAME_STRIDE=1 bash scripts/kitti_vkitti/run_pipeline.sh
```

This yields ~1.5k–2k paired scenarios — closer to SIMPLER’s pool size and enough for MI + local CV.

### Conditions (shared init per seed)

| Mode | Acquisition | Script flag |
|------|-------------|-------------|
| SCOUT with β | MI shortlist + local μ_CV with `proxy_failure` | `--mode with_beta` |
| Real-only (β=0) | MI shortlist + target-BNN mean (no proxy) | `--mode real_only` |

Hyperparameters (mirrors the SIMPLER setup, scaled to N≈2k):

- seeds: `0 1 2`
- `n_init=15`, `n_acq_iters=20`, `batch_acq_size=5` → budget 15+100=115 real labels
- MI shortlist 200; BNN epochs 800; `n_pair_local=8`, `k_unpaired=20`

```bash
for s in 0 1 2; do
  python scripts/kitti_vkitti/run_scout.py --mode with_beta --seed $s \
      --n-init 15 --n-acq-iters 20 --batch-acq-size 5
  python scripts/kitti_vkitti/run_scout.py --mode real_only --seed $s \
      --n-init 15 --n-acq-iters 20 --batch-acq-size 5
done

python scripts/kitti_vkitti/compare_beta_ablation.py --seeds 0 1 2
```

### Why this should show a strong β gain

1. **Paired clone geometry** → proxy/target failures share the same layout; expect solid Spearman correlation.
2. **Cheap dense proxy labels** already computed for the full pool → CV unpaired term is free (unlike online SIMPLER rollouts).
3. Real-only cannot exploit that correlation; with_β reduces acquisition variance exactly where sim/real agree.

Primary plot: cumulative mean `target_failure` vs real budget
(`outputs/kitti_vkitti/ablation_beta/with_beta_vs_real_only.pdf`).

## Layout

```
utils/kitti_vkitti/          # parsers, matching, detector, metrics, diagnostics
scripts/kitti_vkitti/        # download / prepare / pipeline / SCOUT
tests/                       # parsers, matching, smoke (no downloads)
outputs/kitti_vkitti/        # manifests, predictions, diagnostics, SCOUT runs
data/kitti_vkitti/           # DATA_ROOT default
```
