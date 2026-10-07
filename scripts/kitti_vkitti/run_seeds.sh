#!/usr/bin/env bash
# Multi-seed SCOUT ablation with locked hyperparameters (seed-0 sweep best).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/kitti_vkitti}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PY="${PY:-python3}"

TASK="${TASK:-outputs/kitti_vkitti/paired_detection_task_linked.csv}"
OUT="${OUT:-outputs/kitti_vkitti/scout_final}"
SEEDS="${SEEDS:-0 1 2}"

# Locked config: default local-CV SCOUT (standard BNN 96→24→6, 800 ep)
# Matches outputs/kitti_vkitti/ablation_local_cv/ (full high-sev ≈ 0.21/0.18/0.20/0.18)
N_INIT=10
N_PROXY_ONLY=40
N_ACQ=15
BATCH=5
MI_SL=1000
ACQ_PICK=cluster
N_PAIR=10
K_UNP=20
CV_RADIUS=1.5
CV_FALLBACK_K=80
BNN_EPOCHS=800
DROPOUT=0.05

mkdir -p "${OUT}"
LOG="${OUT}/seeds_run.log"
echo "=== SCOUT local-CV seeds ${SEEDS} → ${OUT} ===" | tee -a "${LOG}"
echo "config: n_init=${N_INIT} n_acq=${N_ACQ} mi_sl=${MI_SL} acq=${ACQ_PICK} cv_radius=${CV_RADIUS} bnn_ep=${BNN_EPOCHS} drop=${DROPOUT}" | tee -a "${LOG}"

for s in ${SEEDS}; do
  echo "===== with_beta seed ${s} $(date -Is) =====" | tee -a "${LOG}"
  "${PY}" scripts/kitti_vkitti/run_scout.py --mode with_beta --seed "${s}" \
    --task-csv "${TASK}" --output-dir "${OUT}" \
    --n-init "${N_INIT}" --n-proxy-only "${N_PROXY_ONLY}" \
    --n-acq-iters "${N_ACQ}" --batch-acq-size "${BATCH}" \
    --mi-shortlist "${MI_SL}" --acq-pick "${ACQ_PICK}" \
    --n-pair-local "${N_PAIR}" --k-unpaired "${K_UNP}" \
    --cv-radius "${CV_RADIUS}" --cv-fallback-k "${CV_FALLBACK_K}" \
    --bnn-epochs "${BNN_EPOCHS}" --dropout "${DROPOUT}" 2>&1 | tee -a "${LOG}"

  echo "===== real_only seed ${s} $(date -Is) =====" | tee -a "${LOG}"
  "${PY}" scripts/kitti_vkitti/run_scout.py --mode real_only --seed "${s}" \
    --task-csv "${TASK}" --output-dir "${OUT}" \
    --n-init "${N_INIT}" --n-proxy-only "${N_PROXY_ONLY}" \
    --n-acq-iters "${N_ACQ}" --batch-acq-size "${BATCH}" \
    --mi-shortlist "${MI_SL}" --acq-pick "${ACQ_PICK}" \
    --n-pair-local "${N_PAIR}" --k-unpaired "${K_UNP}" \
    --cv-radius "${CV_RADIUS}" --cv-fallback-k "${CV_FALLBACK_K}" \
    --bnn-epochs "${BNN_EPOCHS}" --dropout "${DROPOUT}" 2>&1 | tee -a "${LOG}"
done

echo "===== compare $(date -Is) =====" | tee -a "${LOG}"
# shellcheck disable=SC2086
"${PY}" scripts/kitti_vkitti/compare_beta_ablation.py --seeds ${SEEDS} \
  --task-csv "${TASK}" --scout-root "${OUT}" \
  --output-dir outputs/kitti_vkitti/ablation_beta_local_cv 2>&1 | tee -a "${LOG}"

echo "===== DONE $(date -Is) =====" | tee -a "${LOG}"
echo "Summary: outputs/kitti_vkitti/ablation_beta_local_cv/summary.json"
