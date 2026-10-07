#!/usr/bin/env bash
# Detector-space ablations with MI shortlist=1000 and sev gate score>=0.3.
# Eval high-sev at y>=0.3.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
PY="${PY:-python3}"
BASE="${BASE:-outputs/kitti_vkitti/ablation_sev03_mi1k}"
mkdir -p "${BASE}"
LOG="${BASE}/launch.log"

run_abl() {
  local name="$1"; shift
  local out="${BASE}/${name}"
  echo "=== ${name} → ${out} ===" | tee -a "${LOG}"
  "${PY}" scripts/kitti_vkitti/ablate_local_cv.py \
    --spaces detector --seeds 0 1 2 \
    --output-dir "${out}" \
    --mi-shortlist 1000 --sev-cutoff 0.3 --high-thr 0.3 \
    "$@" 2>&1 | tee -a "${LOG}"
}

echo "=== detector sev03 / mi1k start $(date -Is) ===" | tee "${LOG}"

# 1) Default: MI shortlist + local CV + cluster + sev gate
run_abl local_cv --acq-pick cluster --acq-score cv

# 2) MI-only (sev gate skipped inside runner)
run_abl mi_only --acq-pick cluster --acq-score mi

# 3) No MI + cluster
run_abl no_mi --acq-pick cluster --acq-score cv --no-mi

# 4) No MI + greedy
run_abl no_mi_greedy --acq-pick greedy --acq-score cv --no-mi

echo "=== done $(date -Is) ===" | tee -a "${LOG}"
