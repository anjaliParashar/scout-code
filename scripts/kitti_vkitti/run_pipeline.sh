#!/usr/bin/env bash
# End-to-end: manifest → detector → metrics → diagnostics
# Skips completed stages; never silently overwrites unless OVERWRITE=1.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/kitti_vkitti}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export KITTI_VKITTI_OUTPUT="${KITTI_VKITTI_OUTPUT:-${REPO_ROOT}/outputs/kitti_vkitti}"
FRAME_STRIDE="${FRAME_STRIDE:-10}"
DEVICE="${DEVICE:-}"
OVERWRITE="${OVERWRITE:-0}"

OUT="${KITTI_VKITTI_OUTPUT}"
mkdir -p "${OUT}"

OW_FLAG=()
if [[ "${OVERWRITE}" == "1" ]]; then
  OW_FLAG=(--overwrite)
fi

echo "=== KITTI–VKITTI pipeline ==="
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUTPUT=${OUT}"
echo "FRAME_STRIDE=${FRAME_STRIDE} DEVICE=${DEVICE:-auto} OVERWRITE=${OVERWRITE}"

# Stage 1: manifest
if [[ -f "${OUT}/paired_manifest.csv" && "${OVERWRITE}" != "1" ]]; then
  echo "[skip] paired_manifest.csv exists"
else
  python3 -m utils.kitti_vkitti.build_manifest \
    --frame-stride "${FRAME_STRIDE}" "${OW_FLAG[@]+"${OW_FLAG[@]}"}"
fi

# Stage 2+3: detector + metrics
if [[ -f "${OUT}/paired_detection_task.csv" && "${OVERWRITE}" != "1" ]]; then
  echo "[skip] paired_detection_task.csv exists"
else
  DEV_FLAG=()
  if [[ -n "${DEVICE}" ]]; then DEV_FLAG=(--device "${DEVICE}"); fi
  python3 -m utils.kitti_vkitti.compute_metrics \
    "${OW_FLAG[@]+"${OW_FLAG[@]}"}" "${DEV_FLAG[@]+"${DEV_FLAG[@]}"}"
fi

# Stage 4: diagnostics
if [[ -f "${OUT}/diagnostics.json" && "${OVERWRITE}" != "1" ]]; then
  echo "[skip] diagnostics.json exists"
else
  python3 -m utils.kitti_vkitti.diagnostics \
    "${OW_FLAG[@]+"${OW_FLAG[@]}"}" --examples
fi

echo "=== pipeline complete ==="
echo "Next (SCOUT with proxy CV vs real-only):"
echo "  python scripts/kitti_vkitti/run_scout.py --mode with_beta --seed 0"
echo "  python scripts/kitti_vkitti/run_scout.py --mode real_only --seed 0"
