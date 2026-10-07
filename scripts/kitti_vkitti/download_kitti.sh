#!/usr/bin/env bash
# Download KITTI tracking training (left images + labels).
# Prefer local copies, then public AWS S3 (avg-kitti, no-sign-request).
# If unavailable, write KITTIDownloadRequired.md and exit 2.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/kitti_vkitti}"
RAW_DIR="${DATA_ROOT}/raw"
KITTI_DIR="${DATA_ROOT}/kitti"
mkdir -p "${RAW_DIR}" "${KITTI_DIR}"

IMG_ZIP_NAME="data_tracking_image_2.zip"
LAB_ZIP_NAME="data_tracking_label_2.zip"
IMG_ZIP="${RAW_DIR}/${IMG_ZIP_NAME}"
LAB_ZIP="${RAW_DIR}/${LAB_ZIP_NAME}"
REQ_MD="${REPO_ROOT}/scripts/kitti_vkitti/KITTIDownloadRequired.md"

SEQUENCES=(0001 0002 0006 0018 0020)

find_existing() {
  local name="$1"
  local found=""
  for d in \
      "${RAW_DIR}" \
      "${DATA_ROOT}" \
      "${REPO_ROOT}/data" \
      "${REPO_ROOT}" \
      "${HOME}/data" \
      "${HOME}/datasets" \
      "/data" \
      "/datasets"; do
    if [[ -f "${d}/${name}" ]]; then
      found="${d}/${name}"
      break
    fi
  done
  echo "${found}"
}

already_extracted() {
  local img="${KITTI_DIR}/training/image_02"
  local lab="${KITTI_DIR}/training/label_02"
  [[ -d "${img}" && -d "${lab}" ]]
}

write_manual_instructions() {
  cat > "${REQ_MD}" <<EOF
# KITTI tracking download required

Automatic download of the KITTI tracking training archives failed or the
archives are not available without authentication in this environment.

## Register once

1. Create a free account at:
   https://www.cvlibs.net/datasets/kitti/user_register.php
2. Open the tracking benchmark page:
   https://www.cvlibs.net/datasets/kitti/eval_tracking.php

## Download exactly these two files

- \`${IMG_ZIP_NAME}\` — left color images of the tracking **training** set
- \`${LAB_ZIP_NAME}\` — tracking **training** labels

Do **not** download test data, right-camera images, LiDAR, GPS, or calibration.

## Place the ZIPs here

\`\`\`
${RAW_DIR}/${IMG_ZIP_NAME}
${RAW_DIR}/${LAB_ZIP_NAME}
\`\`\`

Then re-run:

\`\`\`bash
export DATA_ROOT=${DATA_ROOT}
bash scripts/kitti_vkitti/download_kitti.sh
bash scripts/kitti_vkitti/prepare_data.sh
\`\`\`

The rest of the pipeline (parsers, metrics, smoke tests) works without KITTI
using synthetic fixtures under \`tests/\`.
EOF
  echo "[WARN] Wrote manual instructions → ${REQ_MD}"
}

echo "=== KITTI tracking download ==="
echo "DATA_ROOT=${DATA_ROOT}"

if already_extracted; then
  echo "[skip] Found extracted training/image_02 and training/label_02"
else
  # Search local copies
  local_img="$(find_existing "${IMG_ZIP_NAME}")"
  local_lab="$(find_existing "${LAB_ZIP_NAME}")"
  if [[ -n "${local_img}" && "${local_img}" != "${IMG_ZIP}" ]]; then
    echo "[link] copying ${local_img} → ${IMG_ZIP}"
    cp -n "${local_img}" "${IMG_ZIP}" || true
  fi
  if [[ -n "${local_lab}" && "${local_lab}" != "${LAB_ZIP}" ]]; then
    echo "[link] copying ${local_lab} → ${LAB_ZIP}"
    cp -n "${local_lab}" "${LAB_ZIP}" || true
  fi

  # AWS public bucket
  if [[ ! -f "${IMG_ZIP}" || ! -f "${LAB_ZIP}" ]]; then
    if command -v aws >/dev/null 2>&1; then
      echo "[aws] listing s3://avg-kitti for tracking archives"
      aws s3 ls --no-sign-request s3://avg-kitti/ 2>&1 | grep -E 'data_tracking_(image_2|label_2)\.zip' || true
      if [[ ! -f "${LAB_ZIP}" ]]; then
        echo "[aws] downloading ${LAB_ZIP_NAME}"
        aws s3 cp --no-sign-request "s3://avg-kitti/${LAB_ZIP_NAME}" "${LAB_ZIP}" || true
      fi
      if [[ ! -f "${IMG_ZIP}" ]]; then
        echo "[aws] downloading ${IMG_ZIP_NAME} (~15.8 GB)"
        aws s3 cp --no-sign-request "s3://avg-kitti/${IMG_ZIP_NAME}" "${IMG_ZIP}" || true
      fi
    else
      echo "[WARN] aws CLI not installed"
    fi
  fi
fi

if [[ ! -f "${LAB_ZIP}" ]] && ! already_extracted; then
  write_manual_instructions
  echo "ERROR: ${LAB_ZIP_NAME} not available. See ${REQ_MD}" >&2
  exit 2
fi
if [[ ! -f "${IMG_ZIP}" ]] && ! already_extracted; then
  write_manual_instructions
  echo "ERROR: ${IMG_ZIP_NAME} not available. See ${REQ_MD}" >&2
  exit 2
fi

# Extract (idempotent)
MARKER="${KITTI_DIR}/.extracted_tracking_train5"
if [[ -f "${MARKER}" ]] && already_extracted; then
  echo "[skip] KITTI already extracted"
else
  echo "[extract] labels"
  unzip -n "${LAB_ZIP}" -d "${KITTI_DIR}"
  echo "[extract] images (full zip; will prune unused sequences)"
  unzip -n "${IMG_ZIP}" -d "${KITTI_DIR}"
  # Retain only five sequences
  IMG02="${KITTI_DIR}/training/image_02"
  if [[ -d "${IMG02}" ]]; then
    for d in "${IMG02}"/*; do
      base="$(basename "${d}")"
      keep=0
      for s in "${SEQUENCES[@]}"; do
        [[ "${base}" == "${s}" ]] && keep=1
      done
      if [[ "${keep}" -eq 0 && -d "${d}" ]]; then
        echo "[prune] removing unused sequence ${base}"
        rm -rf "${d}"
      fi
    done
  fi
  # Prune unused label files
  LAB02="${KITTI_DIR}/training/label_02"
  if [[ -d "${LAB02}" ]]; then
    for f in "${LAB02}"/*.txt; do
      base="$(basename "${f}" .txt)"
      keep=0
      for s in "${SEQUENCES[@]}"; do
        [[ "${base}" == "${s}" ]] && keep=1
      done
      if [[ "${keep}" -eq 0 ]]; then
        rm -f "${f}"
      fi
    done
  fi
  touch "${MARKER}"
fi

echo "[integrity] checking five sequences"
for s in "${SEQUENCES[@]}"; do
  n=$(ls -1 "${KITTI_DIR}/training/image_02/${s}" 2>/dev/null | wc -l)
  lab="${KITTI_DIR}/training/label_02/${s}.txt"
  echo "  ${s}: images=${n} label=$([[ -f ${lab} ]] && echo ok || echo MISSING)"
done

echo "=== KITTI ready under ${KITTI_DIR} ==="
