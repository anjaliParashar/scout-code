#!/usr/bin/env bash
# Download Virtual KITTI 2 (official NAVER Labs) and extract only clone/Camera_0
# for the five matched scenes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/kitti_vkitti}"
RAW_DIR="${DATA_ROOT}/raw"
VKITTI_DIR="${DATA_ROOT}/vkitti"
mkdir -p "${RAW_DIR}" "${VKITTI_DIR}"

RGB_URL="https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar"
TEXT_URL="https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_textgt.tar.gz"
RGB_TAR="${RAW_DIR}/vkitti_2.0.3_rgb.tar"
TEXT_TAR="${RAW_DIR}/vkitti_2.0.3_textgt.tar.gz"

SCENES=(Scene01 Scene02 Scene06 Scene18 Scene20)

download_if_needed() {
  local url="$1" dest="$2" min_bytes="$3"
  if [[ -f "${dest}" ]]; then
    local sz
    sz=$(stat -c%s "${dest}" 2>/dev/null || stat -f%z "${dest}")
    if [[ "${sz}" -ge "${min_bytes}" ]]; then
      echo "[skip] ${dest} already present (${sz} bytes)"
      return 0
    fi
    echo "[resume] ${dest} incomplete (${sz} bytes); continuing"
  fi
  echo "[download] ${url}"
  if command -v wget >/dev/null 2>&1; then
    wget -c -O "${dest}" "${url}"
  else
    curl -L -C - -o "${dest}" "${url}"
  fi
}

echo "=== Virtual KITTI 2 download ==="
echo "DATA_ROOT=${DATA_ROOT}"

# textgt ~24MB
download_if_needed "${TEXT_URL}" "${TEXT_TAR}" 1000000
# rgb ~7.5GB
download_if_needed "${RGB_URL}" "${RGB_TAR}" 1000000000

echo "[inspect] textgt archive"
tar -tzf "${TEXT_TAR}" | head -40
echo "..."

MARKER_TEXT="${VKITTI_DIR}/.textgt_extracted"
if [[ -f "${MARKER_TEXT}" ]]; then
  echo "[skip] textgt already extracted"
else
  echo "[extract] textgt → ${VKITTI_DIR}"
  tar -xzf "${TEXT_TAR}" -C "${VKITTI_DIR}"
  touch "${MARKER_TEXT}"
fi

echo "[inspect] rgb archive (first 40 entries)"
tar -tf "${RGB_TAR}" | head -40
echo "..."

# Discover top-level prefix programmatically
PREFIX=$(tar -tf "${RGB_TAR}" | head -1 | awk -F/ '{print $1}')
echo "[info] RGB archive top-level prefix: '${PREFIX}'"

MARKER_RGB="${VKITTI_DIR}/.rgb_clone_cam0_extracted"
# Also treat as incomplete if marker exists but no images on disk
N_EXISTING=$(find "${VKITTI_DIR}" -path '*/clone/frames/rgb/Camera_0/rgb_*.jpg' 2>/dev/null | wc -l)
if [[ -f "${MARKER_RGB}" && "${N_EXISTING}" -gt 0 ]]; then
  echo "[skip] RGB clone/Camera_0 already extracted (${N_EXISTING} images)"
else
  echo "[extract] RGB clone/Camera_0 for matched scenes only"
  # Direct directory members (archive paths are SceneXX/clone/frames/rgb/Camera_0/...)
  EXTRACT_PATHS=()
  for scene in "${SCENES[@]}"; do
    EXTRACT_PATHS+=("${scene}/clone/frames/rgb/Camera_0")
  done
  echo "[info] tar -xf … ${EXTRACT_PATHS[*]}"
  tar -xf "${RGB_TAR}" -C "${VKITTI_DIR}" "${EXTRACT_PATHS[@]}"
  touch "${MARKER_RGB}"
fi

echo "[integrity] checking images + bbox.txt"
python3 - <<'PY'
import sys
from pathlib import Path
import os
root = Path(os.environ.get("DATA_ROOT", "") or Path("data/kitti_vkitti"))
vk = root / "vkitti"
scenes = ["Scene01","Scene02","Scene06","Scene18","Scene20"]
ok = True
for sc in scenes:
    matches = list(vk.glob(f"**/{sc}/clone"))
    if not matches:
        print(f"MISSING scene root: {sc}/clone")
        ok = False
        continue
    clone = matches[0]
    bbox = clone / "bbox.txt"
    info = clone / "info.txt"
    rgb = clone / "frames" / "rgb" / "Camera_0"
    n = len(list(rgb.glob("rgb_*.jpg"))) if rgb.is_dir() else 0
    print(f"{sc}: bbox={bbox.exists()} info={info.exists()} rgb_dir={rgb.is_dir()} n_images={n}")
    if not (bbox.exists() and info.exists() and n > 0):
        ok = False
sys.exit(0 if ok else 1)
PY

echo "=== Virtual KITTI 2 ready under ${VKITTI_DIR} ==="
