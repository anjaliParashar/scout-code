#!/usr/bin/env bash
# Verify datasets and build paired_manifest.csv
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/kitti_vkitti}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
FRAME_STRIDE="${FRAME_STRIDE:-10}"

echo "=== prepare_data ==="
echo "DATA_ROOT=${DATA_ROOT}  FRAME_STRIDE=${FRAME_STRIDE}"

python3 - <<PY
from pathlib import Path
import os, sys
root = Path(os.environ["DATA_ROOT"])
vk = root / "vkitti"
kt = root / "kitti" / "training"
ok = True
for sc in ["Scene01","Scene02","Scene06","Scene18","Scene20"]:
    hits = list(vk.glob(f"**/{sc}/clone/bbox.txt"))
    rgb = list(vk.glob(f"**/{sc}/clone/frames/rgb/Camera_0/rgb_*.jpg"))
    print(f"VKITTI {sc}: bbox={bool(hits)} n_rgb={len(rgb)}")
    if not hits or not rgb:
        ok = False
for seq in ["0001","0002","0006","0018","0020"]:
    img = kt / "image_02" / seq
    lab = kt / "label_02" / f"{seq}.txt"
    n = len(list(img.glob("*"))) if img.is_dir() else 0
    print(f"KITTI {seq}: images={n} label={lab.exists()}")
    if n == 0 or not lab.exists():
        ok = False
if not ok:
    sys.exit("Dataset integrity check failed. Run download_*.sh first.")
print("Integrity OK")
PY

python3 -m utils.kitti_vkitti.build_manifest \
  --frame-stride "${FRAME_STRIDE}" \
  ${OVERWRITE:+--overwrite}

echo "=== prepare_data done ==="
echo "Manifest: outputs/kitti_vkitti/paired_manifest.csv"
