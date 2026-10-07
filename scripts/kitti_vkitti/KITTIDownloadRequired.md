# KITTI tracking download required

Use this only if automatic download via AWS fails.

## Register once

1. https://www.cvlibs.net/datasets/kitti/user_register.php
2. https://www.cvlibs.net/datasets/kitti/eval_tracking.php

## Exact files

- `data_tracking_image_2.zip` — left color images (tracking **training**)
- `data_tracking_label_2.zip` — tracking **training** labels

Do **not** download test data, right-camera images, LiDAR, GPS, or calibration.

## Preferred automatic path (no registration)

```bash
aws s3 cp --no-sign-request s3://avg-kitti/data_tracking_label_2.zip "$DATA_ROOT/raw/"
aws s3 cp --no-sign-request s3://avg-kitti/data_tracking_image_2.zip "$DATA_ROOT/raw/"
```

## Manual placement

```
${DATA_ROOT:-./data/kitti_vkitti}/raw/data_tracking_image_2.zip
${DATA_ROOT:-./data/kitti_vkitti}/raw/data_tracking_label_2.zip
```

Then:

```bash
export DATA_ROOT=${DATA_ROOT:-./data/kitti_vkitti}
bash scripts/kitti_vkitti/download_kitti.sh
bash scripts/kitti_vkitti/prepare_data.sh
```
