"""Frozen YOLO detector with prediction caching (headless)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .config import (
    COCO_CAR_CLASS_ID,
    DEFAULT_MODEL,
    DET_CONF_THRESH,
    FALLBACK_MODEL,
    OUTPUT_ROOT,
    SEED,
)

logger = logging.getLogger(__name__)


def resolve_device(prefer: Optional[str] = None) -> str:
    if prefer:
        return prefer
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_yolo_model(model_name: Optional[str] = None, device: Optional[str] = None):
    """Load Ultralytics YOLO; fall back from yolo11n to yolov8n if needed."""
    from ultralytics import YOLO

    device = resolve_device(device)
    candidates = []
    primary = model_name or DEFAULT_MODEL
    candidates.append(primary)
    if primary != FALLBACK_MODEL:
        candidates.append(FALLBACK_MODEL)

    last_err = None
    for name in candidates:
        try:
            model = YOLO(name)
            # Trigger a tiny warm-up on empty list is unnecessary; just record.
            logger.info("Loaded YOLO model=%s device=%s", name, device)
            return model, name, device
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning("Failed to load %s: %s", name, exc)
    raise RuntimeError(f"Could not load any YOLO model from {candidates}: {last_err}")


def _filter_car_detections(result, conf_thresh: float) -> Dict[str, list]:
    boxes, scores, class_ids = [], [], []
    if result.boxes is None or len(result.boxes) == 0:
        return {"boxes": boxes, "scores": scores, "class_ids": class_ids}
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    for b, s, c in zip(xyxy, conf, cls):
        if int(c) != COCO_CAR_CLASS_ID:
            continue
        if float(s) < conf_thresh:
            continue
        boxes.append([float(x) for x in b.tolist()])
        scores.append(float(s))
        class_ids.append(int(c))
    return {"boxes": boxes, "scores": scores, "class_ids": class_ids}


def predict_images(
    image_paths: Sequence[str | Path],
    *,
    model=None,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    conf_thresh: float = DET_CONF_THRESH,
    batch_size: int = 16,
    domain: str = "unknown",
    scenario_ids: Optional[Sequence[str]] = None,
    cache_path: Optional[str | Path] = None,
    overwrite: bool = False,
) -> List[dict]:
    """
    Run frozen YOLO on images; cache JSONL predictions.

    Each record: scenario_id, domain, image_path, boxes, scores, class_ids
    """
    image_paths = [str(p) for p in image_paths]
    if scenario_ids is None:
        scenario_ids = [Path(p).stem for p in image_paths]
    assert len(scenario_ids) == len(image_paths)

    cache_path = Path(cache_path) if cache_path else (
        OUTPUT_ROOT / "predictions" / f"preds_{domain}.jsonl"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cached: Dict[str, dict] = {}
    if cache_path.exists() and not overwrite:
        with cache_path.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                cached[rec["image_path"]] = rec

    missing_idx = [i for i, p in enumerate(image_paths) if p not in cached]
    if missing_idx:
        if model is None:
            model, model_name, device = load_yolo_model(model_name, device)
        else:
            device = resolve_device(device)
            model_name = model_name or getattr(model, "ckpt_path", DEFAULT_MODEL)

        logger.info(
            "Running detector on %d/%d %s images (model=%s device=%s conf=%.2f)",
            len(missing_idx), len(image_paths), domain, model_name, device, conf_thresh,
        )
        # Deterministic seed for any stochastic ops
        try:
            import torch
            torch.manual_seed(SEED)
        except Exception:
            pass

        new_recs = []
        for start in range(0, len(missing_idx), batch_size):
            batch_i = missing_idx[start:start + batch_size]
            batch_paths = [image_paths[i] for i in batch_i]
            results = model.predict(
                batch_paths,
                conf=conf_thresh,
                device=device,
                verbose=False,
            )
            for i, res in zip(batch_i, results):
                det = _filter_car_detections(res, conf_thresh)
                rec = {
                    "scenario_id": scenario_ids[i],
                    "domain": domain,
                    "image_path": image_paths[i],
                    "boxes": det["boxes"],
                    "scores": det["scores"],
                    "class_ids": det["class_ids"],
                    "model_name": str(model_name),
                    "conf_thresh": float(conf_thresh),
                }
                new_recs.append(rec)
                cached[image_paths[i]] = rec

        with cache_path.open("a") as f:
            for rec in new_recs:
                f.write(json.dumps(rec) + "\n")

    return [cached[p] for p in image_paths]


def load_predictions_jsonl(path: str | Path) -> Dict[str, dict]:
    path = Path(path)
    out: Dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open("r") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["image_path"]] = rec
    return out
