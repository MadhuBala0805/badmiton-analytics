"""
player_detector.py
-------------------
Player (person) detection stage of the pipeline.

Backend: Ultralytics YOLO (YOLOv11 preferred, YOLOv8 fallback, RT-DETR
supported by swapping model_name in config).

This module is intentionally detector-agnostic at the interface level:
`Detection` dataclasses are all downstream code ever sees, so swapping
YOLOv11 -> a custom-trained badminton-specific detector later requires no
changes anywhere else in the pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import numpy as np

from backend.utils.colab_setup import device
from backend.utils.config_loader import get_config, resolve_path


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

    @property
    def bbox(self) -> tuple:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> tuple:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def foot_point(self) -> tuple:
        """Bottom-center of the bbox — used for court-position mapping."""
        return ((self.x1 + self.x2) / 2, self.y2)


class PlayerDetector:
    """Wraps an Ultralytics YOLO model restricted to the 'person' class."""

    def __init__(self, config: dict | None = None):
        self.cfg = (config or get_config())["detection"]
        self.runtime_cfg = (config or get_config())["runtime"]
        self._model = None
        self._device = device(self.runtime_cfg.get("device", "auto"))

    def _load_model(self):
        if self._model is not None:
            return self._model
        from ultralytics import YOLO

        models_dir = resolve_path("models_dir")
        primary = os.path.join(models_dir, self.cfg["model_name"])
        fallback = os.path.join(models_dir, self.cfg["fallback_model_name"])

        weight_path = primary if os.path.exists(primary) else self.cfg["model_name"]
        try:
            self._model = YOLO(weight_path)
        except Exception as e:  # noqa: BLE001
            print(
                f"[player_detector] Failed to load '{weight_path}' ({e}); "
                f"falling back to {self.cfg['fallback_model_name']}."
            )
            fb_path = fallback if os.path.exists(fallback) else self.cfg["fallback_model_name"]
            self._model = YOLO(fb_path)
        return self._model

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Run detection on a single BGR frame, return only 'person' boxes."""
        model = self._load_model()
        results = model.predict(
            source=frame_bgr,
            conf=self.cfg["confidence_threshold"],
            iou=self.cfg["iou_threshold"],
            imgsz=self.cfg["imgsz"],
            classes=[self.cfg["person_class_id"]],
            device=self._device,
            verbose=False,
        )
        detections: List[Detection] = []
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None:
            return detections
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append(Detection(*xyxy, confidence=conf, class_id=cls))
        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Batched inference for throughput on GPU."""
        model = self._load_model()
        results = model.predict(
            source=frames,
            conf=self.cfg["confidence_threshold"],
            iou=self.cfg["iou_threshold"],
            imgsz=self.cfg["imgsz"],
            classes=[self.cfg["person_class_id"]],
            device=self._device,
            verbose=False,
        )
        all_detections: List[List[Detection]] = []
        for r in results:
            frame_dets: List[Detection] = []
            if r.boxes is not None:
                for box in r.boxes:
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    frame_dets.append(Detection(*xyxy, confidence=conf, class_id=cls))
            all_detections.append(frame_dets)
        return all_detections
