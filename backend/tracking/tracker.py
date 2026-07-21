"""
tracker.py
----------
Multi-object tracking stage. Uses Ultralytics' built-in `.track()` API,
which ships BoT-SORT and ByteTrack implementations — both are configured
via the tracker YAML named in configs/config.yaml (tracking.tracker_config).

Design note: tracking gives us short-term, motion-consistent IDs ("track_id")
that are stable frame-to-frame but WILL still switch across occlusions,
side changes, and re-entries. That's expected and fine — ReID
(backend/reid/identity_matcher.py) is what turns a `track_id` into a
persistent `player_id`. Keeping these two concerns separate is the whole
point of the modular design: a stronger tracker can be dropped in later
without touching identity logic, and vice versa.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from backend.utils.colab_setup import device
from backend.utils.config_loader import get_config, resolve_path


@dataclass
class TrackedBox:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def bbox(self) -> tuple:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def foot_point(self) -> tuple:
        return ((self.x1 + self.x2) / 2, self.y2)


class PlayerTracker:
    """
    Stateful tracker: call `.update(frame)` once per frame IN ORDER.
    Internally reuses the same Ultralytics model/tracker instance so
    track IDs persist correctly across the call sequence.
    """

    def __init__(self, config: dict | None = None):
        cfg = config or get_config()
        self.det_cfg = cfg["detection"]
        self.trk_cfg = cfg["tracking"]
        self.runtime_cfg = cfg["runtime"]
        self._model = None
        self._device = device(self.runtime_cfg.get("device", "auto"))
        self._started = False

    def _load_model(self):
        if self._model is not None:
            return self._model
        from ultralytics import YOLO

        models_dir = resolve_path("models_dir")
        primary = os.path.join(models_dir, self.det_cfg["model_name"])
        weight_path = primary if os.path.exists(primary) else self.det_cfg["model_name"]
        self._model = YOLO(weight_path)
        return self._model

    def _tracker_yaml(self) -> str:
        name = self.trk_cfg.get("tracker_config")
        # Ultralytics resolves built-in names like "botsort.yaml"/"bytetrack.yaml"
        # automatically; a custom path can also be supplied via config.
        return name

    def update(self, frame_bgr: np.ndarray) -> List[TrackedBox]:
        model = self._load_model()
        results = model.track(
            source=frame_bgr,
            persist=True,  # keep the tracker's internal state across calls
            conf=self.det_cfg["confidence_threshold"],
            iou=self.det_cfg["iou_threshold"],
            imgsz=self.det_cfg["imgsz"],
            classes=[self.det_cfg["person_class_id"]],
            tracker=self._tracker_yaml(),
            device=self._device,
            verbose=False,
        )
        tracked: List[TrackedBox] = []
        if not results or results[0].boxes is None:
            return tracked
        boxes = results[0].boxes
        if boxes.id is None:
            # Tracker hasn't assigned IDs yet (e.g. first frame with no confirmed track)
            return tracked
        for box, tid in zip(boxes, boxes.id):
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            tracked.append(TrackedBox(int(tid), *xyxy, confidence=conf))
        return tracked

    def reset(self) -> None:
        """Start a fresh tracking session (e.g. new video, or after a hard cut)."""
        self._model = None
        self._started = False
