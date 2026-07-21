"""
pose_estimator.py
------------------
Per-player skeleton keypoints, using Ultralytics' YOLO-pose models (same
family as the detector, so no extra framework dependency).

Pose is used downstream for:
    * Stroke-type heuristics (arm/racket angle at contact -> smash / clear /
      drop / net / drive, see backend/analytics/stats_engine.py)
    * Jump / crouch detection to avoid identity re-evaluation glitches when
      a player's bounding-box aspect ratio changes suddenly
    * Reaction-time approximation (movement onset after opponent's hit)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from backend.utils.colab_setup import device
from backend.utils.config_loader import get_config, resolve_path

# COCO-pose 17 keypoint order, for reference by callers indexing `keypoints`.
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


@dataclass
class PoseResult:
    keypoints: np.ndarray       # (17, 2) pixel coords
    keypoint_conf: np.ndarray   # (17,) confidence per point
    bbox: tuple


class PoseEstimator:
    def __init__(self, config: dict | None = None):
        cfg = config or get_config()
        self.cfg = cfg["pose"]
        self.runtime_cfg = cfg["runtime"]
        self._model = None
        self._device = device(self.runtime_cfg.get("device", "auto"))

    def _load_model(self):
        if self._model is not None:
            return self._model
        from ultralytics import YOLO

        models_dir = resolve_path("models_dir")
        primary = os.path.join(models_dir, self.cfg["model_name"])
        weight_path = primary if os.path.exists(primary) else self.cfg["model_name"]
        self._model = YOLO(weight_path)
        return self._model

    def estimate(self, frame_bgr: np.ndarray) -> List[PoseResult]:
        model = self._load_model()
        results = model.predict(
            source=frame_bgr,
            conf=self.cfg["confidence_threshold"],
            device=self._device,
            verbose=False,
        )
        out: List[PoseResult] = []
        if not results or results[0].keypoints is None:
            return out
        kps = results[0].keypoints
        boxes = results[0].boxes
        for i in range(len(kps)):
            xy = kps.xy[i].cpu().numpy()
            conf = kps.conf[i].cpu().numpy() if kps.conf is not None else np.ones(len(xy))
            bbox = tuple(boxes[i].xyxy[0].tolist()) if boxes is not None else (0, 0, 0, 0)
            out.append(PoseResult(keypoints=xy, keypoint_conf=conf, bbox=bbox))
        return out

    @staticmethod
    def estimate_body_state(pose: PoseResult) -> str:
        """
        Very lightweight jump/crouch/neutral classifier from hip height
        relative to ankle-to-shoulder span. A placeholder heuristic —
        replace with a temporal (multi-frame) model for production use,
        since a single frame can't reliably distinguish a jump-smash
        apex from a deep lunge.
        """
        kp = pose.keypoints
        names = KEYPOINT_NAMES
        hip_y = np.mean([kp[names.index("left_hip")][1], kp[names.index("right_hip")][1]])
        ankle_y = np.mean([kp[names.index("left_ankle")][1], kp[names.index("right_ankle")][1]])
        shoulder_y = np.mean(
            [kp[names.index("left_shoulder")][1], kp[names.index("right_shoulder")][1]]
        )
        span = max(ankle_y - shoulder_y, 1.0)
        hip_ratio = (hip_y - shoulder_y) / span

        if hip_ratio < 0.35:
            return "jump"
        if hip_ratio > 0.65:
            return "crouch"
        return "neutral"
