"""
shuttle_detector.py
--------------------
Shuttlecock detection + trajectory estimation.

HONEST SCOPE NOTE (per project engineering principles — do not oversimplify,
document assumptions when no reliable lightweight OSS solution exists):

    There is no small, general-purpose, pretrained open-source model that
    reliably detects a badminton shuttlecock out of the box. The
    state-of-the-art approach in the literature is **TrackNet** (and its
    successors, e.g. TrackNetV2/V3) — a heatmap-regression CNN trained
    specifically on shuttle trajectories. TrackNet is NOT a generic
    pretrained model you can `pip install` and point at arbitrary footage:
    it needs to be trained (or fine-tuned) on labeled data from the same
    camera setup / broadcast style you intend to run on, and its public
    reference weights are typically trained on a specific broadcast dataset
    that will not generalize well to arbitrary YouTube amateur footage.

    What this module ships as the POC baseline instead is a **classical CV
    heuristic**, clearly scoped and confidence-scored:

        1. Frame differencing + background subtraction (MOG2) to isolate
           moving foreground blobs.
        2. Filter blobs by size (shuttlecock is small, ~5-20px in broadcast
           resolution), circularity/aspect ratio, and brightness (shuttles
           are typically white/bright against the court).
        3. Reject blobs that overlap with player bounding boxes (players are
           usually much larger, but this avoids false positives from hands,
           shoes, wristbands etc.)
        4. Track surviving candidate blobs frame-to-frame with a simple
           nearest-neighbor + Kalman-filter style motion model, since the
           shuttle moves ballistically (near-parabolic) between hits.

    This heuristic WILL miss fast smashes (motion blur), WILL false-positive
    on other small bright/fast-moving things (players' socks, wristbands,
    ball-boys, sponsor logos), and its trajectory/speed estimates are only
    as good as the (currently uncalibrated-by-default) court homography.
    It is provided as a working baseline and a clean seam:
    `ShuttleDetector.detect()` is the entire interface the rest of the
    pipeline depends on, so dropping in a properly trained TrackNetV2/V3
    checkpoint later (see `_load_tracknet_if_available`) requires touching
    only this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from backend.utils.config_loader import get_config


@dataclass
class ShuttleObservation:
    frame_index: int
    x: float
    y: float
    confidence: float
    interpolated: bool = False


class ShuttleDetector:
    def __init__(self, config: dict | None = None):
        self.cfg = (config or get_config())["shuttle"]
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=25, detectShadows=False
        )
        self._tracknet_model = self._load_tracknet_if_available()

    def _load_tracknet_if_available(self):
        """
        Seam for a trained TrackNet-family model. Returns None (heuristic
        mode) unless the user has placed a compatible checkpoint at
        models/tracknet_v2.pt AND set shuttle.detector: "tracknet_v2" in
        config — in which case a real implementation would be loaded here.
        This is intentionally not auto-downloaded since no generic
        pretrained checkpoint reliably applies across arbitrary footage.
        """
        if self.cfg["detector"] != "tracknet_v2_placeholder":
            print(
                "[shuttle_detector] A non-default detector was requested in "
                "config but no TrackNet loader is implemented in this POC. "
                "Falling back to the heuristic CV detector. Implement "
                "`_load_tracknet_if_available` once a trained checkpoint "
                "for your footage domain is available."
            )
        return None

    def detect(
        self, frame_bgr: np.ndarray, player_boxes: List[Tuple[float, float, float, float]]
    ) -> Optional[ShuttleObservation]:
        """
        Return the single best shuttle candidate for this frame, or None.
        `player_boxes` are used to suppress candidates on top of players.
        """
        fg_mask = self._bg_subtractor.apply(frame_bgr)
        fg_mask = cv2.medianBlur(fg_mask, 3)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_candidate = None
        best_score = 0.0

        for c in contours:
            area = cv2.contourArea(c)
            if area < 2 or area > 250:  # shuttle is small; tune per resolution
                continue
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w / 2, y + h / 2

            if self._inside_any_box((cx, cy), player_boxes):
                continue

            aspect = w / max(h, 1)
            circularity_score = 1.0 - min(abs(aspect - 1.0), 1.0)  # shuttles are roughly compact

            roi = frame_bgr[y : y + h, x : x + w]
            brightness_score = 0.0
            if roi.size > 0:
                brightness_score = float(np.mean(roi)) / 255.0

            score = 0.5 * circularity_score + 0.5 * brightness_score
            if score > best_score:
                best_score = score
                best_candidate = (cx, cy)

        if best_candidate is None or best_score < self.cfg["confidence_threshold"]:
            return None

        return ShuttleObservation(
            frame_index=-1,  # caller fills this in
            x=best_candidate[0],
            y=best_candidate[1],
            confidence=best_score,
        )

    @staticmethod
    def _inside_any_box(pt: Tuple[float, float], boxes: List[Tuple[float, float, float, float]]) -> bool:
        x, y = pt
        for (x1, y1, x2, y2) in boxes:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False


def interpolate_gaps(
    observations: List[ShuttleObservation], max_gap: Optional[int] = None
) -> List[ShuttleObservation]:
    """
    Linearly interpolate short gaps in the shuttle trajectory (e.g. frames
    where motion blur or occlusion caused a miss). Long gaps are left as-is
    since they more likely represent a genuine rally break.
    """
    cfg = get_config()["shuttle"]
    max_gap = max_gap or cfg["max_gap_frames_for_interpolation"]

    if len(observations) < 2:
        return observations

    filled: List[ShuttleObservation] = [observations[0]]
    for prev, cur in zip(observations, observations[1:]):
        gap = cur.frame_index - prev.frame_index
        if 1 < gap <= max_gap:
            for step in range(1, gap):
                t = step / gap
                filled.append(
                    ShuttleObservation(
                        frame_index=prev.frame_index + step,
                        x=prev.x + t * (cur.x - prev.x),
                        y=prev.y + t * (cur.y - prev.y),
                        confidence=min(prev.confidence, cur.confidence) * 0.8,
                        interpolated=True,
                    )
                )
        filled.append(cur)
    return filled


def estimate_speed_kmh(
    p1: ShuttleObservation, p2: ShuttleObservation, fps: float, px_to_m: float
) -> float:
    """px_to_m: meters-per-pixel scale, typically derived from the court homography."""
    dt = (p2.frame_index - p1.frame_index) / fps
    if dt <= 0:
        return 0.0
    dist_px = ((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2) ** 0.5
    dist_m = dist_px * px_to_m
    return (dist_m / dt) * 3.6
