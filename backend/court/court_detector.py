"""
court_detector.py
------------------
Court line detection + homography estimation.

HONEST SCOPE NOTE (per project engineering principles: do not oversimplify,
document assumptions):
    Robust automatic court-line detection on arbitrary, unconstrained YouTube
    footage (varying angle, lighting, broadcast overlays, occlusion by
    players) is itself a research problem — there is no small, reliable,
    off-the-shelf open-source model for it the way there is for person
    detection. Two strategies are implemented here, selectable via
    configs/config.yaml (court.detector):

    1. "heuristic_line_detection" (default, fully automatic, best-effort)
       Classical CV: white-line segmentation (HSV threshold + morphology) ->
       Hough line transform -> RANSAC-style filtering for the court's outer
       boundary quadrilateral. Works reasonably on clean, static, top-down or
       high-angle broadcast shots. Degrades on noisy amateur footage — in
       that case `confidence` will be low and the caller should fall back to
       manual calibration.

    2. Manual calibration (UI fallback, always available)
       The user clicks the 4 outer court corners once on a representative
       frame (Streamlit page 2). This is the standard, reliable approach used
       by most real sports-analytics tools for exactly this reason, and it
       is what should be relied on for the POC's court-mapping accuracy.

    This module is designed so a trained keypoint-detection model (e.g. a
    small CNN regressing the 4-6 court keypoints, trained on a labeled
    badminton court dataset) can be swapped in later as a third strategy
    without changing the `CourtModel` interface downstream analytics use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from backend.utils.config_loader import get_config

# Standard BWF doubles court dimensions in meters, used as the "real world"
# reference plane for the homography (court-space coordinates).
COURT_LENGTH_M = 13.4
COURT_WIDTH_M = 6.1


@dataclass
class CourtModel:
    homography: Optional[np.ndarray]     # 3x3, maps image px -> court meters
    corners_image: Optional[List[Tuple[float, float]]]  # 4 pts, clockwise from top-left
    confidence: float
    source: str  # "heuristic" | "manual"

    def image_to_court(self, point_xy: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        if self.homography is None:
            return None
        pt = np.array([point_xy[0], point_xy[1], 1.0])
        mapped = self.homography @ pt
        if mapped[2] == 0:
            return None
        return (mapped[0] / mapped[2], mapped[1] / mapped[2])


_COURT_SPACE_CORNERS = np.array(
    [
        [0, 0],
        [COURT_WIDTH_M, 0],
        [COURT_WIDTH_M, COURT_LENGTH_M],
        [0, COURT_LENGTH_M],
    ],
    dtype=np.float32,
)


def build_from_manual_corners(corners_image: List[Tuple[float, float]]) -> CourtModel:
    """Corners must be given clockwise starting top-left, as clicked in the UI."""
    src = np.array(corners_image, dtype=np.float32)
    H, _ = cv2.findHomography(src, _COURT_SPACE_CORNERS)
    return CourtModel(homography=H, corners_image=corners_image, confidence=1.0, source="manual")


class HeuristicCourtDetector:
    """Best-effort automatic court boundary detection. See module docstring."""

    def __init__(self, config: dict | None = None):
        self.cfg = (config or get_config())["court"]

    def detect(self, frame_bgr: np.ndarray) -> CourtModel:
        quad, conf = self._find_court_quad(frame_bgr)
        if quad is None:
            return CourtModel(None, None, 0.0, "heuristic")
        src = np.array(quad, dtype=np.float32)
        H, _ = cv2.findHomography(src, _COURT_SPACE_CORNERS)
        return CourtModel(homography=H, corners_image=quad, confidence=conf, source="heuristic")

    def _find_court_quad(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[List[Tuple[float, float]]], float]:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        # Court lines are typically white/bright regardless of court color.
        lower = np.array([0, 0, 180])
        upper = np.array([180, 60, 255])
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        lines = cv2.HoughLinesP(
            mask, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=20
        )
        if lines is None or len(lines) < 4:
            return None, 0.0

        # Best-effort outer boundary: convex hull of all detected line endpoints,
        # reduced to a quadrilateral. This is a coarse approximation — good
        # enough to seed a homography for clean broadcast shots, unreliable
        # for cluttered amateur footage (hence low confidence by default).
        pts = lines.reshape(-1, 2, 2).reshape(-1, 2).astype(np.float32)
        hull = cv2.convexHull(pts)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

        if len(approx) != 4:
            return None, 0.2  # detected *something* but not a clean quad -> low confidence

        quad = [tuple(p[0]) for p in approx]
        quad = _order_clockwise_from_top_left(quad)
        confidence = 0.5  # heuristic method never claims high confidence by itself
        return quad, confidence


def _order_clockwise_from_top_left(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    pts_arr = np.array(pts)
    s = pts_arr.sum(axis=1)
    diff = np.diff(pts_arr, axis=1).flatten()
    top_left = pts_arr[np.argmin(s)]
    bottom_right = pts_arr[np.argmax(s)]
    top_right = pts_arr[np.argmin(diff)]
    bottom_left = pts_arr[np.argmax(diff)]
    return [tuple(top_left), tuple(top_right), tuple(bottom_right), tuple(bottom_left)]


def zone_of_point(court_xy: Tuple[float, float]) -> str:
    """
    Classify a court-space point (meters) into a coarse zone used by the
    analytics engine (front/back, left/right court occupancy stats).
    """
    x, y = court_xy
    vertical = "front" if y < COURT_LENGTH_M / 2 else "back"
    horizontal = "left" if x < COURT_WIDTH_M / 2 else "right"
    return f"{vertical}_{horizontal}"
