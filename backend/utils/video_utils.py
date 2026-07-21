"""
video_utils.py
---------------
Thin wrappers around OpenCV for frame extraction, resizing, and writing
annotated output video. Kept separate from the pipeline so any module
(detection, tracking, dashboard preview) can reuse the same primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional, Tuple

import cv2
import numpy as np


@dataclass
class VideoMeta:
    path: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration_sec: float


def probe_video(path: str) -> VideoMeta:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = frame_count / fps if fps else 0.0
    return VideoMeta(path, fps, width, height, frame_count, duration)


def resize_keep_aspect(frame: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    """Resize so the longer side == max_dim. Returns (frame, scale_factor)."""
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return frame, 1.0
    scale = max_dim / longest
    resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return resized, scale


def frame_generator(
    path: str,
    stride: int = 1,
    max_dim: Optional[int] = None,
) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """
    Yields (frame_index, timestamp_sec, frame_bgr) for every `stride`-th frame.
    Handles arbitrarily long videos by never loading more than one frame
    into memory at a time.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            if max_dim:
                frame, _ = resize_keep_aspect(frame, max_dim)
            yield idx, idx / fps, frame
        idx += 1
    cap.release()


class AnnotatedVideoWriter:
    """Convenience wrapper for writing the processed/annotated output video."""

    def __init__(self, out_path: str, fps: float, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        self.out_path = out_path

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()
