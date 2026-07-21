"""
rally_detector.py
------------------
Detects rally start/end by fusing multiple weak signals rather than relying
on any single one (per spec: "Do not rely solely on player movement").

Signals combined, each contributing a vote/weight toward a running
"rally active" confidence score:

    1. Shuttle motion presence  - shuttle observations existing & moving
                                   (from backend/shuttle) is the strongest
                                   signal a rally is live.
    2. Player motion            - aggregate player displacement per frame
                                   (from tracking); near-zero across all
                                   players for a sustained window suggests
                                   a dead ball.
    3. Hit-sequence plausibility - alternating "shuttle direction reversal
                                   near a player" events approximate hits;
                                   a live rally has periodic reversals.
    4. Temporal smoothing        - a hysteresis window (min_rally_frames /
                                   shuttle_stationary_frames_for_end) avoids
                                   flip-flopping on single noisy frames.

This is a heuristic fusion, not a learned classifier — it is deliberately
built as a `RallyStateMachine` with clearly separated signal-scoring
functions so a future trained temporal model (e.g. an LSTM/Transformer over
the same per-frame feature vector this module already computes) can replace
`_fuse_signals` without touching the state machine or anything downstream.
Every transition carries a confidence value; the caller (pipeline.py) is
expected to log low-confidence transitions for manual review rather than
trust them blindly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from backend.utils.config_loader import get_config


class RallyState(Enum):
    IDLE = "idle"
    ACTIVE = "active"


@dataclass
class RallyEvent:
    frame_index: int
    timestamp_sec: float
    event: str          # "rally_start" | "rally_end"
    confidence: float


@dataclass
class FrameSignals:
    frame_index: int
    timestamp_sec: float
    shuttle_present: bool
    shuttle_speed_px: float
    total_player_motion_px: float
    hit_event_detected: bool


class RallyStateMachine:
    def __init__(self, config: dict | None = None):
        self.cfg = (config or get_config())["rally"]
        self.state = RallyState.IDLE
        self._active_run = 0
        self._idle_run = 0
        self.events: List[RallyEvent] = []
        self._current_rally_start: Optional[FrameSignals] = None

    def step(self, signals: FrameSignals) -> Optional[RallyEvent]:
        confidence = self._fuse_signals(signals)
        is_active_frame = confidence >= 0.5

        if self.state == RallyState.IDLE:
            self._active_run = self._active_run + 1 if is_active_frame else 0
            if self._active_run >= self.cfg["min_rally_frames"]:
                self.state = RallyState.ACTIVE
                self._idle_run = 0
                event = RallyEvent(
                    signals.frame_index, signals.timestamp_sec, "rally_start", confidence
                )
                self.events.append(event)
                self._current_rally_start = signals
                return event
        else:  # ACTIVE
            self._idle_run = self._idle_run + 1 if not is_active_frame else 0
            if self._idle_run >= self.cfg["shuttle_stationary_frames_for_end"]:
                self.state = RallyState.IDLE
                self._active_run = 0
                event = RallyEvent(
                    signals.frame_index, signals.timestamp_sec, "rally_end", confidence
                )
                self.events.append(event)
                self._current_rally_start = None
                return event
        return None

    def _fuse_signals(self, s: FrameSignals) -> float:
        """
        Weighted vote in [0, 1]: higher = more likely a rally is currently
        live. Weights are heuristic defaults; tune per venue/footage style.
        """
        score = 0.0
        weight_total = 0.0

        # Shuttle presence & motion — strongest signal
        w = 0.5
        weight_total += w
        if s.shuttle_present:
            score += w * min(s.shuttle_speed_px / 40.0, 1.0)

        # Player motion — weak signal alone (players move between rallies
        # too), included only as corroboration
        w = 0.2
        weight_total += w
        score += w * min(s.total_player_motion_px / 30.0, 1.0)

        # Hit-sequence plausibility
        w = 0.3
        weight_total += w
        if s.hit_event_detected:
            score += w

        return score / weight_total if weight_total else 0.0
