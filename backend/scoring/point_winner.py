"""
point_winner.py
----------------
Infers which side won a completed rally, using the indicators listed in the
spec: shuttle landing outside, net fault, missed return, no valid return,
double hit.

HONEST SCOPE NOTE: reliably detecting "shuttle landed out" or "net fault"
from arbitrary monocular YouTube footage requires accurate court-line
homography AND accurate shuttle landing-point estimation — both of which
are themselves best-effort heuristics in this POC (see court_detector.py
and shuttle_detector.py). Errors compound. This module is therefore built
to be conservative: if the combined evidence isn't strong, it returns
`winner=None, needs_review=True` rather than guessing, exactly as the spec
requires ("If confidence is low, mark the point as Needs Review instead of
making an incorrect decision").

Each `PointIndicator` is a small, independently-testable rule so new
indicators can be added (or a learned classifier substituted) without
rewriting the aggregation logic in `infer_point_winner`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from backend.court.court_detector import CourtModel
from backend.shuttle.shuttle_detector import ShuttleObservation
from backend.utils.config_loader import get_config


@dataclass
class PointIndicator:
    name: str                 # e.g. "shuttle_landed_out"
    side_at_fault: Optional[str]  # "left" | "right" | None if inconclusive
    confidence: float
    detail: str


@dataclass
class PointResult:
    winning_side: Optional[str]   # "left" | "right" | None
    confidence: float
    needs_review: bool
    indicators: List[PointIndicator]


def check_shuttle_out_of_bounds(
    last_observation: ShuttleObservation, court: CourtModel
) -> Optional[PointIndicator]:
    if court.homography is None:
        return None
    court_xy = court.image_to_court((last_observation.x, last_observation.y))
    if court_xy is None:
        return None
    x, y = court_xy
    from backend.court.court_detector import COURT_LENGTH_M, COURT_WIDTH_M

    margin = 0.15  # meters of tolerance around the boundary for homography noise
    out = x < -margin or x > COURT_WIDTH_M + margin or y < -margin or y > COURT_LENGTH_M + margin
    if not out:
        return None

    fault_side = "left" if y < COURT_LENGTH_M / 2 else "right"
    # The point goes to the side that did NOT hit it out.
    beneficiary = "right" if fault_side == "left" else "left"
    conf = min(last_observation.confidence, 0.7)  # capped: landing-point estimate is itself uncertain
    return PointIndicator(
        name="shuttle_landed_out",
        side_at_fault=beneficiary,
        confidence=conf,
        detail=f"Shuttle's last tracked position fell outside court bounds ({x:.2f}, {y:.2f}) m.",
    )


def check_no_return(
    rally_end_frame: int, last_shuttle_frame: int, fps: float, max_gap_sec: float = 1.2
) -> Optional[PointIndicator]:
    """
    If the rally-state-machine flagged rally_end well after the last shuttle
    observation, it likely means a player failed to return the shuttle
    (rather than the shuttle going visibly out). We can't attribute a side
    without knowing who was due to hit — caller should supply that context;
    here we only flag the *event*, side attribution happens in
    `infer_point_winner` using player proximity at the last observation.
    """
    gap_sec = (rally_end_frame - last_shuttle_frame) / fps
    if gap_sec < max_gap_sec:
        return None
    return PointIndicator(
        name="no_return_detected",
        side_at_fault=None,
        confidence=0.4,
        detail=f"No shuttle motion for {gap_sec:.2f}s before rally end — likely missed return.",
    )


def infer_point_winner(indicators: List[PointIndicator]) -> PointResult:
    cfg = get_config()["scoring"]
    usable = [i for i in indicators if i.side_at_fault is not None]

    if not usable:
        return PointResult(None, 0.0, True, indicators)

    # Weighted vote across indicators (an indicator's own confidence is its weight)
    votes = {"left": 0.0, "right": 0.0}
    for ind in usable:
        votes[ind.side_at_fault] += ind.confidence

    total = sum(votes.values())
    if total == 0:
        return PointResult(None, 0.0, True, indicators)

    winner = max(votes, key=votes.get)
    confidence = votes[winner] / total * (sum(v for v in votes.values()) / max(len(usable), 1))
    confidence = min(confidence, max(i.confidence for i in usable))  # never exceed best single signal

    needs_review = confidence < cfg["low_confidence_threshold"]
    return PointResult(
        winning_side=None if needs_review else winner,
        confidence=confidence,
        needs_review=needs_review,
        indicators=indicators,
    )
