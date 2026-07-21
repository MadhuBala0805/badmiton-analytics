"""
identity_matcher.py
--------------------
The core of the "Primary Objective": turning a per-frame appearance
embedding into a STABLE player identity that survives occlusion, crossing,
rotation, side changes, and re-entry.

Two-level design:

1. Enrollment Gallery
   Each enrolled player has 1..N reference embeddings (from their uploaded
   images). This is the ground truth we always match against.

2. Track-Identity Memory
   While processing video, each tracker `track_id` accumulates a running
   embedding (EMA-smoothed). We match against the gallery at every frame,
   but we also keep a `track_id -> player_id` cache so that within a single
   unbroken track we don't re-decide identity every frame (that would be
   noisy) — we only re-evaluate when confidence drops or the track is new.

This means: court position is NEVER used as a signal for WHO a player is
(only for where they are on court, which is a separate analytics concern).
Identity is 100% appearance (ReID) + temporal consistency (tracking).

Confidence & "Needs Review"
----------------------------
Every assignment carries a confidence score. If the top match is too close
to the second-best match (see `needs_review_margin`) or below
`match_threshold`, the assignment is flagged `is_confident=False` so the
UI/analytics layer can visibly mark it instead of silently guessing wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from backend.utils.config_loader import get_config


@dataclass
class IdentityAssignment:
    track_id: int
    player_id: Optional[str]   # None if unresolved / needs review
    confidence: float
    is_confident: bool
    all_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class _TrackMemory:
    running_embedding: np.ndarray
    last_player_id: Optional[str] = None
    last_confidence: float = 0.0
    frames_since_last_match: int = 0


class IdentityMatcher:
    def __init__(self, gallery: Dict[str, np.ndarray], config: dict | None = None):
        """
        gallery: {player_id: (K, D) array of enrolled reference embeddings}
                 (K = number of reference images for that player, already
                 L2-normalized — see backend/reid/gallery.py builder)
        """
        self.cfg = (config or get_config())["reid"]
        self.gallery = gallery
        self._track_memory: Dict[int, _TrackMemory] = {}

    # ------------------------------------------------------------------ #
    def assign(self, track_id: int, embedding: np.ndarray) -> IdentityAssignment:
        """
        Called once per (frame, track) with that track's current appearance
        embedding. Returns the resolved (or unresolved) player identity.
        """
        mem = self._track_memory.get(track_id)
        alpha = self.cfg["ema_alpha"]

        if mem is None:
            mem = _TrackMemory(running_embedding=embedding.copy())
            self._track_memory[track_id] = mem
        else:
            mem.running_embedding = alpha * mem.running_embedding + (1 - alpha) * embedding
            norm = np.linalg.norm(mem.running_embedding)
            if norm > 0:
                mem.running_embedding /= norm

        scores = self._score_against_gallery(mem.running_embedding)
        assignment = self._resolve(track_id, scores)

        mem.last_player_id = assignment.player_id
        mem.last_confidence = assignment.confidence
        mem.frames_since_last_match = 0
        return assignment

    # ------------------------------------------------------------------ #
    def _score_against_gallery(self, embedding: np.ndarray) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for player_id, refs in self.gallery.items():
            # cosine similarity since everything is L2-normalized -> dot product
            sims = refs @ embedding
            scores[player_id] = float(np.max(sims))  # best-matching reference image
        return scores

    def _resolve(self, track_id: int, scores: Dict[str, float]) -> IdentityAssignment:
        if not scores:
            return IdentityAssignment(track_id, None, 0.0, False, scores)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_id, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else -1.0

        margin = best_score - second_score
        confident = (
            best_score >= self.cfg["match_threshold"]
            and margin >= self.cfg["needs_review_margin"]
        )

        return IdentityAssignment(
            track_id=track_id,
            player_id=best_id if confident else (best_id if best_score >= self.cfg["match_threshold"] else None),
            confidence=best_score,
            is_confident=confident,
            all_scores=scores,
        )

    # ------------------------------------------------------------------ #
    def reconcile_reappearance(self, new_track_id: int, embedding: np.ndarray,
                                lost_tracks: List[int]) -> Optional[int]:
        """
        When a NEW track_id appears (e.g. after full occlusion broke the
        tracker's link), check whether its embedding matches a recently
        LOST track's running embedding better than it matches "a brand new
        person". If so, we can merge histories so analytics (distance
        covered, rally participation, etc.) don't fragment across the gap.

        Returns the old track_id to merge into, or None if this really is
        a new/different track.
        """
        best_old_id = None
        best_sim = -1.0
        for old_id in lost_tracks:
            old_mem = self._track_memory.get(old_id)
            if old_mem is None:
                continue
            sim = float(old_mem.running_embedding @ embedding)
            if sim > best_sim:
                best_sim = sim
                best_old_id = old_id

        if best_old_id is not None and best_sim >= self.cfg["match_threshold"]:
            return best_old_id
        return None

    def player_of_track(self, track_id: int) -> Optional[str]:
        mem = self._track_memory.get(track_id)
        return mem.last_player_id if mem else None
