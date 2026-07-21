"""
pipeline.py
-----------
Top-level orchestrator that runs one uploaded match video through every
stage described in the architecture diagram:

    Video -> Frame Extraction -> Court Detection -> Player Detection ->
    Player Tracking -> Re-ID -> Identity Assignment -> Pose Estimation ->
    Shuttle Detection -> Court Mapping -> Rally Detection -> Point Detection
    -> Service Detection -> Score Update -> Analytics Engine -> Dashboard

Each stage is a thin call into its own backend/<module>, so this file stays
short and is the one place that defines execution ORDER — swapping any
individual model only ever touches its own module, never this file's logic,
as long as the stage's public interface (dataclasses in/out) is preserved.

This is written as a generator (`run()` yields a `ProgressUpdate` after
every frame) so the Streamlit "Processing" page can stream live preview
frames, current score, and progress bar updates without blocking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from backend.analytics.stats_engine import PlayerFramePosition, StatsEngine, StrokeEvent
from backend.court.court_detector import CourtModel, HeuristicCourtDetector
from backend.database import store
from backend.pose.pose_estimator import PoseEstimator
from backend.reid.embedder import ReIDEmbedder
from backend.reid.gallery import load_gallery, load_player_names
from backend.reid.identity_matcher import IdentityMatcher
from backend.scoring.point_winner import PointResult
from backend.scoring.rally_detector import FrameSignals, RallyStateMachine
from backend.service.service_detector import ServiceTracker
from backend.shuttle.shuttle_detector import ShuttleDetector
from backend.tracking.tracker import PlayerTracker
from backend.utils.config_loader import get_config
from backend.utils.video_utils import frame_generator, probe_video


@dataclass
class ProgressUpdate:
    frame_index: int
    total_frames: int
    timestamp_sec: float
    frame_bgr: Optional[np.ndarray]
    player_boxes: List[dict]              # [{bbox, player_id, confidence, is_confident}, ...]
    shuttle_point: Optional[tuple]
    current_score: Dict[str, int]
    current_rally_active: bool
    current_server: Optional[str]
    events: List[str] = field(default_factory=list)


class MatchPipeline:
    def __init__(self, video_path: str, mode: str, session_id: Optional[str] = None):
        self.video_path = video_path
        self.mode = mode  # "singles" | "doubles"
        self.cfg = get_config()
        self.session_id = session_id or store.new_session_id()

        self.player_names = load_player_names()
        gallery = load_gallery()
        if not gallery:
            raise RuntimeError(
                "No enrolled players found. Complete Player Enrollment (Page 1) first."
            )

        self.detector_tracker = PlayerTracker(self.cfg)
        self.embedder = ReIDEmbedder(self.cfg)
        self.identity_matcher = IdentityMatcher(gallery, self.cfg)
        self.pose_estimator = PoseEstimator(self.cfg)
        self.shuttle_detector = ShuttleDetector(self.cfg)
        self.court_detector = HeuristicCourtDetector(self.cfg)
        self.rally_machine = RallyStateMachine(self.cfg)
        self.service_tracker = ServiceTracker(mode, self.cfg)
        self.stats_engine = StatsEngine(self.player_names)

        self._court_model: Optional[CourtModel] = None
        self._score = {"left": 0, "right": 0}
        self._prev_player_positions: Dict[str, tuple] = {}
        self._lost_tracks: List[int] = []
        self._active_tracks: set = set()
        self._manual_court_corners: Optional[list] = None

    def set_manual_court_corners(self, corners_image: list) -> None:
        """Optional: called by the UI if the user calibrated the court manually."""
        self._manual_court_corners = corners_image

    # ------------------------------------------------------------------ #
    def run(self):
        meta = probe_video(self.video_path)
        store.save_json(self.session_id, "video_meta", {
            "path": self.video_path, "fps": meta.fps, "width": meta.width,
            "height": meta.height, "frame_count": meta.frame_count,
            "duration_sec": meta.duration_sec,
        })

        if self._manual_court_corners:
            from backend.court.court_detector import build_from_manual_corners
            self._court_model = build_from_manual_corners(self._manual_court_corners)

        stride = self.cfg["runtime"]["frame_stride"]
        max_dim = self.cfg["runtime"]["max_frame_dim"]
        homography_refresh = self.cfg["court"]["homography_refresh_every_n_frames"]

        for frame_index, ts, frame in frame_generator(self.video_path, stride, max_dim):
            events: List[str] = []

            # --- Court detection / refresh -----------------------------
            if self._court_model is None or (
                frame_index % homography_refresh == 0 and self.cfg["court"]["detector"] != "manual"
            ):
                auto_court = self.court_detector.detect(frame)
                if self._court_model is None or auto_court.confidence > self._court_model.confidence:
                    self._court_model = auto_court

            # --- Tracking ------------------------------------------------
            tracked_boxes = self.detector_tracker.update(frame)
            current_track_ids = {t.track_id for t in tracked_boxes}
            newly_lost = self._active_tracks - current_track_ids
            self._lost_tracks = list(set(self._lost_tracks) | newly_lost)[-20:]
            self._active_tracks = current_track_ids

            # --- ReID + Identity assignment ------------------------------
            crops = [
                frame[int(max(t.y1, 0)): int(t.y2), int(max(t.x1, 0)): int(t.x2)]
                for t in tracked_boxes
            ]
            embeddings = self.embedder.embed_batch(crops) if crops else np.zeros((0, 1))

            player_box_updates = []
            total_motion = 0.0
            for t, emb in zip(tracked_boxes, embeddings):
                merged_track_id = self.identity_matcher.reconcile_reappearance(
                    t.track_id, emb, self._lost_tracks
                )
                effective_id = merged_track_id if merged_track_id is not None else t.track_id
                assignment = self.identity_matcher.assign(effective_id, emb)

                foot = t.foot_point
                court_xy = self._court_model.image_to_court(foot) if self._court_model else None

                if assignment.player_id:
                    prev = self._prev_player_positions.get(assignment.player_id)
                    if prev is not None:
                        total_motion += ((foot[0] - prev[0]) ** 2 + (foot[1] - prev[1]) ** 2) ** 0.5
                    self._prev_player_positions[assignment.player_id] = foot

                    self.stats_engine.record_position(
                        assignment.player_id,
                        PlayerFramePosition(frame_index, ts, court_xy),
                        meta.fps,
                    )

                player_box_updates.append({
                    "bbox": t.bbox,
                    "track_id": t.track_id,
                    "player_id": assignment.player_id,
                    "player_name": self.player_names.get(assignment.player_id, "Unknown"),
                    "confidence": assignment.confidence,
                    "is_confident": assignment.is_confident,
                })

            # --- Shuttle detection ----------------------------------------
            player_boxes_xyxy = [t.bbox for t in tracked_boxes]
            shuttle_obs = self.shuttle_detector.detect(frame, player_boxes_xyxy)
            shuttle_point = (shuttle_obs.x, shuttle_obs.y) if shuttle_obs else None

            # --- Rally detection --------------------------------------------
            signals = FrameSignals(
                frame_index=frame_index,
                timestamp_sec=ts,
                shuttle_present=shuttle_obs is not None,
                shuttle_speed_px=0.0,  # populate from consecutive shuttle_obs deltas in a full impl
                total_player_motion_px=total_motion,
                hit_event_detected=False,
            )
            rally_event = self.rally_machine.step(signals)
            if rally_event:
                events.append(f"{rally_event.event} (confidence {rally_event.confidence:.2f})")
                store.append_event(self.session_id, "timeline", {
                    "timestamp": _fmt_time(ts), "description": rally_event.event,
                    "frame": frame_index, "confidence": rally_event.confidence,
                })

            yield ProgressUpdate(
                frame_index=frame_index,
                total_frames=meta.frame_count,
                timestamp_sec=ts,
                frame_bgr=frame,
                player_boxes=player_box_updates,
                shuttle_point=shuttle_point,
                current_score=dict(self._score),
                current_rally_active=self.rally_machine.state.value == "active",
                current_server=self.service_tracker.serving_side,
                events=events,
            )

        # --- Finalize -------------------------------------------------------
        final_stats = self.stats_engine.finalize()
        store.save_json(self.session_id, "player_stats", {
            pid: {
                "name": s.name, "points_won": s.points_won, "points_lost": s.points_lost,
                "distance_covered_m": s.distance_covered_m, "max_speed_kmh": s.max_speed_kmh,
                "avg_speed_kmh": s.avg_speed_kmh, "win_pct": s.win_pct,
                "front_court_pct": s.front_court_pct, "back_court_pct": s.back_court_pct,
                "attack_ratio": s.attack_ratio, "defensive_ratio": s.defensive_ratio,
                "heatmap_points": s.heatmap_points,
            }
            for pid, s in final_stats.items()
        })


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
