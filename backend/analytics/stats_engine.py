"""
stats_engine.py
----------------
Aggregates per-frame tracking/pose/court data into the per-player match
statistics listed in the spec (distance covered, speed, court coverage,
heatmap source data, shot-type counts, attack/defense ratio, etc).

Design: this module never touches raw video or models — it consumes the
structured per-frame records the pipeline already produced (positions in
court-space, rally/point boundaries, stroke labels) and does arithmetic.
That separation keeps it trivially unit-testable and reusable for a
post-hoc "re-run analytics on saved tracking data" mode.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from backend.court.court_detector import zone_of_point


@dataclass
class PlayerFramePosition:
    frame_index: int
    timestamp_sec: float
    court_xy: Optional[Tuple[float, float]]  # None if court mapping unavailable this frame


@dataclass
class StrokeEvent:
    frame_index: int
    player_id: str
    stroke_type: str  # "smash" | "drop" | "clear" | "net" | "drive" | "unknown"


@dataclass
class PlayerStats:
    player_id: str
    name: str
    points_won: int = 0
    points_lost: int = 0
    rallies_won: int = 0
    rallies_lost: int = 0
    distance_covered_m: float = 0.0
    max_speed_kmh: float = 0.0
    avg_speed_kmh: float = 0.0
    zone_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    stroke_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    heatmap_points: List[Tuple[float, float]] = field(default_factory=list)
    movement_path: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def win_pct(self) -> float:
        total = self.points_won + self.points_lost
        return round(100 * self.points_won / total, 1) if total else 0.0

    @property
    def front_court_pct(self) -> float:
        return self._zone_pct(("front_left", "front_right"))

    @property
    def back_court_pct(self) -> float:
        return self._zone_pct(("back_left", "back_right"))

    @property
    def left_court_pct(self) -> float:
        return self._zone_pct(("front_left", "back_left"))

    @property
    def right_court_pct(self) -> float:
        return self._zone_pct(("front_right", "back_right"))

    def _zone_pct(self, zones: tuple) -> float:
        total = sum(self.zone_counts.values())
        if not total:
            return 0.0
        matched = sum(self.zone_counts.get(z, 0) for z in zones)
        return round(100 * matched / total, 1)

    @property
    def attack_ratio(self) -> float:
        attacking = self.stroke_counts.get("smash", 0) + self.stroke_counts.get("drive", 0)
        total = sum(self.stroke_counts.values())
        return round(100 * attacking / total, 1) if total else 0.0

    @property
    def defensive_ratio(self) -> float:
        defensive = self.stroke_counts.get("net", 0) + self.stroke_counts.get("clear", 0)
        total = sum(self.stroke_counts.values())
        return round(100 * defensive / total, 1) if total else 0.0


class StatsEngine:
    def __init__(self, player_names: Dict[str, str]):
        self.players: Dict[str, PlayerStats] = {
            pid: PlayerStats(player_id=pid, name=name) for pid, name in player_names.items()
        }
        self._last_position: Dict[str, PlayerFramePosition] = {}
        self._speed_samples: Dict[str, List[float]] = defaultdict(list)

    # ------------------------------------------------------------------ #
    def record_position(self, player_id: str, pos: PlayerFramePosition, fps: float) -> None:
        stats = self.players.get(player_id)
        if stats is None or pos.court_xy is None:
            return

        stats.heatmap_points.append(pos.court_xy)
        stats.movement_path.append(pos.court_xy)
        stats.zone_counts[zone_of_point(pos.court_xy)] += 1

        prev = self._last_position.get(player_id)
        if prev is not None and prev.court_xy is not None:
            dt = (pos.frame_index - prev.frame_index) / fps
            if dt > 0:
                dist_m = _euclidean(prev.court_xy, pos.court_xy)
                stats.distance_covered_m += dist_m
                speed_kmh = (dist_m / dt) * 3.6
                stats.max_speed_kmh = max(stats.max_speed_kmh, speed_kmh)
                self._speed_samples[player_id].append(speed_kmh)

        self._last_position[player_id] = pos

    def record_stroke(self, event: StrokeEvent) -> None:
        stats = self.players.get(event.player_id)
        if stats:
            stats.stroke_counts[event.stroke_type] += 1

    def record_point(self, winner_id: str, loser_id: str) -> None:
        if winner_id in self.players:
            self.players[winner_id].points_won += 1
        if loser_id in self.players:
            self.players[loser_id].points_lost += 1

    def record_rally_result(self, winner_id: str, loser_id: str) -> None:
        if winner_id in self.players:
            self.players[winner_id].rallies_won += 1
        if loser_id in self.players:
            self.players[loser_id].rallies_lost += 1

    def finalize(self) -> Dict[str, PlayerStats]:
        for pid, samples in self._speed_samples.items():
            if samples and pid in self.players:
                self.players[pid].avg_speed_kmh = round(sum(samples) / len(samples), 2)
        return self.players


def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
