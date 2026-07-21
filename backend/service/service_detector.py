"""
service_detector.py
--------------------
Tracks current server, service court (left/right), and service changes,
following standard badminton rules:

    * Singles: server serves from the right court when their score is even,
      left court when odd. Receiver stands diagonally opposite.
    * Doubles: same odd/even court rule applies to the SERVING SIDE's score;
      service passes between partners on consecutive service turns won by
      that side (first server of a new service turn keeps their position
      from the doubles pair's prior turn).
    * A service change (side-out) happens when the receiving side wins the
      rally.

This module is rules-driven, not vision-driven: it consumes the score state
(from point_winner.py / the scoring engine) and rally boundaries, and
outputs the correct server according to law-of-the-game bookkeeping. The
one heuristic piece is confirming server identity visually at the very
start of a rally (which player was standing in the service box and
initiated the first shuttle motion) — used only as a sanity check against
the rules-based prediction, surfaced as a confidence score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from backend.utils.config_loader import get_config


@dataclass
class ServiceState:
    serving_side: str          # "left" | "right" (which side of the NET, i.e. which team)
    server_player_id: Optional[str]
    service_court: str         # "right" | "left" (which service box, per odd/even score rule)
    confidence: float


class ServiceTracker:
    def __init__(self, mode: str = "singles", config: dict | None = None):
        self.cfg = (config or get_config())["service"]
        self.mode = mode  # "singles" | "doubles"
        self.score: Dict[str, int] = {"left": 0, "right": 0}
        self.serving_side = "left"
        # For doubles: which of the two partners on the serving side serves next.
        self._doubles_server_slot: Dict[str, int] = {"left": 0, "right": 0}
        self._first_serve_of_match = True

    def current_service_court(self) -> str:
        server_score = self.score[self.serving_side]
        return "right" if server_score % 2 == 0 else "left"

    def on_point_won(self, winning_side: str, side_players: Dict[str, list]) -> ServiceState:
        """
        Call after each resolved point. `side_players` = {"left": [player_id,...],
        "right": [player_id,...]} (1 id for singles, 2 for doubles).
        Returns the ServiceState to apply for the NEXT rally.
        """
        was_serving_side_winner = winning_side == self.serving_side

        self.score[winning_side] += 1

        if not was_serving_side_winner:
            # Side-out: service passes to the winning side.
            self.serving_side = winning_side
            if self.mode == "doubles" and not self._first_serve_of_match:
                # Winning side's server slot continues from where it left off
                # (the partner who did NOT just serve serves next), per BWF rules
                # for a new service turn following a side-out.
                self._doubles_server_slot[winning_side] = (
                    self._doubles_server_slot[winning_side] + 1
                ) % 2

        self._first_serve_of_match = False

        court = self.current_service_court()
        server_slot = self._doubles_server_slot[self.serving_side] if self.mode == "doubles" else 0
        players = side_players.get(self.serving_side, [])
        server_id = players[server_slot] if server_slot < len(players) else (players[0] if players else None)

        confidence = 0.9 if server_id is not None else 0.3
        return ServiceState(
            serving_side=self.serving_side,
            server_player_id=server_id,
            service_court=court,
            confidence=confidence,
        )

    def visual_sanity_check(self, predicted_server_id: str, detected_server_id: Optional[str]) -> float:
        """
        Compare the rules-predicted server against a vision-based guess
        (e.g. "which player initiated shuttle motion first this rally").
        Returns an agreement-adjusted confidence in [0,1]; caller decides
        whether to flag "Needs Review" if this disagrees with the rules
        engine (which itself is normally authoritative once score is known
        correctly).
        """
        if detected_server_id is None:
            return 0.5
        return 1.0 if detected_server_id == predicted_server_id else 0.2
