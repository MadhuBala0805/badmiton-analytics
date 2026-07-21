"""
report_generator.py
--------------------
Builds a downloadable match report (HTML, easily "Print to PDF" from the
browser; a direct PDF export via `weasyprint`/`reportlab` can be swapped in
here later without changing the caller in the Streamlit dashboard page).
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List

from backend.analytics.stats_engine import PlayerStats
from backend.utils.config_loader import resolve_path

_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Match Report — {generated_at}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, sans-serif; background:#0e1117; color:#e6e6e6; padding:32px; }}
  h1, h2 {{ color:#f2f2f2; }}
  table {{ border-collapse: collapse; width:100%; margin-bottom:24px; }}
  th, td {{ border:1px solid #333; padding:8px 12px; text-align:left; }}
  th {{ background:#1c1f26; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:10px; font-size:12px; }}
  .win {{ background:#1f8a3a; }}
  .review {{ background:#b58900; }}
  .section {{ margin-bottom:32px; }}
</style>
</head>
<body>
<h1>Badminton Match Report</h1>
<p>Generated: {generated_at} &nbsp;|&nbsp; Session: {session_id}</p>

<div class="section">
<h2>Result</h2>
<p><b>Winner:</b> {winner} &nbsp;&nbsp; <b>Final Score:</b> {final_score}</p>
<p><b>Duration:</b> {duration} &nbsp;&nbsp; <b>Total Rallies:</b> {total_rallies}</p>
<p><b>Longest Rally:</b> {longest_rally} &nbsp;&nbsp; <b>Shortest Rally:</b> {shortest_rally}
&nbsp;&nbsp; <b>Average Rally Length:</b> {avg_rally}</p>
</div>

<div class="section">
<h2>Player Statistics</h2>
<table>
<tr><th>Player</th><th>Points Won</th><th>Points Lost</th><th>Win %</th>
<th>Distance (m)</th><th>Max Speed (km/h)</th><th>Avg Speed (km/h)</th>
<th>Front %</th><th>Back %</th><th>Attack %</th><th>Defense %</th></tr>
{player_rows}
</table>
</div>

<div class="section">
<h2>Timeline</h2>
<table>
<tr><th>Time</th><th>Event</th></tr>
{timeline_rows}
</table>
</div>

</body>
</html>
"""


def _player_row(stats: PlayerStats) -> str:
    return (
        f"<tr><td>{stats.name}</td><td>{stats.points_won}</td><td>{stats.points_lost}</td>"
        f"<td>{stats.win_pct}%</td><td>{stats.distance_covered_m:.1f}</td>"
        f"<td>{stats.max_speed_kmh:.1f}</td><td>{stats.avg_speed_kmh:.1f}</td>"
        f"<td>{stats.front_court_pct}%</td><td>{stats.back_court_pct}%</td>"
        f"<td>{stats.attack_ratio}%</td><td>{stats.defensive_ratio}%</td></tr>"
    )


def generate_html_report(
    session_id: str,
    match_result: Dict,
    player_stats: Dict[str, PlayerStats],
    timeline: List[Dict],
) -> str:
    player_rows = "\n".join(_player_row(s) for s in player_stats.values())
    timeline_rows = "\n".join(
        f"<tr><td>{ev.get('timestamp', '')}</td><td>{ev.get('description', '')}</td></tr>"
        for ev in timeline
    )

    html = _TEMPLATE.format(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        session_id=session_id,
        winner=match_result.get("winner", "N/A"),
        final_score=match_result.get("final_score", "N/A"),
        duration=match_result.get("duration", "N/A"),
        total_rallies=match_result.get("total_rallies", "N/A"),
        longest_rally=match_result.get("longest_rally", "N/A"),
        shortest_rally=match_result.get("shortest_rally", "N/A"),
        avg_rally=match_result.get("avg_rally_length", "N/A"),
        player_rows=player_rows,
        timeline_rows=timeline_rows,
    )

    out_dir = resolve_path("outputs_dir")
    session_dir = os.path.join(out_dir, session_id)
    os.makedirs(session_dir, exist_ok=True)
    out_path = os.path.join(session_dir, "match_report.html")
    with open(out_path, "w") as f:
        f.write(html)
    return out_path
