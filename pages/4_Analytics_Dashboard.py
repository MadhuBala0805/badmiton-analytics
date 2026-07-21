"""
Page 4 — Analytics Dashboard

Final dashboard: player cards, score/winner, heatmaps, movement paths,
player comparison, stat charts, timeline, confidence scores, and a
downloadable HTML match report.
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import store  # noqa: E402
from backend.reports.report_generator import generate_html_report  # noqa: E402

st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")
st.title("📊 Analytics Dashboard")

sessions = store.list_sessions()
if not sessions:
    st.warning("No completed analysis sessions yet. Run the Processing page first.")
    st.stop()

default_idx = 0
if st.session_state.get("session_id") in sessions:
    default_idx = sessions.index(st.session_state["session_id"])
session_id = st.selectbox("Session", sessions, index=default_idx)

player_stats = store.load_json(session_id, "player_stats") or {}
timeline = store.read_events(session_id, "timeline")
video_meta = store.load_json(session_id, "video_meta") or {}

if not player_stats:
    st.info("This session has no saved statistics yet (processing may not have finished).")
    st.stop()

# --------------------------------------------------------------------------- #
st.subheader("Match Result")
sorted_players = sorted(player_stats.items(), key=lambda kv: kv[1]["points_won"], reverse=True)
winner_name = sorted_players[0][1]["name"] if sorted_players else "N/A"
final_score = " / ".join(f"{v['name']}: {v['points_won']}" for _, v in sorted_players)

c1, c2, c3 = st.columns(3)
c1.metric("Leader (by points)", winner_name)
c2.metric("Duration", f"{video_meta.get('duration_sec', 0) / 60:.1f} min")
c3.metric("Total Rally Events", len(timeline))

st.divider()

# --------------------------------------------------------------------------- #
st.subheader("Player Cards")
cols = st.columns(len(player_stats))
for col, (pid, s) in zip(cols, player_stats.items()):
    with col:
        st.markdown(f"### {s['name']}")
        st.metric("Points", s["points_won"])
        st.metric("Win %", f"{s['win_pct']}%")
        st.metric("Distance Covered", f"{s['distance_covered_m']:.1f} m")
        st.metric("Max Speed", f"{s['max_speed_kmh']:.1f} km/h")
        st.metric("Avg Speed", f"{s['avg_speed_kmh']:.1f} km/h")

st.divider()

# --------------------------------------------------------------------------- #
st.subheader("Player Comparison")
df = pd.DataFrame(
    [
        {
            "Player": s["name"],
            "Distance (m)": s["distance_covered_m"],
            "Max Speed (km/h)": s["max_speed_kmh"],
            "Avg Speed (km/h)": s["avg_speed_kmh"],
            "Front Court %": s["front_court_pct"],
            "Back Court %": s["back_court_pct"],
            "Attack %": s["attack_ratio"],
            "Defense %": s["defensive_ratio"],
        }
        for s in player_stats.values()
    ]
)
st.dataframe(df, use_container_width=True)
st.bar_chart(df.set_index("Player")[["Distance (m)", "Max Speed (km/h)", "Avg Speed (km/h)"]])

st.divider()

# --------------------------------------------------------------------------- #
st.subheader("Court Heatmaps")
heat_cols = st.columns(len(player_stats))
for col, (pid, s) in zip(heat_cols, player_stats.items()):
    with col:
        points = s.get("heatmap_points", [])
        if points:
            try:
                from backend.analytics.heatmap import figure_for_player
                fig = figure_for_player(s["name"], [tuple(p) for p in points])
                st.pyplot(fig)
            except Exception as e:  # noqa: BLE001
                st.caption(f"Heatmap unavailable: {e}")
        else:
            st.caption(f"No court-position data recorded for {s['name']} "
                       "(court calibration may be needed — see Video Upload page).")

st.divider()

# --------------------------------------------------------------------------- #
st.subheader("Timeline")
if timeline:
    st.dataframe(
        pd.DataFrame(timeline)[["timestamp", "description", "confidence"]],
        use_container_width=True,
    )
else:
    st.caption("No timeline events recorded.")

st.divider()

# --------------------------------------------------------------------------- #
st.subheader("Download Report")
if st.button("Generate Match Report"):
    match_result = {
        "winner": winner_name,
        "final_score": final_score,
        "duration": f"{video_meta.get('duration_sec', 0) / 60:.1f} min",
        "total_rallies": len([e for e in timeline if e.get("description") == "rally_start"]),
        "longest_rally": "N/A (requires per-rally duration tracking)",
        "shortest_rally": "N/A (requires per-rally duration tracking)",
        "avg_rally_length": "N/A (requires per-rally duration tracking)",
    }
    from backend.analytics.stats_engine import PlayerStats

    stats_objs = {
        pid: PlayerStats(
            player_id=pid, name=s["name"], points_won=s["points_won"],
            points_lost=s["points_lost"], distance_covered_m=s["distance_covered_m"],
            max_speed_kmh=s["max_speed_kmh"], avg_speed_kmh=s["avg_speed_kmh"],
        )
        for pid, s in player_stats.items()
    }
    report_path = generate_html_report(session_id, match_result, stats_objs, timeline)
    with open(report_path, "rb") as f:
        st.download_button(
            "⬇️ Download HTML Report", f, file_name="match_report.html", mime="text/html"
        )
    st.success(f"Report generated: {report_path}")
