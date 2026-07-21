"""
Page 3 — Processing

Runs the full MatchPipeline over the uploaded video and streams live
preview: current frame with bounding boxes + player names, shuttle
position, progress bar, current rally/score/server state, and confidence
indicators — exactly as specified for this page.
"""

import os
import sys
import time

import cv2
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.pipeline import MatchPipeline  # noqa: E402
from backend.utils.colab_setup import free_gpu_memory  # noqa: E402

st.set_page_config(page_title="Processing", page_icon="⚙️", layout="wide")
st.title("⚙️ Processing")

if not st.session_state.get("video_path"):
    st.warning("No video uploaded yet. Go to '2  Video Upload' first.")
    st.stop()

st.write(f"**Video:** `{os.path.basename(st.session_state['video_path'])}`")
st.write(f"**Mode:** {st.session_state.get('match_mode', 'singles')}")

run_button = st.button("▶️ Run Pipeline", type="primary", disabled=st.session_state.get("processing_done", False))

progress_bar = st.progress(0.0)
status_text = st.empty()

col_video, col_state = st.columns([2, 1])
frame_placeholder = col_video.empty()

with col_state:
    score_placeholder = st.empty()
    rally_placeholder = st.empty()
    server_placeholder = st.empty()
    players_placeholder = st.empty()
    events_placeholder = st.empty()

if run_button:
    try:
        pipeline = MatchPipeline(
            video_path=st.session_state["video_path"],
            mode=st.session_state.get("match_mode", "singles"),
        )
        if st.session_state.get("court_corners"):
            pipeline.set_manual_court_corners(st.session_state["court_corners"])

        st.session_state["session_id"] = pipeline.session_id
        recent_events = []
        last_ui_update = 0.0

        for update in pipeline.run():
            progress = update.frame_index / max(update.total_frames, 1)
            progress_bar.progress(min(progress, 1.0))
            status_text.text(
                f"Frame {update.frame_index:,} / {update.total_frames:,}  "
                f"({update.timestamp_sec:.1f}s)"
            )

            # Throttle expensive UI redraws (frame image, tables) to ~4/sec
            now = time.time()
            if now - last_ui_update > 0.25:
                last_ui_update = now
                if update.frame_bgr is not None:
                    vis = update.frame_bgr.copy()
                    for box in update.player_boxes:
                        x1, y1, x2, y2 = [int(v) for v in box["bbox"]]
                        color = (0, 200, 0) if box["is_confident"] else (0, 165, 255)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                        label = f"{box['player_name']} ({box['confidence']:.2f})"
                        cv2.putText(vis, label, (x1, max(y1 - 8, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    if update.shuttle_point:
                        sx, sy = [int(v) for v in update.shuttle_point]
                        cv2.circle(vis, (sx, sy), 6, (0, 255, 255), -1)

                    frame_placeholder.image(
                        cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True
                    )

                score_placeholder.metric(
                    "Score", f"{update.current_score.get('left', 0)} - {update.current_score.get('right', 0)}"
                )
                rally_placeholder.write(
                    f"**Rally active:** {'🟢 Yes' if update.current_rally_active else '⚪ No'}"
                )
                server_placeholder.write(f"**Serving side:** {update.current_server}")

                if update.player_boxes:
                    players_placeholder.table(
                        [
                            {
                                "Player": b["player_name"],
                                "Confidence": f"{b['confidence']:.2f}",
                                "Status": "✅ Confident" if b["is_confident"] else "⚠️ Needs Review",
                            }
                            for b in update.player_boxes
                        ]
                    )

                if update.events:
                    recent_events = (update.events + recent_events)[:8]
                    events_placeholder.write("**Recent events:**\n\n" + "\n\n".join(recent_events))

        progress_bar.progress(1.0)
        status_text.text("Processing complete.")
        st.session_state["processing_done"] = True
        free_gpu_memory()
        st.success("✅ Analysis complete! Go to '4  Analytics Dashboard' to view results.")

    except Exception as e:  # noqa: BLE001
        st.error(f"Pipeline failed: {e}")
        st.exception(e)
