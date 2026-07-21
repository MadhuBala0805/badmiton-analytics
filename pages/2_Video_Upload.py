"""
Page 2 — Video Upload

Lets the user upload the match video (any reasonable length/quality),
choose Singles/Doubles (also settable on Page 1), optionally do a quick
manual court calibration (4-corner click) as a reliability fallback for the
automatic heuristic court detector, and kick off analysis.
"""

import os
import sys

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.reid.gallery import list_enrolled_players  # noqa: E402
from backend.utils.config_loader import resolve_path  # noqa: E402
from backend.utils.video_utils import probe_video  # noqa: E402

st.set_page_config(page_title="Video Upload", page_icon="🎬", layout="wide")
st.title("🎬 Match Video Upload")

enrolled = list_enrolled_players()
n_players_needed = 2 if st.session_state.get("match_mode", "singles") == "singles" else 4
if len(enrolled) < n_players_needed:
    st.warning(
        "Complete Player Enrollment first (need "
        f"{n_players_needed} players for {st.session_state.get('match_mode')} mode)."
    )

mode = st.radio(
    "Match type",
    ["Singles (2 players)", "Doubles (4 players)"],
    horizontal=True,
    index=0 if st.session_state.get("match_mode", "singles") == "singles" else 1,
)
st.session_state["match_mode"] = "singles" if "Singles" in mode else "doubles"

st.divider()
video_file = st.file_uploader(
    "Upload match video (MP4, AVI, MOV, MPEG — no duration limit, "
    "only bounded by available GPU memory / processing time)",
    type=["mp4", "avi", "mov", "mpeg", "mpg"],
)

if video_file is not None:
    uploads_dir = resolve_path("assets_dir")
    video_path = os.path.join(uploads_dir, video_file.name)
    with open(video_path, "wb") as f:
        f.write(video_file.getvalue())
    st.session_state["video_path"] = video_path

    meta = probe_video(video_path)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", f"{meta.duration_sec / 60:.1f} min")
    c2.metric("Resolution", f"{meta.width}x{meta.height}")
    c3.metric("FPS", f"{meta.fps:.1f}")
    c4.metric("Total Frames", f"{meta.frame_count:,}")

    st.video(video_path)

    st.divider()
    st.subheader("Court Calibration (recommended)")
    st.write(
        "Automatic court-line detection is best-effort on unconstrained YouTube "
        "footage (see docs/ARCHITECTURE.md for why this is a hard CV problem). "
        "For reliable court-mapping analytics, click the 4 outer court corners "
        "below on a representative frame, in order: **top-left, top-right, "
        "bottom-right, bottom-left**."
    )

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(meta.frame_count * 0.1))
    ok, frame = cap.read()
    cap.release()

    use_manual = st.checkbox("I want to manually calibrate the court", value=False)
    if use_manual and ok:
        st.write("Enter the 4 corner pixel coordinates (read them off the frame below).")
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Reference frame", use_container_width=True)
        h, w = frame.shape[:2]
        cols = st.columns(4)
        labels = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]
        corners = []
        for c, label in zip(cols, labels):
            with c:
                x = st.number_input(f"{label} X", 0, w, 0, key=f"corner_x_{label}")
                y = st.number_input(f"{label} Y", 0, h, 0, key=f"corner_y_{label}")
                corners.append((x, y))
        if st.button("Save Court Calibration"):
            st.session_state["court_corners"] = corners
            st.success("Court calibration saved.")
    elif not use_manual:
        st.caption("Leaving this unchecked uses the automatic heuristic detector "
                   "(confidence-scored; see the Processing page).")

    st.divider()
    if st.button("▶️ Start Analysis", type="primary", disabled=len(enrolled) < n_players_needed):
        st.session_state["processing_started"] = True
        st.session_state["processing_done"] = False
        st.success("Video ready. Go to '3  Processing' in the sidebar to run the pipeline.")
else:
    st.info("Upload a video to continue.")
