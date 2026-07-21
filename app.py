"""
app.py
------
Entry point for the AI Badminton Analytics Platform Streamlit app.

Run locally:      streamlit run app.py
Run in Colab:      see README.md / notebook cell using `streamlit` + a
                   tunneling tool (e.g. `pyngrok` or Colab's built-in
                   port forwarding), since Colab can't serve a local port
                   directly to your browser.

This file only handles global app config, the session-state bootstrap, and
navigation. Each page under pages/ is a self-contained Streamlit page that
reads/writes shared state via `st.session_state`.
"""

import os
import sys

import streamlit as st

# Make sure `backend.*` imports resolve regardless of the working directory
# Streamlit was launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.utils.colab_setup import gpu_report, in_colab  # noqa: E402
from backend.utils.config_loader import get_config  # noqa: E402

st.set_page_config(
    page_title="AI Badminton Analytics",
    page_icon="🏸",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_session_state():
    defaults = {
        "match_mode": "singles",
        "video_path": None,
        "session_id": None,
        "processing_started": False,
        "processing_done": False,
        "court_corners": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session_state()

st.title("🏸 AI Badminton Analytics Platform")
st.caption("Proof-of-Concept — modular computer-vision pipeline for singles & doubles match analysis")

with st.sidebar:
    st.header("System Status")
    st.write(f"**Environment:** {'Google Colab' if in_colab() else 'Local / Other'}")
    try:
        st.write(gpu_report())
    except Exception as e:  # noqa: BLE001
        st.write(f"GPU check unavailable: {e}")

    cfg = get_config()
    st.write(f"**Match mode:** {st.session_state['match_mode']}")
    st.divider()
    st.markdown(
        "**Pipeline stages**\n\n"
        "1. Player Enrollment\n"
        "2. Video Upload\n"
        "3. Processing (live)\n"
        "4. Analytics Dashboard\n"
    )
    st.divider()
    st.caption(
        "Navigate using the pages in the left menu above ⬆️ "
        "(Streamlit auto-generates them from the `pages/` folder)."
    )

st.markdown(
    """
    ### Welcome
    This POC validates a full badminton video-analytics pipeline:
    detection → tracking → **persistent Re-ID identity** → pose → shuttle
    tracking → court mapping → rally/point/service inference → analytics.

    **Start with "1  Player Enrollment"** in the sidebar to register the
    players who will appear in your match video, then move to
    **"2  Video Upload"**.

    Every automatic decision the system can't make confidently — a point
    winner, a service call, an identity match — is flagged **Needs Review**
    rather than guessed, per this project's design principles.
    """
)

from backend.reid.gallery import list_enrolled_players  # noqa: E402

col1, col2, col3 = st.columns(3)
with col1:
    try:
        n_enrolled = len(list_enrolled_players())
    except Exception:  # noqa: BLE001
        n_enrolled = 0
    st.metric("Enrolled Players", n_enrolled, help="Configured on the Player Enrollment page")
with col2:
    st.metric("Video Loaded", "Yes" if st.session_state["video_path"] else "No")
with col3:
    st.metric("Analysis Complete", "Yes" if st.session_state["processing_done"] else "No")
