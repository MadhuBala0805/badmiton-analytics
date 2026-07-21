"""
Page 1 — Player Enrollment

Lets the user register 2 players (Singles) or 4 players (Doubles), each
with 2-3 reference images. Generates and previews ReID embeddings, and
persists them via backend/reid/gallery.py so the Processing stage can load
them later.
"""

import os
import sys
import uuid

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.reid.embedder import ReIDEmbedder  # noqa: E402
from backend.reid.gallery import (  # noqa: E402
    delete_player,
    enroll_player,
    list_enrolled_players,
)
from backend.utils.config_loader import get_config  # noqa: E402

st.set_page_config(page_title="Player Enrollment", page_icon="🧍", layout="wide")
st.title("🧍 Player Enrollment")

cfg = get_config()["reid"]

mode = st.radio(
    "Match type", ["Singles (2 players)", "Doubles (4 players)"], horizontal=True
)
n_players = 2 if "Singles" in mode else 4
st.session_state["match_mode"] = "singles" if n_players == 2 else "doubles"

st.info(
    f"Upload {cfg['gallery_min_images']}–{cfg['gallery_max_images']} reference images per player "
    "(front view, side view, and any extra angle available). More viewpoints make the "
    "Re-ID matcher noticeably more robust to rotation and occlusion during the match."
)

st.divider()
st.subheader("Currently Enrolled")
enrolled = list_enrolled_players()
if enrolled:
    for p in enrolled:
        c1, c2, c3 = st.columns([3, 2, 1])
        c1.write(f"**{p['name']}**")
        c2.write(f"{p['num_images']} reference image(s)")
        if c3.button("Remove", key=f"remove_{p['player_id']}"):
            delete_player(p["player_id"])
            st.rerun()
else:
    st.write("No players enrolled yet.")

st.divider()
st.subheader(f"Enroll {'2 players' if n_players == 2 else '4 players'}")

embedder = ReIDEmbedder()

for i in range(n_players):
    with st.expander(f"Player {i + 1}", expanded=(i < 2)):
        name = st.text_input(f"Player {i + 1} name", key=f"name_{i}")
        files = st.file_uploader(
            f"Reference images for Player {i + 1} (min {cfg['gallery_min_images']})",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f"files_{i}",
        )

        if files:
            cols = st.columns(len(files))
            for c, f in zip(cols, files):
                c.image(f, use_container_width=True)

        if st.button(f"Save Player {i + 1}", key=f"save_{i}"):
            if not name:
                st.error("Please enter a player name.")
            elif not files or len(files) < cfg["gallery_min_images"]:
                st.error(f"Please upload at least {cfg['gallery_min_images']} images.")
            else:
                images_bgr = []
                for f in files:
                    file_bytes = np.frombuffer(f.getvalue(), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    images_bgr.append(img)

                player_id = f"player_{uuid.uuid4().hex[:8]}"
                try:
                    with st.spinner("Generating embeddings..."):
                        enroll_player(
                            player_id, name, images_bgr,
                            [f.name for f in files], embedder=embedder,
                        )
                    st.success(f"Enrolled {name} with {len(images_bgr)} reference images.")
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Enrollment failed: {e}")

st.divider()
ready = len(list_enrolled_players()) >= n_players
if ready:
    st.success(f"✅ {n_players} players enrolled — you can proceed to '2  Video Upload'.")
else:
    st.warning(
        f"Enroll {n_players} players to continue "
        f"({len(list_enrolled_players())}/{n_players} done)."
    )
