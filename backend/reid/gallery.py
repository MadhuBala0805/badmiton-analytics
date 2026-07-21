"""
gallery.py
----------
Builds and persists the enrollment gallery: player_id -> reference
embeddings, generated from the 2-3 images uploaded on the "Player
Enrollment" Streamlit page.

Storage format: one .npz per player under embeddings/<player_id>.npz
containing:
    embeddings : (K, D) float32, L2-normalized
    name       : player's display name
    image_names: list of source filenames (for the "preview embeddings" UI)
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from backend.reid.embedder import ReIDEmbedder
from backend.utils.config_loader import get_config, resolve_path


def enroll_player(player_id: str, name: str, images_bgr: List[np.ndarray],
                   image_names: List[str], embedder: ReIDEmbedder | None = None) -> str:
    """
    Generate embeddings for all reference images of one player and persist
    them. `images_bgr` should already be person-cropped (or full portrait
    images where the person dominates the frame — fine for a POC).

    Returns the path the gallery entry was saved to.
    """
    cfg = get_config()["reid"]
    if len(images_bgr) < cfg["gallery_min_images"]:
        raise ValueError(
            f"Need at least {cfg['gallery_min_images']} reference images for "
            f"'{name}', got {len(images_bgr)}."
        )

    embedder = embedder or ReIDEmbedder()
    embeddings = embedder.embed_batch(images_bgr[: cfg["gallery_max_images"]])

    out_dir = resolve_path("embeddings_dir")
    out_path = os.path.join(out_dir, f"{player_id}.npz")
    np.savez(
        out_path,
        embeddings=embeddings.astype(np.float32),
        name=name,
        image_names=np.array(image_names, dtype=object),
    )
    return out_path


def load_gallery() -> Dict[str, np.ndarray]:
    """Load every enrolled player's embeddings for the current session."""
    emb_dir = resolve_path("embeddings_dir")
    gallery: Dict[str, np.ndarray] = {}
    for fname in os.listdir(emb_dir):
        if not fname.endswith(".npz"):
            continue
        player_id = fname[: -len(".npz")]
        data = np.load(os.path.join(emb_dir, fname), allow_pickle=True)
        gallery[player_id] = data["embeddings"]
    return gallery


def load_player_names() -> Dict[str, str]:
    emb_dir = resolve_path("embeddings_dir")
    names: Dict[str, str] = {}
    for fname in os.listdir(emb_dir):
        if not fname.endswith(".npz"):
            continue
        player_id = fname[: -len(".npz")]
        data = np.load(os.path.join(emb_dir, fname), allow_pickle=True)
        names[player_id] = str(data["name"])
    return names


def delete_player(player_id: str) -> None:
    emb_dir = resolve_path("embeddings_dir")
    path = os.path.join(emb_dir, f"{player_id}.npz")
    if os.path.exists(path):
        os.remove(path)


def list_enrolled_players() -> List[Dict[str, str]]:
    """Return [{'player_id':..., 'name':..., 'num_images':...}, ...]"""
    emb_dir = resolve_path("embeddings_dir")
    out = []
    for fname in sorted(os.listdir(emb_dir)):
        if not fname.endswith(".npz"):
            continue
        player_id = fname[: -len(".npz")]
        data = np.load(os.path.join(emb_dir, fname), allow_pickle=True)
        out.append(
            {
                "player_id": player_id,
                "name": str(data["name"]),
                "num_images": str(len(data["embeddings"])),
            }
        )
    return out
