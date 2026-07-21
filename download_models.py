"""
download_models.py
-------------------
Downloads/pre-fetches all pretrained weights the pipeline needs, into
models/, so the first pipeline run doesn't stall mid-video on a slow
download and so Colab sessions can cache weights to Google Drive.

Run once after `pip install -r requirements.txt`:

    python download_models.py

What gets downloaded:
    * YOLOv11n (player detection)     -> models/yolo11n.pt
    * YOLOv8n  (detection fallback)   -> models/yolov8n.pt
    * YOLOv11n-pose (pose estimation) -> models/yolo11n-pose.pt

NOT downloaded here (see module docstrings for why):
    * A shuttle-detection model — no reliable generic pretrained checkpoint
      exists (backend/shuttle/shuttle_detector.py runs a classical-CV
      heuristic by default; drop a trained TrackNet checkpoint into
      models/tracknet_v2.pt yourself if you have one for your footage).
    * A court-keypoint model — same reasoning (backend/court/court_detector.py
      uses a heuristic + manual-calibration fallback).
    * torchreid's OSNet weights — downloaded automatically by the `torchreid`
      package itself the first time ReIDEmbedder is used (cached under
      ~/.cache/torch by default).
"""

from __future__ import annotations

import os

from backend.utils.config_loader import get_config, resolve_path


def download_yolo_weights():
    from ultralytics import YOLO

    cfg = get_config()
    models_dir = resolve_path("models_dir")

    targets = [
        cfg["detection"]["model_name"],
        cfg["detection"]["fallback_model_name"],
        cfg["pose"]["model_name"],
    ]

    for name in targets:
        dest = os.path.join(models_dir, name)
        if os.path.exists(dest):
            print(f"[download_models] {name} already present, skipping.")
            continue
        print(f"[download_models] Downloading {name} ...")
        # Instantiating YOLO(name) triggers ultralytics' own download-and-cache
        # logic; we then copy the cached weight into our models/ directory so
        # everything the project needs lives in one predictable place.
        model = YOLO(name)
        cached_path = model.ckpt_path if hasattr(model, "ckpt_path") else None
        if cached_path and os.path.exists(cached_path) and cached_path != dest:
            import shutil

            shutil.copy(cached_path, dest)
        print(f"[download_models] {name} ready at {dest}")


if __name__ == "__main__":
    print("Downloading pretrained model weights for the AI Badminton Analytics Platform...\n")
    download_yolo_weights()
    print(
        "\nDone. Note: shuttle detection and court-keypoint detection use "
        "heuristic/manual methods by default — see the module docstrings in "
        "backend/shuttle/shuttle_detector.py and backend/court/court_detector.py "
        "for why, and how to plug in trained models later."
    )
