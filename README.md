# 🏸 AI Badminton Analytics Platform (POC)

A modular, research-oriented computer-vision platform for analyzing public
badminton match footage — built to run on **Google Colab GPU** with a
**Streamlit** frontend and a cleanly separated **Python backend**.

This is a **Proof of Concept**: it validates the full pipeline end-to-end
(detection → tracking → persistent Re-ID identity → pose → shuttle tracking
→ court mapping → rally/point/service inference → analytics dashboard) with
every stage designed to be independently replaceable as better models
become available — this codebase is meant to grow toward a commercial
platform, not be thrown away.

## Primary design goal: persistent player identity

The system's #1 job is keeping a player's identity stable for the entire
match — through crossing, rotation, side changes, occlusion, jumps, and
disappearance/reappearance — using **Re-Identification (ReID) + tracking**,
**never court position alone**. See `backend/reid/` and
`docs/ARCHITECTURE.md` for how this is implemented.

## Quick start (Google Colab)

```python
# Cell 1 — get the code
!git clone <your-repo-url> badminton-analytics
%cd badminton-analytics

# Cell 2 — install dependencies (GPU runtime required: Runtime > Change
# runtime type > T4 GPU or better)
!pip install -q -r requirements.txt

# Cell 3 — pre-fetch model weights
!python download_models.py

# Cell 4 — (optional) mount Drive so embeddings/outputs persist across sessions
from backend.utils.colab_setup import mount_drive
mount_drive()

# Cell 5 — launch Streamlit and expose it via ngrok (Colab can't serve a
# local port to your browser directly)
!pip install -q pyngrok
from pyngrok import ngrok
import subprocess, time

subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501",
                   "--server.headless", "true"])
time.sleep(5)
print(ngrok.connect(8501))
```

Click the printed ngrok URL to open the app.

## Quick start (local machine)

```bash
git clone <your-repo-url> badminton-analytics
cd badminton-analytics
pip install -r requirements.txt
python download_models.py
streamlit run app.py
```

See `INSTALL.md` for details, GPU notes, and troubleshooting.

## Using the app

1. **Player Enrollment** — register 2 players (Singles) or 4 (Doubles),
   each with 2–3 reference images (front, side, extra angle).
2. **Video Upload** — upload the match video (MP4/AVI/MOV/MPEG, any
   reasonable length), pick Singles/Doubles, optionally calibrate the
   court by clicking its 4 corners for reliable court-mapping analytics.
3. **Processing** — run the pipeline with a live preview: bounding boxes,
   player names, shuttle position, score, current rally/server, and
   per-player identity confidence.
4. **Analytics Dashboard** — player cards, comparison charts, heatmaps,
   movement paths, timeline, and a downloadable HTML match report.

## What's real vs. heuristic in this POC

This project follows one hard rule: **never silently guess**. Where a
robust open-source model doesn't exist for a sub-task, the code says so in
its docstring, implements the best available approach, exposes a
confidence score, and marks low-confidence decisions **"Needs Review"**
instead of pretending to be sure. Summary:

| Module | Approach | Status |
|---|---|---|
| Player detection | YOLOv11 (Ultralytics) | Production-grade pretrained model |
| Player tracking | BoT-SORT / ByteTrack (Ultralytics) | Production-grade |
| Person Re-ID | OSNet via `torchreid`, ResNet50 fallback | Production-grade backbone, POC-scale gallery |
| Pose estimation | YOLOv11-pose (Ultralytics) | Production-grade pretrained model |
| Court detection | Classical CV heuristic + manual-click fallback | Best-effort; see docstring |
| Shuttle detection | Classical CV heuristic (background subtraction) | Best-effort; see docstring — a trained TrackNet checkpoint is the documented upgrade path |
| Rally detection | Multi-signal heuristic fusion, confidence-scored | Best-effort; documented upgrade path to a learned temporal model |
| Point winner | Rule-based indicators + confidence aggregation | Conservative — defers to "Needs Review" when unsure |
| Service tracking | Rules-engine (BWF service rules) + visual sanity check | Deterministic once score is known |

Full detail in `docs/ARCHITECTURE.md`.

## Project structure

```
project/
├── app.py                  # Streamlit entry point
├── pages/                  # Streamlit multi-page UI (Enrollment, Upload, Processing, Dashboard)
├── backend/
│   ├── detection/          # Player detection (YOLO)
│   ├── tracking/           # Multi-object tracking (BoT-SORT/ByteTrack)
│   ├── reid/                # Re-ID embeddings + persistent identity matching
│   ├── shuttle/             # Shuttle detection + trajectory
│   ├── court/                # Court detection + homography + zones
│   ├── pose/                  # Pose estimation
│   ├── analytics/             # Stats engine + heatmaps
│   ├── scoring/                # Rally detection + point-winner inference
│   ├── service/                 # Service/server tracking
│   ├── reports/                  # Match report generation
│   ├── database/                  # Session persistence (JSON-based POC store)
│   ├── utils/                       # Config, Colab/GPU setup, video I/O
│   └── pipeline.py                    # Orchestrates every stage in order
├── models/            # Downloaded model weights (gitignored)
├── embeddings/        # Enrolled player ReID embeddings (gitignored)
├── outputs/            # Per-session analytics/report output (gitignored)
├── configs/config.yaml  # Every tunable parameter, one place
├── requirements.txt
├── download_models.py
├── INSTALL.md
└── docs/ARCHITECTURE.md   # Deep-dive on every module and its assumptions
```

## Extending this POC

Every stage is a swap-in point:

* **Better shuttle detection** → train/obtain a TrackNet-family checkpoint,
  load it in `backend/shuttle/shuttle_detector.py::_load_tracknet_if_available`.
* **Better court detection** → add a keypoint-regression model as a third
  strategy in `backend/court/court_detector.py`.
* **Better ReID** → swap `reid.backend` in `configs/config.yaml` to
  `FastReID`/`TransReID` once integrated in `backend/reid/embedder.py`.
* **Real database** → replace `backend/database/store.py`'s JSON-file
  functions with SQLite/Postgres calls; nothing else needs to change.

## License / model attributions

This project orchestrates open-source models (Ultralytics YOLO family,
`deep-person-reid` OSNet, torchvision) — respect each project's own license
when using this platform, especially for any commercial use.
