# Installation Guide

## Requirements

* Python 3.10 or 3.11
* A CUDA-capable GPU is strongly recommended (the pipeline runs on CPU too,
  but video processing will be dramatically slower — expect single-digit
  FPS or worse on CPU for the detection+tracking+ReID+pose stack combined)
* ~4 GB free disk for model weights + dependencies

## Option A — Google Colab (recommended for this POC)

1. Open a new Colab notebook.
2. `Runtime > Change runtime type > T4 GPU` (or better).
3. Run the Quick Start cells from `README.md`.
4. Because Colab doesn't expose local ports to your browser directly, the
   app is served through `pyngrok`. Free ngrok accounts have session/time
   limits — for longer sessions, sign up for a free ngrok auth token and
   set it with `ngrok.set_auth_token("...")` before calling `ngrok.connect`.

### Google Drive persistence

Call `backend.utils.colab_setup.mount_drive()` early in your notebook, then
point `configs/config.yaml`'s `paths.embeddings_dir` / `paths.outputs_dir`
at folders under `/content/drive/MyDrive/...` if you want enrolled players
and analysis sessions to survive across Colab runtime resets. By default
they live under the ephemeral Colab filesystem and are lost when the
runtime disconnects.

## Option B — Local machine

```bash
git clone <your-repo-url> badminton-analytics
cd badminton-analytics
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python download_models.py
streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`).

## Installing `torchreid` (Person Re-ID backend)

`requirements.txt` installs it from GitHub
(`git+https://github.com/KaiyangZhou/deep-person-reid.git`) since it isn't
reliably published to PyPI. If this fails in your environment (e.g. no
network access to GitHub, or a build issue), the app **will still run** —
`backend/reid/embedder.py` automatically falls back to a torchvision
ResNet50 appearance embedding. Re-ID will still work, just with somewhat
lower matching accuracy than a purpose-built ReID backbone. To confirm
which backend is active, check the console log line printed the first time
a player is enrolled or a video is processed.

## GPU memory notes

* `configs/config.yaml -> runtime.max_frame_dim` controls the resolution
  frames are downscaled to before inference — lower it (e.g. 960 or 720)
  if you hit CUDA out-of-memory errors on long/high-resolution videos.
* `runtime.frame_stride` lets you process every 2nd/3rd frame instead of
  every frame, trading temporal resolution for speed/memory on very long
  matches.
* `backend/utils/colab_setup.py::free_gpu_memory()` is called after each
  pipeline run to release cached CUDA memory between videos.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `RuntimeError: No enrolled players found` | Complete Page 1 (Player Enrollment) before running Page 3. |
| CUDA out of memory | Lower `runtime.max_frame_dim`, increase `runtime.frame_stride`, or restart the runtime to clear fragmented memory. |
| Court heatmaps are empty | No court calibration succeeded — use the manual 4-corner calibration on the Video Upload page. |
| Player identities keep swapping | Add more/better reference images per player at enrollment (varied angles, similar lighting to the match footage); lower `reid.match_threshold` slightly if players are being missed, raise it if they're being confused with each other. |
| Streamlit page not updating live during processing | This is expected — UI redraws are throttled (`~4/sec`) in `pages/3_Processing.py` to keep processing throughput high; the progress bar and score update every frame, the live image every ~0.25s. |
