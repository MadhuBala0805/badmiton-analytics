"""
colab_setup.py
---------------
Environment bootstrap helpers for running the platform inside Google Colab.

Responsibilities:
    * Detect whether we're in Colab.
    * Detect CUDA / GPU availability and report VRAM.
    * Mount Google Drive (optional, for persisting embeddings/outputs).
    * Provide a single `device()` helper used everywhere else in the backend
      so no other module has to re-implement CUDA detection.

This module intentionally has NO hard dependency on `google.colab` — it is
safe to import on a local machine too, it just no-ops the Colab-only bits.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Optional


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def device(preferred: str = "auto") -> str:
    """
    Resolve the torch device string ("cuda" or "cpu") based on config
    preference and actual availability.
    """
    import torch

    if preferred == "cpu":
        return "cpu"
    if preferred in ("auto", "cuda"):
        if torch.cuda.is_available():
            return "cuda"
        if preferred == "cuda":
            print(
                "[colab_setup] WARNING: 'cuda' was requested but no GPU is "
                "available. In Colab: Runtime > Change runtime type > GPU. "
                "Falling back to CPU (will be slow for video processing).",
                file=sys.stderr,
            )
        return "cpu"
    return "cpu"


def gpu_report() -> str:
    """Human-readable summary of the current GPU, for display in the UI."""
    import torch

    if not torch.cuda.is_available():
        return "No GPU detected. Running on CPU — expect significantly slower processing."
    name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return f"GPU: {name} ({total_mem_gb:.1f} GB VRAM)"


def mount_drive(mount_point: str = "/content/drive") -> Optional[str]:
    """
    Mount Google Drive when running in Colab so embeddings/outputs can
    persist across sessions. Returns the mount point, or None if not in Colab.
    """
    if not in_colab():
        print("[colab_setup] Not running in Colab — skipping Drive mount.")
        return None
    from google.colab import drive  # type: ignore

    drive.mount(mount_point)
    return mount_point


def ensure_dependencies(requirements_path: str = "requirements.txt") -> None:
    """
    Install any missing dependencies from requirements.txt. Intended to be
    called once at the top of the Colab notebook / app.py bootstrap cell.
    Safe to call repeatedly — pip skips already-satisfied packages quickly.
    """
    print(f"[colab_setup] Ensuring dependencies from {requirements_path} ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", requirements_path],
        check=True,
    )
    print("[colab_setup] Dependencies OK.")


def free_gpu_memory() -> None:
    """Release cached CUDA memory between heavy stages of the pipeline."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
