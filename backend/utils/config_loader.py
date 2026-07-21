"""
config_loader.py
-----------------
Single source of truth for reading configs/config.yaml.

Usage:
    from backend.utils.config_loader import get_config
    cfg = get_config()
    cfg["detection"]["model_name"]

The loader caches the parsed config in-process so repeated calls are cheap,
and exposes `reload_config()` for the Streamlit UI to pick up edits made
through a settings page without restarting the app.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict

import yaml

_CONFIG_CACHE: Dict[str, Any] | None = None
_LOCK = threading.Lock()

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs",
    "config.yaml",
)


def get_config(path: str | None = None) -> Dict[str, Any]:
    """Return the parsed config dict, loading it once and caching thereafter."""
    global _CONFIG_CACHE
    with _LOCK:
        if _CONFIG_CACHE is None:
            _CONFIG_CACHE = _load(path or _DEFAULT_CONFIG_PATH)
        return _CONFIG_CACHE


def reload_config(path: str | None = None) -> Dict[str, Any]:
    """Force a fresh read from disk (used by the UI after a config edit)."""
    global _CONFIG_CACHE
    with _LOCK:
        _CONFIG_CACHE = _load(path or _DEFAULT_CONFIG_PATH)
        return _CONFIG_CACHE


def _load(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config file not found at {path}. "
            "Make sure you're running from the project root."
        )
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} did not parse into a dict.")
    return data


def resolve_path(relative_key_path: str) -> str:
    """
    Resolve a project-relative path from the `paths` section of the config.
    Example: resolve_path("models_dir") -> "<project_root>/models"
    """
    cfg = get_config()
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    rel = cfg["paths"][relative_key_path]
    full = os.path.join(project_root, rel)
    os.makedirs(full, exist_ok=True)
    return full
