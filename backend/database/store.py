"""
store.py
--------
Lightweight persistence layer for the POC. Uses plain JSON files under
outputs/<session_id>/ rather than a real database — sufficient for a
single-user Colab/Streamlit session, and trivially swappable for SQLite/
Postgres later (every function here is a natural DAO boundary).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

from backend.utils.config_loader import resolve_path


def _session_dir(session_id: str) -> str:
    base = resolve_path("outputs_dir")
    path = os.path.join(base, session_id)
    os.makedirs(path, exist_ok=True)
    return path


def new_session_id() -> str:
    return time.strftime("session_%Y%m%d_%H%M%S")


def _default_encoder(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "tolist"):  # numpy arrays
        return obj.tolist()
    return str(obj)


def save_json(session_id: str, name: str, data: Any) -> str:
    path = os.path.join(_session_dir(session_id), f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_default_encoder)
    return path


def load_json(session_id: str, name: str) -> Optional[Any]:
    path = os.path.join(_session_dir(session_id), f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def append_event(session_id: str, name: str, event: Dict) -> None:
    """Append one event to a growing JSON-lines log (timeline, rally log, etc.)."""
    path = os.path.join(_session_dir(session_id), f"{name}.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(event, default=_default_encoder) + "\n")


def read_events(session_id: str, name: str) -> List[Dict]:
    path = os.path.join(_session_dir(session_id), f"{name}.jsonl")
    if not os.path.exists(path):
        return []
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def list_sessions() -> List[str]:
    base = resolve_path("outputs_dir")
    return sorted(
        [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))],
        reverse=True,
    )
