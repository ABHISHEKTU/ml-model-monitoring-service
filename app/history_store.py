"""
Stores every drift report ever generated, per model_id, as JSON Lines (one report
per line, appended). JSONL is chosen over overwriting a single file because it's
append-only and safe if the process crashes mid-write — you never lose prior history.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path


class HistoryStore:
    def __init__(self, base_dir: str = "./data/history"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, model_id: str) -> Path:
        safe_id = "".join(c for c in model_id if c.isalnum() or c in "-_")
        return self.base_dir / f"{safe_id}.jsonl"

    def append(self, model_id: str, report: dict) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "report": report,
        }
        with self._lock:
            with open(self._path(model_id), "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    def load(self, model_id: str, limit: int | None = None) -> list[dict]:
        path = self._path(model_id)
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        if limit:
            lines = lines[-limit:]
        return lines