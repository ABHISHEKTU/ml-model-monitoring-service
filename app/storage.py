"""
Reference-dataset store, separated from drift logic (single responsibility).
FileStore persists each model's baseline to disk as parquet so it survives restarts.
"""

from __future__ import annotations
import json
import threading
from pathlib import Path

import pandas as pd

from app.exceptions import ReferenceNotSetError


class FileStore:
    def __init__(self, base_dir: str = "./data/reference"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, model_id: str) -> Path:
        safe_id = "".join(c for c in model_id if c.isalnum() or c in "-_")
        return self.base_dir / f"{safe_id}.parquet"

    def save(self, model_id: str, df: pd.DataFrame) -> None:
        with self._lock:
            df.to_parquet(self._path(model_id), index=False)

    def load(self, model_id: str) -> pd.DataFrame:
        path = self._path(model_id)
        if not path.exists():
            raise ReferenceNotSetError(f"No reference set for model_id='{model_id}'.")
        return pd.read_parquet(path)

    def exists(self, model_id: str) -> bool:
        return self._path(model_id).exists()

    def _config_path(self, model_id: str) -> Path:
        safe_id = "".join(c for c in model_id if c.isalnum() or c in "-_")
        return self.base_dir / f"{safe_id}_config.json"

    def save_config(self, model_id: str, config: dict) -> None:
        with self._lock:
            self._config_path(model_id).write_text(json.dumps(config))

    def load_config(self, model_id: str) -> dict:
        path = self._config_path(model_id)
        if not path.exists():
            raise ReferenceNotSetError(f"No feature config for model_id='{model_id}'.")
        return json.loads(path.read_text())

    def _predictions_path(self, model_id: str) -> Path:
        safe_id = "".join(c for c in model_id if c.isalnum() or c in "-_")
        return self.base_dir / f"{safe_id}_predictions.json"

    def save_predictions(self, model_id: str, predictions: list) -> None:
        with self._lock:
            self._predictions_path(model_id).write_text(json.dumps(predictions))

    def load_predictions(self, model_id: str) -> list:
        path = self._predictions_path(model_id)
        if not path.exists():
            raise ReferenceNotSetError(f"No reference predictions stored for model_id='{model_id}'.")
        return json.loads(path.read_text())

    def has_predictions(self, model_id: str) -> bool:
        return self._predictions_path(model_id).exists()