from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any, Dict, List

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None


@dataclass
class ModelRegistry:
    model_dir: Path
    models: Dict[str, Any]

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.models = {}

    def load_all(self) -> None:
        self.models.clear()
        if not self.model_dir.exists():
            return

        for path in self.model_dir.glob("*.pkl"):
            name = path.stem
            self.models[name] = self._load_model(path)

    def list_names(self) -> List[str]:
        return sorted(self.models.keys())

    def get(self, name: str) -> Any:
        return self.models.get(name)

    def _load_model(self, path: Path) -> Any:
        if joblib is not None:
            try:
                return joblib.load(path)
            except Exception:
                pass
        with path.open("rb") as f:
            return pickle.load(f)
