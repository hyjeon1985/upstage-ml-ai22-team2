from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from .path import out_dir

# -------------------------
# Model files
# -------------------------


def save_model(model: Any, name: str = "model") -> Path:
    path = out_dir(f"{name}.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(name: str = "model") -> Any:
    path = out_dir(f"{name}.joblib")

    return joblib.load(path)
