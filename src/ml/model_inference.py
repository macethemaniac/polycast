"""Model loading and inference for supervised opportunity ranking."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np

from ml.calibration import apply_calibration

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "data" / "models"
MODEL_FILE = MODEL_DIR / "opportunity_model.pkl"
SCHEMA_FILE = MODEL_DIR / "feature_schema.json"

_MODEL_CACHE = None
_SCHEMA_CACHE: List[str] | None = None


def _load_schema() -> List[str] | None:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None:
        return _SCHEMA_CACHE
    try:
        if SCHEMA_FILE.exists():
            _SCHEMA_CACHE = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
            return _SCHEMA_CACHE
    except Exception:
        return None
    return None


def load_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    try:
        if MODEL_FILE.exists():
            _MODEL_CACHE = joblib.load(MODEL_FILE)
            return _MODEL_CACHE
    except Exception:
        _MODEL_CACHE = None
    return None


def predict_proba(feature_vec: List[float], feature_order: List[str]) -> float:
    """Return probability [0,1] using trained model; 0 if missing/failed."""
    model = load_model()
    schema = _load_schema()
    if model is None or not schema:
        return 0.0
    if len(schema) != len(feature_order):
        return 0.0
    try:
        arr = np.array(feature_vec, dtype=np.float32).reshape(1, -1)
        proba = model.predict_proba(arr)[0][1]
        calibrated = apply_calibration(proba * 100.0) / 100.0
        return float(calibrated)
    except Exception:
        return 0.0
