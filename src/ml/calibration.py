"""Calibration utilities for model scores."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "data" / "models"
CAL_FILE = MODEL_DIR / "calibrator.pkl"


def fit_calibrator(scores: List[float], labels: List[int]) -> None:
    if len(scores) < 10:
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X = np.array(scores, dtype=np.float32).reshape(-1, 1)
    y = np.array(labels, dtype=np.int32)
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, CAL_FILE)


def load_calibrator():
    try:
        if CAL_FILE.exists():
            return joblib.load(CAL_FILE)
    except Exception:
        return None
    return None


def apply_calibration(score: float) -> float:
    model = load_calibrator()
    if model is None:
        return score
    try:
        prob = model.predict_proba(np.array([[score]], dtype=np.float32))[0][1]
        return float(prob * 100.0)
    except Exception:
        return score
