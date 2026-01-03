"""Train a supervised model on opportunity logs."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support

from ml.feature_builder import build_features, FEATURE_ORDER
from ml.label_store import load_labels, load_label_entries

LOG_FILE = Path(__file__).resolve().parents[1] / "data" / "logs" / "opportunities.jsonl"
MODEL_DIR = Path(__file__).resolve().parents[1] / "data" / "models"
MODEL_FILE = MODEL_DIR / "opportunity_model.pkl"
SCHEMA_FILE = MODEL_DIR / "feature_schema.json"


def load_logs() -> List[Dict]:
    if not LOG_FILE.exists():
        return []
    out = []
    with LOG_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def label_samples(entries: List[Dict], horizon_minutes: int, ev_jump: float, edge_drop: float, price_jump: float, human_labels: Dict[str, str]) -> List[Tuple[Dict, int]]:
    """Label samples using future snapshots within horizon."""
    labeled = []
    entries_sorted = sorted(entries, key=lambda x: x.get("ts", 0))
    for i, e in enumerate(entries_sorted):
        ts = e.get("ts", 0)
        items = e.get("items") or []
        if not items:
            continue
        # Use first item for simplicity in MVP
        item = items[0]
        label = 0
        horizon = ts + horizon_minutes * 60
        future = [x for x in entries_sorted[i+1:] if x.get("ts", 0) <= horizon]
        if not future:
            continue  # unknown label
        # human label takes precedence if available
        if item.get("opportunity_id") in human_labels:
            lbl = human_labels[item["opportunity_id"]]
            if lbl == "good":
                label = 1
            elif lbl == "bad":
                label = 0
            else:
                continue
        else:
            for f in future:
                fit = (f.get("items") or [])[0] if f.get("items") else {}
                if fit.get("ev") and item.get("ev"):
                    if fit["ev"] - item["ev"] >= ev_jump:
                        label = 1
                        break
            if fit.get("edge_pct") and item.get("edge_pct"):
                if (item["edge_pct"] - fit["edge_pct"]) >= edge_drop:
                    label = 1
                    break
            if fit.get("yes_price") and item.get("yes_price"):
                if (fit["yes_price"] - item["yes_price"]) >= price_jump:
                    label = 1
                    break
        labeled.append((item, label))
    return labeled


def to_dataset(labeled: List[Tuple[Dict, int]]) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for item, label in labeled:
        feats, _ = build_features(item)
        X.append(feats)
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon_minutes", type=int, default=120)
    parser.add_argument("--min_samples", type=int, default=200)
    parser.add_argument("--ev_jump", type=float, default=0.05)
    parser.add_argument("--edge_drop", type=float, default=1.0)
    parser.add_argument("--price_jump", type=float, default=0.02)
    args = parser.parse_args()

    entries = load_logs()
    human_labels = load_labels()
    labeled = label_samples(entries, args.horizon_minutes, args.ev_jump, args.edge_drop, args.price_jump, human_labels)
    if len(labeled) < args.min_samples:
        print(f"Not enough samples ({len(labeled)} < {args.min_samples})")
        return
    X, y = to_dataset(labeled)
    # time-based split
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_val, probs)
    except Exception:
        auc = 0.0

    print(f"Val accuracy: {acc:.3f}, precision: {prec:.3f}, recall: {rec:.3f}, f1: {f1:.3f}, auc: {auc:.3f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    SCHEMA_FILE.write_text(json.dumps(FEATURE_ORDER, indent=2), encoding="utf-8")
    print(f"Saved model to {MODEL_FILE}")


if __name__ == "__main__":
    main()
