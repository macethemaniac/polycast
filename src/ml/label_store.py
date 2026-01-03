"""Append-only label storage for opportunities."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parents[2]
LABEL_DIR = BASE_DIR / "data" / "labels"
LABEL_FILE = LABEL_DIR / "opportunity_labels.jsonl"


def _ensure_dir() -> None:
    LABEL_DIR.mkdir(parents=True, exist_ok=True)


def save_label(opportunity_id: str, label: str, timestamp: Optional[int] = None, notes: Optional[str] = None) -> None:
    """Append a label entry."""
    _ensure_dir()
    entry = {
        "ts": int(timestamp or time.time()),
        "opportunity_id": opportunity_id,
        "label": label,
        "notes": notes or "",
    }
    try:
        with LABEL_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        return


def load_labels() -> Dict[str, str]:
    """Return latest label per opportunity_id."""
    try:
        if not LABEL_FILE.exists():
            return {}
        labels: Dict[str, str] = {}
        with LABEL_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    labels[obj.get("opportunity_id")] = obj.get("label")
                except Exception:
                    continue
        return labels
    except Exception:
        return {}


def load_label_entries() -> List[Dict]:
    """Return all label entries."""
    try:
        if not LABEL_FILE.exists():
            return []
        out: List[Dict] = []
        with LABEL_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out
    except Exception:
        return []
