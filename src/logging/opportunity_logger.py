"""Append-only JSONL logger for opportunity outputs."""
from __future__ import annotations

import json
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = BASE_DIR / "data" / "logs"
LOG_FILE = LOG_DIR / "opportunities.jsonl"


def _ensure_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _opportunity_id(item: Dict[str, Any], meta: Dict[str, Any] | None = None) -> str:
    parts = [
        meta.get("platform", "") if meta else "",
        str(item.get("market_id") or item.get("poly_market_id") or ""),
        str(item.get("kalshi_market_id") or ""),
        str(item.get("question") or item.get("question_poly") or "")[:80],
        str(int(time.time()) // 600),
    ]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()


def log_opportunities(command: str, payload: List[Dict[str, Any]], meta: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """Append a JSON line with command, timestamp, and payload."""
    try:
        _ensure_dir()
        enriched = []
        for it in payload:
            oid = _opportunity_id(it, meta)
            it = {**it, "opportunity_id": oid, "short_id": oid[:8]}
            enriched.append(it)
        entry = {
            "ts": int(time.time()),
            "command": command,
            "meta": meta or {},
            "items": enriched,
        }
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return enriched
    except Exception:
        return payload
