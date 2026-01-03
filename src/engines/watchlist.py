"""Watchlist storage and scanning for cross-market mismatches."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from polycast.scanner import scan_cross_market_mismatches

BASE_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE_DIR / "data" / "cache"
WATCH_FILE = CACHE_DIR / "watchlist.json"


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_watchlist() -> Dict[str, List[str]]:
    _ensure_cache_dir()
    try:
        if not WATCH_FILE.exists():
            return {"polymarket": [], "kalshi": []}
        data = json.loads(WATCH_FILE.read_text(encoding="utf-8"))
        return {
            "polymarket": list(dict.fromkeys(data.get("polymarket", []))),
            "kalshi": list(dict.fromkeys(data.get("kalshi", []))),
        }
    except Exception:
        return {"polymarket": [], "kalshi": []}


def save_watchlist(data: Dict[str, List[str]]) -> None:
    _ensure_cache_dir()
    try:
        WATCH_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def add_to_watchlist(platform: str, market_id: str) -> None:
    wl = load_watchlist()
    platform = platform.lower()
    if platform not in wl:
        wl[platform] = []
    if market_id not in wl[platform]:
        wl[platform].append(market_id)
    save_watchlist(wl)


def remove_from_watchlist(platform: str, market_id: str) -> None:
    wl = load_watchlist()
    platform = platform.lower()
    if platform in wl and market_id in wl[platform]:
        wl[platform].remove(market_id)
    save_watchlist(wl)


def list_watchlist() -> Dict[str, List[str]]:
    return load_watchlist()


def scan_watchlist(top_n: int = 5) -> Tuple[List[Dict], str | None]:
    """Filter cross-market mismatches to watchlisted markets."""
    wl = load_watchlist()
    if not wl.get("polymarket") and not wl.get("kalshi"):
        return [], "Watchlist is empty."
    results, err = scan_cross_market_mismatches(limit=200, top_n=50)
    if err:
        return [], err
    filtered = []
    for r in results:
        poly_q = r.get("question_poly", "")
        kal_q = r.get("question_kalshi", "")
        poly_id_in = any(mid in poly_q or mid in r.get("question_poly", "") for mid in wl.get("polymarket", []))
        kal_id_in = any(mid in kal_q or mid in r.get("question_kalshi", "") for mid in wl.get("kalshi", []))
        # Also match if market_id present in fields
        poly_match = any(mid == str(r.get("poly_market_id", "")) for mid in wl.get("polymarket", [])) if r.get("poly_market_id") else False
        kal_match = any(mid == str(r.get("kalshi_market_id", "")) for mid in wl.get("kalshi", [])) if r.get("kalshi_market_id") else False
        if poly_id_in or kal_id_in or poly_match or kal_match:
            filtered.append(r)
    filtered.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return filtered[:top_n], None
