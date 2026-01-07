"""Price history tracking for Polymarket markets.

Stores historical price snapshots to compute 24h and 7d price changes.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_FILE = DATA_DIR / "price_history.json"

# Keep 7 days of history, sample every 30 minutes
MAX_HISTORY_AGE = 7 * 24 * 3600  # 7 days in seconds
SAMPLE_INTERVAL = 30 * 60  # 30 minutes

# In-memory cache
_HISTORY_CACHE: Dict[str, List[Dict]] = {}
_LAST_LOAD_TIME: float = 0
_CACHE_TTL = 60  # Reload from disk every 60 seconds


def _load_history() -> Dict[str, List[Dict]]:
    """Load price history from disk."""
    global _HISTORY_CACHE, _LAST_LOAD_TIME

    now = time.time()
    if _HISTORY_CACHE and now - _LAST_LOAD_TIME < _CACHE_TTL:
        return _HISTORY_CACHE

    try:
        if HISTORY_FILE.exists():
            data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            _HISTORY_CACHE = data
            _LAST_LOAD_TIME = now
            return data
    except Exception as e:
        logger.error(f"Failed to load price history: {e}")

    return {}


def _save_history(history: Dict[str, List[Dict]]) -> None:
    """Save price history to disk."""
    global _HISTORY_CACHE, _LAST_LOAD_TIME

    try:
        HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")
        _HISTORY_CACHE = history
        _LAST_LOAD_TIME = time.time()
    except Exception as e:
        logger.error(f"Failed to save price history: {e}")


def _prune_old_entries(history: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """Remove entries older than MAX_HISTORY_AGE."""
    now = time.time()
    cutoff = now - MAX_HISTORY_AGE

    pruned = {}
    for market_id, snapshots in history.items():
        valid = [s for s in snapshots if s.get("ts", 0) >= cutoff]
        if valid:
            pruned[market_id] = valid

    return pruned


def record_prices(markets: List[Dict]) -> None:
    """Record current prices for markets.

    Call this periodically (e.g., every scan) to build history.
    Will only record if enough time has passed since last sample.

    Args:
        markets: List of normalized market dicts with market_id, yes_price, no_price
    """
    if not markets:
        return

    history = _load_history()
    now = time.time()
    recorded = 0

    for m in markets:
        market_id = m.get("market_id", "")
        if not market_id:
            continue

        yes_price = float(m.get("yes_price", 0) or 0)
        no_price = float(m.get("no_price", 0) or 0)

        if market_id not in history:
            history[market_id] = []

        snapshots = history[market_id]

        # Check if we should record (enough time since last sample)
        if snapshots:
            last_ts = snapshots[-1].get("ts", 0)
            if now - last_ts < SAMPLE_INTERVAL:
                continue

        # Record new snapshot
        snapshots.append({
            "ts": now,
            "yes": yes_price,
            "no": no_price,
        })
        recorded += 1

    if recorded > 0:
        # Prune old entries before saving
        history = _prune_old_entries(history)
        _save_history(history)
        logger.debug(f"Recorded prices for {recorded} markets")


def get_price_change(market_id: str) -> Dict[str, Optional[float]]:
    """Get 24h and 7d price changes for a market.

    Args:
        market_id: Market ID to lookup

    Returns:
        Dict with keys:
        - change_24h: Price change in last 24h (e.g., 0.05 = +5%)
        - change_7d: Price change in last 7d
        - current_price: Current YES price
        - price_24h_ago: YES price 24h ago
        - price_7d_ago: YES price 7d ago
    """
    history = _load_history()

    result = {
        "change_24h": None,
        "change_7d": None,
        "current_price": None,
        "price_24h_ago": None,
        "price_7d_ago": None,
    }

    snapshots = history.get(market_id, [])
    if not snapshots:
        return result

    # Sort by timestamp (newest last)
    snapshots = sorted(snapshots, key=lambda x: x.get("ts", 0))

    # Get current price (most recent)
    current = snapshots[-1]
    result["current_price"] = current.get("yes")

    now = time.time()
    ts_24h_ago = now - 24 * 3600
    ts_7d_ago = now - 7 * 24 * 3600

    # Find closest price to 24h ago
    price_24h = _find_closest_price(snapshots, ts_24h_ago)
    if price_24h is not None and result["current_price"] is not None:
        result["price_24h_ago"] = price_24h
        if price_24h > 0:
            result["change_24h"] = (result["current_price"] - price_24h) / price_24h

    # Find closest price to 7d ago
    price_7d = _find_closest_price(snapshots, ts_7d_ago)
    if price_7d is not None and result["current_price"] is not None:
        result["price_7d_ago"] = price_7d
        if price_7d > 0:
            result["change_7d"] = (result["current_price"] - price_7d) / price_7d

    return result


def _find_closest_price(snapshots: List[Dict], target_ts: float) -> Optional[float]:
    """Find the price closest to target timestamp.

    Only returns if snapshot is within 2 hours of target.
    """
    if not snapshots:
        return None

    closest = None
    closest_diff = float("inf")

    for s in snapshots:
        ts = s.get("ts", 0)
        diff = abs(ts - target_ts)
        if diff < closest_diff:
            closest_diff = diff
            closest = s.get("yes")

    # Only return if within 2 hours of target
    if closest_diff <= 2 * 3600:
        return closest

    return None


def get_price_trend(market_id: str) -> Tuple[str, str]:
    """Get human-readable price trend for a market.

    Returns:
        Tuple of (trend_24h, trend_7d) e.g., ("+5.2%", "-12.1%")
    """
    changes = get_price_change(market_id)

    def format_change(change: Optional[float]) -> str:
        if change is None:
            return "N/A"
        pct = change * 100
        if pct >= 0:
            return f"+{pct:.1f}%"
        return f"{pct:.1f}%"

    return (
        format_change(changes["change_24h"]),
        format_change(changes["change_7d"]),
    )


def get_trending_markets(history: Dict[str, List[Dict]] = None, top_n: int = 10) -> List[Dict]:
    """Get markets with biggest price movements in last 24h.

    Returns:
        List of dicts with market_id, change_24h, current_price
    """
    if history is None:
        history = _load_history()

    results = []
    for market_id in history:
        changes = get_price_change(market_id)
        if changes["change_24h"] is not None:
            results.append({
                "market_id": market_id,
                "change_24h": changes["change_24h"],
                "current_price": changes["current_price"],
            })

    # Sort by absolute change
    results.sort(key=lambda x: abs(x["change_24h"]), reverse=True)
    return results[:top_n]
