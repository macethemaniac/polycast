"""Market freshness filtering.

Filters markets by close time, recency, and activity to focus on
newer, active markets with trending potential.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def parse_close_time(close_time_str: Optional[str]) -> Optional[datetime]:
    """Parse close_time string to datetime.

    Handles ISO 8601 format and Unix timestamps.
    """
    if not close_time_str:
        return None

    try:
        # Try ISO 8601 format first
        if isinstance(close_time_str, str):
            # Handle various ISO formats
            for fmt in [
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d",
            ]:
                try:
                    dt = datetime.strptime(close_time_str, fmt)
                    return dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue

            # Try parsing as Unix timestamp string
            try:
                ts = float(close_time_str)
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            except (ValueError, OSError):
                pass

        # Handle numeric timestamps
        if isinstance(close_time_str, (int, float)):
            return datetime.fromtimestamp(float(close_time_str), tz=timezone.utc)

        return None
    except Exception:
        return None


def parse_updated_at(updated_at_str: Optional[str]) -> Optional[datetime]:
    """Parse updated_at string to datetime."""
    return parse_close_time(updated_at_str)  # Same format handling


def compute_days_until_close(close_time: datetime) -> float:
    """Compute days remaining until market closes."""
    now = datetime.now(timezone.utc)
    delta = close_time - now
    return delta.total_seconds() / 86400.0  # Convert to days


def compute_hours_since_update(updated_at: datetime) -> float:
    """Compute hours since last market update."""
    now = datetime.now(timezone.utc)
    delta = now - updated_at
    return delta.total_seconds() / 3600.0  # Convert to hours


def filter_markets_by_freshness(
    markets: List[Dict],
    min_days_until_close: float = 1.0,
    max_days_until_close: float = 90.0,
    min_volume: float = 100.0,
    max_staleness_hours: float = 168.0,  # 7 days
) -> List[Dict]:
    """Filter normalized Polymarket markets based on freshness criteria.

    Args:
        markets: List of normalized market dicts with close_time, volume, updated_at
        min_days_until_close: Exclude markets closing sooner than this (default 1 day)
        max_days_until_close: Exclude markets closing later than this (default 90 days)
        min_volume: Exclude markets with volume below this (default $100)
        max_staleness_hours: Exclude markets not updated within this period (default 7 days)

    Returns:
        Filtered list of markets meeting all criteria.
    """
    if not markets:
        return []

    filtered = []
    now = datetime.now(timezone.utc)

    for m in markets:
        # Check volume
        vol = float(m.get("volume", 0) or 0)
        if vol < min_volume:
            continue

        # Check close time
        close_time_str = m.get("close_time") or m.get("end_date") or m.get("endDate")
        if close_time_str:
            close_time = parse_close_time(close_time_str)
            if close_time:
                days_until = compute_days_until_close(close_time)

                # Skip if closing too soon
                if days_until < min_days_until_close:
                    continue

                # Skip if closing too far out
                if days_until > max_days_until_close:
                    continue

        # Check staleness (optional - only if updated_at exists)
        updated_at_str = m.get("updated_at") or m.get("updatedAt")
        if updated_at_str and max_staleness_hours > 0:
            updated_at = parse_updated_at(updated_at_str)
            if updated_at:
                hours_since = compute_hours_since_update(updated_at)
                if hours_since > max_staleness_hours:
                    continue

        filtered.append(m)

    logger.debug(
        "filter_markets_by_freshness: %d -> %d markets (vol>=%.0f, %.1f-%.1f days)",
        len(markets), len(filtered), min_volume, min_days_until_close, max_days_until_close
    )

    return filtered
