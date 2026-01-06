"""Google Trends data fetching via pytrends.

Provides trending topics and interest scores for matching to markets.
Primary social trend source (more reliable than Twitter scraping).
"""
from __future__ import annotations

import time
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

_CACHE: Dict[str, Dict] = {}
_TTL_SEC = 30 * 60  # 30 minutes (Google Trends rate limits)
_TIMEOUT = 15


def _from_cache(key: str) -> List | None:
    entry = _CACHE.get(key)
    if not entry:
        return None
    if time.time() - entry["ts"] > _TTL_SEC:
        _CACHE.pop(key, None)
        return None
    return entry["value"]


def _to_cache(key: str, value: List) -> None:
    _CACHE[key] = {"value": value, "ts": time.time()}


def _parse_traffic_score(traffic_str: str) -> float:
    """Parse Google Trends traffic string to numeric score (0-100).

    Examples: "500K+" -> 80, "10K+" -> 50, "1M+" -> 95
    """
    if not traffic_str:
        return 0.0

    traffic_str = str(traffic_str).upper().replace("+", "").replace(",", "").strip()

    try:
        # Handle K/M suffixes
        if "M" in traffic_str:
            num = float(traffic_str.replace("M", ""))
            return min(100.0, 80 + num * 5)
        elif "K" in traffic_str:
            num = float(traffic_str.replace("K", ""))
            if num >= 500:
                return 75.0
            elif num >= 100:
                return 60.0
            elif num >= 10:
                return 45.0
            else:
                return 30.0
        else:
            # Raw number
            num = float(traffic_str)
            if num >= 1000000:
                return 90.0
            elif num >= 100000:
                return 70.0
            elif num >= 10000:
                return 50.0
            else:
                return 30.0
    except (ValueError, TypeError):
        return 0.0


def get_realtime_trends(region: str = "united_states") -> List[Dict]:
    """Get real-time trending searches from Google Trends.

    Args:
        region: Region code (e.g., "united_states", "united_kingdom")

    Returns:
        List of trending topics:
        [{"topic": str, "title": str, "traffic": str, "score": float}]
    """
    cache_key = f"realtime_{region}"
    cached = _from_cache(cache_key)
    if cached is not None:
        return cached

    try:
        from pytrends.request import TrendReq

        pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
        df = pytrends.realtime_trending_searches(pn=region)

        trends = []
        if df is not None and not df.empty:
            for _, row in df.head(25).iterrows():
                topic = str(row.get("title", "") or "")
                traffic = str(row.get("formattedTraffic", "") or "")
                entity_names = row.get("entityNames", [])

                if not topic:
                    continue

                trends.append({
                    "topic": topic,
                    "title": ", ".join(entity_names) if entity_names else topic,
                    "traffic": traffic,
                    "score": _parse_traffic_score(traffic),
                })

        _to_cache(cache_key, trends)
        logger.debug("get_realtime_trends: fetched %d trends for %s", len(trends), region)
        return trends

    except ImportError:
        logger.warning("pytrends not installed, Google Trends unavailable")
        return []
    except Exception as exc:
        logger.debug("Google Trends realtime fetch failed: %s", exc)
        return []


def get_daily_trends(region: str = "united_states") -> List[Dict]:
    """Get daily trending searches from Google Trends.

    Returns:
        List of trending topics:
        [{"topic": str, "traffic": str, "articles": List[str], "score": float}]
    """
    cache_key = f"daily_{region}"
    cached = _from_cache(cache_key)
    if cached is not None:
        return cached

    try:
        from pytrends.request import TrendReq

        pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
        df = pytrends.trending_searches(pn=region)

        trends = []
        if df is not None and not df.empty:
            for i, row in df.head(20).iterrows():
                topic = str(row.iloc[0]) if len(row) > 0 else ""
                if not topic:
                    continue

                # Daily trends don't have traffic numbers, estimate by rank
                score = max(10, 100 - (i * 4))

                trends.append({
                    "topic": topic,
                    "traffic": "",
                    "articles": [],
                    "score": score,
                })

        _to_cache(cache_key, trends)
        logger.debug("get_daily_trends: fetched %d trends for %s", len(trends), region)
        return trends

    except ImportError:
        logger.warning("pytrends not installed, Google Trends unavailable")
        return []
    except Exception as exc:
        logger.debug("Google Trends daily fetch failed: %s", exc)
        return []


def get_interest_score(keyword: str, timeframe: str = "now 7-d") -> float:
    """Get relative interest score (0-100) for a keyword.

    Useful for boosting market scores based on trend interest.

    Args:
        keyword: Search term
        timeframe: pytrends timeframe (e.g., "now 7-d", "today 1-m")

    Returns:
        Interest score 0-100, or 0 on failure
    """
    cache_key = f"interest_{keyword}_{timeframe}"
    cached = _from_cache(cache_key)
    if cached is not None:
        return cached[0] if cached else 0.0

    try:
        from pytrends.request import TrendReq

        pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
        pytrends.build_payload([keyword], timeframe=timeframe)
        df = pytrends.interest_over_time()

        if df is not None and not df.empty and keyword in df.columns:
            # Get the average interest over the period
            score = float(df[keyword].mean())
            _to_cache(cache_key, [score])
            return score

        return 0.0

    except ImportError:
        return 0.0
    except Exception as exc:
        logger.debug("Google Trends interest fetch failed for %s: %s", keyword, exc)
        return 0.0
