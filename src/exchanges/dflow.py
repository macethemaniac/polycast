"""DFlow Prediction Markets API client.

Provides fallback data source when primary APIs (Kalshi, Polymarket) are rate limited.
API Docs: https://pond.dflow.net

Note: Public endpoint has limited rate limits. For production, obtain API key from hello@dflow.net
"""

from __future__ import annotations

import os
import logging
from typing import Dict, List, Optional, Any

from src.utils.rate_limiter import get_with_retry, post_with_retry, RateLimitError

logger = logging.getLogger(__name__)

# DFlow API endpoints
BASE_URL = "https://prediction-markets-api.dflow.net"
PUBLIC_QUOTE_URL = "https://quote-api.dflow.net"
TIMEOUT = 15


def _get_headers() -> Dict[str, str]:
    """Get headers for DFlow API requests."""
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    api_key = os.getenv("DFLOW_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def fetch_dflow_events(limit: int = 100) -> List[Dict]:
    """Fetch prediction market events from DFlow.

    Returns list of event dicts with markets.
    """
    try:
        url = f"{BASE_URL}/api/v1/events"
        params = {"limit": min(limit, 200)}

        resp = get_with_retry(url, headers=_get_headers(), params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return data.get("events", data.get("data", []))
        return []

    except RateLimitError as e:
        logger.warning(f"DFlow rate limited: {e}")
        return []
    except Exception as e:
        logger.error(f"DFlow fetch_events error: {e}")
        return []


def fetch_dflow_markets(limit: int = 100) -> List[Dict]:
    """Fetch prediction markets from DFlow.

    Returns list of normalized market dicts.
    """
    try:
        url = f"{BASE_URL}/api/v1/markets"
        params = {"limit": min(limit, 200)}

        resp = get_with_retry(url, headers=_get_headers(), params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        markets = []
        if isinstance(data, list):
            markets = data
        elif isinstance(data, dict):
            markets = data.get("markets", data.get("data", []))

        return [normalize_dflow_market(m) for m in markets if m]

    except RateLimitError as e:
        logger.warning(f"DFlow rate limited: {e}")
        return []
    except Exception as e:
        logger.error(f"DFlow fetch_markets error: {e}")
        return []


def fetch_dflow_market(market_id: str) -> Optional[Dict]:
    """Fetch a single market by ID/ticker."""
    try:
        url = f"{BASE_URL}/api/v1/market/{market_id}"
        resp = get_with_retry(url, headers=_get_headers(), timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return normalize_dflow_market(data) if data else None

    except Exception as e:
        logger.error(f"DFlow fetch_market error: {e}")
        return None


def fetch_dflow_orderbook(market_ticker: str) -> Optional[Dict]:
    """Fetch orderbook for a market."""
    try:
        url = f"{BASE_URL}/api/v1/orderbook/{market_ticker}"
        resp = get_with_retry(url, headers=_get_headers(), timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    except Exception as e:
        logger.error(f"DFlow fetch_orderbook error: {e}")
        return None


def search_dflow_markets(query: str, limit: int = 20) -> List[Dict]:
    """Search for markets by query string."""
    try:
        url = f"{BASE_URL}/api/v1/search"
        params = {"q": query, "limit": limit}

        resp = get_with_retry(url, headers=_get_headers(), params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        results = []
        if isinstance(data, list):
            results = data
        elif isinstance(data, dict):
            results = data.get("results", data.get("markets", data.get("events", [])))

        return [normalize_dflow_market(m) for m in results if m]

    except Exception as e:
        logger.error(f"DFlow search error: {e}")
        return []


def normalize_dflow_market(m: Dict) -> Optional[Dict]:
    """Normalize DFlow market to common format.

    Maps DFlow fields to the standard market dict used by the app.
    """
    if not m:
        return None

    try:
        market_id = m.get("ticker") or m.get("id") or m.get("market_id")
        if not market_id:
            return None

        # DFlow uses different field names
        question = m.get("title") or m.get("question") or m.get("name") or ""
        description = m.get("description") or ""

        # Price extraction - DFlow may use different structures
        yes_price = None
        no_price = None

        # Try direct price fields
        if "yes_price" in m:
            yes_price = float(m["yes_price"])
        elif "yesPrice" in m:
            yes_price = float(m["yesPrice"])
        elif "price" in m:
            yes_price = float(m["price"])

        # Try outcomes array
        outcomes = m.get("outcomes", [])
        if outcomes and isinstance(outcomes, list):
            for o in outcomes:
                name = (o.get("name") or o.get("title") or "").lower()
                price = o.get("price") or o.get("lastPrice") or o.get("bid")
                if price is not None:
                    if "yes" in name:
                        yes_price = float(price)
                    elif "no" in name:
                        no_price = float(price)

        # Calculate no_price if only yes is available
        if yes_price is not None and no_price is None:
            no_price = 1.0 - yes_price

        # Volume and liquidity
        volume = float(m.get("volume") or m.get("totalVolume") or 0)
        liquidity = float(m.get("liquidity") or m.get("totalLiquidity") or 0)

        # Source platform (DFlow aggregates from multiple sources)
        source = m.get("source") or m.get("platform") or "dflow"

        # Timestamps
        updated_at = m.get("updatedAt") or m.get("updated_at") or m.get("lastUpdated")
        close_time = m.get("closeTime") or m.get("end_date") or m.get("expirationTime")

        return {
            "market_id": str(market_id),
            "question": question,
            "description": description,
            "yes_price": yes_price,
            "no_price": no_price,
            "volume": volume,
            "liquidity": liquidity,
            "updated_at": updated_at,
            "close_time": close_time,
            "source": source,
            "raw": m,
        }

    except Exception as e:
        logger.debug(f"Failed to normalize DFlow market: {e}")
        return None


def is_dflow_available() -> bool:
    """Check if DFlow API is accessible."""
    try:
        url = f"{BASE_URL}/api/v1/markets"
        resp = get_with_retry(url, headers=_get_headers(), params={"limit": 1}, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False
