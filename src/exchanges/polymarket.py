"""Polymarket market fetching via public Gamma API (read-only).

Uses no API keys; fetches open markets and normalizes basic fields.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import requests

BASE_URL = "https://gamma-api.polymarket.com"
TIMEOUT = 10
logger = logging.getLogger(__name__)


def fetch_polymarket_markets(limit: int = 100) -> Dict:
    """Fetch open Polymarket markets via the Gamma API.

    Returns the raw JSON dict from the API.
    """
    params = {"closed": "false", "limit": limit}
    url = f"{BASE_URL}/markets"
    resp = requests.get(url, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def normalize_polymarket_market(m: Dict) -> Dict | None:
    """Normalize a Polymarket market into a simple dict.

    Skips markets that do not have binary yes/no prices.
    """
    try:
        import json

        market_id = m.get("id") or m.get("_id")
        question = m.get("question") or m.get("title") or ""

        # Polymarket returns outcomes and prices as JSON strings
        outcome_prices_str = m.get("outcomePrices") or m.get("prices")
        if not outcome_prices_str:
            return None

        # Parse the JSON string to get price list
        try:
            if isinstance(outcome_prices_str, str):
                prices = json.loads(outcome_prices_str)
            elif isinstance(outcome_prices_str, list):
                prices = outcome_prices_str
            else:
                return None
        except (json.JSONDecodeError, TypeError):
            return None

        if not prices or len(prices) < 2:
            return None

        # Convert string prices to floats
        try:
            yes_price = float(prices[0])
            no_price = float(prices[1])
        except (ValueError, TypeError, IndexError):
            return None

        # Skip if prices are invalid
        if yes_price is None or no_price is None or yes_price < 0 or no_price < 0:
            return None

        volume = m.get("volume") or m.get("volumeNum") or 0
        liquidity = m.get("liquidity") or m.get("liquidityNum") or 0
        updated_at = m.get("updatedAt") or m.get("updated_at")
        close_time = m.get("end_date") or m.get("endDate") or m.get("closeTime")
        description = m.get("description") or ""
        rules = m.get("rules") or ""
        category = m.get("category") or ""
        tags = m.get("tags") or m.get("tickers") or []
        slug = m.get("slug") or ""

        # Build Polymarket URL
        market_url = f"https://polymarket.com/event/{slug}" if slug else ""

        return {
            "market_id": market_id,
            "question": question,
            "yes_price": float(yes_price),
            "no_price": float(no_price),
            "volume": float(volume) if volume is not None else 0.0,
            "liquidity": float(liquidity) if liquidity is not None else 0.0,
            "updated_at": updated_at,
            "close_time": close_time,
            "description": description,
            "rules": rules,
            "category": category,
            "tags": tags,
            "slug": slug,
            "market_url": market_url,
        }
    except Exception as exc:
        logger.debug("Failed to normalize market: %s", exc)
        return None
