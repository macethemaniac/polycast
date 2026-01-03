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
        market_id = m.get("id") or m.get("_id")
        question = m.get("question") or m.get("title") or ""
        outcomes: List[Dict] = m.get("outcomes") or []
        if not outcomes or len(outcomes) < 2:
            return None

        # Attempt to identify yes/no outcomes
        yes = outcomes[0]
        no = outcomes[1]
        yes_price = yes.get("price") or yes.get("bestBid") or yes.get("bid")
        no_price = no.get("price") or no.get("bestBid") or no.get("bid")
        if yes_price is None or no_price is None:
            return None

        volume = m.get("volume") or 0
        liquidity = m.get("liquidity") or 0
        updated_at = m.get("updatedAt") or m.get("updated_at")
        close_time = m.get("end_date") or m.get("endDate") or m.get("closeTime")
        description = m.get("description") or ""
        rules = m.get("rules") or ""
        category = m.get("category") or ""
        tags = m.get("tags") or m.get("tickers") or []

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
        }
    except Exception as exc:
        logger.debug("Failed to normalize market: %s", exc)
        return None
