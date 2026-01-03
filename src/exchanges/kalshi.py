"""Kalshi market fetching (read-only) with optional auth."""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"
TIMEOUT = 10


def _auth_headers() -> Dict[str, str]:
    key = os.getenv("KALSHI_API_KEY")
    secret = os.getenv("KALSHI_API_SECRET") or os.getenv("KALSHI_API_TOKEN")
    if key and secret:
        return {"Authorization": f"Bearer {secret}", "X-Api-Key": key}
    if secret:
        return {"Authorization": f"Bearer {secret}"}
    return {}


def fetch_kalshi_markets(limit: int = 200) -> List[Dict]:
    """Fetch open Kalshi markets. Returns empty list if auth missing or request fails."""
    headers = _auth_headers()
    params = {"status": "Open", "limit": limit}
    url = f"{BASE_URL}/markets"
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
        if resp.status_code == 401 or resp.status_code == 403:
            logger.warning("Kalshi auth missing/invalid; set KALSHI_API_KEY and KALSHI_API_SECRET.")
            return []
        resp.raise_for_status()
        data = resp.json()
        markets = data.get("markets") if isinstance(data, dict) else data
        return markets or []
    except Exception as exc:
        logger.debug("Kalshi fetch failed: %s", exc)
        return []


def normalize_kalshi_market(m: Dict) -> Dict | None:
    """Normalize a Kalshi market into a common schema."""
    try:
        market_id = m.get("id") or m.get("market_id")
        question = m.get("title") or m.get("description") or ""
        description = m.get("description") or ""
        rules = m.get("rules") or m.get("resolution_criteria") or ""
        yes_price = m.get("yes_bid") or m.get("yesPrice") or m.get("yes_bid_price") or m.get("yes_price")
        no_price = m.get("no_bid") or m.get("noPrice") or m.get("no_bid_price") or m.get("no_price")

        # Kalshi prices are typically in cents; normalize to 0-1 if over 1
        if yes_price is not None and yes_price > 1:
            yes_price = float(yes_price) / 100.0
        if no_price is not None and no_price > 1:
            no_price = float(no_price) / 100.0

        if yes_price is None or no_price is None:
            return None

        volume = m.get("volume") or m.get("display_volume") or 0
        liquidity = m.get("liquidity") or 0
        updated_at = m.get("updated_at") or m.get("updatedAt")
        close_time = m.get("close_time") or m.get("closeTime")
        category = m.get("category") or ""
        tags = m.get("tags") or []
        resolution_source = m.get("resolution_source") or ""

        return {
            "market_id": market_id,
            "question": question,
            "description": description,
            "rules": rules,
            "yes_price": float(yes_price),
            "no_price": float(no_price),
            "volume": float(volume) if volume is not None else 0.0,
            "liquidity": float(liquidity) if liquidity is not None else 0.0,
            "updated_at": updated_at,
            "close_time": close_time,
            "category": category,
            "tags": tags,
            "resolution_source": resolution_source,
        }
    except Exception as exc:
        logger.debug("Failed to normalize Kalshi market: %s", exc)
        return None
