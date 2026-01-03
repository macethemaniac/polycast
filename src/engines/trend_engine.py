"""Trend engine for Polymarket markets using unsupervised anomaly scoring."""
from __future__ import annotations

from typing import Dict, List

from exchanges.polymarket import fetch_polymarket_markets, normalize_polymarket_market
from ml.trend_detection import load_state, save_state, compute_trend_scores


def get_trending_polymarket(limit: int = 200, top_n: int = 5) -> List[Dict]:
    """Return top-N trending Polymarket markets based on anomaly scoring."""
    try:
        data = fetch_polymarket_markets(limit=limit)
        markets = data.get("markets") if isinstance(data, dict) else None
        if markets is None and isinstance(data, list):
            markets = data
        if not markets:
            return []
        normalized = []
        for m in markets:
            norm = normalize_polymarket_market(m)
            if norm:
                normalized.append(norm)

        state = load_state()
        scored, new_state = compute_trend_scores(normalized, state)
        save_state(new_state)
        scored.sort(key=lambda x: x.get("trend_score", 0.0), reverse=True)
        return scored[:top_n]
    except Exception:
        return []
