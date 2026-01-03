"""Feature builder for supervised opportunity models."""
from __future__ import annotations

import math
from typing import Dict, List, Tuple


FEATURE_ORDER = [
    "yes_price",
    "no_price",
    "log_volume",
    "liquidity",
    "news_mentions",
    "sentiment",
    "trend_score",
    "similarity",
    "settlement_sim",
    "confidence",
    "edge_pct",
    "time_to_expiry",
]


def _get(d: Dict, key: str, default: float = 0.0) -> float:
    val = d.get(key, default)
    try:
        return float(val)
    except Exception:
        return default


def _time_to_expiry(item: Dict) -> float:
    return _get(item, "time_to_expiry", 0.0)


def build_features(item: Dict) -> Tuple[List[float], List[str]]:
    """Return feature vector and order."""
    yes_price = _get(item, "yes_price")
    no_price = _get(item, "no_price")
    volume = _get(item, "volume") or _get(item, "poly_volume") or _get(item, "kalshi_volume")
    liquidity = _get(item, "liquidity")
    news_mentions = _get(item, "news_mentions")
    sentiment = _get(item, "sentiment")
    trend_score = _get(item, "trend_score")
    similarity = _get(item, "similarity")
    settlement_sim = _get(item, "settlement_sim")
    confidence = _get(item, "confidence")
    edge_pct = _get(item, "edge_pct")
    tte = _time_to_expiry(item)

    feats = [
        yes_price,
        no_price,
        math.log1p(volume),
        liquidity,
        news_mentions,
        sentiment,
        trend_score,
        similarity,
        settlement_sim,
        confidence,
        edge_pct,
        tte,
    ]
    return feats, FEATURE_ORDER
