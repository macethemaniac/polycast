"""Market search and deep-dive analysis engine.

Provides keyword search and comprehensive market analysis.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.exchanges.polymarket import fetch_polymarket_markets, normalize_polymarket_market
from src.ml.embeddings import embed_texts, EmbeddingStore
from src.engines.unified_scorer import compute_unified_score
from src.engines.market_filter import filter_markets_by_freshness, parse_close_time
from src.ml.market_clustering import cluster_markets
from src.data.news_gdelt import get_news_signal
from src.ml.sentiment import score_texts

logger = logging.getLogger(__name__)

# Cache for market data
_MARKET_CACHE: Dict = {"markets": [], "timestamp": 0}
_CACHE_TTL = 300  # 5 minutes


def _get_cached_markets(limit: int = 200) -> List[Dict]:
    """Get markets with caching."""
    import time

    now = time.time()
    if _MARKET_CACHE["markets"] and now - _MARKET_CACHE["timestamp"] < _CACHE_TTL:
        return _MARKET_CACHE["markets"][:limit]

    try:
        data = fetch_polymarket_markets(limit=limit)
        markets = data.get("markets") if isinstance(data, dict) else data
        if not markets:
            return []

        normalized = []
        for m in markets:
            norm = normalize_polymarket_market(m)
            if norm:
                normalized.append(norm)

        _MARKET_CACHE["markets"] = normalized
        _MARKET_CACHE["timestamp"] = now
        return normalized

    except Exception as e:
        logger.error(f"Failed to fetch markets: {e}")
        return _MARKET_CACHE.get("markets", [])


def search_markets(query: str, limit: int = 10) -> List[Dict]:
    """Search markets by keyword using semantic similarity.

    Args:
        query: Search query (keyword or phrase)
        limit: Maximum results to return

    Returns:
        List of matching markets with scores
    """
    if not query or len(query.strip()) < 2:
        return []

    markets = _get_cached_markets(200)
    if not markets:
        return []

    # Apply freshness filter
    markets = filter_markets_by_freshness(markets)

    if not markets:
        return []

    # Get embeddings
    questions = [m.get("question", "") for m in markets]
    try:
        market_embeddings = embed_texts(questions)
        query_embedding = embed_texts([query])

        # Compute similarities
        similarities = cosine_similarity(query_embedding, market_embeddings)[0]

        # Combine with results
        results = []
        for i, m in enumerate(markets):
            sim = float(similarities[i])
            if sim >= 0.25:  # Minimum threshold
                m["search_similarity"] = sim
                results.append(m)

        # Sort by similarity
        results.sort(key=lambda x: x.get("search_similarity", 0), reverse=True)

        return results[:limit]

    except Exception as e:
        logger.error(f"Search failed: {e}")
        # Fallback to keyword matching
        query_lower = query.lower()
        results = []
        for m in markets:
            q = m.get("question", "").lower()
            if query_lower in q:
                m["search_similarity"] = 0.5
                results.append(m)
        return results[:limit]


def get_market_by_id(market_id: str) -> Optional[Dict]:
    """Get a specific market by ID."""
    markets = _get_cached_markets(200)
    for m in markets:
        if m.get("market_id") == market_id:
            return m
    return None


def get_market_details(market: Dict) -> Dict:
    """Get comprehensive analysis for a market.

    Args:
        market: Normalized market dict

    Returns:
        Dict with full analysis including:
        - prices, volume, liquidity
        - news headlines
        - sentiment analysis
        - social trend matches
        - related markets
        - time to close
        - recommendation
    """
    question = market.get("question", "")
    market_id = market.get("market_id", "")

    # Get unified score
    scored = compute_unified_score(market)

    # Get news details
    news = get_news_signal(question)
    headlines = news.get("headlines", [])
    news_mentions = news.get("mentions_24h", 0)

    # Calculate time to close
    close_time_str = market.get("close_time")
    days_to_close = None
    close_date_str = None
    if close_time_str:
        close_time = parse_close_time(close_time_str)
        if close_time:
            now = datetime.now(timezone.utc)
            delta = close_time - now
            days_to_close = delta.total_seconds() / 86400
            close_date_str = close_time.strftime("%Y-%m-%d")

    # Find related markets
    related = find_related_markets(market, limit=3)

    return {
        **scored,
        "headlines": headlines,
        "news_mentions": news_mentions,
        "days_to_close": days_to_close,
        "close_date": close_date_str,
        "related_markets": related,
    }


def find_related_markets(market: Dict, limit: int = 3) -> List[Dict]:
    """Find markets related to the given market.

    Uses semantic similarity to find similar questions.
    """
    question = market.get("question", "")
    market_id = market.get("market_id", "")

    if not question:
        return []

    markets = _get_cached_markets(100)
    # Filter out the current market
    markets = [m for m in markets if m.get("market_id") != market_id]

    if not markets:
        return []

    try:
        # Embed all questions
        questions = [m.get("question", "") for m in markets]
        market_embeddings = embed_texts(questions)
        query_embedding = embed_texts([question])

        # Compute similarities
        similarities = cosine_similarity(query_embedding, market_embeddings)[0]

        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:limit]

        related = []
        for idx in top_indices:
            if similarities[idx] >= 0.4:  # Threshold for "related"
                m = markets[idx].copy()
                m["similarity"] = float(similarities[idx])
                related.append(m)

        return related

    except Exception as e:
        logger.debug(f"Failed to find related markets: {e}")
        return []


def format_market_analysis(details: Dict) -> str:
    """Format market details as readable text for Telegram."""
    lines = []

    # Header
    lines.append("<b>MARKET ANALYSIS</b>")
    lines.append("━━━━━━━━━━━━━━━━━━━━")

    # Question
    question = details.get("question", "")
    if len(question) > 150:
        question = question[:150] + "..."
    lines.append(f"\n<b>{question}</b>")

    # Prices
    yes = details.get("yes_price", 0)
    no = details.get("no_price", 0)
    vol = details.get("volume", 0)

    if vol >= 1_000_000:
        vol_str = f"${vol/1_000_000:.1f}M"
    elif vol >= 1_000:
        vol_str = f"${vol/1_000:.0f}K"
    else:
        vol_str = f"${vol:.0f}"

    lines.append(f"\n<b>PRICES</b>")
    lines.append(f"YES: ${yes:.2f} | NO: ${no:.2f} | Vol: {vol_str}")

    # Time to close
    days = details.get("days_to_close")
    close_date = details.get("close_date")
    if days is not None:
        if days < 1:
            time_str = f"{int(days * 24)}h remaining"
        elif days < 7:
            time_str = f"{days:.1f} days remaining"
        else:
            time_str = f"{int(days)} days ({close_date})"
        lines.append(f"Closes: {time_str}")

    # Signals and score
    confidence = details.get("confidence", 0)
    action = details.get("action", "WATCH")
    action_icon = details.get("action_icon", "[~]")

    lines.append(f"\n<b>SIGNALS</b> (Score: {confidence:.0f}/100)")

    signals = details.get("signals", [])
    for s in signals[:4]:
        name = s.get("name", "")
        positive = s.get("positive", True)
        icon = "[+]" if positive else "[-]"
        lines.append(f"{icon} {name}")

    # News headlines
    headlines = details.get("headlines", [])
    if headlines:
        lines.append(f"\n<b>NEWS</b>")
        for h in headlines[:2]:
            h_text = str(h)[:80]
            lines.append(f"- {h_text}")

    # Recommendation
    lines.append(f"\n<b>RECOMMENDATION: {action_icon} {action}</b>")
    rec = details.get("recommendation", "")
    if rec:
        lines.append(f"<i>{rec}</i>")

    # Related markets
    related = details.get("related_markets", [])
    if related:
        lines.append(f"\n<b>RELATED MARKETS</b>")
        for r in related[:3]:
            rq = r.get("question", "")[:50]
            ry = r.get("yes_price", 0)
            lines.append(f"- {rq}... (YES ${ry:.2f})")

    return "\n".join(lines)
