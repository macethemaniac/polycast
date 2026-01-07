"""Market search and deep-dive analysis engine.

Provides keyword search and comprehensive market analysis.
Fast version - no external API calls or slow embeddings.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.exchanges.polymarket import fetch_polymarket_markets, normalize_polymarket_market
from src.engines.market_filter import filter_markets_by_freshness, parse_close_time

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
    """Search markets by keyword matching (fast).

    Args:
        query: Search query (keyword or phrase)
        limit: Maximum results to return

    Returns:
        List of matching markets with scores
    """
    if not query or len(query.strip()) < 2:
        return []

    markets = _get_cached_markets(500)
    if not markets:
        return []

    # Apply freshness filter
    markets = filter_markets_by_freshness(markets)

    if not markets:
        return []

    query_lower = query.lower().strip()

    # Expand category keywords for broader search
    CATEGORY_KEYWORDS = {
        "politics": ["election", "president", "trump", "biden", "vote", "congress", "senate", "governor", "democrat", "republican", "poll", "political", "primary", "nominee"],
        "crypto": ["bitcoin", "btc", "ethereum", "crypto", "blockchain", "token", "airdrop", "defi", "solana", "xrp", "megaeth", "stablecoin"],
        "sports": ["nfl", "nba", "mlb", "football", "basketball", "soccer", "tennis", "golf", "super bowl", "world cup", "championship", "playoff", "college football", "ncaa", "player of the year"],
        "entertainment": ["movie", "oscar", "grammy", "emmy", "album", "celebrity", "music", "film", "show", "netflix", "taylor swift", "spotify"],
        "tech": ["openai", "google", "apple", "microsoft", "nvidia", "tesla", "spacex", "amazon", "hardware", "software", "chatgpt", "gpt-5"],
        "economy": ["inflation", "fed", "interest rate", "gdp", "recession", "stock", "economy", "unemployment", "trade", "tariff", "deficit", "revenue", "budget"],
        "world": ["war", "russia", "ukraine", "china", "israel", "gaza", "iran", "nato", "military", "sanctions", "ceasefire", "peace"],
        "climate": ["climate", "carbon", "emissions", "temperature", "hottest", "warming", "hurricane", "wildfire", "renewable", "energy", "record"],
    }

    # Check if query is a category name
    if query_lower in CATEGORY_KEYWORDS:
        category_words = CATEGORY_KEYWORDS[query_lower]
    else:
        category_words = None

    query_words = query_lower.split()

    # Fast keyword matching
    results = []
    for m in markets:
        question = m.get("question", "").lower()

        # Calculate match score based on keyword presence
        score = 0.0
        matches = 0

        # If searching by category, check category keywords
        if category_words:
            import re
            for cat_word in category_words:
                # Use word boundary matching to avoid false positives
                # e.g., "eth" shouldn't match "Kenneth", "ai" shouldn't match "Tagovailoa"
                pattern = r'\b' + re.escape(cat_word) + r'\b'
                if re.search(pattern, question, re.IGNORECASE):
                    matches += 1
                    score += 0.3
            if matches > 0:
                # Normalize score for category searches
                score = min(1.0, score)
        else:
            # Exact phrase match (highest score)
            if query_lower in question:
                score = 1.0
                matches = len(query_words)
            else:
                # Individual word matches
                for word in query_words:
                    if len(word) >= 3 and word in question:
                        matches += 1
                        score += 0.3

            if matches > 0:
                # Boost score by match ratio
                score = min(score, 1.0) * (matches / len(query_words))

        if matches > 0:
            m_copy = m.copy()
            m_copy["search_similarity"] = score
            results.append(m_copy)

    # Sort by similarity score
    results.sort(key=lambda x: x.get("search_similarity", 0), reverse=True)

    return results[:limit]


def get_market_by_id(market_id: str) -> Optional[Dict]:
    """Get a specific market by ID."""
    markets = _get_cached_markets(500)
    for m in markets:
        if m.get("market_id") == market_id:
            return m
    return None


def get_market_details(market: Dict) -> Dict:
    """Get analysis for a market (fast version - no external API calls).

    Args:
        market: Normalized market dict

    Returns:
        Dict with analysis including:
        - prices, volume, liquidity
        - time to close
        - price history (24h/7d changes)
        - recommendation
    """
    from src.engines.unified_scorer import compute_fast_score, determine_action
    from src.data.price_history import get_price_change

    question = market.get("question", "")
    yes_price = float(market.get("yes_price", 0.0) or 0.0)
    no_price = float(market.get("no_price", 0.0) or 0.0)
    volume = float(market.get("volume", 0.0) or 0.0)
    market_id = market.get("market_id", "")

    # Use fast scoring (no external API calls)
    scored = compute_fast_score(market)

    # Get price history
    price_data = get_price_change(market_id) if market_id else {}
    change_24h = price_data.get("change_24h")
    change_7d = price_data.get("change_7d")

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

    # Generate recommendation
    confidence = scored.get("confidence", 0)
    action = scored.get("action", "WATCH")
    if confidence >= 70:
        rec = f"Strong signal. Consider {action}."
    elif confidence >= 50:
        rec = f"Moderate signal. {action} with caution."
    else:
        rec = "Monitor for better entry point."

    # Build simple signals list based on price
    signals = []
    if yes_price < 0.3:
        signals.append({"name": "Low YES price", "value": yes_price, "positive": True})
    elif yes_price > 0.7:
        signals.append({"name": "High YES price", "value": yes_price, "positive": False})
    if volume >= 100_000:
        signals.append({"name": "Good volume", "value": volume, "positive": True})

    return {
        **market,
        **scored,
        "signals": signals,
        "headlines": [],
        "news_mentions": 0,
        "days_to_close": days_to_close,
        "close_date": close_date_str,
        "change_24h": change_24h,
        "change_7d": change_7d,
        "related_markets": [],
        "recommendation": rec,
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

    # Price trends
    change_24h = details.get("change_24h")
    change_7d = details.get("change_7d")
    if change_24h is not None or change_7d is not None:
        trend_parts = []
        if change_24h is not None:
            pct_24h = change_24h * 100
            trend_parts.append(f"24h: {'+' if pct_24h >= 0 else ''}{pct_24h:.1f}%")
        if change_7d is not None:
            pct_7d = change_7d * 100
            trend_parts.append(f"7d: {'+' if pct_7d >= 0 else ''}{pct_7d:.1f}%")
        if trend_parts:
            lines.append(f"Trend: {' | '.join(trend_parts)}")

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
