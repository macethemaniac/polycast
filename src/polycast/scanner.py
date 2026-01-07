"""
Shared scanner module for arbitrage checking.

This module provides a unified interface for checking arbitrage opportunities
that can be used by both console and Telegram bot interfaces.
"""

from typing import Dict, Optional, Tuple, List
from src.exchanges.ccxt_client import get_ccxt_prices, DEFAULT_EXCHANGES
from polycast.analytics.arbitrage import check_arbitrage
from src.exchanges.polymarket import fetch_polymarket_markets, normalize_polymarket_market
from src.exchanges.kalshi import fetch_kalshi_markets, normalize_kalshi_market
from src.engines.opportunity_ranker import rank_polymarket_opportunities
from src.engines.trend_engine import get_trending_polymarket
from src.ml.market_clustering import cluster_markets
from src.engines.cross_market_matcher import match_markets_by_embedding
from src.engines.cross_arbitrage import detect_mismatches
from src.engines.market_filter import filter_markets_by_freshness
from src.engines.social_matcher import match_trends_to_markets, get_markets_by_social_relevance
from src.engines.unified_scorer import score_markets, format_signals_text
from src.engines.market_search import search_markets, get_market_details
from src.data.price_history import record_prices, get_price_change


def scan_arbitrage(
    pair: str = 'BTC/USDT',
    exchanges: Optional[list] = None,
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Scan for arbitrage opportunities for a given trading pair.
    
    Args:
        pair: Trading pair symbol (e.g., 'BTC/USDT', 'ETH/USDT')
        exchanges: Optional list of ccxt exchange ids to query
        
    Returns:
        Tuple of (arbitrage_result_dict, error_message)
        If successful: (result_dict, None)
        If error: (None, error_string)
    """
    try:
        exchange_list = exchanges or DEFAULT_EXCHANGES
        prices = get_ccxt_prices(pair, exchange_list)

        if len(prices) < 2:
            raise ValueError(
                f"Need at least 2 exchange prices; got {len(prices)} from {exchange_list}"
            )
        
        # Check for arbitrage opportunity
        arbitrage_result = check_arbitrage(prices)
        
        # Add raw prices to result for convenience
        arbitrage_result['prices'] = prices
        arbitrage_result['pair'] = pair
        
        return arbitrage_result, None
        
    except Exception as e:
        return None, str(e)


def scan_polymarket_raw(limit: int = 200, threshold: float = 0.02) -> List[Dict]:
    """
    Find intra-market mispricing on Polymarket where yes+no > 1+threshold.

    Args:
        limit: number of markets to fetch (open only).
        threshold: minimum excess over 1.0 to consider a candidate (e.g., 0.02 = 2%).
    """
    candidates: List[Dict] = []
    try:
        data = fetch_polymarket_markets(limit=limit)
        markets = data.get("markets") if isinstance(data, dict) else None
        if markets is None and isinstance(data, list):
            markets = data
        if not markets:
            return []

        for m in markets:
            norm = normalize_polymarket_market(m)
            if not norm:
                continue
            total = float(norm["yes_price"]) + float(norm["no_price"])
            if total <= 1.0 + threshold:
                continue
            candidates.append({
                "question": norm["question"],
                "market_id": norm["market_id"],
                "yes_price": norm["yes_price"],
                "no_price": norm["no_price"],
                "volume": norm.get("volume", 0.0),
                "total": total,
                "profit_pct": (total - 1.0) * 100.0,
            })
        candidates.sort(key=lambda x: x["total"], reverse=True)
        return candidates
    except Exception:
        return []


def scan_polymarket_ml(limit: int = 200, top_n: int = 5) -> List[Dict]:
    """
    Rank Polymarket markets using the ML-inspired opportunity ranker.
    Returns top_n entries sorted by opportunity_score.

    Filters out stale/closed markets and focuses on fresh, active markets.
    """
    import logging
    logger = logging.getLogger(__name__)
    try:
        data = fetch_polymarket_markets(limit=limit)
        logger.info(f"scan_polymarket_ml: fetched data type={type(data)}, len={len(data) if isinstance(data, list) else 'N/A'}")
        markets = data.get("markets") if isinstance(data, dict) else None
        if markets is None and isinstance(data, list):
            markets = data
        if not markets:
            logger.warning("scan_polymarket_ml: no markets found")
            return []
        normalized = []
        for m in markets:
            norm = normalize_polymarket_market(m)
            if norm:
                normalized.append(norm)
        logger.info(f"scan_polymarket_ml: normalized {len(normalized)} markets")

        # Apply freshness filter
        normalized = filter_markets_by_freshness(
            normalized,
            min_days_until_close=1.0,
            max_days_until_close=90.0,
            min_volume=100.0,
        )
        logger.info(f"scan_polymarket_ml: {len(normalized)} markets after freshness filter")

        ranked = rank_polymarket_opportunities(normalized)
        logger.info(f"scan_polymarket_ml: ranked {len(ranked)} markets")
        return ranked[:top_n]
    except Exception as e:
        logger.error(f"scan_polymarket_ml error: {e}", exc_info=True)
        return []


def scan_polymarket_trending(limit: int = 200, top_n: int = 5) -> List[Dict]:
    """
    Return top trending Polymarket markets based on anomaly scoring.

    Filters out stale/closed markets and focuses on fresh, active markets.
    """
    import logging
    logger = logging.getLogger(__name__)
    try:
        results = get_trending_polymarket(limit=limit, top_n=top_n * 2)  # Get extra for filtering
        logger.info(f"scan_polymarket_trending: got {len(results)} results")

        # Apply freshness filter
        results = filter_markets_by_freshness(
            results,
            min_days_until_close=1.0,
            max_days_until_close=90.0,
            min_volume=50.0,  # Lower threshold for trending
        )
        logger.info(f"scan_polymarket_trending: {len(results)} after freshness filter")

        return results[:top_n]
    except Exception as e:
        logger.error(f"scan_polymarket_trending error: {e}", exc_info=True)
        return []


def scan_social_trending(limit: int = 100, top_n: int = 5) -> List[Dict]:
    """
    Find markets matching current social media trends (Twitter/Google Trends).

    Returns markets ranked by social trend relevance.
    """
    import logging
    logger = logging.getLogger(__name__)
    try:
        data = fetch_polymarket_markets(limit=limit)
        markets = data.get("markets") if isinstance(data, dict) else data
        if not markets:
            logger.warning("scan_social_trending: no markets found")
            return []

        normalized = []
        for m in markets:
            norm = normalize_polymarket_market(m)
            if norm:
                normalized.append(norm)
        logger.info(f"scan_social_trending: normalized {len(normalized)} markets")

        # Apply freshness filter
        normalized = filter_markets_by_freshness(
            normalized,
            min_days_until_close=1.0,
            max_days_until_close=90.0,
            min_volume=100.0,
        )
        logger.info(f"scan_social_trending: {len(normalized)} after freshness filter")

        # Match to social trends
        matched = match_trends_to_markets(normalized)

        # Get markets sorted by social relevance
        results = get_markets_by_social_relevance(matched, min_boost=5.0)
        logger.info(f"scan_social_trending: {len(results)} markets with trend matches")

        return results[:top_n]
    except Exception as e:
        logger.error(f"scan_social_trending error: {e}", exc_info=True)
        return []


import time

# Cache for scored markets
_SCORED_CACHE: Dict = {"results": [], "timestamp": 0}
_CACHE_TTL = 30  # 30 seconds


def scan_best_opportunities(limit: int = 100, top_n: int = 5, min_confidence: int = 40, use_cache: bool = True) -> List[Dict]:
    """
    Find best opportunities using unified scoring across all signals.

    Combines: EV, social trends, volume into one confidence score.
    Returns markets sorted by confidence with clear BUY/SELL/WATCH actions.

    Uses fast_mode by default for quick response (no external API calls).
    """
    import logging
    logger = logging.getLogger(__name__)

    # Check cache first
    now = time.time()
    if use_cache and _SCORED_CACHE["results"] and now - _SCORED_CACHE["timestamp"] < _CACHE_TTL:
        cached = _SCORED_CACHE["results"]
        filtered = [m for m in cached if m.get("confidence", 0) >= min_confidence]
        logger.info(f"scan_best_opportunities: returning {len(filtered[:top_n])} from cache")
        return filtered[:top_n]

    try:
        data = fetch_polymarket_markets(limit=limit)
        markets = data.get("markets") if isinstance(data, dict) else data
        if not markets:
            logger.warning("scan_best_opportunities: no markets found")
            return []

        normalized = []
        for m in markets:
            norm = normalize_polymarket_market(m)
            if norm:
                normalized.append(norm)
        logger.info(f"scan_best_opportunities: normalized {len(normalized)} markets")

        # Record prices for history tracking
        record_prices(normalized)

        # Apply freshness filter
        normalized = filter_markets_by_freshness(
            normalized,
            min_days_until_close=1.0,
            max_days_until_close=90.0,
            min_volume=100.0,
        )
        logger.info(f"scan_best_opportunities: {len(normalized)} after freshness filter")

        # Score all markets using FAST mode (no external API calls)
        scored = score_markets(normalized, min_confidence=min_confidence, fast_mode=True)
        logger.info(f"scan_best_opportunities: {len(scored)} markets above {min_confidence} confidence")

        # Enrich with price history
        for m in scored:
            market_id = m.get("market_id", "")
            if market_id:
                price_data = get_price_change(market_id)
                m["change_24h"] = price_data.get("change_24h")
                m["change_7d"] = price_data.get("change_7d")

        # Add variety: shuffle within confidence tiers and ensure category diversity
        import random
        random.seed(int(now / 60))  # Changes every minute for variety

        # Group by confidence tier
        high_conf = [m for m in scored if m.get("confidence", 0) >= 60]
        med_conf = [m for m in scored if 40 <= m.get("confidence", 0) < 60]
        low_conf = [m for m in scored if m.get("confidence", 0) < 40]

        # Shuffle within tiers
        random.shuffle(high_conf)
        random.shuffle(med_conf)
        random.shuffle(low_conf)

        # Recombine - high first, then medium, then low
        scored = high_conf + med_conf + low_conf

        # Update cache
        _SCORED_CACHE["results"] = scored
        _SCORED_CACHE["timestamp"] = now

        return scored[:top_n]
    except Exception as e:
        logger.error(f"scan_best_opportunities error: {e}", exc_info=True)
        return []


def scan_trending_news(limit: int = 100, top_n: int = 10) -> List[Dict]:
    """
    Find markets with active news signals using FULL scoring (with news API).

    This is slower than scan_best_opportunities but finds markets with
    actual trending news that may not be priced in yet.
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        data = fetch_polymarket_markets(limit=limit)
        markets = data.get("markets") if isinstance(data, dict) else data
        if not markets:
            logger.warning("scan_trending_news: no markets found")
            return []

        normalized = []
        for m in markets:
            norm = normalize_polymarket_market(m)
            if norm:
                normalized.append(norm)
        logger.info(f"scan_trending_news: normalized {len(normalized)} markets")

        # Apply freshness filter - focus on active markets
        normalized = filter_markets_by_freshness(
            normalized,
            min_days_until_close=1.0,
            max_days_until_close=60.0,  # Tighter window for news relevance
            min_volume=1000.0,  # Higher volume threshold
        )
        logger.info(f"scan_trending_news: {len(normalized)} after freshness filter")

        # Score using FULL mode (with news/sentiment API calls)
        scored = score_markets(normalized, min_confidence=30, fast_mode=False)

        # Filter to only markets with actual news signals
        with_news = [
            m for m in scored
            if m.get("news_mentions", 0) >= 1 or m.get("social_boost", 0) > 10
        ]

        # Sort by news mentions + social boost
        with_news.sort(
            key=lambda x: x.get("news_mentions", 0) * 10 + x.get("social_boost", 0),
            reverse=True
        )

        logger.info(f"scan_trending_news: {len(with_news)} markets with news signals")
        return with_news[:top_n]

    except Exception as e:
        logger.error(f"scan_trending_news error: {e}", exc_info=True)
        return []


def scan_market_details(query: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Get detailed analysis for a specific market by keyword search.

    Returns comprehensive market analysis or error message.
    """
    import logging
    logger = logging.getLogger(__name__)
    try:
        # Search for markets matching the query
        results = search_markets(query, limit=1)
        if not results:
            return None, f"No markets found matching '{query}'"

        # Get full details for the top match
        market = results[0]
        details = get_market_details(market)

        return details, None
    except Exception as e:
        logger.error(f"scan_market_details error: {e}", exc_info=True)
        return None, str(e)


def scan_polymarket_clusters(limit: int = 200, top_k_clusters: int = 5) -> List[Dict]:
    """Return top clusters of Polymarket markets by semantic similarity."""
    import logging
    logger = logging.getLogger(__name__)
    try:
        data = fetch_polymarket_markets(limit=limit)
        logger.info(f"scan_polymarket_clusters: fetched data type={type(data)}")
        markets = data.get("markets") if isinstance(data, dict) else None
        if markets is None and isinstance(data, list):
            markets = data
        if not markets:
            logger.warning("scan_polymarket_clusters: no markets found")
            return []
        normalized = []
        for m in markets:
            norm = normalize_polymarket_market(m)
            if norm:
                normalized.append(norm)
        logger.info(f"scan_polymarket_clusters: normalized {len(normalized)} markets")
        clusters = cluster_markets(normalized, max_markets=limit)
        logger.info(f"scan_polymarket_clusters: got {len(clusters)} clusters")
        return clusters[:top_k_clusters]
    except Exception as e:
        logger.error(f"scan_polymarket_clusters error: {e}", exc_info=True)
        return []


def scan_cross_market_mismatches(limit: int = 200, top_n: int = 5) -> Tuple[List[Dict], Optional[str]]:
    """Find cross-market mismatches between Polymarket and Kalshi."""
    try:
        poly_data = fetch_polymarket_markets(limit=limit)
        poly_markets_raw = poly_data.get("markets") if isinstance(poly_data, dict) else poly_data
        if not poly_markets_raw:
            return [], "No Polymarket markets fetched."
        poly_norm = []
        for m in poly_markets_raw:
            nm = normalize_polymarket_market(m)
            if nm:
                poly_norm.append(nm)
        kal_markets_raw = fetch_kalshi_markets(limit=limit)
        if not kal_markets_raw:
            return [], "Kalshi markets unavailable (missing creds or request failed)."
        kal_norm = []
        for m in kal_markets_raw:
            nm = normalize_kalshi_market(m)
            if nm:
                kal_norm.append(nm)
        if not poly_norm or not kal_norm:
            return [], "Insufficient markets to compare."
        matches = match_markets_by_embedding(poly_norm, kal_norm, top_k=3, sim_threshold=0.72)
        if not matches:
            return [], "No matched markets found."
        mismatches = detect_mismatches(matches)
        return mismatches[:top_n], None
    except Exception as exc:
        return [], str(exc)
