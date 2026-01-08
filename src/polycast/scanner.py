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
from src.engines.unified_scorer import score_markets, format_signals_text, compute_unified_score
from src.engines.market_search import search_markets, get_market_details
from src.data.price_history import record_prices, get_price_change
from src.data.news_gdelt import get_news_signal
from src.ml.sentiment import score_texts
from src.data.twitter_scraper import search_tweets_for_topic


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


def scan_best_opportunities(
    limit: int = 600,
    top_n: int = 5,
    min_confidence: int = 30,
    use_cache: bool = True,
    enrich_top_n: int = 15
) -> List[Dict]:
    """
    Discover NEW opportunities from the full Polymarket catalog.

    Fetches 600+ markets and finds hidden gems across ALL categories.
    Prioritizes variety and novelty over just high-volume markets.
    """
    import logging
    import random
    from collections import defaultdict
    logger = logging.getLogger(__name__)

    now = time.time()

    # Short cache to allow variety on refresh
    if use_cache and _SCORED_CACHE["results"] and now - _SCORED_CACHE["timestamp"] < _CACHE_TTL:
        cached = _SCORED_CACHE["results"]
        # Return DIFFERENT slice each time from cache
        start_idx = int(now / 10) % max(1, len(cached) - top_n)
        result = cached[start_idx:start_idx + top_n]
        logger.info(f"scan_best_opportunities: returning slice [{start_idx}:{start_idx+top_n}] from cache")
        return result

    try:
        # 1. Fetch "main-trending" markets first (high priority)
        trending_data = fetch_polymarket_markets(limit=100, tag_id="964")
        trending_raw = trending_data if isinstance(trending_data, list) else trending_data.get("markets", [])
        logger.info(f"scan_best_opportunities: fetched {len(trending_raw)} trending markets")

        # 2. Fetch LOTS of other markets for discovery
        data = fetch_polymarket_markets(limit=limit, use_pagination=True)
        markets_raw = data if isinstance(data, list) else data.get("markets", [])
        
        # Merge, prioritizing trending
        seen_ids = set()
        merged_raw = []
        for m in (trending_raw + markets_raw):
            mid = m.get("id") or m.get("_id")
            if mid and mid not in seen_ids:
                seen_ids.add(mid)
                merged_raw.append(m)

        if not merged_raw:
            logger.warning("scan_best_opportunities: no markets found")
            return []

        normalized = []
        for m in merged_raw:
            norm = normalize_polymarket_market(m)
            if norm:
                normalized.append(norm)
        logger.info(f"scan_best_opportunities: normalized {len(normalized)} markets")

        # Record prices for history tracking
        record_prices(normalized)

        # RELAXED filter - include more markets for discovery
        normalized = filter_markets_by_freshness(
            normalized,
            min_days_until_close=0.5,
            max_days_until_close=365.0,  # Include long-term markets
            min_volume=50.0,  # Include smaller markets
        )
        logger.info(f"scan_best_opportunities: {len(normalized)} after freshness filter")

        # 3. FAST STAGE: Score all markets without deep API calls
        scored_fast = score_markets(normalized, min_confidence=min_confidence, fast_mode=True)
        logger.info(f"scan_best_opportunities: {len(scored_fast)} markets above fast {min_confidence} confidence")

        # 4. DEEP STAGE: Enrich top N candidates with actual news and social signals
        top_candidates = scored_fast[:enrich_top_n]
        fully_scored = []
        
        for m in top_candidates:
            try:
                # Deep Enrichment: News headlines
                news = get_news_signal(m.get("question", ""))
                headlines = news.get("headlines", [])
                
                # Deep Enrichment: Social sentiment for top trend matches
                sentiment = None
                social_texts = []
                for tm in m.get("trend_matches", []):
                    tweets = search_tweets_for_topic(tm.get("topic", ""), limit=5)
                    social_texts.extend([tw.get("text", "") for tw in tweets])
                
                if social_texts:
                    sentiment = score_texts(social_texts)
                
                # Re-score with deep signals
                deep_result = compute_unified_score(m, sentiment=sentiment, headlines=headlines)
                fully_scored.append(deep_result)
            except Exception as e:
                logger.error(f"Deep enrichment failed for market {m.get('market_id')}: {e}")
                fully_scored.append(m) # Fallback to fast-scored version

        # Combined results: Enriched + remaining fast-scored
        remaining = scored_fast[len(top_candidates):]
        final_pool = fully_scored + remaining
        
        # Enrich with category and history
        for m in final_pool:
            market_id = m.get("market_id", "")
            if market_id:
                price_data = get_price_change(market_id)
                m["change_24h"] = price_data.get("change_24h")
                m["change_7d"] = price_data.get("change_7d")
            m["detected_category"] = _detect_category(m.get("question", ""))

        # TRUE RANDOMIZATION for variety (changes every 10 seconds)
        random.seed(int(now / 10))

        # Group by category
        by_category = defaultdict(list)
        for m in final_pool:
            by_category[m["detected_category"]].append(m)

        # Shuffle within each category for variety
        for cat in by_category:
            random.shuffle(by_category[cat])

        # Build diverse results - round robin from ALL categories
        results = []
        priority_order = ["world", "politics", "crypto", "tech", "economy", "climate", "entertainment", "other", "sports"]
        categories = sorted(by_category.keys(), key=lambda c: priority_order.index(c) if c in priority_order else 99)

        # Round-robin pick ensuring max 2 per category in top results
        cat_counts = defaultdict(int)
        
        # Multiple passes to fill results, ensuring priority to fully scored (enriched) ones
        # First, try to pick enriched markets in round-robin
        enriched_by_cat = defaultdict(list)
        for m in fully_scored:
            enriched_by_cat[m["detected_category"]].append(m)
            
        for pass_num in range(3):
            for cat in categories:
                if len(results) >= top_n:
                    break
                cat_enriched = enriched_by_cat[cat]
                if pass_num < len(cat_enriched):
                    m = cat_enriched[pass_num]
                    if m not in results:
                        results.append(m)

        # Fill remaining with any markets
        for pass_num in range(5):
            for cat in categories:
                if len(results) >= top_n * 4:
                    break
                cat_markets = by_category[cat]
                if pass_num < len(cat_markets):
                    m = cat_markets[pass_num]
                    if m not in results:
                        results.append(m)

        # Final shuffle of results
        random.shuffle(results)

        # Update cache with ALL good results (for variety on next call)
        _SCORED_CACHE["results"] = results
        _SCORED_CACHE["timestamp"] = now

        logger.info(f"scan_best_opportunities: returning {top_n} from {len(results)} diverse markets")
        return results[:top_n]

    except Exception as e:
        logger.error(f"scan_best_opportunities error: {e}", exc_info=True)
        return []


def _detect_category(question: str) -> str:
    """Detect market category from question text."""
    q = question.lower()

    # Sports keywords
    if any(w in q for w in ["super bowl", "nfl", "nba", "mlb", "championship", "playoff",
                            "world series", "stanley cup", "player of the year", "mvp",
                            "premier league", "champions league", "world cup", "f1", "ufc"]):
        return "sports"

    # Politics keywords
    if any(w in q for w in ["trump", "biden", "president", "election", "congress", "senate",
                            "governor", "democrat", "republican", "vote", "poll", "primary",
                            "parliament", "minister", "macron", "putin", "political"]):
        return "politics"

    # Crypto keywords
    if any(w in q for w in ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "xrp",
                            "airdrop", "token", "blockchain", "defi", "nft"]):
        return "crypto"

    # Economy keywords
    if any(w in q for w in ["fed", "inflation", "gdp", "recession", "interest rate", "deficit",
                            "revenue", "budget", "tariff", "unemployment", "stock market"]):
        return "economy"

    # Tech keywords
    if any(w in q for w in ["openai", "chatgpt", "gpt", "ai ", "apple", "google", "microsoft",
                            "nvidia", "tesla", "spacex", "amazon", "meta", "tiktok"]):
        return "tech"

    # World/Geopolitics
    if any(w in q for w in ["russia", "ukraine", "china", "israel", "gaza", "iran", "war",
                            "ceasefire", "nato", "sanctions", "military", "taiwan"]):
        return "world"

    # Climate
    if any(w in q for w in ["climate", "temperature", "hottest", "warming", "hurricane",
                            "wildfire", "emissions", "carbon"]):
        return "climate"

    # Entertainment
    if any(w in q for w in ["oscar", "grammy", "emmy", "movie", "album", "taylor swift",
                            "netflix", "disney", "spotify", "box office"]):
        return "entertainment"

    return "other"


def scan_trending_news(limit: int = 600, top_n: int = 10) -> List[Dict]:
    """
    Find hot/trending markets with CATEGORY DIVERSITY.

    Uses actual 24h volume and price movement data from Polymarket.
    Ensures results include different categories (sports, politics, crypto, etc.)
    Fetches with pagination to get variety beyond just sports markets.
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        # Use pagination to get more variety (not just recent sports)
        data = fetch_polymarket_markets(limit=limit, use_pagination=True)
        markets = data if isinstance(data, list) else data.get("markets", data)
        if not markets:
            logger.warning("scan_trending_news: no markets found")
            return []

        normalized = []
        for m in markets:
            norm = normalize_polymarket_market(m)
            if norm:
                normalized.append(norm)
        logger.info(f"scan_trending_news: normalized {len(normalized)} markets")

        # Apply RELAXED freshness filter for maximum variety
        normalized = filter_markets_by_freshness(
            normalized,
            min_days_until_close=0.5,    # Allow markets closing soon
            max_days_until_close=365.0,  # Allow longer-term markets
            min_volume=100.0,            # Low threshold for variety
        )
        logger.info(f"scan_trending_news: {len(normalized)} after freshness filter")

        # Calculate trending score and detect category
        for m in normalized:
            vol_24h = m.get("volume_24h", 0) or 0
            price_chg = abs(m.get("price_change_24h", 0) or 0)

            # Trending score: high volume + price movement = hot market
            trending_score = (vol_24h / 1000) * (price_chg * 100) + (vol_24h / 10000)
            m["trending_score"] = trending_score

            # Detect category
            m["detected_category"] = _detect_category(m.get("question", ""))

            # Determine trend direction
            raw_chg = m.get("price_change_24h", 0) or 0
            if raw_chg > 0.02:
                m["trend_direction"] = "📈"
            elif raw_chg < -0.02:
                m["trend_direction"] = "📉"
            else:
                m["trend_direction"] = "➡️"

        # Group ALL markets by category (not just those with 24h volume)
        from collections import defaultdict
        by_category = defaultdict(list)
        for m in normalized:
            by_category[m["detected_category"]].append(m)

        # Sort each category - prefer 24h activity, fallback to total volume
        for cat in by_category:
            by_category[cat].sort(
                key=lambda x: (x.get("trending_score", 0), x.get("volume", 0)),
                reverse=True
            )

        logger.info(f"Categories found: {list(by_category.keys())} with counts: {[(c, len(by_category[c])) for c in by_category]}")

        # Round-robin pick - ONE from each category first, then fill with highest scores
        results = []

        # Prioritize non-sports categories first for diversity
        priority_order = ["world", "politics", "crypto", "tech", "economy", "climate", "entertainment", "other", "sports"]
        categories = sorted(by_category.keys(), key=lambda c: priority_order.index(c) if c in priority_order else 99)

        # Phase 1: Get top 1 from each category (ensures diversity)
        for cat in categories:
            if by_category[cat]:
                results.append(by_category[cat][0])

        # Phase 2: Get second best from non-sports categories
        for cat in categories:
            if cat != "sports" and len(by_category[cat]) > 1:
                results.append(by_category[cat][1])

        # Phase 3: Fill remaining with highest trending scores (any category)
        all_remaining = []
        for cat in categories:
            start_idx = 2 if cat != "sports" else 1
            all_remaining.extend(by_category[cat][start_idx:])
        all_remaining.sort(key=lambda x: x.get("trending_score", 0), reverse=True)

        for m in all_remaining:
            if len(results) >= top_n * 2:
                break
            if m not in results:
                results.append(m)

        if not results:
            # Fallback: just return highest volume markets
            normalized.sort(key=lambda x: x.get("volume_24h", 0), reverse=True)
            results = normalized[:top_n]

        # Add confidence and action using fast scoring
        scored = score_markets(results, min_confidence=0, fast_mode=True)

        logger.info(f"scan_trending_news: {len(scored)} hot markets from {len(by_category)} categories")
        return scored[:top_n]

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
