"""
Shared scanner module for arbitrage checking.

This module provides a unified interface for checking arbitrage opportunities
that can be used by both console and Telegram bot interfaces.
"""

from typing import Dict, Optional, Tuple, List
from exchanges.ccxt_client import get_ccxt_prices, DEFAULT_EXCHANGES
from polycast.analytics.arbitrage import check_arbitrage
from exchanges.polymarket import fetch_polymarket_markets, normalize_polymarket_market
from exchanges.kalshi import fetch_kalshi_markets, normalize_kalshi_market
from engines.opportunity_ranker import rank_polymarket_opportunities
from engines.trend_engine import get_trending_polymarket
from ml.market_clustering import cluster_markets
from engines.cross_market_matcher import match_markets_by_embedding
from engines.cross_arbitrage import detect_mismatches


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
    """
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
        ranked = rank_polymarket_opportunities(normalized)
        return ranked[:top_n]
    except Exception:
        return []


def scan_polymarket_trending(limit: int = 200, top_n: int = 5) -> List[Dict]:
    """
    Return top trending Polymarket markets based on anomaly scoring.
    """
    try:
        return get_trending_polymarket(limit=limit, top_n=top_n)
    except Exception:
        return []


def scan_polymarket_clusters(limit: int = 200, top_k_clusters: int = 5) -> List[Dict]:
    """Return top clusters of Polymarket markets by semantic similarity."""
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
        clusters = cluster_markets(normalized, max_markets=limit)
        return clusters[:top_k_clusters]
    except Exception:
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
