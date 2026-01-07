"""Unified scoring engine for market opportunities.

Combines all signals (EV, social, news, sentiment, volume) into a single
confidence score with clear BUY/SELL/WATCH recommendations.
"""
from __future__ import annotations

import math
import logging
from typing import Dict, List, Tuple

from src.data.news_gdelt import get_news_signal
from src.ml.sentiment import score_texts
from src.engines.social_matcher import match_trends_to_markets
from src.engines.market_filter import filter_markets_by_freshness

logger = logging.getLogger(__name__)


def _clip(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def compute_ev_score(yes_price: float, no_price: float, sentiment: float, news_mentions: int) -> Tuple[float, float]:
    """Compute expected value and EV-based score component.

    Returns:
        (ev, ev_score) where ev_score is 0-40 points
    """
    if yes_price <= 0:
        return 0.0, 0.0

    # Adjust probability based on signals
    implied_prob = yes_price
    adjusted_prob = implied_prob + 0.07 * sentiment + 0.02 * math.log1p(news_mentions)
    adjusted_prob = _clip(adjusted_prob, 0.01, 0.99)

    # Calculate EV
    payout_mult = 1.0 / yes_price
    ev = adjusted_prob * (payout_mult - 1.0) - (1.0 - adjusted_prob)

    # Convert EV to score (0-40 points)
    # EV of 0.2+ is excellent, 0.1 is good, 0 is neutral
    ev_score = _clip((ev + 0.1) * 100, 0, 40)

    return ev, ev_score


def compute_volume_score(volume: float, liquidity: float = 0) -> Tuple[float, str]:
    """Compute volume/liquidity score component.

    Returns:
        (score, description) where score is 0-15 points
    """
    # Volume scoring
    if volume >= 1_000_000:
        vol_score = 15.0
        desc = "Very high volume"
    elif volume >= 500_000:
        vol_score = 12.0
        desc = "High volume"
    elif volume >= 100_000:
        vol_score = 8.0
        desc = "Good volume"
    elif volume >= 10_000:
        vol_score = 5.0
        desc = "Moderate volume"
    else:
        vol_score = 2.0
        desc = "Low volume"

    return vol_score, desc


def compute_news_score(mentions: int, headlines: List[str]) -> Tuple[float, str]:
    """Compute news signal score component.

    Returns:
        (score, description) where score is 0-15 points
    """
    if mentions >= 20:
        return 15.0, f"News spike ({mentions} mentions)"
    elif mentions >= 10:
        return 12.0, f"High news activity ({mentions})"
    elif mentions >= 5:
        return 8.0, f"News coverage ({mentions})"
    elif mentions >= 1:
        return 4.0, "Some news"
    else:
        return 0.0, ""


def compute_sentiment_score(sentiment: float) -> Tuple[float, str]:
    """Compute sentiment score component.

    Returns:
        (score, description) where score is 0-10 points
    """
    abs_sent = abs(sentiment)
    if abs_sent >= 0.5:
        score = 10.0
        desc = "Strong sentiment" if sentiment > 0 else "Strong negative sentiment"
    elif abs_sent >= 0.3:
        score = 7.0
        desc = "Positive sentiment" if sentiment > 0 else "Negative sentiment"
    elif abs_sent >= 0.1:
        score = 4.0
        desc = "Slight sentiment"
    else:
        score = 0.0
        desc = ""

    return score, desc


def compute_social_score(social_boost: float, trend_matches: List[Dict]) -> Tuple[float, str]:
    """Compute social trend score component.

    Returns:
        (score, description) where score is 0-20 points
    """
    if social_boost >= 70:
        score = 20.0
    elif social_boost >= 50:
        score = 15.0
    elif social_boost >= 30:
        score = 10.0
    elif social_boost >= 10:
        score = 5.0
    else:
        score = 0.0

    if trend_matches:
        topics = [m.get("topic", "") for m in trend_matches[:2]]
        sources = set(m.get("source", "") for m in trend_matches)
        source_str = "/".join(sources) if sources else "social"
        desc = f"Trending on {source_str}: {', '.join(topics)}"
    else:
        desc = ""

    return score, desc


def determine_action(yes_price: float, no_price: float, ev: float, confidence: float,
                     sentiment: float = 0.0, social_boost: float = 0.0, news_mentions: int = 0) -> Tuple[str, str]:
    """Determine recommended action based on signals, not just price.

    Returns:
        (action, action_icon)

    Logic: Recommend BUY when we have signals (sentiment, social, news) AND the price
    is in a reasonable range (not extreme like $0.01). Extreme prices need stronger signals.
    """
    # Calculate signal strength from available data
    signal_strength = abs(sentiment) * 30 + social_boost + min(news_mentions * 5, 25)
    has_signals = sentiment > 0.15 or social_boost > 10 or news_mentions >= 2
    has_strong_signals = sentiment > 0.3 or social_boost > 25 or news_mentions >= 5

    # Price ranges - be more cautious with extreme prices
    is_extreme_yes = yes_price < 0.10  # Very unlikely according to market
    is_extreme_no = no_price < 0.10
    is_tradeable_yes = 0.10 <= yes_price <= 0.45  # Good value range for YES
    is_tradeable_no = 0.10 <= no_price <= 0.45   # Good value range for NO

    # Strong signals with actual backing - works even at extreme prices
    if confidence >= 70 and has_strong_signals:
        if sentiment > 0 and is_tradeable_yes:
            return "STRONG BUY YES", "[++]"
        elif sentiment < 0 and is_tradeable_no:
            return "STRONG BUY NO", "[--]"
        elif social_boost > 25 and is_tradeable_yes:
            return "STRONG BUY YES", "[++]"
        elif social_boost > 25 and is_tradeable_no:
            return "STRONG BUY NO", "[--]"

    # Moderate signals - good opportunities in tradeable range
    if confidence >= 50 and has_signals:
        if is_tradeable_yes and not is_extreme_yes:
            return "BUY YES", "[+]"
        elif is_tradeable_no and not is_extreme_no:
            return "BUY NO", "[-]"
        # Even extreme prices can be opportunities with strong enough signals
        elif is_extreme_yes and signal_strength >= 30:
            return "BUY YES", "[+]"
        elif is_extreme_no and signal_strength >= 30:
            return "BUY NO", "[-]"

    # Trending - market has social buzz, worth monitoring
    if social_boost > 5 or news_mentions >= 1:
        return "TRENDING", "[^]"

    return "WATCH", "[~]"


def compute_unified_score(market: Dict) -> Dict:
    """Compute unified confidence score combining all signals.

    Args:
        market: Normalized market dict with prices, volume, etc.

    Returns:
        Dict with:
        - confidence: float (0-100)
        - action: str (BUY YES/BUY NO/SELL/WATCH)
        - action_icon: str
        - signals: list of signal descriptions
        - recommendation: str
        - ev: float
        - All original market fields
    """
    question = market.get("question", "") or ""
    yes_price = float(market.get("yes_price", 0.0) or 0.0)
    no_price = float(market.get("no_price", 0.0) or 0.0)
    volume = float(market.get("volume", 0.0) or 0.0)
    liquidity = float(market.get("liquidity", 0.0) or 0.0)

    # Get existing signals if already computed, otherwise compute fresh
    social_boost = float(market.get("social_boost", 0.0) or 0.0)
    trend_matches = market.get("trend_matches", [])

    # Get news signal
    news = get_news_signal(question)
    news_mentions = int(news.get("mentions_24h", 0) or 0)
    headlines = news.get("headlines", [])

    # Get sentiment
    sentiment = score_texts([question] + headlines)

    # Compute component scores
    ev, ev_score = compute_ev_score(yes_price, no_price, sentiment, news_mentions)
    vol_score, vol_desc = compute_volume_score(volume, liquidity)
    news_score, news_desc = compute_news_score(news_mentions, headlines)
    sent_score, sent_desc = compute_sentiment_score(sentiment)
    social_score, social_desc = compute_social_score(social_boost, trend_matches)

    # Total confidence score (0-100)
    confidence = ev_score + vol_score + news_score + sent_score + social_score
    confidence = _clip(confidence, 0, 100)

    # Build signals list
    signals = []
    if ev > 0.05:
        signals.append({"name": "Positive EV", "value": ev, "positive": True})
    if social_desc:
        signals.append({"name": social_desc, "value": social_boost, "positive": True})
    if news_desc:
        signals.append({"name": news_desc, "value": news_mentions, "positive": news_mentions > 0})
    if vol_desc and volume >= 100_000:
        signals.append({"name": vol_desc, "value": volume, "positive": True})
    if sent_desc:
        signals.append({"name": sent_desc, "value": sentiment, "positive": sentiment > 0})

    # Determine action - pass actual signals so recommendation is based on real data
    action, action_icon = determine_action(
        yes_price, no_price, ev, confidence,
        sentiment=sentiment, social_boost=social_boost, news_mentions=news_mentions
    )

    # Build recommendation text
    if action.startswith("STRONG"):
        rec = f"Strong signals support this trade. {action} recommended."
    elif action.startswith("BUY"):
        rec = f"Good signals detected. Consider {action}."
    elif action == "MONITOR":
        rec = "Interesting activity. Monitor for clearer entry."
    else:
        rec = "No strong signals. Watch for developments."

    return {
        **market,  # Include all original fields
        "confidence": confidence,
        "action": action,
        "action_icon": action_icon,
        "signals": signals,
        "recommendation": rec,
        "ev": ev,
        "news_mentions": news_mentions,
        "sentiment": sentiment,
        "headlines": headlines,
    }


def compute_fast_score(market: Dict) -> Dict:
    """Fast scoring without external API calls.

    Uses only local data: prices, volume, social boost (if already computed).
    Much faster than compute_unified_score.

    NOTE: Without external signals (news, sentiment), we can only score based on
    volume and social activity. We do NOT give bonus points for extreme prices
    since those often reflect legitimate market consensus.
    """
    yes_price = float(market.get("yes_price", 0.0) or 0.0)
    no_price = float(market.get("no_price", 0.0) or 0.0)
    volume = float(market.get("volume", 0.0) or 0.0)
    social_boost = float(market.get("social_boost", 0.0) or 0.0)

    # Without external signals, we can't reliably estimate EV
    # Just set to 0 - we need actual signals to claim positive EV
    ev = 0.0
    ev_score = 0.0

    # Volume score (0-30) - high volume means active/liquid market
    if volume >= 1_000_000:
        vol_score = 30.0
    elif volume >= 500_000:
        vol_score = 25.0
    elif volume >= 100_000:
        vol_score = 20.0
    elif volume >= 50_000:
        vol_score = 15.0
    elif volume >= 10_000:
        vol_score = 10.0
    else:
        vol_score = 5.0

    # Social score (0-40) - this is our main signal in fast mode
    social_score = min(social_boost * 1.5, 40.0)

    # Liquidity bonus - prefer markets with prices in tradeable range (0-20)
    # Markets at extreme prices (0.01 or 0.99) are hard to profit from
    if 0.15 <= yes_price <= 0.85:
        liquidity_score = 20.0  # Good trading range
    elif 0.05 <= yes_price <= 0.95:
        liquidity_score = 10.0  # Acceptable range
    else:
        liquidity_score = 0.0  # Extreme prices - likely resolved or very certain

    confidence = vol_score + social_score + liquidity_score
    confidence = _clip(confidence, 0, 100)

    # Pass social_boost to determine_action - it's our only signal in fast mode
    action, action_icon = determine_action(
        yes_price, no_price, ev, confidence,
        sentiment=0.0, social_boost=social_boost, news_mentions=0
    )

    return {
        **market,
        "confidence": confidence,
        "action": action,
        "action_icon": action_icon,
        "signals": [],
        "recommendation": "",
        "ev": ev,
    }


def score_markets(markets: List[Dict], min_confidence: int = 40, fast_mode: bool = True) -> List[Dict]:
    """Score multiple markets and filter by confidence.

    Args:
        markets: List of normalized market dicts
        min_confidence: Minimum confidence to include
        fast_mode: If True, use fast scoring (no external API calls)

    Returns:
        List of scored markets, sorted by confidence descending
    """
    # First enrich with social trends (cached, relatively fast)
    try:
        markets = match_trends_to_markets(markets)
    except Exception:
        for m in markets:
            m["trend_matches"] = []
            m["social_boost"] = 0.0

    scored = []
    score_func = compute_fast_score if fast_mode else compute_unified_score

    for m in markets:
        try:
            result = score_func(m)
            if result["confidence"] >= min_confidence:
                scored.append(result)
        except Exception as e:
            logger.debug(f"Failed to score market: {e}")
            continue

    # Sort by confidence
    scored.sort(key=lambda x: x.get("confidence", 0), reverse=True)

    return scored


def format_signals_text(signals: List[Dict], max_signals: int = 3) -> str:
    """Format signals list as human-readable text."""
    if not signals:
        return ""

    parts = []
    for s in signals[:max_signals]:
        name = s.get("name", "")
        if name:
            parts.append(name)

    return ", ".join(parts) if parts else ""
