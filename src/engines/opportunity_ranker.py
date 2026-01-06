"""MVP opportunity ranking for Polymarket markets.

Uses simple heuristics combining price-implied probability, news signal, sentiment,
and social trend signals (Twitter/Google Trends).
"""
from __future__ import annotations

import math
import os
from typing import Dict, List

from data.news_gdelt import get_news_signal
from ml.sentiment import score_texts
from ml.feature_builder import build_features
from ml.model_inference import predict_proba
from src.engines.social_matcher import match_trends_to_markets


def _clip(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def rank_polymarket_opportunities(markets: List[Dict]) -> List[Dict]:
    """Rank normalized Polymarket markets with a lightweight EV heuristic.

    Each input market should come from normalize_polymarket_market and include:
      - question, market_id, yes_price, no_price, volume
    Returns a list sorted by opportunity_score descending.

    Includes social trend signals from Twitter/Google Trends for boosting
    markets that match current trending topics.
    """
    out: List[Dict] = []
    if not markets:
        return out

    # Enrich markets with social trend matches
    try:
        markets = match_trends_to_markets(markets)
    except Exception:
        # Graceful fallback if social matching fails
        for m in markets:
            m["trend_matches"] = []
            m["social_boost"] = 0.0

    for m in markets:
        try:
            q = m.get("question", "") or ""
            yes = float(m.get("yes_price", 0.0) or 0.0)
            no = float(m.get("no_price", 0.0) or 0.0)
            vol = float(m.get("volume", 0.0) or 0.0)

            news = get_news_signal(q)
            news_mentions = int(news.get("mentions_24h", 0) or 0)
            headlines = news.get("headlines") or []
            sentiment = score_texts([q] + headlines)

            implied_prob_yes = yes
            p_win = implied_prob_yes + 0.07 * sentiment + 0.02 * math.log1p(news_mentions)
            p_win = _clip(p_win, 0.01, 0.99)

            payout_mult = 1.0 / yes if yes > 0 else 0.0
            stake = 1.0
            ev = p_win * (payout_mult - 1.0) - (1.0 - p_win) * stake

            # Get social boost from trend matching
            social_boost = float(m.get("social_boost", 0.0) or 0.0)
            trend_matches = m.get("trend_matches", [])

            opp_score = (
                ev * 50.0  # scale EV
                + math.log1p(vol)
                + math.log1p(news_mentions + 1)
                + abs(sentiment) * 10.0
                + social_boost * 0.5  # Social trend boost
            )

            # optional supervised model
            model_score = 0.0
            if os.getenv("USE_SUPERVISED_MODEL", "false").lower() == "true":
                feats, order = build_features({
                    "yes_price": yes,
                    "no_price": no,
                    "volume": vol,
                    "liquidity": liq,
                    "news_mentions": news_mentions,
                    "sentiment": sentiment,
                    "trend_score": 0.0,
                    "edge_pct": (ev * 100) if ev else 0.0,
                    "time_to_expiry": 0.0,
                })
                model_score = predict_proba(feats, order) * 100.0
                opp_score = 0.6 * opp_score + 0.4 * model_score

            out.append({
                "question": q,
                "market_id": m.get("market_id"),
                "yes_price": yes,
                "no_price": no,
                "volume": vol,
                "news_mentions": news_mentions,
                "sentiment": sentiment,
                "p_win": p_win,
                "ev": ev,
                "opportunity_score": opp_score,
                "model_score": model_score,
                "social_boost": social_boost,
                "trend_matches": trend_matches,
                "opportunity_id": "",
            })
        except Exception:
            continue

    out.sort(key=lambda x: x.get("opportunity_score", 0), reverse=True)
    return out
