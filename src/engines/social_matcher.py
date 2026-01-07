"""Match social trending topics to Polymarket markets using embeddings.

Uses sentence-transformers embeddings for semantic similarity matching
to find markets related to current trending topics.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.ml.embeddings import embed_texts
from src.data.social_trends import get_trending_topics, topic_matches_text

logger = logging.getLogger(__name__)


def match_trends_to_markets(
    markets: List[Dict],
    min_similarity: float = 0.35,
    top_k_matches: int = 3,
    region: str = "united_states",
) -> List[Dict]:
    """Match trending topics to markets using semantic similarity.

    Enriches each market with trend matches and a social boost score.

    Args:
        markets: Normalized Polymarket markets with 'question' field
        min_similarity: Minimum cosine similarity threshold (0-1)
        top_k_matches: Max trend matches per market
        region: Region for trending topics

    Returns:
        Markets enriched with:
        - trend_matches: List of matching trends with similarity scores
        - social_boost: Combined trend signal (0-100)
    """
    if not markets:
        return []

    # Get current trending topics
    trends = get_trending_topics(region=region)

    if not trends:
        # No trends available, return markets with empty trend data (FAST PATH)
        for m in markets:
            m["trend_matches"] = []
            m["social_boost"] = 0.0
        logger.debug("match_trends_to_markets: no trends, skipping embeddings")
        return markets

    # First pass: quick keyword matching (FAST - no ML)
    has_any_match = False
    for m in markets:
        m["_keyword_matches"] = []
        question = m.get("question", "")
        for t in trends:
            topic = t.get("topic", "")
            if topic_matches_text(topic, question):
                m["_keyword_matches"].append(t)
                has_any_match = True

    # If no keyword matches found, skip expensive embeddings
    if not has_any_match:
        for m in markets:
            m["trend_matches"] = []
            m["social_boost"] = 0.0
        logger.debug("match_trends_to_markets: no keyword matches, skipping embeddings")
        return markets

    # Embed market questions (only if we have keyword matches)
    market_texts = [m.get("question", "") for m in markets]
    try:
        market_embeddings = embed_texts(market_texts)
    except Exception as exc:
        logger.error("Failed to embed market texts: %s", exc)
        for m in markets:
            m["trend_matches"] = m.pop("_keyword_matches", [])[:top_k_matches]
            m["social_boost"] = _compute_social_boost(m["trend_matches"])
        return markets

    # Embed trend topics
    trend_texts = [t.get("topic", "") for t in trends]
    try:
        trend_embeddings = embed_texts(trend_texts)
    except Exception as exc:
        logger.error("Failed to embed trend texts: %s", exc)
        for m in markets:
            m["trend_matches"] = m.pop("_keyword_matches", [])[:top_k_matches]
            m["social_boost"] = _compute_social_boost(m["trend_matches"])
        return markets

    # Compute similarity matrix
    sim_matrix = cosine_similarity(market_embeddings, trend_embeddings)

    # Match each market to top trends
    for i, m in enumerate(markets):
        sims = sim_matrix[i]
        matches = []

        # Get top-k matches above threshold
        top_indices = np.argsort(sims)[::-1][:top_k_matches * 2]  # Get extra for filtering

        for j in top_indices:
            if sims[j] >= min_similarity:
                matches.append({
                    "topic": trends[j].get("topic", ""),
                    "source": trends[j].get("source", ""),
                    "similarity": float(sims[j]),
                    "score": float(trends[j].get("score", 0.0)),
                })

            if len(matches) >= top_k_matches:
                break

        # Include keyword matches that weren't caught by embedding
        keyword_matches = m.pop("_keyword_matches", [])
        existing_topics = {match["topic"].lower() for match in matches}

        for km in keyword_matches:
            topic = km.get("topic", "")
            if topic.lower() not in existing_topics:
                matches.append({
                    "topic": topic,
                    "source": km.get("source", ""),
                    "similarity": 0.5,  # Assume moderate similarity for keyword match
                    "score": float(km.get("score", 50.0)),
                })

        # Sort by similarity * score
        matches.sort(key=lambda x: x["similarity"] * x["score"], reverse=True)
        m["trend_matches"] = matches[:top_k_matches]
        m["social_boost"] = _compute_social_boost(m["trend_matches"])

    # Log summary
    markets_with_matches = sum(1 for m in markets if m.get("trend_matches"))
    logger.info(
        "match_trends_to_markets: %d/%d markets matched to trends",
        markets_with_matches, len(markets)
    )

    return markets


def _compute_social_boost(matches: List[Dict]) -> float:
    """Compute combined social boost score from trend matches.

    Higher similarity + higher trend score = higher boost.

    Args:
        matches: List of trend match dicts with similarity and score

    Returns:
        Social boost score 0-100
    """
    if not matches:
        return 0.0

    # Weighted average: similarity * trend_score
    total = 0.0
    for m in matches:
        sim = float(m.get("similarity", 0))
        score = float(m.get("score", 0))
        # Weight by position (first match is most relevant)
        total += sim * score

    # Normalize: divide by number of matches, cap at 100
    boost = total / len(matches)
    return min(100.0, boost)


def get_markets_by_social_relevance(
    markets: List[Dict],
    min_boost: float = 10.0,
) -> List[Dict]:
    """Filter and sort markets by social trend relevance.

    Args:
        markets: Markets enriched with trend_matches and social_boost
        min_boost: Minimum social_boost to include

    Returns:
        Markets with trend matches, sorted by social_boost descending
    """
    # Filter to markets with trend matches
    with_matches = [
        m for m in markets
        if m.get("trend_matches") and m.get("social_boost", 0) >= min_boost
    ]

    # Sort by social boost
    with_matches.sort(key=lambda x: x.get("social_boost", 0), reverse=True)

    return with_matches
