"""Unified social trends interface.

Combines multiple trend sources (Google Trends, Twitter/X) with fallback.
Provides a single API for fetching trending topics.
"""
from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def get_trending_topics(region: str = "united_states", max_topics: int = 30) -> List[Dict]:
    """Get trending topics from all available sources.

    Priority: Google Trends (most reliable) > Twitter/X (fallback)
    Deduplicates topics and normalizes scores.

    Args:
        region: Region for trends (e.g., "united_states", "USA")
        max_topics: Maximum number of topics to return

    Returns:
        List of trending topics:
        [
            {
                "topic": str,
                "source": str,  # "google" or "twitter"
                "score": float, # Normalized 0-100
                "metadata": Dict
            }
        ]
    """
    all_trends = []
    seen_topics = set()

    # Try Google Trends first (more reliable, no auth)
    try:
        from src.data.google_trends import get_realtime_trends, get_daily_trends

        # Try realtime first
        google_trends = get_realtime_trends(region)

        # Fall back to daily if realtime is empty
        if not google_trends:
            google_trends = get_daily_trends(region)

        for t in google_trends:
            topic = t.get("topic", "").strip()
            if not topic:
                continue

            topic_lower = topic.lower()
            if topic_lower in seen_topics:
                continue
            seen_topics.add(topic_lower)

            all_trends.append({
                "topic": topic,
                "source": "google",
                "score": float(t.get("score", 50.0)),
                "metadata": t,
            })

        logger.debug("get_trending_topics: got %d from Google Trends", len(google_trends))

    except ImportError:
        logger.debug("Google Trends module not available")
    except Exception as exc:
        logger.debug("Google Trends unavailable: %s", exc)

    # Try Twitter as supplement (may have different topics)
    try:
        from src.data.twitter_scraper import get_twitter_trends

        # Map region names
        twitter_region = "USA" if "united" in region.lower() else region
        twitter_trends = get_twitter_trends(twitter_region)

        for t in twitter_trends:
            topic = t.get("topic", "").strip()
            if not topic:
                continue

            topic_lower = topic.lower()
            if topic_lower in seen_topics:
                continue
            seen_topics.add(topic_lower)

            all_trends.append({
                "topic": topic,
                "source": "twitter",
                "score": float(t.get("score", 40.0)),
                "metadata": t,
            })

        logger.debug("get_trending_topics: got %d from Twitter", len(twitter_trends))

    except ImportError:
        logger.debug("Twitter scraper module not available")
    except Exception as exc:
        logger.debug("Twitter trends unavailable: %s", exc)

    # Sort by score descending
    all_trends.sort(key=lambda x: x.get("score", 0), reverse=True)

    logger.info("get_trending_topics: returning %d total trends", min(len(all_trends), max_topics))

    return all_trends[:max_topics]


def get_topic_keywords(topics: List[Dict]) -> List[str]:
    """Extract keywords from trending topics for matching.

    Useful for quick string matching before embedding-based matching.

    Args:
        topics: List of trend dicts from get_trending_topics()

    Returns:
        List of lowercase topic strings
    """
    return [t.get("topic", "").lower().strip() for t in topics if t.get("topic")]


def topic_matches_text(topic: str, text: str) -> bool:
    """Check if a trending topic appears in text.

    Simple substring matching for quick filtering.

    Args:
        topic: Trending topic string
        text: Text to search (e.g., market question)

    Returns:
        True if topic found in text
    """
    if not topic or not text:
        return False

    topic_lower = topic.lower().strip()
    text_lower = text.lower()

    # Direct match
    if topic_lower in text_lower:
        return True

    # Match individual words (for multi-word topics)
    words = topic_lower.split()
    if len(words) > 1:
        # If all significant words match
        significant_words = [w for w in words if len(w) > 3]
        if significant_words and all(w in text_lower for w in significant_words):
            return True

    return False
