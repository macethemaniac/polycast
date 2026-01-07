"""Unified social trends interface.

Combines multiple trend sources (Google Trends, Twitter/X, GDELT) with fallback.
Provides a single API for fetching trending topics.
"""
from __future__ import annotations

import logging
import time
import requests
from typing import Dict, List

logger = logging.getLogger(__name__)

# GDELT cache
_GDELT_CACHE: Dict = {"topics": [], "timestamp": 0}
_GDELT_TTL = 10 * 60  # 10 minutes


def _get_gdelt_trending_topics() -> List[Dict]:
    """Get trending news topics from GDELT API.

    Extracts trending entities/topics from recent news headlines.
    """
    now = time.time()
    if _GDELT_CACHE["topics"] and now - _GDELT_CACHE["timestamp"] < _GDELT_TTL:
        return _GDELT_CACHE["topics"]

    try:
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            "query": "sourcelang:english",
            "mode": "ArtList",
            "maxrecords": 150,
            "format": "json",
            "timespan": "24h",
        }

        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()

        if not resp.text:
            return []

        data = resp.json()
        articles = data.get("articles", []) or []

        # Extract entities/topics from headlines
        topic_counts: Dict[str, int] = {}
        # Common words to skip
        skip_words = {
            "the", "and", "for", "that", "this", "with", "from", "have", "has",
            "will", "are", "was", "were", "been", "being", "could", "would",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "about", "after", "before", "into", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "each", "few", "more", "most", "other", "some", "such",
            "only", "own", "same", "than", "too", "very", "just", "also", "now",
            "new", "first", "last", "says", "said", "report", "news", "today",
            "year", "years", "time", "world", "people", "week", "month", "day",
        }

        for art in articles:
            title = art.get("title", "")
            # Extract capitalized words/phrases (likely entities)
            words = title.split()
            for i, word in enumerate(words):
                word = word.strip(".,!?\"'()[]:-")
                word_lower = word.lower()

                # Skip short words, common words, and non-capitalized
                if len(word) < 4 or word_lower in skip_words:
                    continue
                if not word[0].isupper():
                    continue

                # Check for two-word phrases (e.g., "Donald Trump")
                if i + 1 < len(words):
                    next_word = words[i + 1].strip(".,!?\"'()[]:-")
                    if next_word and next_word[0].isupper() and len(next_word) > 2:
                        phrase = f"{word} {next_word}"
                        if phrase.lower() not in skip_words:
                            topic_counts[phrase] = topic_counts.get(phrase, 0) + 1

                topic_counts[word] = topic_counts.get(word, 0) + 1

        # Sort by frequency and take top topics
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)

        topics = []
        seen_lower = set()
        for topic, count in sorted_topics:
            if count < 3:  # Must appear at least 3 times
                continue

            topic_lower = topic.lower()
            # Skip if we already have this topic (case-insensitive)
            if topic_lower in seen_lower:
                continue
            # Skip if a longer version of this topic exists
            if any(topic_lower in s for s in seen_lower if len(s) > len(topic_lower)):
                continue

            seen_lower.add(topic_lower)
            # Score: higher count = higher score (50-90 range)
            score = min(90, 50 + count * 2)

            topics.append({
                "topic": topic,
                "score": score,
                "count": count,
            })

            if len(topics) >= 25:
                break

        _GDELT_CACHE["topics"] = topics
        _GDELT_CACHE["timestamp"] = now
        logger.info("_get_gdelt_trending_topics: fetched %d topics", len(topics))
        return topics

    except Exception as exc:
        logger.debug("GDELT trending topics failed: %s", exc)
        return []


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

    # GDELT news as fallback when other sources fail
    if not all_trends:
        try:
            gdelt_trends = _get_gdelt_trending_topics()
            for t in gdelt_trends:
                topic = t.get("topic", "").strip()
                if not topic:
                    continue

                topic_lower = topic.lower()
                if topic_lower in seen_topics:
                    continue
                seen_topics.add(topic_lower)

                all_trends.append({
                    "topic": topic,
                    "source": "gdelt",
                    "score": float(t.get("score", 50.0)),
                    "metadata": t,
                })

            logger.debug("get_trending_topics: got %d from GDELT", len(gdelt_trends))

        except Exception as exc:
            logger.debug("GDELT trends unavailable: %s", exc)

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

    Flexible matching for better coverage.

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

    # Single word partial matching (e.g., "Trump" matches "Trump's")
    if len(words) == 1 and len(topic_lower) >= 4:
        # Check if topic appears as a word boundary
        import re
        pattern = r'\b' + re.escape(topic_lower[:4])  # Match first 4 chars at word boundary
        if re.search(pattern, text_lower):
            return True

    # Common name/entity matching
    # Handle cases like "Bitcoin" matching "BTC", "Trump" matching "Donald Trump"
    aliases = {
        "bitcoin": ["btc", "bitcoin"],
        "ethereum": ["eth", "ethereum"],
        "trump": ["trump", "donald trump"],
        "biden": ["biden", "joe biden"],
        "nvidia": ["nvidia", "nvda"],
        "tesla": ["tesla", "tsla"],
        "openai": ["openai", "chatgpt", "gpt"],
        "meta": ["meta", "facebook", "zuckerberg"],
        "google": ["google", "alphabet", "googl"],
        "amazon": ["amazon", "amzn"],
        "apple": ["apple", "aapl"],
        "microsoft": ["microsoft", "msft"],
    }

    topic_base = topic_lower.split()[0] if words else topic_lower
    if topic_base in aliases:
        for alias in aliases[topic_base]:
            if alias in text_lower:
                return True

    return False
