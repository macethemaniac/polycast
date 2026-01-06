"""Twitter/X trending topic scraping via Nitter (no API key required).

Uses ntscraper library which scrapes Nitter instances (Twitter mirrors).
Falls back gracefully if unavailable.
"""
from __future__ import annotations

import time
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

_CACHE: Dict[str, Dict] = {}
_TTL_SEC = 15 * 60  # 15 minutes
_TIMEOUT = 15


def _from_cache(key: str) -> List | None:
    entry = _CACHE.get(key)
    if not entry:
        return None
    if time.time() - entry["ts"] > _TTL_SEC:
        _CACHE.pop(key, None)
        return None
    return entry["value"]


def _to_cache(key: str, value: List) -> None:
    _CACHE[key] = {"value": value, "ts": time.time()}


def _normalize_tweet_volume(volume: int | None) -> float:
    """Normalize tweet volume to 0-100 scale."""
    if not volume:
        return 30.0  # Default score for trends without volume

    if volume >= 1000000:
        return 95.0
    elif volume >= 500000:
        return 85.0
    elif volume >= 100000:
        return 70.0
    elif volume >= 50000:
        return 55.0
    elif volume >= 10000:
        return 40.0
    else:
        return 25.0


def get_twitter_trends(region: str = "USA") -> List[Dict]:
    """Fetch trending topics from Twitter/X via Nitter scraping.

    Args:
        region: Region/country code (not always supported by Nitter)

    Returns:
        List of trending topics:
        [{"topic": str, "tweet_volume": int | None, "score": float}]
    """
    cache_key = f"twitter_{region}"
    cached = _from_cache(cache_key)
    if cached is not None:
        return cached

    trends = []

    # Try ntscraper first
    try:
        from ntscraper import Nitter

        scraper = Nitter(log_level=logging.WARNING)

        # Get trending topics
        trending = scraper.get_trending_topics()

        if trending:
            for i, topic in enumerate(trending[:25]):
                if isinstance(topic, dict):
                    name = topic.get("name", "") or topic.get("topic", "")
                    volume = topic.get("tweet_volume")
                elif isinstance(topic, str):
                    name = topic
                    volume = None
                else:
                    continue

                if not name:
                    continue

                # Clean up hashtags
                name = name.lstrip("#")

                # Score by rank if no volume
                score = _normalize_tweet_volume(volume) if volume else max(20, 90 - (i * 3))

                trends.append({
                    "topic": name,
                    "tweet_volume": volume,
                    "score": score,
                })

        _to_cache(cache_key, trends)
        logger.debug("get_twitter_trends: fetched %d trends via ntscraper", len(trends))
        return trends

    except ImportError:
        logger.debug("ntscraper not installed, Twitter trends unavailable")
    except Exception as exc:
        logger.debug("ntscraper fetch failed: %s", exc)

    # Cache empty result to avoid hammering on failure
    _to_cache(cache_key, trends)
    return trends


def search_tweets_for_topic(topic: str, limit: int = 10) -> List[Dict]:
    """Search recent tweets for a topic.

    Args:
        topic: Search query
        limit: Max tweets to return

    Returns:
        List of tweets:
        [{"text": str, "created_at": str, "likes": int, "retweets": int}]
    """
    cache_key = f"search_{topic}"
    cached = _from_cache(cache_key)
    if cached is not None:
        return cached

    tweets = []

    try:
        from ntscraper import Nitter

        scraper = Nitter(log_level=logging.WARNING)
        results = scraper.get_tweets(topic, mode="term", number=limit)

        if results and "tweets" in results:
            for tweet in results["tweets"][:limit]:
                tweets.append({
                    "text": tweet.get("text", ""),
                    "created_at": tweet.get("date", ""),
                    "likes": tweet.get("stats", {}).get("likes", 0),
                    "retweets": tweet.get("stats", {}).get("retweets", 0),
                })

        _to_cache(cache_key, tweets)
        return tweets

    except ImportError:
        logger.debug("ntscraper not installed")
        return []
    except Exception as exc:
        logger.debug("Tweet search failed for %s: %s", topic, exc)
        return []
