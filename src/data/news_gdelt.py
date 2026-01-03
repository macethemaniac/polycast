"""GDELT-based news signal (free, no auth).

Provides a lightweight mentions count and a few headlines for a query.
"""
from __future__ import annotations

import time
import logging
import urllib.parse
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)

_CACHE: Dict[str, Dict] = {}
_TTL_SEC = 15 * 60  # 15 minutes
_TIMEOUT = 10
_GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"


def _from_cache(key: str) -> Dict | None:
    entry = _CACHE.get(key)
    if not entry:
        return None
    if time.time() - entry["ts"] > _TTL_SEC:
        _CACHE.pop(key, None)
        return None
    return entry["value"]


def _to_cache(key: str, value: Dict) -> None:
    _CACHE[key] = {"value": value, "ts": time.time()}


def get_news_signal(query: str) -> Dict:
    """Return a simple news signal for a query using GDELT (no auth).

    Response keys:
        mentions_24h (int): Count of returned articles (up to GDELT maxrecords) in last 24h.
        headlines (list[str]): Up to 3 headlines.
    """
    if not query:
        return {"mentions_24h": 0, "headlines": []}

    cache_key = query.lower().strip()
    cached = _from_cache(cache_key)
    if cached is not None:
        return cached

    try:
        encoded_query = urllib.parse.quote_plus(query.strip())
        params = {
            "query": encoded_query,
            "maxrecords": 50,  # keep it light
            "format": "json",
            "mode": "ArtList",
            "timespan": "24h",
        }
        resp = requests.get(_GDELT_DOC_ENDPOINT, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        arts: List[Dict] = data.get("articles") or []
        mentions = len(arts)
        headlines = []
        for a in arts[:3]:
            h = a.get("title") or a.get("semtags")
            if h:
                headlines.append(str(h))
        result = {"mentions_24h": mentions, "headlines": headlines}
    except Exception as exc:
        logger.debug("GDELT fetch failed for %s: %s", query, exc)
        result = {"mentions_24h": 0, "headlines": []}

    _to_cache(cache_key, result)
    return result
