import os
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

from . import Market, Outcome, OrderBook
from src.utils.rate_limiter import get_with_retry, RateLimitError, check_rate_limit_cooldown

logger = logging.getLogger(__name__)

_MARKET_CACHE: Dict[str, Market] = {}
_ORDERBOOK_CACHE: Dict[Tuple[str, str], OrderBook] = {}
_CACHE_TIMESTAMP: float = 0
_CACHE_TTL: float = 300  # 5 minute cache TTL


def _norm_prob(val: Optional[float]) -> Optional[float]:
    if val is None:
        return None
    try:
        v = float(val)
    except Exception:
        return None
    if v > 1.0:
        v = v / 100.0
    if v < 0:
        v = 0.0
    if v > 1:
        v = 1.0
    return v


def _extract_outcomes(m: dict) -> List[Outcome]:
    yes_bid = _norm_prob(m.get("yes_bid") or m.get("yesBid") or m.get("best_bid_yes"))
    yes_ask = _norm_prob(m.get("yes_ask") or m.get("yesAsk") or m.get("best_ask_yes"))
    no_bid = _norm_prob(m.get("no_bid") or m.get("noBid") or m.get("best_bid_no"))
    no_ask = _norm_prob(m.get("no_ask") or m.get("noAsk") or m.get("best_ask_no"))
    outcomes: List[Outcome] = []
    outcomes.append(Outcome(name="YES", best_bid=yes_bid, best_ask=yes_ask, raw=m))
    outcomes.append(Outcome(name="NO", best_bid=no_bid, best_ask=no_ask, raw=m))
    return outcomes


def _build_market(obj: dict) -> Optional[Market]:
    mid = obj.get("id") or obj.get("ticker") or obj.get("event_ticker") or obj.get("marketTicker")
    if not mid:
        return None
    title = obj.get("title") or obj.get("name") or obj.get("ticker") or ""
    ts = obj.get("updated") or obj.get("timestamp") or obj.get("ts")
    try:
        ts = float(ts) if ts is not None else None
    except Exception:
        ts = None
    outcomes = _extract_outcomes(obj)
    market = Market(id=str(mid), title=title, outcomes=outcomes, updated_ts=ts, raw=obj)
    return market


def list_markets(limit: int = 100, return_debug: bool = False) -> Union[List[Market], Tuple[List[Market], Dict[str, Any]]]:
    global _CACHE_TIMESTAMP

    # Return cached data if still valid
    if _MARKET_CACHE and (time.time() - _CACHE_TIMESTAMP) < _CACHE_TTL:
        cached = list(_MARKET_CACHE.values())
        meta_cached = {"attempts": [], "count": len(cached), "source": "cache"}
        return (cached, meta_cached) if return_debug else cached

    headers = _get_kalshi_headers()
    urls = [
        f"https://api.elections.kalshi.com/trade-api/v2/markets?limit={limit}&status=open",
        f"https://api.kalshi.com/v1/markets?limit={limit}",
    ]
    attempts = []
    markets: List[Market] = []
    rate_limited = False

    for url in urls:
        # Check if this domain is rate limited
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        cooldown = check_rate_limit_cooldown(domain)
        if cooldown and cooldown > 5:
            attempts.append({"url": url, "status_code": 429, "ok": False, "error": f"Rate limited, {cooldown:.0f}s remaining"})
            rate_limited = True
            continue

        start = time.monotonic()
        try:
            resp = get_with_retry(url, headers=headers, timeout=10, max_retries=2)
            latency_ms = int((time.monotonic() - start) * 1000)
            attempts.append({"url": url, "status_code": resp.status_code, "ok": resp.ok, "latency_ms": latency_ms})

            if not resp.ok:
                continue

            data = resp.json()
            container = []
            if isinstance(data, dict):
                if isinstance(data.get("markets"), list):
                    container = data["markets"]
                elif isinstance(data.get("data"), list):
                    container = data["data"]
                else:
                    for v in data.values():
                        if isinstance(v, list):
                            container = v
                            break
            elif isinstance(data, list):
                container = data

            for m in container:
                market = _build_market(m)
                if market:
                    markets.append(market)
                    _MARKET_CACHE[market.id] = market

            if markets:
                _CACHE_TIMESTAMP = time.time()
                break

        except RateLimitError as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            attempts.append({"url": url, "status_code": 429, "ok": False, "error": str(exc), "latency_ms": latency_ms})
            rate_limited = True
            logger.warning(f"Kalshi rate limited: {exc}")
            continue

        except requests.RequestException as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            attempts.append({"url": url, "status_code": None, "ok": False, "error": str(exc), "latency_ms": latency_ms})
            continue

    # Fallback to DFlow if rate limited or no results
    if (rate_limited or not markets) and not _MARKET_CACHE:
        logger.info("Trying DFlow fallback for Kalshi markets...")
        try:
            from src.exchanges.dflow import fetch_dflow_markets
            dflow_markets = fetch_dflow_markets(limit=limit)
            for dm in dflow_markets:
                if dm and dm.get("source", "").lower() in ("kalshi", "dflow"):
                    market = _build_market_from_normalized(dm)
                    if market:
                        markets.append(market)
                        _MARKET_CACHE[market.id] = market
            if markets:
                _CACHE_TIMESTAMP = time.time()
                attempts.append({"url": "dflow_fallback", "ok": True, "count": len(markets)})
        except Exception as e:
            logger.error(f"DFlow fallback failed: {e}")
            attempts.append({"url": "dflow_fallback", "ok": False, "error": str(e)})

    meta = {"attempts": attempts, "count": len(markets), "rate_limited": rate_limited}
    if return_debug:
        return markets, meta
    return markets


def _build_market_from_normalized(dm: Dict) -> Optional[Market]:
    """Build Market from normalized DFlow dict."""
    mid = dm.get("market_id")
    if not mid:
        return None

    title = dm.get("question") or dm.get("title") or ""
    yes_price = dm.get("yes_price")
    no_price = dm.get("no_price")

    outcomes = []
    if yes_price is not None:
        outcomes.append(Outcome(name="YES", best_bid=yes_price, best_ask=yes_price, raw=dm))
    if no_price is not None:
        outcomes.append(Outcome(name="NO", best_bid=no_price, best_ask=no_price, raw=dm))

    return Market(id=str(mid), title=title, outcomes=outcomes, updated_ts=None, raw=dm)


def get_orderbook(market_id: str, side: str) -> Optional[OrderBook]:
    cache_key = (market_id, side.lower())
    if cache_key in _ORDERBOOK_CACHE:
        return _ORDERBOOK_CACHE[cache_key]
    market = _MARKET_CACHE.get(market_id)
    if market is None:
        list_markets(return_debug=False)
        market = _MARKET_CACHE.get(market_id)
        if market is None:
            return None
    target = None
    side_l = side.lower()
    for o in market.outcomes:
        if side_l == o.name.lower():
            target = o
            break
    if target is None and market.outcomes:
        target = market.outcomes[0]
    ob = OrderBook(
        best_bid=target.best_bid if target else None,
        bid_size=target.bid_size if target else None,
        best_ask=target.best_ask if target else None,
        ask_size=target.ask_size if target else None,
        ts=market.updated_ts if market else None,
    )
    _ORDERBOOK_CACHE[cache_key] = ob
    return ob
def _get_kalshi_headers() -> Dict[str, str]:
    """Get headers for Kalshi API requests, including auth if available."""
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }
    api_key = os.getenv('KALSHI_API_KEY')
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    return headers
