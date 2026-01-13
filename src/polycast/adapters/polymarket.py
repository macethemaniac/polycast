import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

from . import Market, Outcome, OrderBook
from src.utils.rate_limiter import get_with_retry, RateLimitError, check_rate_limit_cooldown

logger = logging.getLogger(__name__)

_MARKET_CACHE: Dict[str, Market] = {}
_ORDERBOOK_CACHE: Dict[Tuple[str, str], OrderBook] = {}
_MARKETS_META: Dict[str, Any] = {}
_CACHE_TIMESTAMP: float = 0
_CACHE_TTL: float = 300  # 5 minute cache TTL


def _normalize_price(val: Optional[float]) -> Optional[float]:
    if val is None:
        return None
    try:
        v = float(val)
    except Exception:
        return None
    if v < 0:
        v = 0.0
    if v > 1:
        v = 1.0
    return v


def _build_market(obj: dict) -> Optional[Market]:
    market_id = str(obj.get("id") or obj.get("_id") or obj.get("slug") or "")
    title = obj.get("question") or obj.get("title") or obj.get("slug") or ""
    if not market_id:
        return None
    outcomes_raw = obj.get("outcomes") or []
    outcomes: List[Outcome] = []
    for o in outcomes_raw:
        if not isinstance(o, dict):
            continue
        name = o.get("title") or o.get("id") or ""
        bid = _normalize_price(o.get("bestBid") or o.get("bid") or o.get("price"))
        ask = _normalize_price(o.get("bestAsk") or o.get("ask") or o.get("price"))
        outcomes.append(
            Outcome(
                name=name,
                best_bid=bid,
                best_ask=ask,
                bid_size=None,
                ask_size=None,
                raw=o,
            )
        )
    updated_ts = obj.get("timestamp") or obj.get("updated") or obj.get("ts")
    try:
        updated_ts = float(updated_ts) if updated_ts is not None else None
    except Exception:
        updated_ts = None
    return Market(id=market_id, title=title, outcomes=outcomes, updated_ts=updated_ts, raw=obj)


def list_markets(limit: int = 100, return_debug: bool = False) -> Union[List[Market], Tuple[List[Market], Dict[str, Any]]]:
    """Fetch Polymarket markets as standardized Market objects."""
    global _CACHE_TIMESTAMP

    # Return cached data if still valid
    if _MARKET_CACHE and (time.time() - _CACHE_TIMESTAMP) < _CACHE_TTL:
        cached = list(_MARKET_CACHE.values())
        if return_debug:
            meta = _MARKETS_META or {"attempts": [], "count": len(cached), "source": "cache"}
            return cached, meta
        return cached

    url = f"https://gamma-api.polymarket.com/markets?closed=false&limit={limit}"
    attempts = []
    markets: List[Market] = []
    rate_limited = False

    # Check if domain is rate limited
    domain = "gamma-api.polymarket.com"
    cooldown = check_rate_limit_cooldown(domain)
    if cooldown and cooldown > 5:
        attempts.append({"url": url, "status_code": 429, "ok": False, "error": f"Rate limited, {cooldown:.0f}s remaining"})
        rate_limited = True
    else:
        start = time.monotonic()
        try:
            resp = get_with_retry(url, timeout=10, max_retries=3)
            latency_ms = int((time.monotonic() - start) * 1000)
            attempts.append({"url": url, "status_code": resp.status_code, "ok": resp.ok, "latency_ms": latency_ms})

            if resp.ok:
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

        except RateLimitError as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            attempts.append({"url": url, "status_code": 429, "ok": False, "error": str(exc), "latency_ms": latency_ms})
            rate_limited = True
            logger.warning(f"Polymarket rate limited: {exc}")

        except requests.RequestException as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            attempts.append({"url": url, "status_code": None, "ok": False, "error": str(exc), "latency_ms": latency_ms})

    # Fallback to DFlow if rate limited or no results
    if (rate_limited or not markets) and not _MARKET_CACHE:
        logger.info("Trying DFlow fallback for Polymarket markets...")
        try:
            from src.exchanges.dflow import fetch_dflow_markets
            dflow_markets = fetch_dflow_markets(limit=limit)
            for dm in dflow_markets:
                if dm and dm.get("source", "").lower() in ("polymarket", "dflow"):
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

    _MARKETS_META["attempts"] = attempts
    _MARKETS_META["count"] = len(markets)
    _MARKETS_META["rate_limited"] = rate_limited

    if return_debug:
        return markets, {"attempts": attempts, "count": len(markets), "rate_limited": rate_limited}
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
    """Return a standardized OrderBook for a Polymarket outcome (side ~ outcome title)."""
    cache_key = (market_id, side.lower())
    if cache_key in _ORDERBOOK_CACHE:
        return _ORDERBOOK_CACHE[cache_key]
    market = _MARKET_CACHE.get(market_id)
    if market is None:
        # populate cache by refetching
        list_markets(return_debug=False)
        market = _MARKET_CACHE.get(market_id)
        if market is None:
            return None
    target = None
    side_l = side.lower()
    for o in market.outcomes:
        if side_l in o.name.lower():
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
