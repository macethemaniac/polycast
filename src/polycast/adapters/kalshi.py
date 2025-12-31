import os
import time
from typing import Dict, List, Optional, Tuple, Union

import requests

from . import Market, Outcome, OrderBook

_MARKET_CACHE: Dict[str, Market] = {}
_ORDERBOOK_CACHE: Dict[Tuple[str, str], OrderBook] = {}


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


def list_markets(limit: int = 100, return_debug: bool = False) -> Union[List[Market], Tuple[List[Market], Dict[str, any]]]:
    if _MARKET_CACHE:
        cached = list(_MARKET_CACHE.values())
        meta_cached = {"attempts": [], "count": len(cached)}
        return (cached, meta_cached) if return_debug else cached
    headers = _get_kalshi_headers()
    urls = [
        f"https://api.kalshi.com/v1/markets?limit={limit}",
        f"https://api.elections.kalshi.com/trade-api/v2/markets?limit={limit}&status=open",
        f"https://www.kalshi.com/api/markets?limit={limit}",
    ]
    attempts = []
    markets: List[Market] = []
    max_retries = 2
    for url in urls:
        for attempt in range(max_retries):
            start = time.monotonic()
            try:
                resp = requests.get(url, headers=headers, timeout=10)
                latency_ms = int((time.monotonic() - start) * 1000)
                attempts.append({"url": url, "status_code": resp.status_code, "ok": resp.ok, "latency_ms": latency_ms})
                resp.raise_for_status()
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
                break
            except requests.RequestException as exc:
                attempts.append({"url": url, "status_code": None, "ok": False, "error": str(exc), "latency_ms": int((time.monotonic() - start) * 1000)})
                if attempt == max_retries - 1:
                    continue
                time.sleep(1)
    meta = {"attempts": attempts, "count": len(markets)}
    if return_debug:
        return markets, meta
    return markets


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
