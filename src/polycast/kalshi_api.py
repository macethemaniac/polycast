"""Simple Kalshi public API helpers.

Attempts unauthenticated fetches of Kalshi market lists. The Kalshi
public API may require authentication for some endpoints; this module
tries a common endpoint and falls back cleanly to returning an empty list
if unavailable.
"""
from typing import List, Dict, Any, Optional, Tuple
import requests
import os
import time


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


def fetch_kalshi_markets(limit: int, return_debug: bool = False) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Attempt to fetch Kalshi markets. Returns list of market dicts or [] on error.

    Note: Kalshi's API may require API keys for full access; this function
    uses common public endpoints and is resilient to errors.
    Set KALSHI_API_KEY environment variable for authenticated access.
    """
    headers = _get_kalshi_headers()

    # common guess for public listing endpoint
    urls = [
        f"https://api.kalshi.com/v1/markets?limit={limit}",
        f"https://api.elections.kalshi.com/trade-api/v2/markets?limit={limit}&status=open",
        f"https://www.kalshi.com/api/markets?limit={limit}",
    ]

    attempts = []
    for url in urls:
        try:
            start = time.monotonic()
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            latency_ms = int((time.monotonic() - start) * 1000)
            attempts.append(
                {
                    "url": url,
                    "status_code": resp.status_code,
                    "ok": True,
                    "latency_ms": latency_ms,
                }
            )
            data = resp.json()

            # try common shapes
            if isinstance(data, dict):
                if 'markets' in data and isinstance(data['markets'], list):
                    markets = data['markets']
                    return (markets, {"attempts": attempts, "count": len(markets)}) if return_debug else markets
                if 'data' in data and isinstance(data['data'], list):
                    markets = data['data']
                    return (markets, {"attempts": attempts, "count": len(markets)}) if return_debug else markets
                # sometimes API returns a list in a key
                for v in data.values():
                    if isinstance(v, list):
                        markets = v
                        return (markets, {"attempts": attempts, "count": len(markets)}) if return_debug else markets

            if isinstance(data, list):
                markets = data
                return (markets, {"attempts": attempts, "count": len(markets)}) if return_debug else markets

        except requests.RequestException as exc:
            attempts.append(
                {
                    "url": url,
                    "status_code": getattr(exc.response, "status_code", None) if hasattr(exc, "response") else None,
                    "ok": False,
                    "error": str(exc),
                    "latency_ms": int((time.monotonic() - start) * 1000) if 'start' in locals() else None,
                }
            )
            continue

    return ([], {"attempts": attempts, "count": 0}) if return_debug else []


def fetch_kalshi_series(series_id: str) -> Dict[str, Any]:
    """
    Fetch Kalshi series information by series id.

    Example endpoint (provided):
      https://api.elections.kalshi.com/trade-api/v2/series/KXHIGHNY

    Returns the parsed JSON (empty dict on error).
    """
    headers = _get_kalshi_headers()
    url = f"https://api.elections.kalshi.com/trade-api/v2/series/{series_id}"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # Prefer 'series' payload if present
        if isinstance(data, dict) and 'series' in data:
            return data['series']
        return data
    except requests.RequestException:
        return {}


def series_to_market(series: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a Kalshi series payload into the market shape used elsewhere.

    This is best-effort: Kalshi series payloads may not include live prices.
    """
    if not series:
        return {}
    title = series.get('title') or series.get('name') or ''
    # attempt to extract outcomes/contracts
    outcomes = []
    # common keys to inspect
    for key in ('contracts', 'outcomes', 'events'):
        if key in series and isinstance(series[key], list):
            for c in series[key]:
                o_title = c.get('title') or c.get('name') or c.get('contract') or ''
                price = None
                for pk in ('price', 'lastPrice', 'midpoint'):
                    if pk in c and c[pk] is not None:
                        try:
                            price = float(c[pk])
                            break
                        except Exception:
                            continue
                outcomes.append({'title': o_title, 'price': price})
            break

    return {
        'title': title,
        'series_id': series.get('id') or series.get('seriesId') or '',
        'frequency': series.get('frequency'),
        'category': series.get('category'),
        'outcomes': outcomes,
        'raw': series,
    }


def fetch_kalshi_series_markets(series_ticker: str, status: str = 'open') -> List[Dict[str, Any]]:
    """Fetch active markets for a Kalshi series (by ticker).

    Example endpoint (from user):
      https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXHIGHNY&status=open

    Returns list of market dicts (may be empty).
    """
    headers = _get_kalshi_headers()
    url = f"https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker={series_ticker}&status={status}"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and 'markets' in data and isinstance(data['markets'], list):
            return data['markets']
        if isinstance(data, list):
            return data
        return []
    except requests.RequestException:
        return []


def fetch_kalshi_event(event_ticker: str) -> Dict[str, Any]:
    """Fetch event details for a given Kalshi event ticker.

    Endpoint example: https://api.elections.kalshi.com/trade-api/v2/events/{event_ticker}
    """
    headers = _get_kalshi_headers()
    url = f"https://api.elections.kalshi.com/trade-api/v2/events/{event_ticker}"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and 'event' in data:
            return data['event']
        return data
    except requests.RequestException:
        return {}


def _norm_prob(val: Optional[float]) -> Optional[float]:
    """Normalize a Kalshi price to probability [0,1], handling cents."""
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


def kalshi_market_to_generic(m: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a Kalshi market record into the generic market shape used by cross_arb.

    Kalshi often reports `yes_price` in cents; convert to decimal (0-1) if present.
    """
    if not isinstance(m, dict):
        return {}
    title = m.get('title') or m.get('ticker') or ''
    yes_bid = _norm_prob(m.get('yes_bid') or m.get('yesBid'))
    yes_ask = _norm_prob(m.get('yes_ask') or m.get('yesAsk'))
    no_bid = _norm_prob(m.get('no_bid') or m.get('noBid'))
    no_ask = _norm_prob(m.get('no_ask') or m.get('noAsk'))

    yes = None
    no = None
    try:
        if 'last_price' in m and m.get('last_price') is not None:
            lp = float(m.get('last_price'))
            yes = _norm_prob(lp)

        if yes is None and 'yes_bid' in m and m.get('yes_bid') is not None:
            yes = _norm_prob(m.get('yes_bid'))
        if no is None and 'no_bid' in m and m.get('no_bid') is not None:
            no = _norm_prob(m.get('no_bid'))

        if yes is None and 'yes_ask' in m and m.get('yes_ask') is not None:
            yes = _norm_prob(m.get('yes_ask'))
        if no is None and 'no_ask' in m and m.get('no_ask') is not None:
            no = _norm_prob(m.get('no_ask'))

        if yes is None:
            for k in ('last_price_dollars', 'yes_ask_dollars', 'yes_bid_dollars'):
                if k in m and m.get(k) is not None:
                    try:
                        yes = float(str(m.get(k)))
                        break
                    except Exception:
                        continue
        if no is None:
            for k in ('no_ask_dollars', 'no_bid_dollars'):
                if k in m and m.get(k) is not None:
                    try:
                        no = float(str(m.get(k)))
                        break
                    except Exception:
                        continue

        if (yes is None or no is None) and 'price' in m and m.get('price') is not None:
            try:
                p = float(m.get('price'))
                yes = yes if yes is not None else p
                no = no if no is not None else (1.0 - p)
            except Exception:
                pass
    except Exception:
        pass

    outcomes = []
    if yes is not None or yes_bid is not None or yes_ask is not None:
        outcomes.append({'title': 'YES', 'price': yes, 'best_bid': yes_bid, 'best_ask': yes_ask})
    if no is not None or no_bid is not None or no_ask is not None:
        outcomes.append({'title': 'NO', 'price': no, 'best_bid': no_bid, 'best_ask': no_ask})

    return {
        'title': title,
        'ticker': m.get('ticker') or m.get('event_ticker') or '',
        'outcomes': outcomes,
        'raw': m,
    }
