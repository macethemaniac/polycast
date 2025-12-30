"""Simple Kalshi public API helpers.

Attempts unauthenticated fetches of Kalshi market lists. The Kalshi
public API may require authentication for some endpoints; this module
tries a common endpoint and falls back cleanly to returning an empty list
if unavailable.
"""
from typing import List, Dict, Any
import requests


def fetch_kalshi_markets(limit: int) -> List[Dict[str, Any]]:
    """
    Attempt to fetch Kalshi markets. Returns list of market dicts or [] on error.

    Note: Kalshi's API may require API keys for full access; this function
    uses common public endpoints and is resilient to errors.
    """
    # common guess for public listing endpoint
    urls = [
        f"https://api.kalshi.com/v1/markets?limit={limit}",
        f"https://www.kalshi.com/api/markets?limit={limit}",
    ]

    for url in urls:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            # try common shapes
            if isinstance(data, dict):
                if 'markets' in data and isinstance(data['markets'], list):
                    return data['markets']
                if 'data' in data and isinstance(data['data'], list):
                    return data['data']
                # sometimes API returns a list in a key
                for v in data.values():
                    if isinstance(v, list):
                        return v

            if isinstance(data, list):
                return data

        except requests.RequestException:
            continue

    return []


def fetch_kalshi_series(series_id: str) -> Dict[str, Any]:
    """
    Fetch Kalshi series information by series id.

    Example endpoint (provided):
      https://api.elections.kalshi.com/trade-api/v2/series/KXHIGHNY

    Returns the parsed JSON (empty dict on error).
    """
    url = f"https://api.elections.kalshi.com/trade-api/v2/series/{series_id}"
    try:
        resp = requests.get(url, timeout=10)
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
    url = f"https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker={series_ticker}&status={status}"
    try:
        resp = requests.get(url, timeout=10)
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
    url = f"https://api.elections.kalshi.com/trade-api/v2/events/{event_ticker}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and 'event' in data:
            return data['event']
        return data
    except requests.RequestException:
        return {}


def kalshi_market_to_generic(m: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a Kalshi market record into the generic market shape used by cross_arb.

    Kalshi often reports `yes_price` in cents; convert to decimal (0-1) if present.
    """
    if not isinstance(m, dict):
        return {}
    title = m.get('title') or m.get('ticker') or ''
    # Try to find yes/no prices
    yes = None
    no = None
    # Try common fields in Kalshi market records. Prices often in cents.
    # Prefer `last_price`, then `yes_bid`/`yes_ask`, then `yes_price`/`no_price`.
    try:
        if 'last_price' in m and m.get('last_price') is not None:
            # last_price often in cents (integer)
            try:
                lp = float(m.get('last_price'))
                # if value > 1 assume cents
                yes = lp / 100.0 if lp > 1.0 else lp
            except Exception:
                pass

        # yes_bid/yes_ask are typically in cents integer
        if yes is None and 'yes_bid' in m and m.get('yes_bid') is not None:
            try:
                yb = float(m.get('yes_bid'))
                yes = yb / 100.0 if yb > 1.0 else yb
            except Exception:
                pass
        if no is None and 'no_bid' in m and m.get('no_bid') is not None:
            try:
                nb = float(m.get('no_bid'))
                no = nb / 100.0 if nb > 1.0 else nb
            except Exception:
                pass

        # yes_ask / no_ask fallback
        if yes is None and 'yes_ask' in m and m.get('yes_ask') is not None:
            try:
                ya = float(m.get('yes_ask'))
                yes = ya / 100.0 if ya > 1.0 else ya
            except Exception:
                pass
        if no is None and 'no_ask' in m and m.get('no_ask') is not None:
            try:
                na = float(m.get('no_ask'))
                no = na / 100.0 if na > 1.0 else na
            except Exception:
                pass

        # dollar-string fields like last_price_dollars or yes_ask_dollars (e.g. '0.0300')
        if yes is None:
            for k in ('last_price_dollars', 'yes_ask_dollars', 'yes_bid_dollars'):
                if k in m and m.get(k) is not None:
                    try:
                        v = float(str(m.get(k)))
                        yes = v
                        break
                    except Exception:
                        continue
        if no is None:
            for k in ('no_ask_dollars', 'no_bid_dollars'):
                if k in m and m.get(k) is not None:
                    try:
                        v = float(str(m.get(k)))
                        no = v
                        break
                    except Exception:
                        continue

        # If still missing, and a 'price' midpoint exists use it
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
    if yes is not None:
        outcomes.append({'title': 'YES', 'price': yes})
    if no is not None:
        outcomes.append({'title': 'NO', 'price': no})

    return {
        'title': title,
        'ticker': m.get('ticker') or m.get('event_ticker') or '',
        'outcomes': outcomes,
        'raw': m,
    }
