"""
CoinGecko data source integration module.

This module provides functionality to fetch cryptocurrency prices from CoinGecko
using their public API.
"""

from typing import Dict, List
import time
import requests

_COIN_LIST_CACHE: Dict[str, object] = {"timestamp": 0, "symbols": {}}
_COIN_LIST_TTL = 60 * 60 * 24

_COIN_ID_OVERRIDES = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "DOGE": "dogecoin",
    "USDT": "tether",
    "USDC": "usd-coin",
    "DAI": "dai",
    "BUSD": "binance-usd",
    "USDP": "pax-dollar",
    "TUSD": "true-usd",
}

_COIN_ID_PREFERENCES = {
    "ARB": ["arbitrum"],
    "OP": ["optimism"],
}


def _fetch_coin_list() -> Dict[str, List[str]]:
    now = int(time.time())
    cached = _COIN_LIST_CACHE.get("symbols", {})
    if cached and (now - int(_COIN_LIST_CACHE.get("timestamp", 0)) < _COIN_LIST_TTL):
        return cached

    url = "https://api.coingecko.com/api/v3/coins/list"
    response = requests.get(url, params={"include_platform": "false"}, timeout=10)
    response.raise_for_status()
    data = response.json()
    symbol_map: Dict[str, List[str]] = {}
    for coin in data:
        symbol = str(coin.get("symbol", "")).upper()
        coin_id = coin.get("id")
        if not symbol or not coin_id:
            continue
        symbol_map.setdefault(symbol, []).append(coin_id)

    _COIN_LIST_CACHE["symbols"] = symbol_map
    _COIN_LIST_CACHE["timestamp"] = now
    return symbol_map


def _resolve_coin_id(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if symbol in _COIN_ID_OVERRIDES:
        return _COIN_ID_OVERRIDES[symbol]

    symbol_map = _fetch_coin_list()
    candidates = symbol_map.get(symbol, [])
    if not candidates:
        raise Exception(f"Symbol {symbol} not found on CoinGecko")

    preferred = _COIN_ID_PREFERENCES.get(symbol, [])
    for pref in preferred:
        if pref in candidates:
            return pref

    return candidates[0]


def _fetch_simple_prices(coin_ids: List[str]) -> Dict[str, Dict[str, float]]:
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": ",".join(coin_ids), "vs_currencies": "usd"}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def get_coingecko_price(pair: str) -> float:
    """
    Fetch the current price for a trading pair from CoinGecko.

    Args:
        pair: Trading pair symbol (e.g., 'BTC/USDT', 'ETH/SOL')

    Returns:
        Current price as a float

    Raises:
        Exception: If price fetching fails
    """
    base, quote = pair.split('/')
    base_symbol = base.strip().upper()
    quote_symbol = quote.strip().upper()

    if base_symbol == quote_symbol:
        return 1.0

    base_id = _resolve_coin_id(base_symbol) if base_symbol != "USD" else None
    quote_id = _resolve_coin_id(quote_symbol) if quote_symbol != "USD" else None

    coin_ids = []
    if base_id:
        coin_ids.append(base_id)
    if quote_id and quote_id not in coin_ids:
        coin_ids.append(quote_id)

    try:
        prices = _fetch_simple_prices(coin_ids) if coin_ids else {}
        base_usd = 1.0 if base_symbol == "USD" else float(prices[base_id]["usd"])
        quote_usd = 1.0 if quote_symbol == "USD" else float(prices[quote_id]["usd"])
        return base_usd / quote_usd
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch CoinGecko price for {pair}: {str(e)}")
    except (KeyError, ValueError) as e:
        raise Exception(f"Failed to parse CoinGecko response for {pair}: {str(e)}")
