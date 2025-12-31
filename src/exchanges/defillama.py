"""
DeFiLlama data source integration module.

This module provides functionality to fetch cryptocurrency prices from DeFiLlama
using their public API.
"""

from typing import Dict, List
import requests

from exchanges.coingecko import _resolve_coin_id


def _defillama_id(coin_id: str) -> str:
    if coin_id.startswith("coingecko:"):
        return coin_id
    return f"coingecko:{coin_id}"


def _fetch_defillama_prices(coin_ids: List[str]) -> Dict[str, Dict[str, float]]:
    url = f"https://coins.llama.fi/prices/current/{','.join(coin_ids)}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    if "coins" not in data:
        raise Exception("Invalid response format from DeFiLlama")
    return data["coins"]


def get_defillama_price(pair: str) -> float:
    """
    Fetch the current price for a trading pair from DeFiLlama.

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

    defillama_ids = []
    if base_id:
        defillama_ids.append(_defillama_id(base_id))
    if quote_id:
        defillama_id = _defillama_id(quote_id)
        if defillama_id not in defillama_ids:
            defillama_ids.append(defillama_id)

    try:
        prices = _fetch_defillama_prices(defillama_ids) if defillama_ids else {}
        base_usd = (
            1.0
            if base_symbol == "USD"
            else float(prices[_defillama_id(base_id)]["price"])
        )
        quote_usd = (
            1.0
            if quote_symbol == "USD"
            else float(prices[_defillama_id(quote_id)]["price"])
        )
        return base_usd / quote_usd
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch DeFiLlama price for {pair}: {str(e)}")
    except (KeyError, ValueError) as e:
        raise Exception(f"Failed to parse DeFiLlama response for {pair}: {str(e)}")
