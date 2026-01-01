"""
CCXT data source integration module.

This module provides functionality to fetch cryptocurrency prices from
centralized exchanges via CCXT.
"""

from typing import Dict, Iterable, List
import ccxt

# Expand default CEX coverage beyond two venues for better arbitrage visibility.
DEFAULT_EXCHANGES: List[str] = [
    "binance",
    "kraken",
    "okx",
    "bybit",
    "kucoin",
]


def _normalize_pair(pair: str) -> str:
    """Normalize pair formatting for CCXT (e.g., btc/usdt -> BTC/USDT)."""
    return pair.strip().upper()


def get_ccxt_prices(pair: str, exchanges: Iterable[str]) -> Dict[str, float]:
    """
    Fetch last traded prices for a trading pair from multiple CCXT exchanges.

    Args:
        pair: Trading pair symbol (e.g., 'BTC/USDT', 'ETH/USDC')
        exchanges: Iterable of CCXT exchange ids

    Returns:
        Dictionary mapping exchange ids to last price

    Raises:
        Exception: If no prices can be fetched
    """
    symbol = _normalize_pair(pair)
    prices: Dict[str, float] = {}
    errors: List[str] = []

    for ex_id in exchanges:
        try:
            exchange_class = getattr(ccxt, ex_id)
            exchange = exchange_class({"enableRateLimit": True})
            ticker = exchange.fetch_ticker(symbol)
            price = ticker.get("last") or ticker.get("close")
            if price is None:
                raise Exception("Missing last/close price")
            prices[ex_id] = float(price)
        except Exception as exc:
            errors.append(f"{ex_id}: {exc}")
            continue

    if not prices:
        error_text = "; ".join(errors) if errors else "No prices returned"
        raise Exception(f"Failed to fetch prices via CCXT: {error_text}")

    return prices
