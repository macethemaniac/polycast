"""
Shared scanner module for arbitrage checking.

This module provides a unified interface for checking arbitrage opportunities
that can be used by both console and Telegram bot interfaces.
"""

from typing import Dict, Optional, Tuple
from exchanges.ccxt_client import get_ccxt_prices, DEFAULT_EXCHANGES
from polycast.analytics.arbitrage import check_arbitrage


def scan_arbitrage(
    pair: str = 'BTC/USDT',
    exchanges: Optional[list] = None,
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Scan for arbitrage opportunities for a given trading pair.
    
    Args:
        pair: Trading pair symbol (e.g., 'BTC/USDT', 'ETH/USDT')
        exchanges: Optional list of ccxt exchange ids to query
        
    Returns:
        Tuple of (arbitrage_result_dict, error_message)
        If successful: (result_dict, None)
        If error: (None, error_string)
    """
    try:
        exchange_list = exchanges or DEFAULT_EXCHANGES
        prices = get_ccxt_prices(pair, exchange_list)

        if len(prices) < 2:
            raise ValueError(
                f"Need at least 2 exchange prices; got {len(prices)} from {exchange_list}"
            )
        
        # Check for arbitrage opportunity
        arbitrage_result = check_arbitrage(prices)
        
        # Add raw prices to result for convenience
        arbitrage_result['prices'] = prices
        arbitrage_result['pair'] = pair
        
        return arbitrage_result, None
        
    except Exception as e:
        return None, str(e)
