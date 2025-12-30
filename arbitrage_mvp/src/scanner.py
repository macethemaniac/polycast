"""
Shared scanner module for arbitrage checking.

This module provides a unified interface for checking arbitrage opportunities
that can be used by both console and Telegram bot interfaces.
"""

from typing import Dict, Optional, Tuple
from exchanges.coingecko import get_coingecko_price
from exchanges.defillama import get_defillama_price
from analytics.arbitrage import check_arbitrage


def scan_arbitrage(pair: str = 'BTC/USDT') -> Tuple[Optional[Dict], Optional[str]]:
    """
    Scan for arbitrage opportunities for a given trading pair.
    
    Args:
        pair: Trading pair symbol (e.g., 'BTC/USDT', 'ETH/USDT')
        
    Returns:
        Tuple of (arbitrage_result_dict, error_message)
        If successful: (result_dict, None)
        If error: (None, error_string)
    """
    try:
        # Fetch prices from both data sources
        coingecko_price = get_coingecko_price(pair)
        defillama_price = get_defillama_price(pair)
        
        # Check for arbitrage opportunity
        prices = {
            'coingecko': coingecko_price,
            'defillama': defillama_price,
        }
        
        arbitrage_result = check_arbitrage(prices)
        
        # Add raw prices to result for convenience
        arbitrage_result['coingecko_price'] = coingecko_price
        arbitrage_result['defillama_price'] = defillama_price
        arbitrage_result['pair'] = pair
        
        return arbitrage_result, None
        
    except Exception as e:
        return None, str(e)


