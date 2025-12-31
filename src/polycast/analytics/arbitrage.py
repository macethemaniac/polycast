"""
Arbitrage analytics module.

This module provides functionality to calculate arbitrage opportunities
between different exchange prices.
"""

from typing import Dict


def check_arbitrage(prices: Dict[str, float]) -> Dict[str, float]:
    """
    Check for arbitrage opportunity between exchange prices.
    
    Calculates the price spread and percentage difference between the
    highest and lowest prices from different exchanges.
    
    Args:
        prices: Dictionary mapping exchange names to their prices
                (e.g., {'binance': 50000.0, 'bybit': 50100.0})
        
    Returns:
        Dictionary containing:
        - 'spread': Absolute price difference (highest - lowest)
        - 'spread_percent': Percentage difference ((spread / lowest) * 100)
        - 'buy_exchange': Exchange with lowest price (where to buy)
        - 'sell_exchange': Exchange with highest price (where to sell)
        - 'buy_price': Lowest price
        - 'sell_price': Highest price
    """
    if not prices or len(prices) < 2:
        raise ValueError("Need at least 2 prices to check arbitrage")
    
    # Find the highest and lowest prices
    sorted_prices = sorted(prices.items(), key=lambda x: x[1])
    buy_exchange, buy_price = sorted_prices[0]
    sell_exchange, sell_price = sorted_prices[-1]
    
    # Calculate spread
    spread = sell_price - buy_price
    spread_percent = (spread / buy_price) * 100 if buy_price > 0 else 0.0
    
    return {
        'spread': spread,
        'spread_percent': spread_percent,
        'buy_exchange': buy_exchange,
        'sell_exchange': sell_exchange,
        'buy_price': buy_price,
        'sell_price': sell_price,
    }

