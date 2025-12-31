"""
Prediction market arbitrage checker.

Provides utilities to analyze Polymarket outcome prices for arbitrage
opportunities (overround / mispricing among outcomes).
"""
from typing import Dict, List


def check_prediction_arbitrage(prices: Dict[str, float], threshold: float = 0.005) -> Dict:
    """
    Analyze prediction market outcome prices for arbitrage.

    Args:
        prices: Mapping outcome title -> price (0-1)
        threshold: Minimum absolute overround to flag as arbitrage (default 0.5%)

    Returns:
        Dict with keys:
        - 'total': sum of outcome prices
        - 'overround': total - 1.0
        - 'is_arbitrage': bool if abs(overround) > threshold
        - 'threshold': threshold
        - 'outcomes_sorted': list of tuples (outcome, price) sorted by price asc
    """
    if not prices:
        raise ValueError("No prices provided for prediction arbitrage check")

    total = sum(float(p) for p in prices.values())
    overround = total - 1.0
    is_arbitrage = abs(overround) > threshold

    outcomes_sorted = sorted(prices.items(), key=lambda x: x[1])

    return {
        'total': total,
        'overround': overround,
        'is_arbitrage': is_arbitrage,
        'threshold': threshold,
        'outcomes_sorted': outcomes_sorted,
        'raw_prices': prices,
    }
