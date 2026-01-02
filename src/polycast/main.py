"""
Main entry point for the arbitrage scanner MVP.

This script fetches BTC/USDT prices from CCXT exchanges, performs an arbitrage
check, and prints the results to the console.
"""

from exchanges.ccxt_client import get_ccxt_prices, DEFAULT_EXCHANGES
from polycast.analytics.arbitrage import check_arbitrage


def main():
    """
    Main function that orchestrates the arbitrage check.
    
    Fetches BTC/USDT prices from CCXT exchanges, then calculates
    and displays the arbitrage opportunity.
    """
    pair = 'BTC/USDT'
    
    print(f"Fetching {pair} prices from data sources...")
    print("-" * 50)
    
    try:
        prices = get_ccxt_prices(pair, DEFAULT_EXCHANGES)
        for exchange, price in prices.items():
            print(f"{exchange} {pair}: ${price:,.2f}")
        print("-" * 50)
        
        arbitrage_result = check_arbitrage(prices)
        
        # Display results
        print("\nArbitrage Analysis:")
        print(f"  Buy on:  {arbitrage_result['buy_exchange'].upper()} at ${arbitrage_result['buy_price']:,.2f}")
        print(f"  Sell on: {arbitrage_result['sell_exchange'].upper()} at ${arbitrage_result['sell_price']:,.2f}")
        print(f"  Spread:  ${arbitrage_result['spread']:,.2f} ({arbitrage_result['spread_percent']:.4f}%)")
        
        if arbitrage_result['spread_percent'] > 0:
            print(f"\n[+] Arbitrage opportunity detected!")
        else:
            print(f"\n[-] No arbitrage opportunity (prices are equal)")
            
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print("\nFull error details:")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
