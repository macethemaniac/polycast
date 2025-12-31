"""
Main entry point for the arbitrage scanner MVP.

This script fetches BTC/USDT prices from CoinGecko and DeFiLlama data sources,
performs an arbitrage check, and prints the results to the console.
"""

from exchanges.coingecko import get_coingecko_price
from exchanges.defillama import get_defillama_price
from analytics.arbitrage import check_arbitrage


def main():
    """
    Main function that orchestrates the arbitrage check.
    
    Fetches BTC/USDT prices from CoinGecko and DeFiLlama, then calculates
    and displays the arbitrage opportunity.
    """
    pair = 'BTC/USDT'
    
    print(f"Fetching {pair} prices from data sources...")
    print("-" * 50)
    
    try:
        # Fetch prices from both data sources
        coingecko_price = get_coingecko_price(pair)
        defillama_price = get_defillama_price(pair)
        
        print(f"CoinGecko {pair}: ${coingecko_price:,.2f}")
        print(f"DeFiLlama {pair}: ${defillama_price:,.2f}")
        print("-" * 50)
        
        # Check for arbitrage opportunity
        prices = {
            'coingecko': coingecko_price,
            'defillama': defillama_price,
        }
        
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

