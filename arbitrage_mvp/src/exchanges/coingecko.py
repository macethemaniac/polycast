"""
CoinGecko data source integration module.

This module provides functionality to fetch cryptocurrency prices from CoinGecko
using their public API.
"""

import requests


def get_coingecko_price(pair: str) -> float:
    """
    Fetch the current price for a trading pair from CoinGecko.
    
    Args:
        pair: Trading pair symbol (e.g., 'BTC/USDT', 'BTC/USD')
              Currently supports BTC/USD or BTC/USDT
        
    Returns:
        Current price as a float
        
    Raises:
        Exception: If price fetching fails
    """
    # Parse the pair (e.g., 'BTC/USDT' -> 'bitcoin', 'usd')
    base, quote = pair.split('/')
    
    # CoinGecko uses 'bitcoin' as the coin ID for BTC
    coin_id_map = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
    }
    
    # Convert quote currency to CoinGecko format
    quote_map = {
        'USDT': 'usd',  # CoinGecko treats USDT price as USD
        'USD': 'usd',
    }
    
    coin_id = coin_id_map.get(base.upper(), base.lower())
    currency = quote_map.get(quote.upper(), quote.lower())
    
    url = 'https://api.coingecko.com/api/v3/simple/price'
    params = {
        'ids': coin_id,
        'vs_currencies': currency
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if coin_id not in data:
            raise Exception(f"Coin {coin_id} not found in CoinGecko response")
        
        if currency not in data[coin_id]:
            raise Exception(f"Currency {currency} not found for {coin_id}")
        
        return float(data[coin_id][currency])
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch CoinGecko price for {pair}: {str(e)}")
    except (KeyError, ValueError) as e:
        raise Exception(f"Failed to parse CoinGecko response for {pair}: {str(e)}")

