"""
DeFiLlama data source integration module.

This module provides functionality to fetch cryptocurrency prices from DeFiLlama
using their public API.
"""

import requests


def get_defillama_price(pair: str) -> float:
    """
    Fetch the current price for a trading pair from DeFiLlama.
    
    Args:
        pair: Trading pair symbol (e.g., 'BTC/USDT', 'BTC/USD')
              Currently supports BTC/USD or BTC/USDT
        
    Returns:
        Current price as a float
        
    Raises:
        Exception: If price fetching fails
    """
    # Parse the pair (e.g., 'BTC/USDT' -> 'bitcoin')
    base, quote = pair.split('/')
    
    # DeFiLlama uses coin IDs like 'coingecko:bitcoin' or just 'bitcoin'
    coin_id_map = {
        'BTC': 'coingecko:bitcoin',
        'ETH': 'coingecko:ethereum',
    }
    
    coin_id = coin_id_map.get(base.upper(), f'coingecko:{base.lower()}')
    
    url = f'https://coins.llama.fi/prices/current/{coin_id}'
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'coins' not in data:
            raise Exception("Invalid response format from DeFiLlama")
        
        if coin_id not in data['coins']:
            raise Exception(f"Coin {coin_id} not found in DeFiLlama response")
        
        price = data['coins'][coin_id].get('price')
        if price is None:
            raise Exception(f"Price not available for {coin_id}")
        
        return float(price)
    except requests.RequestException as e:
        raise Exception(f"Failed to fetch DeFiLlama price for {pair}: {str(e)}")
    except (KeyError, ValueError) as e:
        raise Exception(f"Failed to parse DeFiLlama response for {pair}: {str(e)}")

