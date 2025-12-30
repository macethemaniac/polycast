"""
Exploratory script to discover Polymarket Gamma API endpoints and data structure.

This script helps understand what data is available from the Polymarket API
for building the arbitrage scanner.
"""

import os
import json
from exchanges.polymarket import PolymarketAPI


def explore_schema():
    """Explore the GraphQL schema to see available queries."""
    api = PolymarketAPI()
    
    print("=" * 60)
    print("Exploring Polymarket Gamma API Schema")
    print("=" * 60)
    
    # Try to get the schema
    schema_query = """
    query {
        __schema {
            queryType {
                name
                fields {
                    name
                    description
                    args {
                        name
                        type {
                            name
                            kind
                        }
                    }
                }
            }
        }
    }
    """
    
    try:
        result = api._query(schema_query)
        print("\n[+] Schema introspection successful!")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"\n[-] Schema introspection failed: {e}")
        print("This might require authentication or a different endpoint.")


def explore_markets():
    """Explore markets endpoint."""
    api = PolymarketAPI()
    
    print("\n" + "=" * 60)
    print("Exploring Markets Endpoint")
    print("=" * 60)
    
    try:
        markets = api.get_markets(limit=5)
        print(f"\n[+] Found {len(markets)} markets")
        
        if markets:
            print("\nSample market structure:")
            print(json.dumps(markets[0], indent=2))
            
            # Try to get prices for the first market
            if markets[0].get('id'):
                market_id = markets[0]['id']
                print(f"\n" + "-" * 60)
                print(f"Getting prices for market: {market_id}")
                print("-" * 60)
                
                prices = api.get_market_prices(market_id)
                if prices:
                    print("\nMarket prices:")
                    for outcome, price in prices.items():
                        print(f"  {outcome}: {price:.4f}")
        
    except Exception as e:
        print(f"\n[-] Markets exploration failed: {e}")
        print("\nNote: This API may require authentication.")
        print("If you have an API key, set it as:")
        print("  export POLYMARKET_API_KEY='your_key_here'")


def explore_events():
    """Explore events endpoint."""
    api = PolymarketAPI()
    
    print("\n" + "=" * 60)
    print("Exploring Events Endpoint")
    print("=" * 60)
    
    try:
        events = api.get_events(limit=3)
        print(f"\n[+] Found {len(events)} events")
        
        if events:
            print("\nSample event structure:")
            print(json.dumps(events[0], indent=2))
    except Exception as e:
        print(f"\n[-] Events exploration failed: {e}")


def explore_search():
    """Explore search functionality."""
    api = PolymarketAPI()
    
    print("\n" + "=" * 60)
    print("Exploring Search Functionality")
    print("=" * 60)
    
    try:
        results = api.search_markets("bitcoin", limit=3)
        print(f"\n[+] Search for 'bitcoin' returned {len(results)} results")
        
        if results:
            print("\nSample search result:")
            print(json.dumps(results[0], indent=2))
    except Exception as e:
        print(f"\n[-] Search exploration failed: {e}")


def main():
    """Main exploration function."""
    print("\n" + "=" * 60)
    print("POLYMARKET GAMMA API EXPLORER")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv('POLYMARKET_API_KEY')
    if api_key:
        print(f"\n[+] API key found (length: {len(api_key)})")
        api = PolymarketAPI(api_key=api_key)
    else:
        print("\n[!] No API key found (using unauthenticated requests)")
        print("Set POLYMARKET_API_KEY environment variable if you have one")
        api = PolymarketAPI()
    
    # Run explorations
    try:
        explore_schema()
    except Exception as e:
        print(f"Schema exploration error: {e}")
    
    try:
        explore_markets()
    except Exception as e:
        print(f"Markets exploration error: {e}")
    
    try:
        explore_events()
    except Exception as e:
        print(f"Events exploration error: {e}")
    
    try:
        explore_search()
    except Exception as e:
        print(f"Search exploration error: {e}")
    
    print("\n" + "=" * 60)
    print("Exploration Complete")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the data structures above")
    print("2. Identify arbitrage opportunities between:")
    print("   - Polymarket prices vs other sources")
    print("   - Different outcomes in the same market")
    print("   - Same markets across different platforms")
    print("3. Integrate with existing arbitrage scanner")


if __name__ == '__main__':
    main()

