"""
Polymarket Gamma API integration module.

This module provides functionality to fetch market data from Polymarket
using their Gamma API (GraphQL-based).

API Documentation: https://gamma-api.polymarket.com
"""

import os
import requests
from typing import Dict, List, Optional, Any


class PolymarketAPI:
    """
    Client for interacting with Polymarket Gamma API.
    
    The Gamma API uses GraphQL for queries. Common endpoints:
    - Markets data
    - Event information
    - Order book (CLOB) data
    - Price feeds
    """
    
    BASE_URL = "https://gamma-api.polymarket.com"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Polymarket API client.
        
        Args:
            api_key: Optional API key for authenticated requests
        """
        # Accept either passed api_key or environment variables used by other Polymarket clients
        self.api_key = api_key or os.getenv('POLYMARKET_API_KEY') or os.getenv('CLOB_API_KEY') or os.getenv('POLY_API_KEY')
        self.session = requests.Session()
        # Default content type for GraphQL
        self.session.headers.update({'Content-Type': 'application/json'})

        # Set authentication headers in multiple possible forms to maximize compatibility
        if self.api_key:
            # Common GraphQL pattern: Bearer token
            self.session.headers.update({'Authorization': f'Bearer {self.api_key}'})
            # CLOB / internal Polymarket headers used by py-clob-client
            self.session.headers.update({'POLY_API_KEY': self.api_key})
    
    def _query(self, query: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute a GraphQL query against the Polymarket API.
        
        Args:
            query: GraphQL query string
            variables: Optional query variables
            
        Returns:
            JSON response as dictionary
            
        Raises:
            Exception: If the API request fails
        """
        payload = {'query': query}
        if variables:
            payload['variables'] = variables
        
        try:
            response = self.session.post(
                f"{self.BASE_URL}/query",
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            
            if 'errors' in data:
                error_messages = [err.get('message', 'Unknown error') for err in data['errors']]
                raise Exception(f"GraphQL errors: {', '.join(error_messages)}")
            
            return data.get('data', {})
            
        except requests.RequestException as e:
            raise Exception(f"Failed to execute GraphQL query: {str(e)}")
    
    def get_markets(self, limit: int = 10, offset: int = 0, 
                   active: Optional[bool] = True) -> List[Dict]:
        """
        Fetch markets from Polymarket.
        
        Args:
            limit: Maximum number of markets to return
            offset: Number of markets to skip
            active: Filter for active markets only
            
        Returns:
            List of market dictionaries
        """
        Polymarket Gamma API integration module.

        This module provides functionality to fetch market data from Polymarket
        using their Gamma API (GraphQL-based).

        API Documentation: https://gamma-api.polymarket.com
        """

        import requests
        from typing import Dict, List, Optional, Any


        class PolymarketAPI:
            """
            Client for interacting with Polymarket Gamma API.
    
            The Gamma API uses GraphQL for queries. Common endpoints:
            - Markets data
            - Event information
            - Order book (CLOB) data
            - Price feeds
            """
    
            BASE_URL = "https://gamma-api.polymarket.com"
    
            def __init__(self, api_key: Optional[str] = None):
                """
                Initialize Polymarket API client.
        
                Args:
                    api_key: Optional API key for authenticated requests
                """
                self.api_key = api_key
                self.session = requests.Session()
                if api_key:
                    self.session.headers.update({
                        'Authorization': f'Bearer {api_key}',
                        'Content-Type': 'application/json'
                    })

            def _query(self, query: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
                """
                Execute a GraphQL query against the Polymarket API.
        
                Args:
                    query: GraphQL query string
                    variables: Optional query variables
            
                Returns:
                    JSON response as dictionary
            
                Raises:
                    Exception: If the API request fails
                """
                payload = {'query': query}
                if variables:
                    payload['variables'] = variables
        
                try:
                    response = self.session.post(
                        f"{self.BASE_URL}/query",
                        json=payload,
                        timeout=10
                    )
                    response.raise_for_status()
            
                    data = response.json()
            
                    if 'errors' in data:
                        error_messages = [err.get('message', 'Unknown error') for err in data['errors']]
                        raise Exception(f"GraphQL errors: {', '.join(error_messages)}")
            
                    return data.get('data', {})
            
                except requests.RequestException as e:
                    raise Exception(f"Failed to execute GraphQL query: {str(e)}")

            def get_markets(self, limit: int = 10, offset: int = 0, 
                           active: Optional[bool] = True) -> List[Dict]:
                """
                Fetch markets from Polymarket.
        
                Args:
                    limit: Maximum number of markets to return
                    offset: Number of markets to skip
                    active: Filter for active markets only
            
                Returns:
                    List of market dictionaries
                """
                query = """
                query GetMarkets($limit: Int, $offset: Int, $active: Boolean) {
                    markets(limit: $limit, offset: $offset, active: $active) {
                        id
                        question
                        slug
                        outcomes
                        image
                        endDate
                        liquidity
                        volume
                        marketMakerAddress
                        active
                        archived
                    }
                }
                """
        
                variables = {
                    'limit': limit,
                    'offset': offset,
                    'active': active
                }
        
                result = self._query(query, variables)
                return result.get('markets', [])
    
            def get_market_by_id(self, market_id: str) -> Optional[Dict]:
                """
                Fetch a specific market by ID.
        
                Args:
                    market_id: The market ID or slug
            
                Returns:
                    Market dictionary or None if not found
                """
                query = """
                query GetMarket($id: String!) {
                    market(id: $id) {
                        id
                        question
                        slug
                        outcomes
                        image
                        endDate
                        liquidity
                        volume
                        marketMakerAddress
                        active
                        archived
                    }
                }
                """
        
                variables = {'id': market_id}
        
                result = self._query(query, variables)
                return result.get('market')
    
            def get_market_prices(self, market_id: str) -> Dict[str, float]:
                """
                Get current prices for a market's outcomes.
        
                Args:
                    market_id: The market ID
            
                Returns:
                    Dictionary mapping outcome to price (0-1 range)
                """
                query = """
                query GetMarketPrices($id: String!) {
                    market(id: $id) {
                        id
                        outcomes {
                            id
                            title
                            price
                            volume
                        }
                    }
                }
                """
        
                variables = {'id': market_id}
        
                result = self._query(query, variables)
                market = result.get('market')
        
                if not market:
                    return {}
        
                prices = {}
                for outcome in market.get('outcomes', []):
                    prices[outcome.get('title', outcome.get('id'))] = float(outcome.get('price', 0))
        
                return prices
    
            def get_events(self, limit: int = 10, offset: int = 0) -> List[Dict]:
                """
                Fetch events from Polymarket.
        
                Args:
                    limit: Maximum number of events to return
                    offset: Number of events to skip
            
                Returns:
                    List of event dictionaries
                """
                query = """
                query GetEvents($limit: Int, $offset: Int) {
                    events(limit: $limit, offset: $offset) {
                        id
                        title
                        slug
                        description
                        image
                        startDate
                        endDate
                        markets {
                            id
                            question
                            outcomes
                        }
                    }
                }
                """
        
                variables = {
                    'limit': limit,
                    'offset': offset
                }
        
                result = self._query(query, variables)
                return result.get('events', [])
    
            def search_markets(self, query_text: str, limit: int = 10) -> List[Dict]:
                """
                Search for markets by query text.
        
                Args:
                    query_text: Search query string
                    limit: Maximum number of results
            
                Returns:
                    List of matching markets
                """
                query = """
                query SearchMarkets($query: String!, $limit: Int) {
                    searchMarkets(query: $query, limit: $limit) {
                        id
                        question
                        slug
                        outcomes
                        active
                        liquidity
                        volume
                    }
                }
                """
        
                variables = {
                    'query': query_text,
                    'limit': limit
                }
        
                result = self._query(query, variables)
                return result.get('searchMarkets', [])


        def get_polymarket_markets(limit: int = 10, api_key: Optional[str] = None) -> List[Dict]:
            """
            Convenience function to fetch Polymarket markets.
    
            Args:
                limit: Maximum number of markets to return
                api_key: Optional API key for authenticated requests
        
            Returns:
                List of market dictionaries
            """
            api = PolymarketAPI(api_key=api_key)
            return api.get_markets(limit=limit)
    

        def get_polymarket_market_prices(market_id: str, 
                                          api_key: Optional[str] = None) -> Dict[str, float]:
            """
            Convenience function to get market prices.
    
            Args:
                market_id: The market ID
                api_key: Optional API key for authenticated requests
        
            Returns:
                Dictionary mapping outcome to price
            """
            api = PolymarketAPI(api_key=api_key)
            return api.get_market_prices(market_id)

