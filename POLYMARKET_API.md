# Polymarket Gamma API Integration Guide

## Overview

The Polymarket Gamma API is a GraphQL-based API that provides access to prediction market data, including:
- Market information and prices
- Event data
- Order book (CLOB) data
- Market statistics (liquidity, volume)

## Base URL

```
https://gamma-api.polymarket.com
```

## Authentication

The API may require authentication for certain endpoints. Authentication can be done via:
- API Key (Bearer token in Authorization header)
- Cookies (for browser-based access)

**Note:** Some endpoints may work without authentication, but full access typically requires an API key.

## Key Endpoints

### GraphQL Query Endpoint

**Endpoint:** `POST /query`

**Content-Type:** `application/json`

**Request Format:**
```json
{
  "query": "GraphQL query string",
  "variables": {
    "variableName": "value"
  }
}
```

## Common Queries

### 1. Get Markets

Fetch a list of markets with their basic information.

**Query:**
```graphql
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
```

**Variables:**
```json
{
  "limit": 10,
  "offset": 0,
  "active": true
}
```

### 2. Get Market by ID

Fetch detailed information about a specific market.

**Query:**
```graphql
query GetMarket($id: String!) {
  market(id: $id) {
    id
    question
    slug
    outcomes {
      id
      title
      price
      volume
    }
    liquidity
    volume
    active
    endDate
  }
}
```

### 3. Get Market Prices

Get current prices for all outcomes in a market.

**Query:**
```graphql
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
```

### 4. Search Markets

Search for markets by query text.

**Query:**
```graphql
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
```

### 5. Get Events

Fetch events and their associated markets.

**Query:**
```graphql
query GetEvents($limit: Int, $offset: Int) {
  events(limit: $limit, offset: $offset) {
    id
    title
    slug
    description
    startDate
    endDate
    markets {
      id
      question
      outcomes
    }
  }
}
```

## Data Structures

### Market Object

```json
{
  "id": "string",
  "question": "string",
  "slug": "string",
  "outcomes": [
    {
      "id": "string",
      "title": "string",
      "price": 0.5,
      "volume": 1000.0
    }
  ],
  "image": "url",
  "endDate": "ISO8601",
  "liquidity": 10000.0,
  "volume": 5000.0,
  "marketMakerAddress": "0x...",
  "active": true,
  "archived": false
}
```

### Outcome Object

```json
{
  "id": "string",
  "title": "string",
  "price": 0.5,
  "volume": 1000.0
}
```

**Note:** Prices in Polymarket are typically in the range 0-1, representing probability or likelihood.

## Integration for Arbitrage Scanner

### Use Cases

1. **Cross-Market Arbitrage:**
   - Compare prices for similar questions across different markets
   - Find discrepancies in market maker prices

2. **Outcome Arbitrage:**
   - Within a single market, ensure outcomes sum to 1.0 (or expected value)
   - Find mispriced outcomes

3. **Cross-Platform Arbitrage:**
   - Compare Polymarket prices with other prediction markets
   - Compare with traditional betting odds

4. **Time-Based Arbitrage:**
   - Track price movements over time
   - Identify early market entry opportunities

### Implementation Notes

- **Price Format:** Polymarket prices are probabilities (0-1), convert to percentages for comparison
- **Liquidity:** Check liquidity before considering arbitrage (low liquidity = high slippage risk)
- **Market Status:** Only consider active markets (archived markets may not be tradeable)
- **Rate Limits:** Be mindful of API rate limits, especially when polling frequently

## Example Python Usage

```python
from exchanges.polymarket import PolymarketAPI

# Initialize API (with or without key)
api = PolymarketAPI()  # or PolymarketAPI(api_key="your_key")

# Get markets
markets = api.get_markets(limit=10, active=True)

# Get prices for a specific market
prices = api.get_market_prices(market_id="0x...")

# Search for markets
results = api.search_markets("bitcoin", limit=5)
```

## Rate Limits

Check Polymarket documentation for current rate limits. Implement:
- Request throttling
- Exponential backoff for errors
- Caching for frequently accessed data

## Error Handling

Common errors:
- `401 Unauthorized` - Missing or invalid authentication
- `400 Bad Request` - Invalid GraphQL query
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server-side issue

Implement proper error handling and retry logic.

## Next Steps

1. Test API access (with/without authentication)
2. Explore available queries via schema introspection
3. Identify specific arbitrage opportunities
4. Integrate with existing arbitrage scanner
5. Add real-time price monitoring
6. Implement alerting for arbitrage opportunities

