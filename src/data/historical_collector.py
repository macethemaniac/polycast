"""Historical data collection for Polymarket.

Fetches resolved and active markets for ML training data.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from src.utils.rate_limiter import get_with_retry, RateLimitError
from src.data.schemas import HistoricalMarket, MarketSnapshot, PricePoint, generate_id

logger = logging.getLogger(__name__)

# Polymarket Gamma API
BASE_URL = "https://gamma-api.polymarket.com"
TIMEOUT = 15

# Data storage
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "historical"
DATA_DIR.mkdir(parents=True, exist_ok=True)

POLYMARKET_MARKETS_FILE = DATA_DIR / "polymarket_markets.jsonl"
POLYMARKET_SNAPSHOTS_FILE = DATA_DIR / "polymarket_snapshots.jsonl"


class PolymarketHistoricalCollector:
    """Collects historical market data from Polymarket."""

    def __init__(self):
        self.markets_cache: Dict[str, HistoricalMarket] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing markets from disk."""
        if POLYMARKET_MARKETS_FILE.exists():
            try:
                with open(POLYMARKET_MARKETS_FILE, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            market = HistoricalMarket.from_dict(data)
                            self.markets_cache[market.market_id] = market
                logger.info(f"Loaded {len(self.markets_cache)} existing Polymarket markets")
            except Exception as e:
                logger.error(f"Error loading existing markets: {e}")

    def _save_market(self, market: HistoricalMarket) -> None:
        """Append market to storage."""
        try:
            with open(POLYMARKET_MARKETS_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(market.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Error saving market: {e}")

    def _save_snapshot(self, snapshot: MarketSnapshot) -> None:
        """Append snapshot to storage."""
        try:
            with open(POLYMARKET_SNAPSHOTS_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(snapshot.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")

    def fetch_all_markets(
        self,
        include_closed: bool = True,
        limit: int = 1000,
        offset: int = 0
    ) -> List[Dict]:
        """Fetch markets from Polymarket API."""
        all_markets = []
        current_offset = offset
        batch_size = 100

        while len(all_markets) < limit:
            try:
                params = {
                    "limit": min(batch_size, limit - len(all_markets)),
                    "offset": current_offset,
                }
                if not include_closed:
                    params["closed"] = "false"
                    params["active"] = "true"

                url = f"{BASE_URL}/markets"
                resp = get_with_retry(url, params=params, timeout=TIMEOUT, max_retries=3)

                if not resp.ok:
                    logger.warning(f"API returned {resp.status_code}")
                    break

                data = resp.json()
                if not data:
                    break

                markets = data if isinstance(data, list) else data.get('markets', [])
                if not markets:
                    break

                all_markets.extend(markets)
                current_offset += len(markets)

                logger.info(f"Fetched {len(all_markets)} markets so far...")

                if len(markets) < batch_size:
                    break

                time.sleep(0.5)  # Rate limiting

            except RateLimitError as e:
                logger.warning(f"Rate limited: {e}")
                time.sleep(60)
            except Exception as e:
                logger.error(f"Error fetching markets: {e}")
                break

        return all_markets

    def fetch_resolved_markets(self, limit: int = 500) -> List[Dict]:
        """Fetch resolved (closed) markets."""
        all_markets = []
        current_offset = 0
        batch_size = 100

        while len(all_markets) < limit:
            try:
                params = {
                    "limit": min(batch_size, limit - len(all_markets)),
                    "offset": current_offset,
                    "closed": "true",
                }

                url = f"{BASE_URL}/markets"
                resp = get_with_retry(url, params=params, timeout=TIMEOUT, max_retries=3)

                if not resp.ok:
                    break

                data = resp.json()
                markets = data if isinstance(data, list) else data.get('markets', [])

                if not markets:
                    break

                all_markets.extend(markets)
                current_offset += len(markets)

                logger.info(f"Fetched {len(all_markets)} resolved markets...")

                if len(markets) < batch_size:
                    break

                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error fetching resolved markets: {e}")
                break

        return all_markets

    def normalize_market(self, raw: Dict) -> Optional[HistoricalMarket]:
        """Normalize raw API response to HistoricalMarket."""
        try:
            market_id = raw.get("id") or raw.get("_id") or raw.get("conditionId")
            if not market_id:
                return None

            question = raw.get("question") or raw.get("title") or ""
            description = raw.get("description") or ""

            # Parse prices
            outcome_prices = raw.get("outcomePrices") or raw.get("prices") or "[]"
            if isinstance(outcome_prices, str):
                try:
                    prices = json.loads(outcome_prices)
                except:
                    prices = [0.5, 0.5]
            else:
                prices = outcome_prices

            yes_price = float(prices[0]) if len(prices) > 0 else 0.5
            no_price = float(prices[1]) if len(prices) > 1 else 0.5

            # Volume
            volume = float(raw.get("volume") or raw.get("volumeNum") or 0)
            liquidity = float(raw.get("liquidity") or raw.get("liquidityNum") or 0)

            # Timestamps
            created_at = raw.get("createdAt") or raw.get("created_at")
            close_time = raw.get("endDate") or raw.get("end_date") or raw.get("closeTime")
            resolved_at = raw.get("resolvedAt") or raw.get("resolved_at")

            # Parse timestamps
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00')).timestamp()
                except:
                    created_at = None

            if isinstance(close_time, str):
                try:
                    close_time = datetime.fromisoformat(close_time.replace('Z', '+00:00')).timestamp()
                except:
                    close_time = None

            if isinstance(resolved_at, str):
                try:
                    resolved_at = datetime.fromisoformat(resolved_at.replace('Z', '+00:00')).timestamp()
                except:
                    resolved_at = None

            # Resolution
            resolution = "pending"
            if raw.get("closed") or raw.get("resolved"):
                resolution_value = raw.get("resolution") or raw.get("resolutionValue")
                if resolution_value is not None:
                    if str(resolution_value).lower() in ["yes", "1", "true"]:
                        resolution = "yes"
                    elif str(resolution_value).lower() in ["no", "0", "false"]:
                        resolution = "no"
                    else:
                        resolution = "cancelled"

            # Category and tags
            category = raw.get("category") or ""
            tags = raw.get("tags") or raw.get("tickers") or []
            if isinstance(tags, str):
                tags = [tags]

            # Create price point
            price_point = PricePoint(
                timestamp=time.time(),
                yes_price=yes_price,
                no_price=no_price,
                volume=volume,
                liquidity=liquidity
            )

            return HistoricalMarket(
                market_id=str(market_id),
                platform="polymarket",
                question=question,
                description=description,
                outcomes=["YES", "NO"],
                price_history=[price_point],
                resolution=resolution,
                resolved_at=resolved_at,
                created_at=created_at,
                close_time=close_time,
                category=category,
                tags=tags,
                final_yes_price=yes_price,
                final_no_price=no_price,
                total_volume=volume
            )

        except Exception as e:
            logger.debug(f"Error normalizing market: {e}")
            return None

    def collect_historical_data(
        self,
        include_resolved: bool = True,
        include_active: bool = True,
        max_markets: int = 2000
    ) -> int:
        """Collect historical market data."""
        new_count = 0

        if include_resolved:
            logger.info("Fetching resolved markets...")
            resolved = self.fetch_resolved_markets(limit=max_markets // 2)

            for raw in resolved:
                market = self.normalize_market(raw)
                if market and market.market_id not in self.markets_cache:
                    self.markets_cache[market.market_id] = market
                    self._save_market(market)
                    new_count += 1

            logger.info(f"Added {new_count} resolved markets")

        if include_active:
            logger.info("Fetching active markets...")
            active = self.fetch_all_markets(include_closed=False, limit=max_markets // 2)

            active_count = 0
            for raw in active:
                market = self.normalize_market(raw)
                if market and market.market_id not in self.markets_cache:
                    self.markets_cache[market.market_id] = market
                    self._save_market(market)
                    active_count += 1
                    new_count += 1

            logger.info(f"Added {active_count} active markets")

        logger.info(f"Total markets in cache: {len(self.markets_cache)}")
        return new_count

    def take_snapshot(self) -> int:
        """Take snapshot of current active markets."""
        logger.info("Taking market snapshot...")
        markets = self.fetch_all_markets(include_closed=False, limit=500)
        snapshot_count = 0

        for raw in markets:
            try:
                market_id = raw.get("id") or raw.get("_id")
                if not market_id:
                    continue

                # Parse prices
                outcome_prices = raw.get("outcomePrices") or "[]"
                if isinstance(outcome_prices, str):
                    prices = json.loads(outcome_prices)
                else:
                    prices = outcome_prices

                yes_price = float(prices[0]) if len(prices) > 0 else 0.5
                no_price = float(prices[1]) if len(prices) > 1 else 0.5

                snapshot = MarketSnapshot(
                    market_id=str(market_id),
                    platform="polymarket",
                    question=raw.get("question") or "",
                    description=raw.get("description") or "",
                    yes_price=yes_price,
                    no_price=no_price,
                    volume=float(raw.get("volume") or 0),
                    liquidity=float(raw.get("liquidity") or 0),
                    category=raw.get("category") or "",
                    snapshot_ts=time.time()
                )

                self._save_snapshot(snapshot)
                snapshot_count += 1

                # Update cache if market exists
                if market_id in self.markets_cache:
                    market = self.markets_cache[market_id]
                    market.price_history.append(PricePoint(
                        timestamp=time.time(),
                        yes_price=yes_price,
                        no_price=no_price,
                        volume=float(raw.get("volume") or 0),
                        liquidity=float(raw.get("liquidity") or 0)
                    ))

            except Exception as e:
                logger.debug(f"Error creating snapshot: {e}")
                continue

        logger.info(f"Saved {snapshot_count} snapshots")
        return snapshot_count

    def get_market(self, market_id: str) -> Optional[HistoricalMarket]:
        """Get market from cache."""
        return self.markets_cache.get(market_id)

    def get_all_markets(self) -> List[HistoricalMarket]:
        """Get all cached markets."""
        return list(self.markets_cache.values())

    def get_resolved_markets(self) -> List[HistoricalMarket]:
        """Get only resolved markets."""
        return [m for m in self.markets_cache.values() if m.resolution != "pending"]

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        markets = list(self.markets_cache.values())
        resolved = [m for m in markets if m.resolution != "pending"]
        pending = [m for m in markets if m.resolution == "pending"]

        return {
            "total_markets": len(markets),
            "resolved_markets": len(resolved),
            "pending_markets": len(pending),
            "yes_resolutions": len([m for m in resolved if m.resolution == "yes"]),
            "no_resolutions": len([m for m in resolved if m.resolution == "no"]),
            "cancelled": len([m for m in resolved if m.resolution == "cancelled"]),
            "total_volume": sum(m.total_volume for m in markets),
            "categories": list(set(m.category for m in markets if m.category)),
        }


# Convenience functions
_collector: Optional[PolymarketHistoricalCollector] = None


def get_collector() -> PolymarketHistoricalCollector:
    """Get or create singleton collector."""
    global _collector
    if _collector is None:
        _collector = PolymarketHistoricalCollector()
    return _collector


def collect_polymarket_history(max_markets: int = 2000) -> int:
    """Collect Polymarket historical data."""
    collector = get_collector()
    return collector.collect_historical_data(max_markets=max_markets)


def take_polymarket_snapshot() -> int:
    """Take snapshot of current Polymarket state."""
    collector = get_collector()
    return collector.take_snapshot()


def get_polymarket_stats() -> Dict[str, Any]:
    """Get Polymarket collection statistics."""
    collector = get_collector()
    return collector.get_stats()
