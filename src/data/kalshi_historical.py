"""Historical data collection for Kalshi.

Fetches resolved and active markets for ML training data.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from src.utils.rate_limiter import get_with_retry, RateLimitError
from src.data.schemas import HistoricalMarket, MarketSnapshot, PricePoint, generate_id

logger = logging.getLogger(__name__)

# Kalshi API endpoints
API_URLS = [
    "https://api.elections.kalshi.com/trade-api/v2",
    "https://api.kalshi.com/v1",
]
TIMEOUT = 15

# Data storage
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "historical"
DATA_DIR.mkdir(parents=True, exist_ok=True)

KALSHI_MARKETS_FILE = DATA_DIR / "kalshi_markets.jsonl"
KALSHI_SNAPSHOTS_FILE = DATA_DIR / "kalshi_snapshots.jsonl"


def _get_headers() -> Dict[str, str]:
    """Get headers for Kalshi API requests."""
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }
    api_key = os.getenv('KALSHI_API_KEY')
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    return headers


class KalshiHistoricalCollector:
    """Collects historical market data from Kalshi."""

    def __init__(self):
        self.markets_cache: Dict[str, HistoricalMarket] = {}
        self.base_url = API_URLS[0]  # Default to elections API
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing markets from disk."""
        if KALSHI_MARKETS_FILE.exists():
            try:
                with open(KALSHI_MARKETS_FILE, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            market = HistoricalMarket.from_dict(data)
                            self.markets_cache[market.market_id] = market
                logger.info(f"Loaded {len(self.markets_cache)} existing Kalshi markets")
            except Exception as e:
                logger.error(f"Error loading existing markets: {e}")

    def _save_market(self, market: HistoricalMarket) -> None:
        """Append market to storage."""
        try:
            with open(KALSHI_MARKETS_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(market.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Error saving market: {e}")

    def _save_snapshot(self, snapshot: MarketSnapshot) -> None:
        """Append snapshot to storage."""
        try:
            with open(KALSHI_SNAPSHOTS_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(snapshot.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")

    def _try_fetch(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Try fetching from multiple API URLs."""
        for base_url in API_URLS:
            try:
                url = f"{base_url}/{endpoint}"
                resp = get_with_retry(
                    url,
                    headers=_get_headers(),
                    params=params,
                    timeout=TIMEOUT,
                    max_retries=2
                )

                if resp.ok:
                    self.base_url = base_url
                    return resp.json()

            except RateLimitError:
                logger.warning(f"Rate limited on {base_url}")
                continue
            except Exception as e:
                logger.debug(f"Error fetching from {base_url}: {e}")
                continue

        return None

    def fetch_all_markets(
        self,
        status: str = "open",
        limit: int = 500,
        exclude_mve: bool = False
    ) -> List[Dict]:
        """Fetch markets from Kalshi API.

        Args:
            status: Market status ("open", "settled", "closed")
            limit: Maximum markets to fetch
            exclude_mve: If True, filter out multivariate (combo) markets
        """
        all_markets = []
        cursor = None

        while len(all_markets) < limit:
            try:
                params = {
                    "limit": min(100, limit - len(all_markets)),
                    "status": status,
                }
                if cursor:
                    params["cursor"] = cursor

                data = self._try_fetch("markets", params)
                if not data:
                    break

                markets = data.get("markets", [])
                if not markets:
                    break

                # Filter out MVE markets if requested
                if exclude_mve:
                    markets = [m for m in markets if not m.get("ticker", "").startswith("KXMV")]

                all_markets.extend(markets)
                logger.info(f"Fetched {len(all_markets)} Kalshi markets...")

                cursor = data.get("cursor")
                if not cursor or len(markets) < 100:
                    break

                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error fetching markets: {e}")
                break

        return all_markets

    def fetch_events(self, limit: int = 200) -> List[Dict]:
        """Fetch events (which contain multiple markets)."""
        all_events = []
        cursor = None

        while len(all_events) < limit:
            try:
                params = {"limit": min(50, limit - len(all_events))}
                if cursor:
                    params["cursor"] = cursor

                data = self._try_fetch("events", params)
                if not data:
                    break

                events = data.get("events", [])
                if not events:
                    break

                all_events.extend(events)
                logger.info(f"Fetched {len(all_events)} Kalshi events...")

                cursor = data.get("cursor")
                if not cursor:
                    break

                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error fetching events: {e}")
                break

        return all_events

    def normalize_market(self, raw: Dict) -> Optional[HistoricalMarket]:
        """Normalize raw API response to HistoricalMarket."""
        try:
            market_id = raw.get("ticker") or raw.get("id") or raw.get("market_ticker")
            if not market_id:
                return None

            # Get title - prefer yes_sub_title or title over no_sub_title
            question = raw.get("title") or raw.get("yes_sub_title") or raw.get("question") or ""
            if not question or question.startswith("yes "):
                # For MVE markets, construct a readable title
                event_ticker = raw.get("event_ticker") or ""
                if event_ticker:
                    question = f"{event_ticker}: {raw.get('subtitle', question)}"
            description = raw.get("rules") or raw.get("description") or ""

            # Parse prices (Kalshi uses cents, 0-100)
            yes_price = raw.get("yes_ask") or raw.get("last_price") or raw.get("yes_bid") or 50
            no_price = raw.get("no_ask") or raw.get("no_bid") or (100 - yes_price)

            # Convert from cents to decimal
            if yes_price > 1:
                yes_price = yes_price / 100.0
            if no_price > 1:
                no_price = no_price / 100.0

            # Volume
            volume = float(raw.get("volume") or raw.get("dollar_volume") or 0)
            liquidity = float(raw.get("open_interest") or raw.get("liquidity") or 0)

            # Timestamps
            created_at = raw.get("open_time") or raw.get("created_time")
            close_time = raw.get("close_time") or raw.get("expiration_time")
            resolved_at = raw.get("result_time") or raw.get("determination_time")

            # Parse ISO timestamps
            for ts_var in ['created_at', 'close_time', 'resolved_at']:
                ts_val = locals().get(ts_var)
                if isinstance(ts_val, str):
                    try:
                        locals()[ts_var] = datetime.fromisoformat(
                            ts_val.replace('Z', '+00:00')
                        ).timestamp()
                    except:
                        locals()[ts_var] = None

            # Resolution
            resolution = "pending"
            status = raw.get("status", "").lower()
            result = raw.get("result") or raw.get("settlement_value")

            if status in ["settled", "closed", "finalized"] or result is not None:
                if result is not None:
                    if str(result).lower() in ["yes", "1", "true"]:
                        resolution = "yes"
                    elif str(result).lower() in ["no", "0", "false"]:
                        resolution = "no"
                    else:
                        resolution = "cancelled"
                else:
                    resolution = "cancelled"

            # Category
            category = raw.get("category") or raw.get("series_ticker") or ""
            tags = [raw.get("event_ticker")] if raw.get("event_ticker") else []

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
                platform="kalshi",
                question=question,
                description=description,
                outcomes=["YES", "NO"],
                price_history=[price_point],
                resolution=resolution,
                resolved_at=resolved_at if isinstance(resolved_at, float) else None,
                created_at=created_at if isinstance(created_at, float) else None,
                close_time=close_time if isinstance(close_time, float) else None,
                category=category,
                tags=tags,
                final_yes_price=yes_price,
                final_no_price=no_price,
                total_volume=volume
            )

        except Exception as e:
            logger.debug(f"Error normalizing Kalshi market: {e}")
            return None

    def collect_historical_data(
        self,
        include_settled: bool = True,
        include_open: bool = True,
        max_markets: int = 1000
    ) -> int:
        """Collect historical market data."""
        new_count = 0

        if include_settled:
            logger.info("Fetching settled Kalshi markets...")
            settled = self.fetch_all_markets(status="settled", limit=max_markets // 2)

            for raw in settled:
                market = self.normalize_market(raw)
                if market and market.market_id not in self.markets_cache:
                    self.markets_cache[market.market_id] = market
                    self._save_market(market)
                    new_count += 1

            logger.info(f"Added {new_count} settled markets")

        if include_open:
            logger.info("Fetching open Kalshi markets...")
            open_markets = self.fetch_all_markets(status="open", limit=max_markets // 2)

            open_count = 0
            for raw in open_markets:
                market = self.normalize_market(raw)
                if market and market.market_id not in self.markets_cache:
                    self.markets_cache[market.market_id] = market
                    self._save_market(market)
                    open_count += 1
                    new_count += 1

            logger.info(f"Added {open_count} open markets")

        logger.info(f"Total Kalshi markets in cache: {len(self.markets_cache)}")
        return new_count

    def take_snapshot(self) -> int:
        """Take snapshot of current open markets."""
        logger.info("Taking Kalshi market snapshot...")
        markets = self.fetch_all_markets(status="open", limit=300)
        snapshot_count = 0

        for raw in markets:
            try:
                market_id = raw.get("ticker") or raw.get("id")
                if not market_id:
                    continue

                yes_price = raw.get("yes_ask") or raw.get("last_price") or 50
                no_price = raw.get("no_ask") or (100 - yes_price)

                if yes_price > 1:
                    yes_price = yes_price / 100.0
                if no_price > 1:
                    no_price = no_price / 100.0

                snapshot = MarketSnapshot(
                    market_id=str(market_id),
                    platform="kalshi",
                    question=raw.get("title") or "",
                    description=raw.get("rules") or "",
                    yes_price=yes_price,
                    no_price=no_price,
                    volume=float(raw.get("volume") or 0),
                    liquidity=float(raw.get("open_interest") or 0),
                    category=raw.get("category") or "",
                    snapshot_ts=time.time()
                )

                self._save_snapshot(snapshot)
                snapshot_count += 1

                # Update cache
                if market_id in self.markets_cache:
                    market = self.markets_cache[market_id]
                    market.price_history.append(PricePoint(
                        timestamp=time.time(),
                        yes_price=yes_price,
                        no_price=no_price,
                        volume=float(raw.get("volume") or 0),
                        liquidity=float(raw.get("open_interest") or 0)
                    ))

            except Exception as e:
                logger.debug(f"Error creating snapshot: {e}")
                continue

        logger.info(f"Saved {snapshot_count} Kalshi snapshots")
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
_collector: Optional[KalshiHistoricalCollector] = None


def get_kalshi_collector() -> KalshiHistoricalCollector:
    """Get or create singleton collector."""
    global _collector
    if _collector is None:
        _collector = KalshiHistoricalCollector()
    return _collector


def collect_kalshi_history(max_markets: int = 1000) -> int:
    """Collect Kalshi historical data."""
    collector = get_kalshi_collector()
    return collector.collect_historical_data(max_markets=max_markets)


def take_kalshi_snapshot() -> int:
    """Take snapshot of current Kalshi state."""
    collector = get_kalshi_collector()
    return collector.take_snapshot()


def get_kalshi_stats() -> Dict[str, Any]:
    """Get Kalshi collection statistics."""
    collector = get_kalshi_collector()
    return collector.get_stats()
