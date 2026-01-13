"""Continuous data collection for ML training.

Collects price snapshots from active markets at regular intervals.
Run this script continuously to build training data.
"""

import sys
sys.path.insert(0, '.')

import json
import logging
import time
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# API
POLYMARKET_API = "https://gamma-api.polymarket.com"
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"


class MarketSnapshotCollector:
    """Collects market snapshots for ML training."""

    def __init__(self, snapshot_interval: int = 3600):
        """
        Args:
            snapshot_interval: Seconds between snapshots (default 1 hour)
        """
        self.snapshot_interval = snapshot_interval
        self.snapshot_count = 0
        self.markets_seen = set()

        # Daily file for snapshots
        self.current_date = None
        self.current_file = None

    def _get_snapshot_file(self) -> Path:
        """Get current snapshot file (one per day)."""
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self.current_date:
            self.current_date = today
            self.current_file = SNAPSHOTS_DIR / f"snapshots_{today}.jsonl"
        return self.current_file

    def fetch_polymarket_markets(self, limit: int = 500) -> List[Dict]:
        """Fetch active Polymarket markets."""
        markets = []
        offset = 0
        batch_size = 100

        while len(markets) < limit:
            try:
                params = {
                    "limit": min(batch_size, limit - len(markets)),
                    "offset": offset,
                    "active": "true",
                    "closed": "false",
                }
                resp = requests.get(
                    f"{POLYMARKET_API}/markets",
                    params=params,
                    timeout=15
                )
                if not resp.ok:
                    logger.warning(f"Polymarket API error: {resp.status_code}")
                    break

                data = resp.json()
                batch = data if isinstance(data, list) else data.get("markets", [])
                if not batch:
                    break

                markets.extend(batch)
                offset += len(batch)

                if len(batch) < batch_size:
                    break

                time.sleep(0.3)  # Rate limiting

            except Exception as e:
                logger.error(f"Error fetching Polymarket: {e}")
                break

        return markets

    def fetch_kalshi_markets(self, limit: int = 200) -> List[Dict]:
        """Fetch active Kalshi markets."""
        markets = []
        try:
            # Kalshi public API
            resp = requests.get(
                f"{KALSHI_API}/markets",
                params={"limit": limit, "status": "open"},
                timeout=15
            )
            if resp.ok:
                data = resp.json()
                markets = data.get("markets", [])
        except Exception as e:
            logger.debug(f"Kalshi fetch error: {e}")
        return markets

    def normalize_polymarket(self, raw: Dict) -> Optional[Dict]:
        """Normalize Polymarket market to snapshot format."""
        try:
            market_id = raw.get("id") or raw.get("conditionId")
            if not market_id:
                return None

            # Parse prices
            outcome_prices = raw.get("outcomePrices") or "[]"
            if isinstance(outcome_prices, str):
                try:
                    prices = json.loads(outcome_prices)
                except:
                    prices = [0.5, 0.5]
            else:
                prices = outcome_prices

            yes_price = float(prices[0]) if len(prices) > 0 else 0.5
            no_price = float(prices[1]) if len(prices) > 1 else 1.0 - yes_price

            return {
                "timestamp": time.time(),
                "market_id": str(market_id),
                "platform": "polymarket",
                "question": raw.get("question", "")[:500],
                "yes_price": yes_price,
                "no_price": no_price,
                "volume": float(raw.get("volume") or raw.get("volumeNum") or 0),
                "liquidity": float(raw.get("liquidity") or raw.get("liquidityNum") or 0),
                "category": raw.get("category", ""),
                "end_date": raw.get("endDate"),
            }
        except Exception as e:
            logger.debug(f"Normalize error: {e}")
            return None

    def normalize_kalshi(self, raw: Dict) -> Optional[Dict]:
        """Normalize Kalshi market to snapshot format."""
        try:
            market_id = raw.get("ticker")
            if not market_id:
                return None

            yes_price = float(raw.get("yes_ask", 50)) / 100
            no_price = float(raw.get("no_ask", 50)) / 100

            return {
                "timestamp": time.time(),
                "market_id": str(market_id),
                "platform": "kalshi",
                "question": raw.get("title", "")[:500],
                "yes_price": yes_price,
                "no_price": no_price,
                "volume": float(raw.get("volume") or 0),
                "liquidity": float(raw.get("open_interest") or 0),
                "category": raw.get("category", ""),
                "end_date": raw.get("close_time"),
            }
        except Exception as e:
            logger.debug(f"Kalshi normalize error: {e}")
            return None

    def save_snapshot(self, snapshot: Dict) -> None:
        """Save snapshot to daily file."""
        file_path = self._get_snapshot_file()
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot) + "\n")

    def collect_snapshot(self) -> int:
        """Collect one round of snapshots from all platforms."""
        count = 0
        timestamp = time.time()

        # Polymarket
        logger.info("Fetching Polymarket markets...")
        poly_markets = self.fetch_polymarket_markets()
        for raw in poly_markets:
            snapshot = self.normalize_polymarket(raw)
            if snapshot:
                self.save_snapshot(snapshot)
                self.markets_seen.add(snapshot["market_id"])
                count += 1

        # Kalshi
        logger.info("Fetching Kalshi markets...")
        kalshi_markets = self.fetch_kalshi_markets()
        for raw in kalshi_markets:
            snapshot = self.normalize_kalshi(raw)
            if snapshot:
                self.save_snapshot(snapshot)
                self.markets_seen.add(snapshot["market_id"])
                count += 1

        self.snapshot_count += 1
        logger.info(f"Snapshot #{self.snapshot_count}: {count} markets captured")
        return count

    def get_stats(self) -> Dict:
        """Get collection statistics."""
        # Count total snapshots
        total_snapshots = 0
        files = list(SNAPSHOTS_DIR.glob("snapshots_*.jsonl"))
        for f in files:
            with open(f, "r") as fp:
                total_snapshots += sum(1 for _ in fp)

        return {
            "total_snapshots": total_snapshots,
            "unique_markets": len(self.markets_seen),
            "snapshot_rounds": self.snapshot_count,
            "snapshot_files": len(files),
        }

    def run_continuous(self, duration_hours: Optional[float] = None):
        """Run continuous collection."""
        logger.info(f"Starting continuous collection (interval: {self.snapshot_interval}s)")
        if duration_hours:
            logger.info(f"Will run for {duration_hours} hours")

        start_time = time.time()
        end_time = start_time + (duration_hours * 3600) if duration_hours else None

        try:
            while True:
                self.collect_snapshot()
                stats = self.get_stats()
                logger.info(f"Stats: {stats['total_snapshots']} total snapshots, {stats['unique_markets']} unique markets")

                if end_time and time.time() >= end_time:
                    logger.info("Duration reached, stopping")
                    break

                logger.info(f"Next snapshot in {self.snapshot_interval}s...")
                time.sleep(self.snapshot_interval)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        final_stats = self.get_stats()
        logger.info(f"Final stats: {final_stats}")
        return final_stats


def collect_once():
    """Collect a single snapshot."""
    collector = MarketSnapshotCollector()
    count = collector.collect_snapshot()
    stats = collector.get_stats()
    print(f"Collected {count} market snapshots")
    print(f"Total snapshots: {stats['total_snapshots']}")
    return count


def run_collector(interval_minutes: int = 60, duration_hours: Optional[float] = None):
    """Run continuous collection."""
    collector = MarketSnapshotCollector(snapshot_interval=interval_minutes * 60)
    return collector.run_continuous(duration_hours=duration_hours)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect market snapshots for ML training")
    parser.add_argument("--once", action="store_true", help="Collect single snapshot and exit")
    parser.add_argument("--interval", type=int, default=60, help="Minutes between snapshots (default: 60)")
    parser.add_argument("--duration", type=float, help="Hours to run (default: indefinite)")

    args = parser.parse_args()

    if args.once:
        collect_once()
    else:
        print("=" * 60)
        print("MARKET SNAPSHOT COLLECTOR")
        print("=" * 60)
        print(f"Interval: {args.interval} minutes")
        print(f"Duration: {args.duration or 'indefinite'} hours")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        run_collector(interval_minutes=args.interval, duration_hours=args.duration)
