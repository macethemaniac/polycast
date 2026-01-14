"""Run a daily training data collection session.

Usage: python scripts/daily_training_session.py [hours]
Default: 3 hours of collection at 15-minute intervals
"""

import sys
sys.path.insert(0, '.')

import argparse
import logging
from datetime import datetime

from scripts.collect_snapshots import MarketSnapshotCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_session(duration_hours: float = 3.0, interval_minutes: int = 15):
    """Run a collection session."""
    print("=" * 60)
    print("DAILY TRAINING DATA COLLECTION")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration_hours} hours")
    print(f"Interval: {interval_minutes} minutes")
    print(f"Expected snapshots: {int(duration_hours * 60 / interval_minutes)} rounds")
    print("=" * 60)
    print()

    collector = MarketSnapshotCollector(snapshot_interval=interval_minutes * 60)

    try:
        stats = collector.run_continuous(duration_hours=duration_hours)
        print()
        print("Session complete!")
        print(f"Total snapshots: {stats['total_snapshots']}")
        print(f"Unique markets: {stats['unique_markets']}")
        return stats
    except KeyboardInterrupt:
        print("\nSession interrupted")
        return collector.get_stats()


def main():
    parser = argparse.ArgumentParser(description="Run daily training collection")
    parser.add_argument("hours", type=float, nargs="?", default=3.0,
                        help="Hours to collect (default: 3)")
    parser.add_argument("--interval", type=int, default=15,
                        help="Minutes between snapshots (default: 15)")

    args = parser.parse_args()
    run_session(duration_hours=args.hours, interval_minutes=args.interval)


if __name__ == "__main__":
    main()
