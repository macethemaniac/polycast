#!/usr/bin/env python3
"""CLI runner for cross-market arbitrage checks.

Usage examples:
  python scripts/run_cross_arb.py --pol 100 --kal 100 --min-sim 0.3
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cross_arb import find_cross_market_arbitrage


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pol', type=int, default=50, help='Polymarket limit')
    p.add_argument('--kal', type=int, default=50, help='Kalshi limit')
    p.add_argument('--threshold', type=float, default=1.02, help='Arbitrage threshold')
    p.add_argument('--min-sim', type=float, default=0.4, help='Minimum title similarity (0-1)')
    args = p.parse_args()

    res = find_cross_market_arbitrage(limit_pol=args.pol, limit_kal=args.kal, threshold=args.threshold, min_similarity=args.min_sim)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
