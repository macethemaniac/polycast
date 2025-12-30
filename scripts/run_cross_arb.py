#!/usr/bin/env python3
"""Workspace-level CLI wrapper for cross-market arbitrage checks.

This script calls the project cross-arb finder by adding the src path
so it can be executed from the repo root as `python scripts/run_cross_arb.py`.
"""
import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
PROJ_ROOT = ROOT.parent
SRC = PROJ_ROOT / 'arbitrage_mvp' / 'src'
sys.path.insert(0, str(SRC))

from cross_arb import find_cross_market_arbitrage


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pol', type=int, default=50, help='Polymarket limit')
    p.add_argument('--kal', type=int, default=50, help='Kalshi limit')
    p.add_argument('--threshold', type=float, default=1.02, help='Arbitrage threshold')
    p.add_argument('--min-sim', type=float, default=0.4, help='Minimum title similarity (0-1)')
    p.add_argument('--kal-series', nargs='*', default=None, help='Optional Kalshi series ids to include (e.g. KXHIGHNY)')
    args = p.parse_args()

    res = find_cross_market_arbitrage(limit_pol=args.pol, limit_kal=args.kal, threshold=args.threshold, min_similarity=args.min_sim, kal_series_ids=args.kal_series)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
