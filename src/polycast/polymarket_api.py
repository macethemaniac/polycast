"""Polymarket Gamma public API helpers.

Provides simple unauthenticated REST calls to fetch markets and
detect binary (YES/NO) arbitrage opportunities.
"""
from typing import List, Dict, Any, Tuple
import time
import requests
from polycast.adapters import Market
from polycast.adapters import polymarket as pm_adapter


def fetch_polymarket_markets(limit: int, return_debug: bool = False) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Fetch public markets from the Polymarket Gamma REST endpoint (adapter-backed).
    """
    markets, meta = pm_adapter.list_markets(limit=limit, return_debug=True)
    raw_list = [m.raw for m in markets if isinstance(m, Market)]
    if return_debug:
        return raw_list, {"attempts": meta.get("attempts", []), "count": meta.get("count", len(raw_list)), "error": meta.get("error")}
    return raw_list


def find_polymarket_arbitrage(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Find simple arbitrage opportunities among binary (YES/NO) markets.

    Criteria: if yes_price + no_price > 1.02 it's considered an opportunity.

    Returns a list of dicts with: question, yes_price, no_price, total, profit_pct
    """
    markets = fetch_polymarket_markets(limit)
    opportunities: List[Dict[str, Any]] = []

    for m in markets:
        outcomes = m.get('outcomes') or []
        if not isinstance(outcomes, list) or len(outcomes) != 2:
            continue

        # Extract prices and titles robustly
        try:
            o0 = outcomes[0]
            o1 = outcomes[1]
            title0 = (o0.get('title') or o0.get('id') or '').strip()
            title1 = (o1.get('title') or o1.get('id') or '').strip()

            # Attempt different keys for price
            def _price(outcome):
                for k in ('price', 'lastPrice', 'last_price', 'probability'):
                    if k in outcome and outcome[k] is not None:
                        try:
                            return float(outcome[k])
                        except Exception:
                            continue
                return None

            p0 = _price(o0)
            p1 = _price(o1)
            if p0 is None or p1 is None:
                continue

            # Determine which outcome is YES and which is NO by title if possible
            title0_l = title0.lower()
            title1_l = title1.lower()
            yes_price = None
            no_price = None
            if 'yes' in title0_l and 'no' in title1_l:
                yes_price = p0
                no_price = p1
            elif 'no' in title0_l and 'yes' in title1_l:
                yes_price = p1
                no_price = p0
            else:
                # Fallback: treat first as YES, second as NO
                yes_price = p0
                no_price = p1

            total = yes_price + no_price
            if total > 1.02:
                profit_pct = (total - 1.0) * 100.0
                opportunities.append({
                    'question': m.get('question') or m.get('title') or m.get('slug') or '',
                    'yes_price': yes_price,
                    'no_price': no_price,
                    'total': total,
                    'profit_pct': profit_pct,
                })

        except Exception:
            continue

    # sort by profit desc
    opportunities.sort(key=lambda x: x['profit_pct'], reverse=True)
    return opportunities
