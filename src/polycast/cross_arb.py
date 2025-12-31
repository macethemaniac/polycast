"""Cross-market arbitrage utilities combining Polymarket and Kalshi.

This module fetches binary markets from both platforms and looks for
cross-market mismatches, e.g. Polymarket YES vs Kalshi NO that sum
to greater than a threshold.
"""
from typing import List, Dict, Any, Optional, Tuple, Union
from difflib import SequenceMatcher
import re
import datetime
import json
import time
from pathlib import Path

from polycast.adapters import Market, Outcome
from polycast.adapters import polymarket as pm_adapter
from polycast.adapters import kalshi as kal_adapter
from polycast.matching.engine import match_markets
from polycast.arbitrage.engine import compute_opportunities, ArbConfig, save_opportunities, format_console_table
from polymarket_api import fetch_polymarket_markets
from kalshi_api import (
    fetch_kalshi_markets,
    fetch_kalshi_series,
    fetch_kalshi_series_markets,
    kalshi_market_to_generic,
    series_to_market,
)


def normalize_price_to_prob(val: Optional[float]) -> Optional[float]:
    """Normalize a price (possibly in cents) to [0,1]."""
    if val is None:
        return None
    try:
        v = float(val)
    except Exception:
        return None
    if v > 1.0:
        v = v / 100.0
    if v < 0:
        v = 0.0
    if v > 1:
        v = 1.0
    return v


def _extract_binary_prices_from_market(market: Dict[str, Any]) -> Dict[str, float]:
    """Return {'yes_bid','yes_ask','no_bid','no_ask'} (probs) or {} if cannot parse."""
    outcomes = market.get('outcomes') or []
    if not isinstance(outcomes, list) or len(outcomes) != 2:
        return {}

    def _bid(o: Dict[str, Any]) -> Optional[float]:
        for k in ('bestBid', 'bid', 'yesBid', 'noBid', 'best_bid', 'bidPrice', 'price', 'lastPrice', 'midpoint'):
            if k in o and o[k] is not None:
                return normalize_price_to_prob(o[k])
        return None

    def _ask(o: Dict[str, Any]) -> Optional[float]:
        for k in ('bestAsk', 'ask', 'yesAsk', 'noAsk', 'best_ask', 'askPrice', 'price', 'lastPrice', 'midpoint'):
            if k in o and o[k] is not None:
                return normalize_price_to_prob(o[k])
        return None

    o0, o1 = outcomes[0], outcomes[1]
    title0 = (o0.get('title') or o0.get('id') or '').lower()
    title1 = (o1.get('title') or o1.get('id') or '').lower()

    bids_asks = {}
    def _set_outcome(as_yes: bool, outcome: Dict[str, Any]):
        bid = _bid(outcome)
        ask = _ask(outcome)
        if as_yes:
            bids_asks['yes_bid'] = bid
            bids_asks['yes_ask'] = ask
        else:
            bids_asks['no_bid'] = bid
            bids_asks['no_ask'] = ask

    if 'yes' in title0 and 'no' in title1:
        _set_outcome(True, o0)
        _set_outcome(False, o1)
    elif 'no' in title0 and 'yes' in title1:
        _set_outcome(True, o1)
        _set_outcome(False, o0)
    else:
        _set_outcome(True, o0)
        _set_outcome(False, o1)

    return bids_asks


def find_cross_market_arbitrage(
    limit_pol: int = 50,
    limit_kal: int = 50,
    threshold: float = 1.02,
    min_similarity: float = 0.4,
    kal_series_ids: List[str] = None,
    collect_debug: bool = False,
    verbose: bool = False,
    dump_debug: bool = False,
    debug_dir: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Find cross-market arbitrage opportunities between Polymarket and Kalshi.

    For each Polymarket binary and each Kalshi binary, check two combos:
      - pol_yes + kal_no
      - pol_no + kal_yes
    If total > threshold, record opportunity.
    """
    debug: Dict[str, Any] = {
        "polymarket": {},
        "kalshi": {},
        "pairs_compared": 0,
        "threshold": threshold,
        "min_similarity": min_similarity,
        "candidates": [],
        "errors": [],
    }

    # Adapter-backed market fetch with diagnostics
    pol_markets_raw, pol_meta = fetch_polymarket_markets(limit_pol, return_debug=True)
    pol_markets = []
    for m in pol_markets_raw:
        if isinstance(m, Market):
            pol_markets.append(m.raw)
        else:
            pol_markets.append(m)
    debug["polymarket"] = {
        "count": pol_meta.get("count", len(pol_markets_raw)),
        "attempts": pol_meta.get("attempts", []),
        "error": pol_meta.get("error"),
    }

    kal_markets_raw, kal_meta = fetch_kalshi_markets(limit_kal, return_debug=True)
    kal_markets = []
    for m in kal_markets_raw:
        if isinstance(m, dict) and m.get('outcomes'):
            kal_markets.append(m)
            continue
        converted = kalshi_market_to_generic(m)
        if converted and converted.get('outcomes'):
            kal_markets.append(converted)
    debug["kalshi"] = {
        "count": kal_meta.get("count", len(kal_markets_raw)) if isinstance(kal_meta, dict) else len(kal_markets_raw),
        "attempts": kal_meta.get("attempts", []) if isinstance(kal_meta, dict) else [],
        "error": kal_meta.get("error") if isinstance(kal_meta, dict) else None,
    }

    # Build adapter-based market lists for matching
    pm_markets_for_match = pm_adapter.list_markets(limit=limit_pol, return_debug=False)
    kal_markets_for_match = kal_adapter.list_markets(limit=limit_kal, return_debug=False)
    match_results = match_markets(pm_markets_for_match, kal_markets_for_match, top_k=5, min_confidence=min_similarity)
    debug["matching"] = match_results
    # If explicit Kalshi series IDs are provided, fetch and include them
    if kal_series_ids:
        for sid in kal_series_ids:
            try:
                # Fetch markets for the series ticker and convert each
                km = []
                try:
                    km = fetch_kalshi_series_markets(sid)
                except Exception:
                    km = []

                for kmarket in km:
                    try:
                        gm = kalshi_market_to_generic(kmarket)
                        if gm and gm.get('outcomes'):
                            kal_markets.append(gm)
                    except Exception:
                        continue
                # Also include the series meta as a market if no individual markets
                try:
                    s = fetch_kalshi_series(sid)
                    if s:
                        m = series_to_market(s)
                        if m and m.get('outcomes'):
                            kal_markets.append(m)
                except Exception:
                    pass
            except Exception:
                continue

    opportunities: List[Dict[str, Any]] = []
    candidates_log: List[Dict[str, Any]] = []

    def _sizes(market: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        """Best-effort extract of bid/ask size fields (may be None)."""
        bid_size = market.get("bid_size") or market.get("yes_size") or market.get("size")
        ask_size = market.get("ask_size") or market.get("no_size") or market.get("size")
        try:
            bid_size = float(bid_size) if bid_size is not None else None
        except Exception:
            bid_size = None
        try:
            ask_size = float(ask_size) if ask_size is not None else None
        except Exception:
            ask_size = None
        return bid_size, ask_size

    def _is_stale(market: Dict[str, Any], max_age_seconds: int = 3600) -> bool:
        """Heuristic to flag stale markets based on timestamp-ish fields."""
        now = time.time()
        for key in ("timestamp", "ts", "updated", "last_traded", "lastTraded"):
            val = market.get(key)
            if val is None:
                continue
            try:
                ts = float(val)
                if ts > 1e12:  # ms to seconds
                    ts = ts / 1000.0
                if now - ts > max_age_seconds:
                    return True
            except Exception:
                continue
        return False

    def _normalize_text(t: str) -> str:
        if not t:
            return ''
        t = t.lower()
        t = re.sub(r"[^a-z0-9\s]", ' ', t)
        t = re.sub(r"\s+", ' ', t).strip()
        return t

    def _similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()
    
    def _extract_tokens(t: str) -> set:
        """Extract meaningful tokens from a title: words, years, numbers, quoted phrases."""
        if not t:
            return set()
        # normalized words
        s = _normalize_text(t)
        tokens = set(w for w in s.split() if len(w) > 2)
        # extract years like 2026
        years = set(re.findall(r"\b20\d{2}\b", t))
        tokens.update(years)
        # extract dates like Jan 1, 2026 or 01/01/2026 (simple numeric sequences)
        nums = set(re.findall(r"\b\d{1,4}\b", t))
        tokens.update(n for n in nums if len(n) >= 2)
        return tokens

    def _jaccard(a: str, b: str) -> float:
        ta = _extract_tokens(a)
        tb = _extract_tokens(b)
        if not ta or not tb:
            return 0.0
        inter = ta.intersection(tb)
        union = ta.union(tb)
        if len(union) == 0:
            return 0.0
        return len(inter) / len(union)


    _MONTHS = {
        'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
        'jul':7,'aug':8,'sep':9,'sept':9,'oct':10,'nov':11,'dec':12
    }

    def _extract_date_components(t: str) -> Dict[str, Optional[int]]:
        """Attempt to extract (year, month, day) from a title string.

        Returns dict with keys 'year','month','day' (ints or None).
        """
        res = {'year': None, 'month': None, 'day': None}
        if not t:
            return res
        s = t
        # look for formats like 'Dec 31, 2025' or 'Dec 31 2025'
        m = re.search(r"\b([A-Za-z]{3,9})\s+(\d{1,2}),?\s*(\d{4})\b", s)
        if m:
            mn = m.group(1)[:3].lower()
            day = int(m.group(2))
            year = int(m.group(3))
            month = _MONTHS.get(mn)
            res.update({'year': year, 'month': month, 'day': day})
            return res
        # look for numeric dates like 12/31/2025 or 31/12/2025
        m2 = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", s)
        if m2:
            a = int(m2.group(1)); b = int(m2.group(2)); y = int(m2.group(3))
            if y < 100: y += 2000
            # Heuristic: if a>12 then a=day else a=month
            if a > 12:
                day = a; month = b
            else:
                month = a; day = b
            res.update({'year': y, 'month': month, 'day': day})
            return res
        # look for compact forms like '25DEC31' or '25DEC2025' or '2025'
        m3 = re.search(r"\b(\d{2})([A-Za-z]{3})(\d{2,4})\b", s)
        if m3:
            day = int(m3.group(1))
            mn = m3.group(2)[:3].lower()
            y = int(m3.group(3))
            if y < 100: y += 2000
            month = _MONTHS.get(mn)
            res.update({'year': y, 'month': month, 'day': day})
            return res
        # fallback: year-only
        m4 = re.search(r"\b(20\d{2})\b", s)
        if m4:
            res['year'] = int(m4.group(1))
        return res

    def _date_score(a: str, b: str) -> float:
        ca = _extract_date_components(a)
        cb = _extract_date_components(b)
        if not ca and not cb:
            return 0.0
        # if both have year
        if ca.get('year') and cb.get('year'):
            if ca['year'] == cb['year']:
                # month+day match
                if ca.get('month') and cb.get('month') and ca.get('day') and cb.get('day'):
                    if ca['month'] == cb['month'] and ca['day'] == cb['day']:
                        return 1.0
                # month match
                if ca.get('month') and cb.get('month') and ca['month'] == cb['month']:
                    return 0.7
                # only year match
                return 0.5
        # one-side year match
        if (ca.get('year') and not cb.get('year')) or (cb.get('year') and not ca.get('year')):
            return 0.3
        return 0.0

    # Pre-extract parsed binaries to speed matching
    pol_binaries = []
    for m in pol_markets:
        p = _extract_binary_prices_from_market(m)
        if p:
            # store a normalized title for fuzzy matching
            title = (m.get('question') or m.get('title') or m.get('slug') or '')
            pol_binaries.append((m, p, _normalize_text(title)))

    kal_by_id: Dict[str, Dict[str, Any]] = {}
    kal_binaries = []
    for m in kal_markets:
        p = _extract_binary_prices_from_market(m)
        if p:
            title = (m.get('question') or m.get('title') or m.get('slug') or '')
            mid = m.get('id') or m.get('ticker') or m.get('event_ticker') or title
            kal_by_id[mid] = m
            kal_binaries.append((m, p, _normalize_text(title)))

    # Compare every pair (could be optimized with fuzzy matching)
    for pol_m, pol_p, pol_title in pol_binaries:
        pm_id = pol_m.get('id') or pol_m.get('slug') or pol_m.get('title') or pol_m.get('question') or pol_title
        candidates = match_results.get(pm_id, {}).get("matches", [])
        allowed_ids = {c.get("kal_id") for c in candidates if c.get("kal_id")}
        kal_candidates = [kb for kb in kal_binaries if (kb[0].get('id') or kb[0].get('ticker') or kb[0].get('event_ticker') or kb[2]) in allowed_ids] if allowed_ids else []
        if not kal_candidates:
            continue
        for kal_m, kal_p, kal_title in kal_candidates:
            debug["pairs_compared"] += 1
            try:
                seq_sim = _similarity(pol_title, kal_title)
                jacc = _jaccard(pol_title, kal_title)
                dscore = _date_score(pol_title, kal_title)
                # combine date score (highest weight) + sequence + token overlap
                combined = 0.5 * dscore + 0.35 * seq_sim + 0.15 * jacc
                reasons_base = []
                if combined < min_similarity:
                    reasons_base.append("NO_MATCH")
                if _is_stale(pol_m) or _is_stale(kal_m):
                    reasons_base.append("STALE")

                pol_yes_bid = pol_p.get("yes_bid")
                pol_yes_ask = pol_p.get("yes_ask")
                pol_no_bid = pol_p.get("no_bid")
                pol_no_ask = pol_p.get("no_ask")
                kal_yes_bid = kal_p.get("yes_bid")
                kal_yes_ask = kal_p.get("yes_ask")
                kal_no_bid = kal_p.get("no_bid")
                kal_no_ask = kal_p.get("no_ask")
                pol_bid_size, pol_ask_size = _sizes(pol_m)
                kal_bid_size, kal_ask_size = _sizes(kal_m)

                # Combo A: buy Polymarket YES (ask), sell Kalshi NO (bid)
                buy_ask_a = pol_yes_ask
                sell_bid_a = kal_no_bid
                prob_buy_a = normalize_price_to_prob(buy_ask_a)
                prob_sell_a = normalize_price_to_prob(sell_bid_a)
                total_a = (buy_ask_a or 0) + (sell_bid_a or 0)
                edge_a = (sell_bid_a or 0) - (buy_ask_a or 0)
                reasons_a = list(reasons_base)
                if buy_ask_a is None or sell_bid_a is None:
                    reasons_a.append("NO_LIQUIDITY")
                if prob_buy_a is None or prob_sell_a is None:
                    reasons_a.append("NORMALIZATION_ERR")
                if total_a <= threshold:
                    reasons_a.append("LOW_EDGE")
                if pol_bid_size is None or pol_ask_size is None or kal_bid_size is None or kal_ask_size is None:
                    reasons_a.append("LOW_LIQUIDITY")
                if not reasons_a:
                    reasons_a.append("NONE")

                cand_a = {
                    "combo": "pol_yes+kal_no",
                    "pol_question": pol_m.get("question") or pol_m.get("title") or "",
                    "kal_question": kal_m.get("question") or kal_m.get("title") or "",
                    "pol": {
                        "best_bid": pol_yes_bid,
                        "best_ask": pol_yes_ask,
                        "bid_size": pol_bid_size,
                        "ask_size": pol_ask_size,
                        "prob_bid": normalize_price_to_prob(pol_yes_bid),
                        "prob_ask": normalize_price_to_prob(pol_yes_ask),
                    },
                    "kal": {
                        "best_bid": kal_no_bid,
                        "best_ask": kal_no_ask,
                        "bid_size": kal_bid_size,
                        "ask_size": kal_ask_size,
                        "prob_bid": normalize_price_to_prob(kal_no_bid),
                        "prob_ask": normalize_price_to_prob(kal_no_ask),
                    },
                    "edge": edge_a,
                    "total": total_a,
                    "reasons": reasons_a or ["NONE"],
                    "similarity": {
                        "combined": combined,
                        "seq": seq_sim,
                        "jaccard": jacc,
                        "date": dscore,
                    },
                }

                if total_a > threshold and combined >= min_similarity:
                    opportunities.append(
                        {
                            "pol_question": cand_a["pol_question"],
                            "kal_question": cand_a["kal_question"],
                            "pol_yes": pol_yes_ask,
                            "pol_no": pol_no_ask,
                            "kal_yes": kal_yes_bid,
                            "kal_no": kal_no_bid,
                            "type": "pol_yes + kal_no",
                            "total": total_a,
                            "profit_pct": (total_a - 1.0) * 100.0,
                            "title_similarity": combined,
                            "seq_similarity": seq_sim,
                            "jaccard_similarity": jacc,
                            "date_score": dscore,
                        }
                    )
                candidates_log.append(cand_a)

                # Combo B: buy Polymarket NO (ask), sell Kalshi YES (bid)
                buy_ask_b = pol_no_ask
                sell_bid_b = kal_yes_bid
                prob_buy_b = normalize_price_to_prob(buy_ask_b)
                prob_sell_b = normalize_price_to_prob(sell_bid_b)
                total_b = (buy_ask_b or 0) + (sell_bid_b or 0)
                edge_b = (sell_bid_b or 0) - (buy_ask_b or 0)
                reasons_b = list(reasons_base)
                if buy_ask_b is None or sell_bid_b is None:
                    reasons_b.append("NO_LIQUIDITY")
                if prob_buy_b is None or prob_sell_b is None:
                    reasons_b.append("NORMALIZATION_ERR")
                if total_b <= threshold:
                    reasons_b.append("LOW_EDGE")
                if pol_bid_size is None or pol_ask_size is None or kal_bid_size is None or kal_ask_size is None:
                    reasons_b.append("LOW_LIQUIDITY")
                if not reasons_b:
                    reasons_b.append("NONE")

                cand_b = {
                    "combo": "pol_no+kal_yes",
                    "pol_question": pol_m.get("question") or pol_m.get("title") or "",
                    "kal_question": kal_m.get("question") or kal_m.get("title") or "",
                    "pol": {
                        "best_bid": pol_no_bid,
                        "best_ask": pol_no_ask,
                        "bid_size": pol_bid_size,
                        "ask_size": pol_ask_size,
                        "prob_bid": normalize_price_to_prob(pol_no_bid),
                        "prob_ask": prob_buy_b,
                    },
                    "kal": {
                        "best_bid": kal_yes_bid,
                        "best_ask": kal_yes_ask,
                        "bid_size": kal_bid_size,
                        "ask_size": kal_ask_size,
                        "prob_bid": prob_sell_b,
                        "prob_ask": normalize_price_to_prob(kal_yes_ask),
                    },
                    "edge": edge_b,
                    "total": total_b,
                    "reasons": reasons_b or ["NONE"],
                    "similarity": {
                        "combined": combined,
                        "seq": seq_sim,
                        "jaccard": jacc,
                        "date": dscore,
                    },
                }

                if total_b > threshold and combined >= min_similarity:
                    opportunities.append(
                        {
                            "pol_question": cand_b["pol_question"],
                            "kal_question": cand_b["kal_question"],
                            "pol_yes": pol_yes_bid,
                            "pol_no": pol_no_ask,
                            "kal_yes": kal_yes_bid,
                            "kal_no": kal_no_bid,
                            "type": "pol_no + kal_yes",
                            "total": total_b,
                            "profit_pct": (total_b - 1.0) * 100.0,
                            "title_similarity": combined,
                            "seq_similarity": seq_sim,
                            "jaccard_similarity": jacc,
                            "date_score": dscore,
                        }
                    )
                candidates_log.append(cand_b)

            except Exception as exc:
                candidates_log.append(
                    {
                        "combo": "unknown",
                        "pol_question": pol_m.get("question") or pol_m.get("title") or "",
                        "kal_question": kal_m.get("question") or kal_m.get("title") or "",
                        "edge": None,
                        "total": None,
                        "reasons": ["FETCH_FAIL"],
                        "similarity": {},
                        "error": str(exc),
                    }
                )
                continue

    opportunities.sort(key=lambda x: x['profit_pct'], reverse=True)

    # New arbitrage engine scoring with orderbooks
    arb_cfg = ArbConfig(
        min_edge=float(os.getenv("ARB_MIN_EDGE", "0.01")),
        min_size=float(os.getenv("ARB_MIN_SIZE", "0")),
        max_staleness=int(os.getenv("ARB_MAX_STALENESS", "3600")),
        fee_buy=float(os.getenv("ARB_FEE_BUY", "0.005")),
        fee_sell=float(os.getenv("ARB_FEE_SELL", "0.005")),
        slippage=float(os.getenv("ARB_SLIPPAGE", "0.01")),
    )
    opps, rejection_counts = compute_opportunities(match_results, arb_cfg)
    debug["new_engine_opps"] = opps
    debug["rejection_counts"] = rejection_counts

    if candidates_log:
        debug["candidates"] = candidates_log

    if dump_debug or collect_debug:
        # keep only top 200 candidates to avoid huge payloads
        if len(debug.get("candidates", [])) > 200:
            debug["candidates"] = debug["candidates"][:200]
        if dump_debug:
            dbg_dir = Path(debug_dir) if debug_dir else Path(__file__).resolve().parents[1] / "debug"
            dbg_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            dbg_path = dbg_dir / f"run_{ts}.json"
            try:
                with dbg_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "debug": debug,
                            "opportunities_count": len(opps),
                            "pairs_compared": debug.get("pairs_compared"),
                            "rejection_counts": rejection_counts,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
                debug["debug_file"] = str(dbg_path)
            except Exception as exc:
                debug["debug_file_error"] = str(exc)

    # Save and print console table if not in collect_debug-only mode
    if dump_debug:
        try:
            ts = int(time.time())
            out_path = (Path(__file__).resolve().parents[1] / "outputs") / f"opps_{ts}.json"
            save_opportunities(opps, out_path)
            debug["opps_file"] = str(out_path)
        except Exception as exc:
            debug["opps_file_error"] = str(exc)

    if collect_debug:
        return opps, debug
    return opps
