"""Cross-market arbitrage utilities combining Polymarket and Kalshi.

This module fetches binary markets from both platforms and looks for
cross-market mismatches, e.g. Polymarket YES vs Kalshi NO that sum
to greater than a threshold.
"""
from typing import List, Dict, Any
from difflib import SequenceMatcher
import re
from typing import Optional
import datetime

from polymarket_api import fetch_polymarket_markets
from kalshi_api import (
    fetch_kalshi_markets,
    fetch_kalshi_series,
    fetch_kalshi_series_markets,
    kalshi_market_to_generic,
    series_to_market,
)


def _extract_binary_prices_from_market(market: Dict[str, Any]) -> Dict[str, float]:
    """Return {'yes': float, 'no': float} or {} if cannot parse."""
    outcomes = market.get('outcomes') or []
    if not isinstance(outcomes, list) or len(outcomes) != 2:
        return {}

    def _price(o: Dict[str, Any]):
        for k in ('price', 'lastPrice', 'last_price', 'probability', 'midpoint'):
            if k in o and o[k] is not None:
                try:
                    return float(o[k])
                except Exception:
                    continue
        return None

    o0, o1 = outcomes[0], outcomes[1]
    title0 = (o0.get('title') or o0.get('id') or '').lower()
    title1 = (o1.get('title') or o1.get('id') or '').lower()
    p0 = _price(o0)
    p1 = _price(o1)
    if p0 is None or p1 is None:
        return {}

    # Determine mapping to yes/no
    if 'yes' in title0 and 'no' in title1:
        return {'yes': p0, 'no': p1}
    if 'no' in title0 and 'yes' in title1:
        return {'yes': p1, 'no': p0}
    # fallback: treat first as yes
    return {'yes': p0, 'no': p1}


def find_cross_market_arbitrage(limit_pol: int = 50, limit_kal: int = 50, threshold: float = 1.02, min_similarity: float = 0.4, kal_series_ids: List[str] = None) -> List[Dict[str, Any]]:
    """
    Find cross-market arbitrage opportunities between Polymarket and Kalshi.

    For each Polymarket binary and each Kalshi binary, check two combos:
      - pol_yes + kal_no
      - pol_no + kal_yes
    If total > threshold, record opportunity.
    """
    pol_markets = fetch_polymarket_markets(limit_pol)
    kal_raw_markets = fetch_kalshi_markets(limit_kal)
    kal_markets = []
    for m in kal_raw_markets:
        if isinstance(m, dict) and m.get('outcomes'):
            kal_markets.append(m)
            continue
        converted = kalshi_market_to_generic(m)
        if converted and converted.get('outcomes'):
            kal_markets.append(converted)
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

    kal_binaries = []
    for m in kal_markets:
        p = _extract_binary_prices_from_market(m)
        if p:
            title = (m.get('question') or m.get('title') or m.get('slug') or '')
            kal_binaries.append((m, p, _normalize_text(title)))

    # Compare every pair (could be optimized with fuzzy matching)
    for pol_m, pol_p, pol_title in pol_binaries:
        for kal_m, kal_p, kal_title in kal_binaries:
            try:
                seq_sim = _similarity(pol_title, kal_title)
                jacc = _jaccard(pol_title, kal_title)
                dscore = _date_score(pol_title, kal_title)
                # combine date score (highest weight) + sequence + token overlap
                combined = 0.5 * dscore + 0.35 * seq_sim + 0.15 * jacc
                if combined < min_similarity:
                    continue
                # include similarity metrics in reported results
                # Option A: buy YES on Polymarket, sell NO on Kalshi
                total_a = pol_p['yes'] + kal_p['no']
                if total_a > threshold:
                    opportunities.append({
                        'pol_question': pol_m.get('question') or pol_m.get('title') or '',
                        'kal_question': kal_m.get('question') or kal_m.get('title') or '',
                        'pol_yes': pol_p['yes'],
                        'pol_no': pol_p['no'],
                        'kal_yes': kal_p['yes'],
                        'kal_no': kal_p['no'],
                        'type': 'pol_yes + kal_no',
                        'total': total_a,
                        'profit_pct': (total_a - 1.0) * 100.0,
                        'title_similarity': combined,
                        'seq_similarity': seq_sim,
                        'jaccard_similarity': jacc,
                        'date_score': dscore,
                    })

                # Option B: buy NO on Polymarket, sell YES on Kalshi
                total_b = pol_p['no'] + kal_p['yes']
                if total_b > threshold:
                    opportunities.append({
                        'pol_question': pol_m.get('question') or pol_m.get('title') or '',
                        'kal_question': kal_m.get('question') or kal_m.get('title') or '',
                        'pol_yes': pol_p['yes'],
                        'pol_no': pol_p['no'],
                        'kal_yes': kal_p['yes'],
                        'kal_no': kal_p['no'],
                        'type': 'pol_no + kal_yes',
                        'total': total_b,
                        'profit_pct': (total_b - 1.0) * 100.0,
                        'title_similarity': combined,
                        'seq_similarity': seq_sim,
                        'jaccard_similarity': jacc,
                        'date_score': dscore,
                    })

            except Exception:
                continue

    opportunities.sort(key=lambda x: x['profit_pct'], reverse=True)
    return opportunities
