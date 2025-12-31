#!/usr/bin/env python3
import sys, json
sys.path.insert(0, 'arbitrage_mvp/src')
from polymarket_api import fetch_polymarket_markets
from kalshi_api import fetch_kalshi_markets
from difflib import SequenceMatcher
import re


def _normalize_text(t: str) -> str:
    if not t:
        return ''
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", ' ', t)
    t = re.sub(r"\s+", ' ', t).strip()
    return t


def _extract_tokens(t: str) -> set:
    if not t:
        return set()
    s = _normalize_text(t)
    tokens = set(w for w in s.split() if len(w) > 2)
    years = set(re.findall(r"\b20\d{2}\b", t))
    tokens.update(years)
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
    return len(inter) / len(union)


_MONTHS = {
    'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
    'jul':7,'aug':8,'sep':9,'sept':9,'oct':10,'nov':11,'dec':12
}


def _extract_date_components(t: str):
    res = {'year': None, 'month': None, 'day': None}
    if not t:
        return res
    s = t
    m = re.search(r"\b([A-Za-z]{3,9})\s+(\d{1,2}),?\s*(\d{4})\b", s)
    if m:
        mn = m.group(1)[:3].lower()
        day = int(m.group(2))
        year = int(m.group(3))
        month = _MONTHS.get(mn)
        res.update({'year': year, 'month': month, 'day': day})
        return res
    m2 = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", s)
    if m2:
        a = int(m2.group(1)); b = int(m2.group(2)); y = int(m2.group(3))
        if y < 100: y += 2000
        if a > 12:
            day = a; month = b
        else:
            month = a; day = b
        res.update({'year': y, 'month': month, 'day': day})
        return res
    m3 = re.search(r"\b(\d{2})([A-Za-z]{3})(\d{2,4})\b", s)
    if m3:
        day = int(m3.group(1))
        mn = m3.group(2)[:3].lower()
        y = int(m3.group(3))
        if y < 100: y += 2000
        month = _MONTHS.get(mn)
        res.update({'year': y, 'month': month, 'day': day})
        return res
    m4 = re.search(r"\b(20\d{2})\b", s)
    if m4:
        res['year'] = int(m4.group(1))
    return res


def _date_score(a: str, b: str) -> float:
    ca = _extract_date_components(a)
    cb = _extract_date_components(b)
    if not ca and not cb:
        return 0.0
    if ca.get('year') and cb.get('year'):
        if ca['year'] == cb['year']:
            if ca.get('month') and cb.get('month') and ca.get('day') and cb.get('day'):
                if ca['month'] == cb['month'] and ca['day'] == cb['day']:
                    return 1.0
            if ca.get('month') and cb.get('month') and ca['month'] == cb['month']:
                return 0.7
            return 0.5
    if (ca.get('year') and not cb.get('year')) or (cb.get('year') and not ca.get('year')):
        return 0.3
    return 0.0


def main():
    pol = fetch_polymarket_markets(100)
    kal = fetch_kalshi_markets(200)

    pairs = []
    for pm in pol:
        ptitle = (pm.get('question') or pm.get('title') or pm.get('slug') or '')
        npt = _normalize_text(ptitle)
        for km in kal:
            ktitle = (km.get('question') or km.get('title') or km.get('slug') or '')
            nkt = _normalize_text(ktitle)
            seq = SequenceMatcher(None, npt, nkt).ratio()
            jac = _jaccard(npt, nkt)
            ds = _date_score(npt, nkt)
            combined = 0.5 * ds + 0.35 * seq + 0.15 * jac
            pairs.append({
                'pol_title': ptitle,
                'kal_title': ktitle,
                'combined': combined,
                'seq': seq,
                'jaccard': jac,
                'date_score': ds,
            })

    pairs.sort(key=lambda x: x['combined'], reverse=True)
    top = pairs[:30]
    print(json.dumps(top, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
