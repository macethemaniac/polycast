"""Trend detection utilities for Polymarket markets using IsolationForest."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.ensemble import IsolationForest

from data.news_gdelt import get_news_signal


BASE_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = BASE_DIR / "data" / "cache"
STATE_FILE = CACHE_DIR / "polymarket_state.json"


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_state() -> Dict[str, Dict]:
    """Load last known market state (prices/volume/timestamp) from cache."""
    try:
        _ensure_cache_dir()
        if not STATE_FILE.exists():
            return {}
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state: Dict[str, Dict]) -> None:
    """Persist market state."""
    try:
        _ensure_cache_dir()
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception:
        pass


def _scale_scores(raw: List[float]) -> List[float]:
    if not raw:
        return []
    mn = min(raw)
    mx = max(raw)
    if math.isclose(mx, mn):
        return [50.0 for _ in raw]
    return [((v - mn) / (mx - mn)) * 100.0 for v in raw]


def compute_trend_scores(markets: List[Dict], prev_state: Dict[str, Dict]) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Compute trend scores for markets and return updated state."""
    if prev_state is None:
        prev_state = {}
    current_ts = int(time.time())
    features = []
    meta = []

    for m in markets:
        try:
            market_id = m.get("market_id")
            q = m.get("question", "") or ""
            yes = float(m.get("yes_price", 0.0) or 0.0)
            no = float(m.get("no_price", 0.0) or 0.0)
            vol = float(m.get("volume", 0.0) or 0.0)
            liq = float(m.get("liquidity", 0.0) or 0.0)

            prev = prev_state.get(str(market_id), {})
            prev_yes = float(prev.get("yes_price", 0.0) or 0.0)
            prev_no = float(prev.get("no_price", 0.0) or 0.0)
            prev_vol = float(prev.get("volume", 0.0) or 0.0)

            price_delta = abs(yes - prev_yes) + abs(no - prev_no)
            volume_delta = vol - prev_vol

            news = get_news_signal(q)
            news_mentions = int(news.get("mentions_24h", 0) or 0)

            feat = [
                math.log1p(vol),
                liq,
                abs(price_delta),
                abs(volume_delta),
                news_mentions,
            ]
            features.append(feat)
            meta.append({
                "market_id": market_id,
                "question": q,
                "yes_price": yes,
                "no_price": no,
                "volume": vol,
                "liquidity": liq,
                "price_delta": price_delta,
                "volume_delta": volume_delta,
                "news_mentions": news_mentions,
            })
        except Exception:
            continue

    if len(features) < 2:
        # not enough data to fit a model
        results = []
        new_state = prev_state.copy()
        for m in meta:
            results.append({
                **m,
                "trend_score": 0.0,
                "reasons": [],
            })
            new_state[str(m["market_id"])] = {
                "yes_price": m["yes_price"],
                "no_price": m["no_price"],
                "volume": m["volume"],
                "ts": current_ts,
            }
        return results, new_state

    try:
        model = IsolationForest(
            n_estimators=50,
            contamination="auto",
            random_state=42,
        )
        model.fit(features)
        raw_scores = [-s for s in model.decision_function(features)]
        scaled = _scale_scores(raw_scores)
    except Exception:
        scaled = [0.0 for _ in features]

    results = []
    new_state = prev_state.copy()
    for m, score in zip(meta, scaled):
        reasons = []
        # Looser thresholds: any positive volume change or modest relative lift
        if m["volume_delta"] > 0 or (m["volume"] > 0 and m["volume_delta"] / max(m["volume"], 1.0) > 0.02):
            reasons.append("volume spike")
        # Smaller price move threshold
        if m["price_delta"] >= 0.02:
            reasons.append("price jump")
        # Lower news bar
        if m["news_mentions"] >= 1:
            reasons.append("news spike")

        results.append({
            **m,
            "trend_score": score,
            "reasons": reasons,
        })
        new_state[str(m["market_id"])] = {
            "yes_price": m["yes_price"],
            "no_price": m["no_price"],
            "volume": m["volume"],
            "ts": current_ts,
        }

    return results, new_state
