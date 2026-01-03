"""Cross-market mismatch detection between Polymarket and Kalshi."""
from __future__ import annotations

import datetime as dt
import math
from typing import Dict, List

from engines.outcome_normalizer import normalize_binary_outcomes
from engines.settlement_guardrails import settlement_similarity
from ml.feature_builder import build_features
from ml.model_inference import predict_proba
import os


def _liquidity_score(pm_vol: float, km_vol: float) -> float:
    return min(1.0, 0.5 * math.log1p(pm_vol) / 5.0 + 0.5 * math.log1p(km_vol) / 5.0)


def _time_score(close_time) -> float:
    """Favor not-too-soon (>=1h) and not-too-far (<=120d)."""
    if not close_time:
        return 0.4
    try:
        if isinstance(close_time, (int, float)):
            t = dt.datetime.fromtimestamp(float(close_time), tz=dt.timezone.utc)
        else:
            t = dt.datetime.fromisoformat(str(close_time).replace("Z", "+00:00"))
        now = dt.datetime.now(tz=dt.timezone.utc)
        days = (t - now).total_seconds() / 86400.0
        if days < 0:
            return 0.0
        if days < 1:
            return 0.3
        if days > 120:
            return 0.3
        return 0.7
    except Exception:
        return 0.4


def _confidence_label(val: float) -> str:
    if val >= 75:
        return "HIGH"
    if val >= 55:
        return "MEDIUM"
    return "LOW"


def detect_mismatches(matches: List[Dict]) -> List[Dict]:
    """Compute mismatches for matched pairs with confidence scoring."""
    out: List[Dict] = []
    for m in matches:
        pm = m.get("poly_market", {})
        km = m.get("kalshi_market", {})
        sim = float(m.get("similarity", 0.0) or 0.0)
        pm_norm = normalize_binary_outcomes("polymarket", pm)
        km_norm = normalize_binary_outcomes("kalshi", km)
        poly_yes = pm_norm["yes_prob"]
        kal_yes = km_norm["yes_prob"]
        diff_yes = abs(poly_yes - kal_yes)
        edge_pct = diff_yes * 100.0

        pm_vol = float(pm.get("volume", 0.0) or 0.0)
        km_vol = float(km.get("volume", 0.0) or 0.0)

        settlement_sim = settlement_similarity(pm, km)
        liq_score = _liquidity_score(pm_vol, km_vol)
        time_score = _time_score(pm.get("close_time") or km.get("close_time"))
        confidence = (
            40.0 * settlement_sim
            + 20.0 * sim
            + 20.0 * liq_score
            + 20.0 * time_score
        )

        score = (
            edge_pct * max(settlement_sim, 0.2)
            + 0.1 * math.log1p(pm_vol)
            + 0.1 * math.log1p(km_vol)
            + 5.0 * confidence / 100.0
        )

        model_score = 0.0
        if os.getenv("USE_SUPERVISED_MODEL", "false").lower() == "true":
            feats, order = build_features({
                "yes_price": poly_yes,
                "no_price": kal_yes,
                "volume": pm_vol + km_vol,
                "liquidity": (pm.get("liquidity", 0.0) or 0.0) + (km.get("liquidity", 0.0) or 0.0),
                "news_mentions": (pm.get("news_mentions", 0.0) or 0.0),
                "sentiment": (pm.get("sentiment", 0.0) or 0.0),
                "trend_score": 0.0,
                "similarity": sim,
                "settlement_sim": settlement_sim,
                "confidence": confidence,
                "edge_pct": edge_pct,
                "time_to_expiry": 0.0,
            })
            model_score = predict_proba(feats, order) * 100.0
            score = 0.6 * score + 0.4 * model_score

        out.append({
            "question_poly": pm.get("question", ""),
            "question_kalshi": km.get("question", ""),
            "poly_market_id": pm.get("market_id"),
            "kalshi_market_id": km.get("market_id"),
            "poly_yes": poly_yes,
            "kalshi_yes": kal_yes,
            "edge_pct": edge_pct,
            "similarity": sim,
            "poly_volume": pm_vol,
            "kalshi_volume": km_vol,
            "settlement_sim": settlement_sim,
            "liquidity_score": liq_score,
            "time_score": time_score,
            "confidence": confidence,
            "confidence_label": _confidence_label(confidence),
            "score": score,
            "model_score": model_score,
            # include ids for logging
            "opportunity_id": "",
        })
    out.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return out
