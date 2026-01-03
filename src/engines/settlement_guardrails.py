"""Guardrails to estimate settlement/rule similarity for cross-market matches."""
from __future__ import annotations

import datetime as dt
import logging
from typing import Dict

import numpy as np

from ml.embeddings import embed_texts

logger = logging.getLogger(__name__)


def extract_resolution_text(market: Dict) -> str:
    """Combine question + description + rules to form resolution text."""
    parts = [
        market.get("question", "") or "",
        market.get("description", "") or "",
        market.get("rules", "") or "",
        market.get("resolution_source", "") or "",
    ]
    return "\n".join(p for p in parts if p)


def _parse_time(val) -> dt.datetime | None:
    if not val:
        return None
    if isinstance(val, (int, float)):
        try:
            return dt.datetime.fromtimestamp(float(val), tz=dt.timezone.utc)
        except Exception:
            return None
    try:
        return dt.datetime.fromisoformat(str(val).replace("Z", "+00:00"))
    except Exception:
        return None


def settlement_similarity(poly_market: Dict, kalshi_market: Dict) -> float:
    """Compute settlement similarity in [0,1] using embeddings and timing guardrails."""
    try:
        texts = [
            extract_resolution_text(poly_market),
            extract_resolution_text(kalshi_market),
        ]
        if not texts[0] or not texts[1]:
            base_sim = 0.4  # conservative cap when missing text
        else:
            embs = embed_texts(texts)
            v0, v1 = embs[0], embs[1]
            denom = (np.linalg.norm(v0) * np.linalg.norm(v1)) + 1e-9
            base_sim = float(np.dot(v0, v1) / denom)
        base_sim = max(0.0, min(1.0, base_sim))

        # Time guardrail
        t0 = _parse_time(poly_market.get("close_time"))
        t1 = _parse_time(kalshi_market.get("close_time"))
        time_penalty = 0.0
        if t0 and t1:
            diff_days = abs((t0 - t1).total_seconds()) / 86400.0
            if diff_days > 7:
                time_penalty = min(0.3, diff_days / 90.0)  # penalize large gaps
        else:
            # Missing time info, be conservative
            time_penalty = 0.1

        sim = max(0.0, min(1.0, base_sim - time_penalty))
        return sim
    except Exception as exc:
        logger.debug("settlement_similarity failed: %s", exc)
        return 0.0
