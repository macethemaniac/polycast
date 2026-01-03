"""Outcome normalization for binary markets across platforms."""
from __future__ import annotations

import re
from typing import Dict

NEGATION_PATTERNS = [
    r"\bnot\b",
    r"\bno\b",
    r"won't",
    r"will not",
    r"fail to",
    r"without",
    r"never",
]


def _has_negation(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(re.search(p, t) for p in NEGATION_PATTERNS)


def normalize_binary_outcomes(platform: str, market: Dict) -> Dict:
    """Return normalized yes/no probabilities, flipping if negation suggests inverted meaning."""
    yes = float(market.get("yes_price", 0.0) or 0.0)
    no = float(market.get("no_price", 0.0) or 0.0)
    q = market.get("question", "") or ""

    # Basic clamp
    yes = max(0.0, min(1.0, yes))
    no = max(0.0, min(1.0, no))

    # Detect negation; if strong, flip yes/no
    if _has_negation(q):
        yes, no = 1.0 - yes, 1.0 - no

    return {"yes_prob": yes, "no_prob": no}
