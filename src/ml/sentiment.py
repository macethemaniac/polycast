"""Lightweight sentiment scoring using VaderSentiment.

Provides convenience helpers for single text and list-of-texts scoring.
"""
from __future__ import annotations

from typing import List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def score_text(text: str) -> float:
    """Return compound sentiment in [-1, 1] for a single text."""
    if not text:
        return 0.0
    scores = _analyzer.polarity_scores(text)
    return float(scores.get("compound", 0.0))


def score_texts(texts: List[str]) -> float:
    """Return average compound sentiment for a list of texts (ignores empty)."""
    if not texts:
        return 0.0
    vals = []
    for t in texts:
        if not t:
            continue
        vals.append(score_text(t))
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))
