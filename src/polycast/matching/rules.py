import re
from typing import Dict, List, Set, Tuple

from rapidfuzz import fuzz


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and normalize spaces."""
    if not text:
        return ""
    s = text.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_overlap(a: str, b: str) -> float:
    """Jaccard overlap of tokens."""
    ta = set(normalize_text(a).split())
    tb = set(normalize_text(b).split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def fuzzy_score(a: str, b: str) -> float:
    """Fuzzy ratio normalized to [0,1]."""
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0


_MONTHS = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "sept": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}
_DATE_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_DATE_MONTH_DAY_RE = re.compile(r"\b(" + "|".join(_MONTHS.keys()) + r")\.?\s+(\d{1,2})\b", re.I)
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_THRESH_RE = re.compile(r"(>=|<=|>|<)\s*(\d+(?:\.\d+)?)")
_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)")
_ENTITY_RE = re.compile(r"\b(biden|trump|gop|dem|fed|inflation|rate|election|btc|bitcoin|eth|ethereum)\b", re.I)


def extract_dates(text: str) -> Set[str]:
    """Extract year or month-day signatures."""
    dates: Set[str] = set(_DATE_YEAR_RE.findall(text or ""))
    for m in _DATE_MONTH_DAY_RE.findall(text or ""):
        month = _MONTHS.get(m[0].lower(), "")
        day = m[1].zfill(2)
        dates.add(f"{month}-{day}")
    return dates


def extract_thresholds(text: str) -> List[str]:
    """Extract thresholds like >50, <=3.5, or ranges 5-10."""
    thresholds: List[str] = []
    for op, num in _THRESH_RE.findall(text or ""):
        thresholds.append(f"{op}{num}")
    for a, b in _RANGE_RE.findall(text or ""):
        thresholds.append(f"{a}-{b}")
    return thresholds


def extract_features(text: str) -> Dict[str, Set[str]]:
    """Extract dates, numbers, thresholds, entities."""
    norm = normalize_text(text)
    dates = extract_dates(text)
    nums = set(_NUMBER_RE.findall(norm))
    ths = set(extract_thresholds(text))
    ents = set(x.lower() for x in _ENTITY_RE.findall(text or ""))
    return {"dates": dates, "numbers": nums, "thresholds": ths, "entities": ents}


def feature_agreement(a: str, b: str) -> Tuple[float, Dict[str, float]]:
    """Score based on overlap of extracted features; returns score and parts."""
    fa = extract_features(a)
    fb = extract_features(b)

    def _ov(x: Set[str], y: Set[str]) -> float:
        if not x or not y:
            return 0.0
        inter = len(x & y)
        union = len(x | y)
        return inter / union if union else 0.0

    dates = _ov(fa["dates"], fb["dates"])
    thresholds = _ov(fa["thresholds"], fb["thresholds"])
    numbers = _ov(fa["numbers"], fb["numbers"])
    entities = _ov(fa["entities"], fb["entities"])

    score = 0.35 * dates + 0.35 * thresholds + 0.15 * numbers + 0.15 * entities
    if dates > 0 and thresholds > 0:
        score += 0.2  # heavy boost when both date and threshold match

    return score, {"dates": dates, "thresholds": thresholds, "numbers": numbers, "entities": entities}
