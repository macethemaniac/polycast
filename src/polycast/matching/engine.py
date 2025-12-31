import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from polycast.adapters import Market
from .rules import (
    normalize_text,
    token_overlap,
    fuzzy_score,
    feature_agreement,
    extract_features,
)
from .vector_index import get_index


def _title(m: Any) -> str:
    if isinstance(m, Market):
        return m.title
    if isinstance(m, dict):
        return m.get("question") or m.get("title") or m.get("slug") or ""
    return str(m)


def match_markets(
    pm_markets: List[Any],
    kal_markets: List[Any],
    top_k: int = 5,
    min_confidence: float = 0.25,
) -> Dict[str, Dict[str, Any]]:
    """
    For each Polymarket market, find top Kalshi candidates with scores.

    Returns mapping: pm_id -> { "matches": [ {id, score, parts}, ... ] }
    """
    data_dir = Path(__file__).resolve().parents[3] / "data"
    manual_path = data_dir / "manual_matches.json"
    memory_path = data_dir / "match_memory.json"

    def _load_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    manual = _load_json(manual_path)
    memory = _load_json(memory_path)

    # Build TF-IDF index over Kalshi titles/descriptions
    kal_texts: List[str] = []
    kal_ids: List[str] = []
    for kal in kal_markets:
        kal_id = getattr(kal, "id", None) or (kal.get("id") if isinstance(kal, dict) else None) or str(kal)
        kal_title = _title(kal)
        desc = ""
        if isinstance(kal, dict):
            desc = kal.get("description") or kal.get("question") or ""
        kal_ids.append(kal_id)
        kal_texts.append(f"{kal_title} {desc}".strip())
    tfidf_index = get_index(kal_texts) if kal_texts else None

    results: Dict[str, Dict[str, Any]] = {}
    for pm in pm_markets:
        pm_id = getattr(pm, "id", None) or (pm.get("id") if isinstance(pm, dict) else None) or str(pm)
        pm_title = _title(pm)
        pm_norm = normalize_text(pm_title)
        # Manual override
        if pm_id in manual:
            m = manual[pm_id]
            results[pm_id] = {
                "pm_title": pm_title,
                "matches": [
                    {
                        "kal_id": m.get("kalshi_id"),
                        "kal_title": m.get("kal_title", ""),
                        "score": 1.0,
                        "parts": {"manual": True},
                    }
                ],
                "all_top": [],
                "reason": None,
                "manual": True,
            }
            continue

        tfidf_candidates: List[Tuple[int, float]] = []
        if tfidf_index:
            tfidf_candidates = tfidf_index.query(pm_title, top_k=max(top_k * 3, 10))
        best: List[Tuple[float, Any]] = []
        for idx, tfidf_score in tfidf_candidates:
            if idx >= len(kal_markets):
                continue
            kal = kal_markets[idx]
            kal_id = getattr(kal, "id", None) or (kal.get("id") if isinstance(kal, dict) else None) or str(kal)
            kal_title = _title(kal)
            kal_norm = normalize_text(kal_title)
            fuzzy = fuzzy_score(pm_norm, kal_norm)
            tok = token_overlap(pm_norm, kal_norm)
            feat_score, feat_parts = feature_agreement(pm_title, kal_title)
            combined = 0.6 * fuzzy + 0.3 * tok + 0.1 * feat_score

            # Boost if seen in memory or shared keywords/entities
            mem = memory.get(pm_id)
            if mem:
                if mem.get("kalshi_id") == kal_id:
                    combined += 0.15
                mem_kw = set(mem.get("keywords", []))
                mem_ent = set(mem.get("entities", []))
                feats = extract_features(pm_title)
                kal_feats = extract_features(kal_title)
                if mem_kw & set(normalize_text(kal_title).split()):
                    combined += 0.05
                if mem_ent & (feats["entities"] | kal_feats["entities"]):
                    combined += 0.05

            best.append((combined, {
                "kal_id": kal_id,
                "kal_title": kal_title,
                "score": combined,
                "parts": {
                    "fuzzy": fuzzy,
                    "token_overlap": tok,
                    "features": feat_score,
                    "feature_parts": feat_parts,
                    "tfidf": tfidf_score,
                    "memory_boost": mem.get("kalshi_id") == kal_id if mem else False,
                },
            }))
        best.sort(key=lambda x: x[0], reverse=True)
        top_matches = [m for _, m in best[:top_k]]
        accepted = [m for m in top_matches if m["score"] >= min_confidence]
        results[pm_id] = {
            "pm_title": pm_title,
            "matches": accepted,
            "all_top": top_matches,
            "reason": None if accepted else "LOW_CONF",
        }

        # Persist learned association for accepted top match
        if accepted:
            top = accepted[0]
            feats_pm = extract_features(pm_title)
            feats_kal = extract_features(top.get("kal_title", ""))
            memory[pm_id] = {
                "kalshi_id": top["kal_id"],
                "keywords": list(set(normalize_text(pm_title).split()) | set(normalize_text(top.get("kal_title","")).split())),
                "entities": list((feats_pm["entities"] | feats_kal["entities"])),
            }

    # Save memory
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        memory_path.write_text(json.dumps(memory, indent=2), encoding="utf-8")
    except Exception:
        pass
    return results
