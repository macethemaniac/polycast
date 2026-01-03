"""Cross-platform market matcher using semantic embeddings."""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

from ml.embeddings import EmbeddingStore


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def build_embeddings_for_markets(markets: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
    items = [(m.get("market_id", f"m{i}"), m.get("question", "") or "") for i, m in enumerate(markets)]
    store = EmbeddingStore()
    emb = store.embed(items)
    return emb, markets


def match_markets_by_embedding(
    poly_markets: List[Dict],
    kal_markets: List[Dict],
    top_k: int = 3,
    sim_threshold: float = 0.72,
) -> List[Dict]:
    """Return matched pairs with similarity."""
    if not poly_markets or not kal_markets:
        return []

    poly_emb, _ = build_embeddings_for_markets(poly_markets)
    kal_emb, _ = build_embeddings_for_markets(kal_markets)

    matches: List[Dict] = []
    for i, pm in enumerate(poly_markets):
        pm_vec = poly_emb[i]
        sims = []
        for j, km in enumerate(kal_markets):
            sim = _cosine_sim(pm_vec, kal_emb[j])
            sims.append((sim, km))
        sims.sort(key=lambda x: x[0], reverse=True)
        for sim, km in sims[:top_k]:
            if sim < sim_threshold:
                continue
            matches.append({"poly_market": pm, "kalshi_market": km, "similarity": sim})
    return matches
