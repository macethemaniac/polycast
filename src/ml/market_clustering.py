"""Market clustering using semantic embeddings."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from ml.embeddings import EmbeddingStore


def _build_clusters(labels: np.ndarray, markets: List[Dict]) -> List[Dict]:
    clusters: Dict[int, List[Dict]] = {}
    for lbl, m in zip(labels, markets):
        clusters.setdefault(int(lbl), []).append(m)

    out = []
    cid = 0
    for members in clusters.values():
        if not members:
            continue
        vol_sum = sum(float(m.get("volume", 0.0) or 0.0) for m in members)
        rep = max((m.get("question", "") or "" for m in members), key=len, default="")
        out.append({
            "cluster_id": cid,
            "representative_question": rep,
            "markets": [
                {
                    "market_id": m.get("market_id"),
                    "question": m.get("question", ""),
                    "yes_price": m.get("yes_price"),
                    "no_price": m.get("no_price"),
                    "volume": m.get("volume", 0.0),
                }
                for m in members
            ],
            "cluster_volume_sum": vol_sum,
        })
        cid += 1
    out.sort(key=lambda x: x.get("cluster_volume_sum", 0.0), reverse=True)
    return out


def cluster_markets(markets: List[Dict], max_markets: int = 200) -> List[Dict]:
    """Cluster markets by semantic similarity of questions."""
    if not markets:
        return []
    subset = markets[:max_markets]

    store = EmbeddingStore()
    items = [(m.get("market_id", f"m{i}"), m.get("question", "") or "") for i, m in enumerate(subset)]
    embeddings = store.embed(items)
    if embeddings.shape[0] < 2:
        return []

    try:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.4,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embeddings)
    except Exception:
        return []

    return _build_clusters(labels, subset)
