"""Embedding utilities for Polymarket questions (cached, CPU-friendly)."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = BASE_DIR / "data" / "cache"
EMB_FILE = CACHE_DIR / "embeddings.pkl"

logger = logging.getLogger(__name__)


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


class EmbeddingStore:
    """Simple cache for text embeddings, keyed by (market_id, text hash)."""

    def __init__(self) -> None:
        _ensure_cache_dir()
        self.cache: Dict[str, Dict[str, List[float]]] = {}
        self.model: SentenceTransformer | None = None
        self._load()

    def _load(self) -> None:
        try:
            if EMB_FILE.exists():
                self.cache = json.loads(EMB_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("Failed to load embeddings cache: %s", exc)
            self.cache = {}

    def _save(self) -> None:
        try:
            EMB_FILE.write_text(json.dumps(self.cache), encoding="utf-8")
        except Exception as exc:
            logger.debug("Failed to save embeddings cache: %s", exc)

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return self.model

    def embed(self, items: List[Tuple[str, str]]) -> np.ndarray:
        """
        Embed a list of (market_id, text) and return an array.
        Caches by market_id + text hash; reuses cached vectors when possible.
        """
        texts_to_embed = []
        indices = []
        vectors: List[np.ndarray] = []

        for idx, (mid, text) in enumerate(items):
            h = _hash_text(text or "")
            cached = self.cache.get(mid)
            if cached and cached.get("hash") == h and "vec" in cached:
                try:
                    vectors.append(np.array(cached["vec"], dtype=np.float32))
                    continue
                except Exception:
                    pass
            texts_to_embed.append(text or "")
            indices.append(len(vectors))
            vectors.append(None)  # placeholder

        if texts_to_embed:
            model = self._get_model()
            try:
                new_vecs = model.encode(texts_to_embed, convert_to_numpy=True, show_progress_bar=False)
            except Exception as exc:
                logger.error("Embedding failed: %s", exc)
                # fallback zeros
                new_vecs = np.zeros((len(texts_to_embed), 384), dtype=np.float32)
            # fill placeholders and cache
            ptr = 0
            for i, v in zip(indices, new_vecs):
                vectors[i] = np.array(v, dtype=np.float32)
                mid, text = items[i]
                self.cache[mid] = {"hash": _hash_text(text or ""), "vec": vectors[i].tolist()}

        # replace any remaining None with zeros
        for i, v in enumerate(vectors):
            if v is None:
                vectors[i] = np.zeros((384,), dtype=np.float32)

        self._save()
        return np.stack(vectors, axis=0)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of texts without caching (utility)."""
    store = EmbeddingStore()
    items = [(f"txt-{i}", t) for i, t in enumerate(texts)]
    return store.embed(items)
