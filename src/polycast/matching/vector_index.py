from typing import Any, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class TfidfIndex:
    def __init__(self, documents: List[str]):
        self.vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(documents)
        self.docs = documents

    def query(self, text: str, top_k: int = 10) -> List[Tuple[int, float]]:
        if not text:
            return []
        q = self.vectorizer.transform([text])
        sims = linear_kernel(q, self.matrix).flatten()
        top_idx = sims.argsort()[::-1][:top_k]
        return [(i, float(sims[i])) for i in top_idx if sims[i] > 0]


_INDEX_CACHE: Tuple[Any, List[str]] | None = None


def get_index(documents: List[str]) -> TfidfIndex:
    global _INDEX_CACHE
    if _INDEX_CACHE and len(_INDEX_CACHE[1]) == len(documents):
        return _INDEX_CACHE[0]
    idx = TfidfIndex(documents)
    _INDEX_CACHE = (idx, documents)
    return idx
