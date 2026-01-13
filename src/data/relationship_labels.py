"""Relationship labels storage for ML training.

Stores labeled relationships between markets for training the relationship discovery model.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from src.data.schemas import MarketRelationship, RelationshipType, HistoricalMarket

logger = logging.getLogger(__name__)

# Data storage
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "relationships"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RELATIONSHIPS_FILE = DATA_DIR / "market_relationships.jsonl"
GROUND_TRUTH_FILE = DATA_DIR / "ground_truth.jsonl"
AUTO_LABELED_FILE = DATA_DIR / "auto_labeled.jsonl"


class RelationshipStore:
    """Manages market relationship labels."""

    def __init__(self):
        self.relationships: Dict[str, MarketRelationship] = {}
        self.by_market: Dict[str, List[str]] = defaultdict(list)  # market_id -> [pair_ids]
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing relationships from disk."""
        for filepath in [RELATIONSHIPS_FILE, GROUND_TRUTH_FILE, AUTO_LABELED_FILE]:
            if filepath.exists():
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                rel = MarketRelationship.from_dict(data)
                                pair_id = rel.get_pair_id()
                                self.relationships[pair_id] = rel
                                self.by_market[rel.market_a_id].append(pair_id)
                                self.by_market[rel.market_b_id].append(pair_id)
                except Exception as e:
                    logger.error(f"Error loading {filepath}: {e}")

        logger.info(f"Loaded {len(self.relationships)} existing relationships")

    def _save_relationship(self, rel: MarketRelationship, filepath: Path) -> None:
        """Append relationship to file."""
        try:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rel.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Error saving relationship: {e}")

    def add_relationship(
        self,
        market_a_id: str,
        market_b_id: str,
        platform_a: str,
        platform_b: str,
        relationship_type: str,
        confidence: float = 0.0,
        verified_by: str = "auto",
        notes: str = "",
        semantic_similarity: float = 0.0,
        entity_overlap: float = 0.0,
        time_overlap: float = 0.0,
        historical_correlation: float = 0.0,
    ) -> MarketRelationship:
        """Add a new relationship."""
        rel = MarketRelationship(
            market_a_id=market_a_id,
            market_b_id=market_b_id,
            platform_a=platform_a,
            platform_b=platform_b,
            relationship_type=relationship_type,
            confidence=confidence,
            verified=verified_by in ["human", "llm"],
            verified_by=verified_by,
            notes=notes,
            semantic_similarity=semantic_similarity,
            entity_overlap=entity_overlap,
            time_overlap=time_overlap,
            historical_correlation=historical_correlation,
        )

        pair_id = rel.get_pair_id()
        self.relationships[pair_id] = rel
        self.by_market[market_a_id].append(pair_id)
        self.by_market[market_b_id].append(pair_id)

        # Save to appropriate file
        if verified_by == "human":
            self._save_relationship(rel, GROUND_TRUTH_FILE)
        elif verified_by == "auto":
            self._save_relationship(rel, AUTO_LABELED_FILE)
        else:
            self._save_relationship(rel, RELATIONSHIPS_FILE)

        return rel

    def get_relationship(self, market_a_id: str, market_b_id: str) -> Optional[MarketRelationship]:
        """Get relationship between two markets."""
        # Create temp relationship to get pair_id
        temp = MarketRelationship(
            market_a_id=market_a_id,
            market_b_id=market_b_id,
            platform_a="",
            platform_b="",
            relationship_type="independent"
        )
        pair_id = temp.get_pair_id()
        return self.relationships.get(pair_id)

    def get_market_relationships(self, market_id: str) -> List[MarketRelationship]:
        """Get all relationships for a market."""
        pair_ids = self.by_market.get(market_id, [])
        return [self.relationships[pid] for pid in pair_ids if pid in self.relationships]

    def get_relationships_by_type(self, rel_type: str) -> List[MarketRelationship]:
        """Get all relationships of a specific type."""
        return [r for r in self.relationships.values() if r.relationship_type == rel_type]

    def get_cross_platform_pairs(self) -> List[MarketRelationship]:
        """Get relationships where markets are on different platforms."""
        return [r for r in self.relationships.values() if r.platform_a != r.platform_b]

    def get_training_data(self) -> List[Tuple[Dict, int]]:
        """Get labeled data for training.

        Returns list of (features_dict, label) where:
        - label=1 for correlated, same_event, subset
        - label=0 for independent
        - label=-1 for anti_correlated, exclusive (inverse relationship)
        """
        training_data = []

        for rel in self.relationships.values():
            if not rel.verified:
                continue

            features = {
                "semantic_similarity": rel.semantic_similarity,
                "entity_overlap": rel.entity_overlap,
                "time_overlap": rel.time_overlap,
                "historical_correlation": rel.historical_correlation,
                "confidence": rel.confidence,
                "cross_platform": 1.0 if rel.platform_a != rel.platform_b else 0.0,
            }

            # Map relationship type to label
            if rel.relationship_type in ["correlated", "same_event", "subset", "sequential"]:
                label = 1
            elif rel.relationship_type in ["anti_correlated", "exclusive"]:
                label = -1
            else:  # independent
                label = 0

            training_data.append((features, label))

        return training_data

    def get_stats(self) -> Dict[str, Any]:
        """Get relationship statistics."""
        rels = list(self.relationships.values())
        verified = [r for r in rels if r.verified]

        type_counts = defaultdict(int)
        for r in rels:
            type_counts[r.relationship_type] += 1

        return {
            "total_relationships": len(rels),
            "verified_relationships": len(verified),
            "cross_platform": len([r for r in rels if r.platform_a != r.platform_b]),
            "by_type": dict(type_counts),
            "by_verifier": {
                "human": len([r for r in rels if r.verified_by == "human"]),
                "llm": len([r for r in rels if r.verified_by == "llm"]),
                "auto": len([r for r in rels if r.verified_by == "auto"]),
            },
            "avg_confidence": sum(r.confidence for r in rels) / len(rels) if rels else 0,
        }


class AutoLabeler:
    """Automatically labels relationships based on market outcomes."""

    def __init__(self, store: RelationshipStore):
        self.store = store

    def label_from_resolutions(
        self,
        markets: List[HistoricalMarket],
        similarity_threshold: float = 0.6
    ) -> int:
        """Auto-label relationships based on resolved market outcomes.

        If two resolved markets have similar questions:
        - Both YES or both NO -> CORRELATED
        - One YES, one NO -> ANTI_CORRELATED
        - Similar topics, independent outcomes -> INDEPENDENT
        """
        from src.ml.embeddings import EmbeddingStore

        embedding_store = EmbeddingStore()
        resolved = [m for m in markets if m.resolution in ["yes", "no"]]
        labeled_count = 0

        logger.info(f"Auto-labeling from {len(resolved)} resolved markets...")

        # Group by platform
        by_platform: Dict[str, List[HistoricalMarket]] = defaultdict(list)
        for m in resolved:
            by_platform[m.platform].append(m)

        # Compare within and across platforms
        for i, market_a in enumerate(resolved):
            # Get embedding
            emb_a = embedding_store.get_embedding(market_a.market_id, market_a.question)
            if emb_a is None:
                continue

            for market_b in resolved[i+1:]:
                # Skip if already labeled
                existing = self.store.get_relationship(market_a.market_id, market_b.market_id)
                if existing and existing.verified:
                    continue

                # Get embedding
                emb_b = embedding_store.get_embedding(market_b.market_id, market_b.question)
                if emb_b is None:
                    continue

                # Calculate similarity
                import numpy as np
                similarity = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))

                if similarity < similarity_threshold:
                    continue

                # Determine relationship from outcomes
                if market_a.resolution == market_b.resolution:
                    rel_type = "correlated"
                else:
                    rel_type = "anti_correlated"

                # Check if same event (very high similarity)
                if similarity > 0.9 and market_a.platform != market_b.platform:
                    rel_type = "same_event"

                self.store.add_relationship(
                    market_a_id=market_a.market_id,
                    market_b_id=market_b.market_id,
                    platform_a=market_a.platform,
                    platform_b=market_b.platform,
                    relationship_type=rel_type,
                    confidence=similarity,
                    verified_by="auto",
                    notes=f"Auto-labeled from resolutions: {market_a.resolution} vs {market_b.resolution}",
                    semantic_similarity=similarity,
                )
                labeled_count += 1

        logger.info(f"Auto-labeled {labeled_count} relationships")
        return labeled_count

    def label_cross_platform_matches(
        self,
        poly_markets: List[HistoricalMarket],
        kalshi_markets: List[HistoricalMarket],
        similarity_threshold: float = 0.75
    ) -> int:
        """Find and label cross-platform market pairs."""
        from src.ml.embeddings import EmbeddingStore

        embedding_store = EmbeddingStore()
        labeled_count = 0

        logger.info(f"Finding cross-platform matches: {len(poly_markets)} Poly vs {len(kalshi_markets)} Kalshi")

        # Get embeddings for all markets
        poly_embeddings = {}
        for m in poly_markets:
            emb = embedding_store.get_embedding(m.market_id, m.question)
            if emb is not None:
                poly_embeddings[m.market_id] = (m, emb)

        kalshi_embeddings = {}
        for m in kalshi_markets:
            emb = embedding_store.get_embedding(m.market_id, m.question)
            if emb is not None:
                kalshi_embeddings[m.market_id] = (m, emb)

        # Compare all pairs
        import numpy as np

        for poly_id, (poly_market, poly_emb) in poly_embeddings.items():
            for kalshi_id, (kalshi_market, kalshi_emb) in kalshi_embeddings.items():
                # Skip if already labeled
                existing = self.store.get_relationship(poly_id, kalshi_id)
                if existing:
                    continue

                # Calculate similarity
                similarity = float(np.dot(poly_emb, kalshi_emb) / (np.linalg.norm(poly_emb) * np.linalg.norm(kalshi_emb)))

                if similarity < similarity_threshold:
                    continue

                # High similarity cross-platform = same event
                rel_type = "same_event" if similarity > 0.85 else "correlated"

                # Check resolutions if available
                if poly_market.resolution != "pending" and kalshi_market.resolution != "pending":
                    if poly_market.resolution != kalshi_market.resolution:
                        # Different resolutions = might be anti-correlated or exclusive
                        rel_type = "anti_correlated"

                self.store.add_relationship(
                    market_a_id=poly_id,
                    market_b_id=kalshi_id,
                    platform_a="polymarket",
                    platform_b="kalshi",
                    relationship_type=rel_type,
                    confidence=similarity,
                    verified_by="auto",
                    notes=f"Cross-platform match (sim={similarity:.3f})",
                    semantic_similarity=similarity,
                )
                labeled_count += 1

        logger.info(f"Found {labeled_count} cross-platform matches")
        return labeled_count


# Convenience functions
_store: Optional[RelationshipStore] = None


def get_relationship_store() -> RelationshipStore:
    """Get or create singleton store."""
    global _store
    if _store is None:
        _store = RelationshipStore()
    return _store


def add_relationship(
    market_a_id: str,
    market_b_id: str,
    platform_a: str,
    platform_b: str,
    relationship_type: str,
    confidence: float = 0.0,
    verified_by: str = "auto",
    **kwargs
) -> MarketRelationship:
    """Add a relationship to the store."""
    store = get_relationship_store()
    return store.add_relationship(
        market_a_id=market_a_id,
        market_b_id=market_b_id,
        platform_a=platform_a,
        platform_b=platform_b,
        relationship_type=relationship_type,
        confidence=confidence,
        verified_by=verified_by,
        **kwargs
    )


def get_relationship_stats() -> Dict[str, Any]:
    """Get relationship statistics."""
    store = get_relationship_store()
    return store.get_stats()


def auto_label_from_history() -> int:
    """Auto-label relationships from historical data."""
    from src.data.historical_collector import get_collector as get_poly_collector
    from src.data.kalshi_historical import get_kalshi_collector

    store = get_relationship_store()
    labeler = AutoLabeler(store)

    # Get resolved markets
    poly_collector = get_poly_collector()
    kalshi_collector = get_kalshi_collector()

    poly_markets = poly_collector.get_resolved_markets()
    kalshi_markets = kalshi_collector.get_resolved_markets()

    # Label within platforms
    all_markets = poly_markets + kalshi_markets
    count1 = labeler.label_from_resolutions(all_markets)

    # Label cross-platform
    count2 = labeler.label_cross_platform_matches(poly_markets, kalshi_markets)

    return count1 + count2
