"""High-performance inference engine for real-time arbitrage detection.

Optimized for low latency with caching and batch processing.
"""

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from functools import lru_cache

import numpy as np

from src.ml.ensemble import ArbitrageEnsemble, PredictionResult, get_ensemble
from src.ml.feature_builder_v2 import AdvancedFeatureBuilder, get_feature_builder
from src.ml.market_clustering_v2 import HierarchicalClusterer, ClusterTree, get_clusterer
from src.ml.relationship_discovery import RelationshipDiscoveryEngine, RelationshipPrediction, get_discovery_engine
from src.ml.logical_consistency import LogicalConsistencyChecker, Inconsistency, get_consistency_checker
from src.ml.semantic_analyzer import SemanticAnalyzer, MarketAnalysis, get_semantic_analyzer
from src.ml.models.specialized import predict_all_types, SpecializedPrediction

logger = logging.getLogger(__name__)


class TTLCache:
    """Simple time-to-live cache."""

    def __init__(self, maxsize: int = 1000, ttl_seconds: int = 300):
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}

    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None

        # Check TTL
        if time.time() - self._timestamps[key] > self.ttl:
            del self._cache[key]
            del self._timestamps[key]
            return None

        # Move to end (LRU)
        self._cache.move_to_end(key)
        return self._cache[key]

    def set(self, key: str, value: Any) -> None:
        # Evict if full
        while len(self._cache) >= self.maxsize:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            del self._timestamps[oldest_key]

        self._cache[key] = value
        self._timestamps[key] = time.time()

    def clear(self) -> None:
        self._cache.clear()
        self._timestamps.clear()


@dataclass
class ScoredOpportunity:
    """Fully scored arbitrage opportunity."""
    opportunity_id: str
    opportunity_type: str
    timestamp: float = field(default_factory=time.time)

    # Markets involved
    markets: List[Dict] = field(default_factory=list)
    market_ids: List[str] = field(default_factory=list)

    # Scoring
    ensemble_score: float = 0.0
    specialized_scores: Dict[str, float] = field(default_factory=dict)
    final_score: float = 0.0

    # Edge and confidence
    edge_pct: float = 0.0
    confidence: float = 0.0

    # Relationship context
    relationship_type: Optional[str] = None
    relationship_confidence: float = 0.0

    # Cluster context
    cluster_id: Optional[str] = None
    cluster_deviation: float = 0.0

    # Inconsistency
    inconsistency_type: Optional[str] = None
    inconsistency_severity: Optional[str] = None

    # Action
    recommended_action: str = ""
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "opportunity_id": self.opportunity_id,
            "opportunity_type": self.opportunity_type,
            "timestamp": self.timestamp,
            "market_ids": self.market_ids,
            "final_score": self.final_score,
            "edge_pct": self.edge_pct,
            "confidence": self.confidence,
            "recommended_action": self.recommended_action,
            "explanation": self.explanation,
        }


class InferenceEngine:
    """
    High-performance inference engine for real-time scoring.

    Combines all ML components for comprehensive opportunity scoring.
    """

    def __init__(
        self,
        cache_ttl: int = 300,
        cache_size: int = 1000,
        min_score_threshold: float = 0.5,
        min_edge_threshold: float = 0.03
    ):
        """
        Initialize inference engine.

        Args:
            cache_ttl: Cache time-to-live in seconds
            cache_size: Maximum cache entries
            min_score_threshold: Minimum score to return opportunity
            min_edge_threshold: Minimum edge % to consider
        """
        self.min_score = min_score_threshold
        self.min_edge = min_edge_threshold

        # Initialize components
        self.ensemble = get_ensemble()
        self.feature_builder = get_feature_builder()
        self.clusterer = get_clusterer()
        self.discovery_engine = get_discovery_engine()
        self.consistency_checker = get_consistency_checker()
        self.semantic_analyzer = get_semantic_analyzer()

        # Caches
        self.cluster_cache = TTLCache(maxsize=100, ttl_seconds=cache_ttl)
        self.relationship_cache = TTLCache(maxsize=cache_size, ttl_seconds=cache_ttl)
        self.analysis_cache = TTLCache(maxsize=cache_size, ttl_seconds=cache_ttl * 2)
        self.score_cache = TTLCache(maxsize=cache_size, ttl_seconds=60)  # Short TTL for scores

    def _get_market_id(self, market: Dict) -> str:
        return market.get("id") or market.get("market_id") or market.get("ticker") or ""

    def _get_cached_analysis(self, market: Dict) -> Optional[MarketAnalysis]:
        """Get cached semantic analysis."""
        mid = self._get_market_id(market)
        return self.analysis_cache.get(mid)

    def _cache_analysis(self, market: Dict, analysis: MarketAnalysis) -> None:
        """Cache semantic analysis."""
        mid = self._get_market_id(market)
        self.analysis_cache.set(mid, analysis)

    def _get_cached_clusters(self, cache_key: str) -> Optional[ClusterTree]:
        """Get cached cluster tree."""
        return self.cluster_cache.get(cache_key)

    def _cache_clusters(self, cache_key: str, tree: ClusterTree) -> None:
        """Cache cluster tree."""
        self.cluster_cache.set(cache_key, tree)

    def analyze_markets(self, markets: List[Dict]) -> List[MarketAnalysis]:
        """Analyze markets with caching."""
        results = []
        to_analyze = []

        for market in markets:
            cached = self._get_cached_analysis(market)
            if cached:
                results.append(cached)
            else:
                to_analyze.append(market)

        # Batch analyze uncached
        if to_analyze:
            new_analyses = self.semantic_analyzer.analyze_batch(to_analyze)
            for market, analysis in zip(to_analyze, new_analyses):
                self._cache_analysis(market, analysis)
                results.append(analysis)

        return results

    def cluster_markets(self, markets: List[Dict]) -> ClusterTree:
        """Cluster markets with caching."""
        # Create cache key from market IDs
        market_ids = sorted([self._get_market_id(m) for m in markets])
        cache_key = hash(tuple(market_ids[:20]))  # Use first 20 for key

        cached = self._get_cached_clusters(str(cache_key))
        if cached:
            return cached

        tree = self.clusterer.cluster_markets(markets)
        self._cache_clusters(str(cache_key), tree)
        return tree

    def discover_relationships(self, markets: List[Dict]) -> List[RelationshipPrediction]:
        """Discover relationships with caching."""
        return self.discovery_engine.discover_relationships(markets)

    def check_consistency(
        self,
        markets: List[Dict],
        relationships: List[RelationshipPrediction]
    ) -> List[Inconsistency]:
        """Check logical consistency."""
        return self.consistency_checker.check_all(markets, relationships)

    def score_opportunity(
        self,
        item: Dict,
        cluster_info: Optional[Dict] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Score a single opportunity.

        Returns:
            (final_score, component_scores)
        """
        # Check cache
        cache_key = f"{self._get_market_id(item)}:{item.get('yes_price', 0)}"
        cached = self.score_cache.get(cache_key)
        if cached:
            return cached

        # Ensemble score
        ensemble_result = self.ensemble.predict_single(item)
        ensemble_score = ensemble_result.probability

        # Specialized model scores
        specialized_results = predict_all_types(item)
        specialized_scores = {r.opportunity_type: r.probability for r in specialized_results}

        # Combine scores
        # Weight: ensemble 40%, best specialized 30%, edge factor 30%
        edge = float(item.get("edge_pct", 0)) / 100.0
        edge_factor = min(edge / 0.1, 1.0)  # Normalize edge to 0-1

        best_specialized = max(specialized_scores.values()) if specialized_scores else 0.5

        final_score = (
            0.4 * ensemble_score +
            0.3 * best_specialized +
            0.3 * edge_factor
        )

        component_scores = {
            "ensemble": ensemble_score,
            "specialized_max": best_specialized,
            "edge_factor": edge_factor,
            **specialized_scores
        }

        result = (final_score, component_scores)
        self.score_cache.set(cache_key, result)

        return result

    def score_all_opportunities(
        self,
        markets: List[Dict]
    ) -> List[ScoredOpportunity]:
        """
        Score all opportunities in a set of markets.

        Full pipeline:
        1. Cluster markets
        2. Discover relationships
        3. Check consistency (find inconsistencies)
        4. Score each inconsistency as opportunity
        """
        logger.info(f"Scoring opportunities from {len(markets)} markets...")

        # Step 1: Cluster
        tree = self.cluster_markets(markets)

        # Step 2: Discover relationships
        relationships = self.discover_relationships(markets)

        # Step 3: Check consistency
        inconsistencies = self.check_consistency(markets, relationships)

        # Step 4: Score each inconsistency
        opportunities = []

        for inc in inconsistencies:
            # Build opportunity dict
            opp_dict = {
                "id": f"opp_{inc.market_a.get('id', '')}_{time.time()}",
                "market_id": self._get_market_id(inc.market_a),
                "platform": inc.market_a.get("platform", ""),
                "opportunity_type": inc.inconsistency_type.value,
                "yes_price": inc.price_a,
                "no_price": 1.0 - inc.price_a,
                "edge_pct": inc.edge_pct,
                "confidence": inc.confidence,
                "severity": inc.severity,
                "recommended_action": inc.recommended_action,
            }

            # Add relationship info if available
            if inc.relationship:
                opp_dict["relationship_type"] = inc.relationship.relationship_type
                opp_dict["semantic_similarity"] = inc.relationship.features.get("semantic_similarity", 0)

            # Score
            final_score, component_scores = self.score_opportunity(opp_dict)

            # Filter by thresholds
            if final_score < self.min_score or inc.edge_pct / 100.0 < self.min_edge:
                continue

            # Build scored opportunity
            scored = ScoredOpportunity(
                opportunity_id=opp_dict["id"],
                opportunity_type=inc.inconsistency_type.value,
                markets=[inc.market_a, inc.market_b] if inc.market_b else [inc.market_a],
                market_ids=[self._get_market_id(inc.market_a)],
                ensemble_score=component_scores.get("ensemble", 0),
                specialized_scores={k: v for k, v in component_scores.items() if k not in ["ensemble", "edge_factor", "specialized_max"]},
                final_score=final_score,
                edge_pct=inc.edge_pct,
                confidence=inc.confidence,
                relationship_type=inc.relationship.relationship_type if inc.relationship else None,
                inconsistency_type=inc.inconsistency_type.value,
                inconsistency_severity=inc.severity,
                recommended_action=inc.recommended_action,
                explanation=inc.explanation,
            )

            if inc.market_b:
                scored.market_ids.append(self._get_market_id(inc.market_b))

            opportunities.append(scored)

        # Sort by final score
        opportunities.sort(key=lambda x: -x.final_score)

        logger.info(f"Found {len(opportunities)} scored opportunities")
        return opportunities

    def get_top_opportunities(
        self,
        markets: List[Dict],
        n: int = 10
    ) -> List[ScoredOpportunity]:
        """Get top N opportunities from markets."""
        all_opps = self.score_all_opportunities(markets)
        return all_opps[:n]

    async def score_async(self, markets: List[Dict]) -> List[ScoredOpportunity]:
        """Async wrapper for scoring."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.score_all_opportunities, markets)

    def clear_caches(self) -> None:
        """Clear all caches."""
        self.cluster_cache.clear()
        self.relationship_cache.clear()
        self.analysis_cache.clear()
        self.score_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "ensemble_fitted": self.ensemble.is_fitted,
            "min_score_threshold": self.min_score,
            "min_edge_threshold": self.min_edge,
            "cache_sizes": {
                "cluster": len(self.cluster_cache._cache),
                "relationship": len(self.relationship_cache._cache),
                "analysis": len(self.analysis_cache._cache),
                "score": len(self.score_cache._cache),
            }
        }


# Convenience functions
_engine: Optional[InferenceEngine] = None


def get_inference_engine() -> InferenceEngine:
    """Get or create singleton inference engine."""
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine


def score_markets(markets: List[Dict]) -> List[ScoredOpportunity]:
    """Score markets for opportunities."""
    engine = get_inference_engine()
    return engine.score_all_opportunities(markets)


def get_top_opportunities(markets: List[Dict], n: int = 10) -> List[ScoredOpportunity]:
    """Get top N opportunities."""
    engine = get_inference_engine()
    return engine.get_top_opportunities(markets, n)


async def score_markets_async(markets: List[Dict]) -> List[ScoredOpportunity]:
    """Async market scoring."""
    engine = get_inference_engine()
    return await engine.score_async(markets)
