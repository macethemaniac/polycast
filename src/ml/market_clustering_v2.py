"""Advanced hierarchical market clustering.

Multi-level clustering combining semantic analysis and embeddings,
based on arXiv paper 2512.02436 "Semantic Trading".
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from src.ml.embeddings import EmbeddingStore
from src.ml.semantic_analyzer import SemanticAnalyzer, MarketAnalysis, get_semantic_analyzer

logger = logging.getLogger(__name__)


@dataclass
class MarketNode:
    """Node in the cluster hierarchy."""
    market_id: str
    platform: str
    question: str
    embedding: Optional[np.ndarray] = None
    analysis: Optional[MarketAnalysis] = None
    raw: Dict = field(default_factory=dict)


@dataclass
class Cluster:
    """A cluster of related markets."""
    cluster_id: str
    level: str  # "category", "topic", "event", "outcome"
    label: str  # Human-readable cluster name
    markets: List[MarketNode] = field(default_factory=list)
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None

    # Cluster stats
    avg_price: float = 0.0
    total_volume: float = 0.0
    price_std: float = 0.0

    def get_market_ids(self) -> List[str]:
        return [m.market_id for m in self.markets]

    def compute_stats(self) -> None:
        """Compute cluster statistics."""
        if not self.markets:
            return

        prices = []
        volumes = []

        for m in self.markets:
            yes_price = m.raw.get("yes_price") or m.raw.get("outcomePrices", [0.5])[0] if isinstance(m.raw.get("outcomePrices"), list) else 0.5
            if isinstance(yes_price, str):
                try:
                    yes_price = float(yes_price)
                except:
                    yes_price = 0.5
            prices.append(float(yes_price) if yes_price else 0.5)
            volumes.append(float(m.raw.get("volume", 0) or 0))

        self.avg_price = np.mean(prices) if prices else 0.5
        self.price_std = np.std(prices) if len(prices) > 1 else 0.0
        self.total_volume = sum(volumes)


@dataclass
class ClusterTree:
    """Hierarchical tree of market clusters."""
    categories: Dict[str, Cluster] = field(default_factory=dict)  # Level 1
    topics: Dict[str, Cluster] = field(default_factory=dict)      # Level 2
    events: Dict[str, Cluster] = field(default_factory=dict)      # Level 3
    outcomes: Dict[str, Cluster] = field(default_factory=dict)    # Level 4

    # Indexes
    market_to_clusters: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))

    def get_cluster(self, cluster_id: str) -> Optional[Cluster]:
        """Get cluster by ID from any level."""
        for level_dict in [self.categories, self.topics, self.events, self.outcomes]:
            if cluster_id in level_dict:
                return level_dict[cluster_id]
        return None

    def get_market_clusters(self, market_id: str) -> List[Cluster]:
        """Get all clusters containing a market."""
        cluster_ids = self.market_to_clusters.get(market_id, [])
        return [c for c in [self.get_cluster(cid) for cid in cluster_ids] if c is not None]

    def get_cluster_peers(self, market_id: str) -> List[MarketNode]:
        """Get all markets in the same clusters."""
        peers = {}
        for cluster in self.get_market_clusters(market_id):
            for m in cluster.markets:
                if m.market_id != market_id:
                    peers[m.market_id] = m
        return list(peers.values())


class HierarchicalClusterer:
    """Multi-level hierarchical market clustering."""

    def __init__(
        self,
        embedding_store: Optional[EmbeddingStore] = None,
        semantic_analyzer: Optional[SemanticAnalyzer] = None,
        use_llm: bool = False
    ):
        self.embedding_store = embedding_store or EmbeddingStore()
        self.semantic_analyzer = semantic_analyzer or get_semantic_analyzer(use_llm)
        self.use_llm = use_llm

    def _prepare_markets(self, markets: List[Dict]) -> List[MarketNode]:
        """Convert raw market dicts to MarketNode objects with embeddings and analysis."""
        nodes = []

        # Batch embed all markets
        items = []
        for m in markets:
            market_id = m.get("id") or m.get("market_id") or m.get("ticker") or ""
            question = m.get("question") or m.get("title") or ""
            if market_id and question:
                items.append((market_id, question))

        embeddings = self.embedding_store.embed(items) if items else np.array([])

        # Create nodes
        for i, m in enumerate(markets):
            market_id = m.get("id") or m.get("market_id") or m.get("ticker") or ""
            question = m.get("question") or m.get("title") or ""
            platform = m.get("platform") or "unknown"

            if not market_id or not question:
                continue

            # Get embedding
            emb = embeddings[i] if i < len(embeddings) else None

            # Get semantic analysis
            analysis = self.semantic_analyzer.analyze_market_fast(m)

            node = MarketNode(
                market_id=market_id,
                platform=platform,
                question=question,
                embedding=emb,
                analysis=analysis,
                raw=m
            )
            nodes.append(node)

        return nodes

    def _cluster_by_category(self, nodes: List[MarketNode]) -> Dict[str, Cluster]:
        """Level 1: Cluster by category (using semantic analysis)."""
        category_groups: Dict[str, List[MarketNode]] = defaultdict(list)

        for node in nodes:
            category = node.analysis.category if node.analysis else "other"
            category_groups[category].append(node)

        clusters = {}
        for category, markets in category_groups.items():
            cluster_id = f"cat_{category}"
            cluster = Cluster(
                cluster_id=cluster_id,
                level="category",
                label=category.title(),
                markets=markets
            )

            # Compute centroid
            embeddings = [m.embedding for m in markets if m.embedding is not None]
            if embeddings:
                cluster.centroid = np.mean(embeddings, axis=0)

            cluster.compute_stats()
            clusters[cluster_id] = cluster

        logger.info(f"Created {len(clusters)} category clusters")
        return clusters

    def _cluster_by_topic(
        self,
        nodes: List[MarketNode],
        category_clusters: Dict[str, Cluster],
        distance_threshold: float = 0.4
    ) -> Dict[str, Cluster]:
        """Level 2: Cluster by topic within categories (using embeddings)."""
        topic_clusters = {}

        for cat_id, cat_cluster in category_clusters.items():
            if len(cat_cluster.markets) < 2:
                # Single market = its own topic cluster
                topic_id = f"topic_{cat_cluster.markets[0].market_id[:8]}"
                topic_cluster = Cluster(
                    cluster_id=topic_id,
                    level="topic",
                    label=cat_cluster.markets[0].question[:50],
                    markets=cat_cluster.markets.copy(),
                    parent_id=cat_id
                )
                topic_cluster.compute_stats()
                topic_clusters[topic_id] = topic_cluster
                cat_cluster.children.append(topic_id)
                continue

            # Get embeddings for this category
            embeddings = []
            valid_markets = []
            for m in cat_cluster.markets:
                if m.embedding is not None:
                    embeddings.append(m.embedding)
                    valid_markets.append(m)

            if len(embeddings) < 2:
                continue

            # Agglomerative clustering
            X = np.array(embeddings)
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=distance_threshold,
                    metric='cosine',
                    linkage='average'
                )
                labels = clustering.fit_predict(X)
            except Exception as e:
                logger.debug(f"Clustering failed for {cat_id}: {e}")
                labels = [0] * len(valid_markets)

            # Group by cluster label
            label_groups: Dict[int, List[MarketNode]] = defaultdict(list)
            for idx, label in enumerate(labels):
                label_groups[label].append(valid_markets[idx])

            # Create topic clusters
            for label, group_markets in label_groups.items():
                # Use most common entity/topic as label
                topics = []
                for m in group_markets:
                    if m.analysis:
                        topics.extend(m.analysis.topics)
                        topics.extend(m.analysis.entities[:2])

                topic_label = max(set(topics), key=topics.count) if topics else f"Topic {label}"

                topic_id = f"topic_{cat_id}_{label}"
                topic_cluster = Cluster(
                    cluster_id=topic_id,
                    level="topic",
                    label=topic_label,
                    markets=group_markets,
                    parent_id=cat_id
                )

                # Centroid
                group_embeddings = [m.embedding for m in group_markets if m.embedding is not None]
                if group_embeddings:
                    topic_cluster.centroid = np.mean(group_embeddings, axis=0)

                topic_cluster.compute_stats()
                topic_clusters[topic_id] = topic_cluster
                cat_cluster.children.append(topic_id)

        logger.info(f"Created {len(topic_clusters)} topic clusters")
        return topic_clusters

    def _cluster_by_event(
        self,
        topic_clusters: Dict[str, Cluster],
        similarity_threshold: float = 0.8
    ) -> Dict[str, Cluster]:
        """Level 3: Group markets that are about the same event (very high similarity)."""
        event_clusters = {}

        for topic_id, topic_cluster in topic_clusters.items():
            if len(topic_cluster.markets) < 2:
                # Single market = its own event
                event_id = f"event_{topic_cluster.markets[0].market_id[:8]}"
                event_cluster = Cluster(
                    cluster_id=event_id,
                    level="event",
                    label=topic_cluster.markets[0].question[:80],
                    markets=topic_cluster.markets.copy(),
                    parent_id=topic_id
                )
                event_cluster.compute_stats()
                event_clusters[event_id] = event_cluster
                topic_cluster.children.append(event_id)
                continue

            # High-similarity clustering for same-event detection
            embeddings = [m.embedding for m in topic_cluster.markets if m.embedding is not None]
            valid_markets = [m for m in topic_cluster.markets if m.embedding is not None]

            if len(embeddings) < 2:
                continue

            X = np.array(embeddings)
            sim_matrix = cosine_similarity(X)

            # Greedy clustering by similarity
            assigned = set()
            event_groups = []

            for i in range(len(valid_markets)):
                if i in assigned:
                    continue

                group = [i]
                assigned.add(i)

                for j in range(i + 1, len(valid_markets)):
                    if j in assigned:
                        continue
                    if sim_matrix[i, j] >= similarity_threshold:
                        group.append(j)
                        assigned.add(j)

                event_groups.append(group)

            # Create event clusters
            for idx, group_indices in enumerate(event_groups):
                group_markets = [valid_markets[i] for i in group_indices]

                # Use first market's question as label
                event_label = group_markets[0].question[:80]

                event_id = f"event_{topic_id}_{idx}"
                event_cluster = Cluster(
                    cluster_id=event_id,
                    level="event",
                    label=event_label,
                    markets=group_markets,
                    parent_id=topic_id
                )

                # Centroid
                group_embeddings = [m.embedding for m in group_markets if m.embedding is not None]
                if group_embeddings:
                    event_cluster.centroid = np.mean(group_embeddings, axis=0)

                event_cluster.compute_stats()
                event_clusters[event_id] = event_cluster
                topic_cluster.children.append(event_id)

        logger.info(f"Created {len(event_clusters)} event clusters")
        return event_clusters

    def _cluster_by_outcome(
        self,
        event_clusters: Dict[str, Cluster]
    ) -> Dict[str, Cluster]:
        """Level 4: Group markets by specific outcome conditions."""
        outcome_clusters = {}

        for event_id, event_cluster in event_clusters.items():
            # For now, each market in an event is its own outcome
            # In the future, could group by YES/NO conditions
            for i, market in enumerate(event_cluster.markets):
                outcome_id = f"outcome_{market.market_id[:12]}"
                outcome_cluster = Cluster(
                    cluster_id=outcome_id,
                    level="outcome",
                    label=market.question[:100],
                    markets=[market],
                    parent_id=event_id
                )
                outcome_cluster.centroid = market.embedding
                outcome_cluster.compute_stats()
                outcome_clusters[outcome_id] = outcome_cluster
                event_cluster.children.append(outcome_id)

        logger.info(f"Created {len(outcome_clusters)} outcome clusters")
        return outcome_clusters

    def cluster_markets(
        self,
        markets: List[Dict],
        topic_distance: float = 0.4,
        event_similarity: float = 0.8
    ) -> ClusterTree:
        """
        Build hierarchical cluster tree from markets.

        Args:
            markets: List of raw market dicts
            topic_distance: Distance threshold for topic clustering (lower = more clusters)
            event_similarity: Similarity threshold for same-event detection (higher = stricter)

        Returns:
            ClusterTree with all hierarchy levels
        """
        logger.info(f"Clustering {len(markets)} markets...")

        # Prepare nodes
        nodes = self._prepare_markets(markets)
        logger.info(f"Prepared {len(nodes)} valid market nodes")

        # Build hierarchy
        categories = self._cluster_by_category(nodes)
        topics = self._cluster_by_topic(nodes, categories, topic_distance)
        events = self._cluster_by_event(topics, event_similarity)
        outcomes = self._cluster_by_outcome(events)

        # Build tree
        tree = ClusterTree(
            categories=categories,
            topics=topics,
            events=events,
            outcomes=outcomes
        )

        # Build market index
        for level_dict in [categories, topics, events, outcomes]:
            for cluster in level_dict.values():
                for market in cluster.markets:
                    tree.market_to_clusters[market.market_id].append(cluster.cluster_id)

        return tree

    def find_cross_platform_matches(
        self,
        tree: ClusterTree,
        similarity_threshold: float = 0.85
    ) -> List[Tuple[MarketNode, MarketNode, float]]:
        """Find markets on different platforms that are about the same event."""
        matches = []

        # Check event clusters for cross-platform markets
        for cluster in tree.events.values():
            platforms = set(m.platform for m in cluster.markets)
            if len(platforms) < 2:
                continue

            # Group by platform
            by_platform: Dict[str, List[MarketNode]] = defaultdict(list)
            for m in cluster.markets:
                by_platform[m.platform].append(m)

            # Cross-platform matching
            platform_list = list(by_platform.keys())
            for i, p1 in enumerate(platform_list):
                for p2 in platform_list[i+1:]:
                    for m1 in by_platform[p1]:
                        for m2 in by_platform[p2]:
                            if m1.embedding is not None and m2.embedding is not None:
                                sim = float(np.dot(m1.embedding, m2.embedding) /
                                          (np.linalg.norm(m1.embedding) * np.linalg.norm(m2.embedding)))
                                if sim >= similarity_threshold:
                                    matches.append((m1, m2, sim))

        # Sort by similarity
        matches.sort(key=lambda x: -x[2])
        return matches

    def find_price_divergences(
        self,
        tree: ClusterTree,
        min_divergence: float = 0.1
    ) -> List[Dict]:
        """Find clusters where prices diverge significantly."""
        divergences = []

        # Check topic and event clusters
        for level_name, level_dict in [("topic", tree.topics), ("event", tree.events)]:
            for cluster in level_dict.values():
                if len(cluster.markets) < 2:
                    continue

                # Get prices
                prices = []
                for m in cluster.markets:
                    yes_price = m.raw.get("yes_price")
                    if yes_price is None:
                        outcome_prices = m.raw.get("outcomePrices")
                        if outcome_prices:
                            if isinstance(outcome_prices, str):
                                import json
                                try:
                                    outcome_prices = json.loads(outcome_prices)
                                except:
                                    outcome_prices = [0.5]
                            yes_price = float(outcome_prices[0]) if outcome_prices else 0.5

                    prices.append((m, float(yes_price) if yes_price else 0.5))

                if len(prices) < 2:
                    continue

                # Find max divergence
                prices.sort(key=lambda x: x[1])
                low_market, low_price = prices[0]
                high_market, high_price = prices[-1]

                divergence = high_price - low_price
                if divergence >= min_divergence:
                    divergences.append({
                        "cluster_id": cluster.cluster_id,
                        "cluster_level": level_name,
                        "cluster_label": cluster.label,
                        "divergence": divergence,
                        "low_market": low_market,
                        "low_price": low_price,
                        "high_market": high_market,
                        "high_price": high_price,
                        "num_markets": len(cluster.markets)
                    })

        # Sort by divergence
        divergences.sort(key=lambda x: -x["divergence"])
        return divergences


# Convenience functions
_clusterer: Optional[HierarchicalClusterer] = None


def get_clusterer(use_llm: bool = False) -> HierarchicalClusterer:
    """Get or create singleton clusterer."""
    global _clusterer
    if _clusterer is None:
        _clusterer = HierarchicalClusterer(use_llm=use_llm)
    return _clusterer


def cluster_markets(markets: List[Dict], use_llm: bool = False) -> ClusterTree:
    """Cluster markets into hierarchical tree."""
    clusterer = get_clusterer(use_llm)
    return clusterer.cluster_markets(markets)


def find_arbitrage_clusters(markets: List[Dict], min_divergence: float = 0.1) -> List[Dict]:
    """Find clusters with price divergences (potential arbitrage)."""
    clusterer = get_clusterer()
    tree = clusterer.cluster_markets(markets)
    return clusterer.find_price_divergences(tree, min_divergence)
