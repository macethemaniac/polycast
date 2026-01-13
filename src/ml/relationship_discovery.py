"""Relationship discovery between prediction markets.

Classifies relationships between market pairs for arbitrage detection,
based on arXiv paper 2512.02436 "Semantic Trading".
"""

import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np

from src.ml.embeddings import EmbeddingStore
from src.ml.semantic_analyzer import SemanticAnalyzer, MarketAnalysis, get_semantic_analyzer
from src.data.schemas import MarketRelationship, RelationshipType

logger = logging.getLogger(__name__)

# Model storage
MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RELATIONSHIP_MODEL_FILE = MODEL_DIR / "relationship_classifier.pkl"


@dataclass
class RelationshipPrediction:
    """Prediction result for a market pair relationship."""
    market_a_id: str
    market_b_id: str
    relationship_type: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class MarketPair:
    """A pair of markets for relationship analysis."""
    market_a: Dict
    market_b: Dict
    embedding_a: Optional[np.ndarray] = None
    embedding_b: Optional[np.ndarray] = None
    analysis_a: Optional[MarketAnalysis] = None
    analysis_b: Optional[MarketAnalysis] = None


class RelationshipFeatureBuilder:
    """Builds feature vectors for market pair relationship classification."""

    def __init__(
        self,
        embedding_store: Optional[EmbeddingStore] = None,
        semantic_analyzer: Optional[SemanticAnalyzer] = None
    ):
        self.embedding_store = embedding_store or EmbeddingStore()
        self.semantic_analyzer = semantic_analyzer or get_semantic_analyzer()

    def _get_market_id(self, market: Dict) -> str:
        return market.get("id") or market.get("market_id") or market.get("ticker") or ""

    def _get_question(self, market: Dict) -> str:
        return market.get("question") or market.get("title") or ""

    def _get_platform(self, market: Dict) -> str:
        return market.get("platform") or "unknown"

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if a is None or b is None:
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _entity_overlap(self, entities_a: List[str], entities_b: List[str]) -> float:
        """Calculate Jaccard similarity of entity sets."""
        if not entities_a or not entities_b:
            return 0.0
        set_a = set(e.lower() for e in entities_a)
        set_b = set(e.lower() for e in entities_b)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _topic_overlap(self, topics_a: List[str], topics_b: List[str]) -> float:
        """Calculate topic overlap score."""
        if not topics_a or not topics_b:
            return 0.0
        set_a = set(t.lower() for t in topics_a)
        set_b = set(t.lower() for t in topics_b)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _time_overlap(self, market_a: Dict, market_b: Dict) -> float:
        """Calculate temporal overlap between markets."""
        # Get close times
        close_a = market_a.get("close_time") or market_a.get("endDate")
        close_b = market_b.get("close_time") or market_b.get("endDate")

        if not close_a or not close_b:
            return 0.5  # Unknown, neutral

        # Parse if string
        if isinstance(close_a, str):
            try:
                from datetime import datetime
                close_a = datetime.fromisoformat(close_a.replace('Z', '+00:00')).timestamp()
            except:
                return 0.5
        if isinstance(close_b, str):
            try:
                from datetime import datetime
                close_b = datetime.fromisoformat(close_b.replace('Z', '+00:00')).timestamp()
            except:
                return 0.5

        # Calculate overlap score based on time difference
        diff_days = abs(close_a - close_b) / 86400  # Convert to days

        if diff_days < 1:
            return 1.0  # Same day
        elif diff_days < 7:
            return 0.8  # Same week
        elif diff_days < 30:
            return 0.5  # Same month
        elif diff_days < 90:
            return 0.3  # Same quarter
        else:
            return 0.1  # Different time frames

    def _category_match(self, analysis_a: MarketAnalysis, analysis_b: MarketAnalysis) -> float:
        """Check if markets are in the same category."""
        if not analysis_a or not analysis_b:
            return 0.0

        if analysis_a.category == analysis_b.category:
            if analysis_a.subcategory == analysis_b.subcategory:
                return 1.0
            return 0.7
        return 0.0

    def _price_correlation(self, market_a: Dict, market_b: Dict) -> float:
        """Estimate price correlation from current prices."""
        # Get yes prices
        price_a = market_a.get("yes_price")
        if price_a is None:
            prices = market_a.get("outcomePrices", [0.5])
            if isinstance(prices, str):
                import json
                try:
                    prices = json.loads(prices)
                except:
                    prices = [0.5]
            price_a = float(prices[0]) if prices else 0.5

        price_b = market_b.get("yes_price")
        if price_b is None:
            prices = market_b.get("outcomePrices", [0.5])
            if isinstance(prices, str):
                import json
                try:
                    prices = json.loads(prices)
                except:
                    prices = [0.5]
            price_b = float(prices[0]) if prices else 0.5

        # Simple correlation proxy: how close are the prices?
        diff = abs(float(price_a) - float(price_b))
        return 1.0 - min(diff, 1.0)

    def build_features(self, pair: MarketPair) -> Dict[str, float]:
        """Build feature dictionary for a market pair."""
        features = {}

        # Semantic similarity (embedding cosine)
        features["semantic_similarity"] = self._cosine_similarity(
            pair.embedding_a, pair.embedding_b
        )

        # Entity overlap
        entities_a = pair.analysis_a.entities if pair.analysis_a else []
        entities_b = pair.analysis_b.entities if pair.analysis_b else []
        features["entity_overlap"] = self._entity_overlap(entities_a, entities_b)

        # Topic overlap
        topics_a = pair.analysis_a.topics if pair.analysis_a else []
        topics_b = pair.analysis_b.topics if pair.analysis_b else []
        features["topic_overlap"] = self._topic_overlap(topics_a, topics_b)

        # Time overlap
        features["time_overlap"] = self._time_overlap(pair.market_a, pair.market_b)

        # Category match
        features["category_match"] = self._category_match(pair.analysis_a, pair.analysis_b)

        # Cross-platform indicator
        platform_a = self._get_platform(pair.market_a)
        platform_b = self._get_platform(pair.market_b)
        features["cross_platform"] = 1.0 if platform_a != platform_b else 0.0

        # Price correlation proxy
        features["price_correlation"] = self._price_correlation(pair.market_a, pair.market_b)

        # Question type match
        qtype_a = pair.analysis_a.question_type if pair.analysis_a else ""
        qtype_b = pair.analysis_b.question_type if pair.analysis_b else ""
        features["question_type_match"] = 1.0 if qtype_a == qtype_b else 0.0

        # Specificity scores
        features["specificity_a"] = pair.analysis_a.specificity_score if pair.analysis_a else 0.0
        features["specificity_b"] = pair.analysis_b.specificity_score if pair.analysis_b else 0.0
        features["specificity_diff"] = abs(features["specificity_a"] - features["specificity_b"])

        return features

    def prepare_pair(self, market_a: Dict, market_b: Dict) -> MarketPair:
        """Prepare a market pair with embeddings and analysis."""
        id_a = self._get_market_id(market_a)
        id_b = self._get_market_id(market_b)
        q_a = self._get_question(market_a)
        q_b = self._get_question(market_b)

        # Get embeddings
        emb_a = self.embedding_store.get_embedding(id_a, q_a)
        emb_b = self.embedding_store.get_embedding(id_b, q_b)

        # Get semantic analysis
        analysis_a = self.semantic_analyzer.analyze_market_fast(market_a)
        analysis_b = self.semantic_analyzer.analyze_market_fast(market_b)

        return MarketPair(
            market_a=market_a,
            market_b=market_b,
            embedding_a=emb_a,
            embedding_b=emb_b,
            analysis_a=analysis_a,
            analysis_b=analysis_b
        )


class RelationshipClassifier:
    """Classifies relationships between market pairs."""

    RELATIONSHIP_TYPES = [
        "correlated",       # If A happens, B more likely
        "anti_correlated",  # If A happens, B less likely
        "same_event",       # Same event, different platforms
        "subset",           # A implies B
        "exclusive",        # A and B cannot both happen
        "independent",      # No relationship
    ]

    def __init__(self):
        self.feature_builder = RelationshipFeatureBuilder()
        self.model = None
        self.feature_names: List[str] = []
        self._load_model()

    def _load_model(self) -> None:
        """Load trained model from disk."""
        if RELATIONSHIP_MODEL_FILE.exists():
            try:
                with open(RELATIONSHIP_MODEL_FILE, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data.get("model")
                    self.feature_names = data.get("feature_names", [])
                logger.info("Loaded relationship classifier model")
            except Exception as e:
                logger.debug(f"Error loading model: {e}")

    def _save_model(self) -> None:
        """Save trained model to disk."""
        try:
            with open(RELATIONSHIP_MODEL_FILE, 'wb') as f:
                pickle.dump({
                    "model": self.model,
                    "feature_names": self.feature_names
                }, f)
            logger.info("Saved relationship classifier model")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def _rule_based_classify(self, features: Dict[str, float]) -> Tuple[str, float]:
        """Rule-based classification when no ML model is available."""
        sem_sim = features.get("semantic_similarity", 0)
        entity_overlap = features.get("entity_overlap", 0)
        cross_platform = features.get("cross_platform", 0)
        price_corr = features.get("price_correlation", 0)
        category_match = features.get("category_match", 0)

        # Same event detection (high similarity, cross-platform)
        if sem_sim > 0.85 and cross_platform > 0.5:
            return "same_event", sem_sim

        # Correlated (high similarity, same platform or cross-platform)
        if sem_sim > 0.7 and entity_overlap > 0.3:
            return "correlated", sem_sim * 0.9

        # Anti-correlated (moderate similarity but prices move opposite)
        if sem_sim > 0.5 and price_corr < 0.3 and category_match > 0.5:
            return "anti_correlated", sem_sim * 0.7

        # Subset (one question contains the other conceptually)
        if sem_sim > 0.6 and entity_overlap > 0.5:
            # Check if one is more specific
            spec_diff = features.get("specificity_diff", 0)
            if spec_diff > 0.3:
                return "subset", sem_sim * 0.8

        # Exclusive (same category, similar topics, but likely mutually exclusive)
        if category_match > 0.8 and entity_overlap > 0.4 and sem_sim < 0.7:
            return "exclusive", 0.6

        # Default: independent
        confidence = 1.0 - sem_sim  # More confident it's independent if low similarity
        return "independent", max(confidence, 0.5)

    def train(
        self,
        training_data: List[Tuple[Dict, int]],
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the relationship classifier.

        Args:
            training_data: List of (features_dict, label) tuples
                          label: 1=correlated/same_event, 0=independent, -1=anti_correlated
            validation_split: Fraction of data to use for validation

        Returns:
            Training metrics
        """
        if len(training_data) < 10:
            logger.warning("Insufficient training data, need at least 10 examples")
            return {"error": "insufficient_data"}

        # Prepare data
        X = []
        y = []
        self.feature_names = list(training_data[0][0].keys())

        for features, label in training_data:
            feature_vec = [features.get(fn, 0.0) for fn in self.feature_names]
            X.append(feature_vec)
            y.append(label)

        X = np.array(X)
        y = np.array(y)

        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )

        # Train model (gradient boosting)
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
            self.model.fit(X_train, y_train)

            # Evaluate
            train_acc = self.model.score(X_train, y_train)
            val_acc = self.model.score(X_val, y_val)

            # Save model
            self._save_model()

            return {
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_features": len(self.feature_names)
            }

        except ImportError:
            logger.warning("sklearn not available, using rule-based classification")
            return {"error": "sklearn_not_available"}

    def predict(self, market_a: Dict, market_b: Dict) -> RelationshipPrediction:
        """Predict the relationship between two markets."""
        # Prepare pair
        pair = self.feature_builder.prepare_pair(market_a, market_b)
        features = self.feature_builder.build_features(pair)

        id_a = market_a.get("id") or market_a.get("market_id") or market_a.get("ticker") or ""
        id_b = market_b.get("id") or market_b.get("market_id") or market_b.get("ticker") or ""

        # Use ML model if available
        if self.model is not None and self.feature_names:
            try:
                feature_vec = np.array([[features.get(fn, 0.0) for fn in self.feature_names]])
                proba = self.model.predict_proba(feature_vec)[0]
                pred_label = self.model.predict(feature_vec)[0]

                # Map label to relationship type
                if pred_label == 1:
                    # Check if same_event or correlated
                    if features.get("semantic_similarity", 0) > 0.85 and features.get("cross_platform", 0) > 0.5:
                        rel_type = "same_event"
                    else:
                        rel_type = "correlated"
                elif pred_label == -1:
                    rel_type = "anti_correlated"
                else:
                    rel_type = "independent"

                confidence = float(max(proba))

                return RelationshipPrediction(
                    market_a_id=id_a,
                    market_b_id=id_b,
                    relationship_type=rel_type,
                    confidence=confidence,
                    probabilities={
                        "correlated": float(proba[2]) if len(proba) > 2 else 0,
                        "independent": float(proba[1]) if len(proba) > 1 else 0,
                        "anti_correlated": float(proba[0]) if len(proba) > 0 else 0,
                    },
                    features=features
                )
            except Exception as e:
                logger.debug(f"ML prediction failed: {e}, falling back to rules")

        # Fallback to rule-based
        rel_type, confidence = self._rule_based_classify(features)

        return RelationshipPrediction(
            market_a_id=id_a,
            market_b_id=id_b,
            relationship_type=rel_type,
            confidence=confidence,
            probabilities={rel_type: confidence},
            features=features
        )

    def predict_batch(
        self,
        markets: List[Dict],
        min_similarity: float = 0.3
    ) -> List[RelationshipPrediction]:
        """
        Predict relationships for all market pairs above similarity threshold.

        Args:
            markets: List of market dicts
            min_similarity: Only return pairs with semantic similarity above this

        Returns:
            List of relationship predictions
        """
        predictions = []

        # Pre-compute embeddings
        items = []
        for m in markets:
            mid = m.get("id") or m.get("market_id") or m.get("ticker") or ""
            q = m.get("question") or m.get("title") or ""
            if mid and q:
                items.append((mid, q))

        embeddings = self.feature_builder.embedding_store.embed(items)

        # Create market -> embedding mapping
        emb_map = {}
        for i, (mid, _) in enumerate(items):
            emb_map[mid] = embeddings[i] if i < len(embeddings) else None

        # Compare all pairs
        for i, market_a in enumerate(markets):
            id_a = market_a.get("id") or market_a.get("market_id") or market_a.get("ticker") or ""
            emb_a = emb_map.get(id_a)

            for market_b in markets[i+1:]:
                id_b = market_b.get("id") or market_b.get("market_id") or market_b.get("ticker") or ""
                emb_b = emb_map.get(id_b)

                # Quick similarity check
                if emb_a is not None and emb_b is not None:
                    sim = self.feature_builder._cosine_similarity(emb_a, emb_b)
                    if sim < min_similarity:
                        continue

                # Full prediction
                pred = self.predict(market_a, market_b)
                if pred.relationship_type != "independent" or pred.confidence < 0.8:
                    predictions.append(pred)

        return predictions

    def find_related_markets(
        self,
        target_market: Dict,
        candidate_markets: List[Dict],
        min_confidence: float = 0.5
    ) -> List[RelationshipPrediction]:
        """Find all markets related to a target market."""
        related = []

        for candidate in candidate_markets:
            pred = self.predict(target_market, candidate)
            if pred.relationship_type != "independent" and pred.confidence >= min_confidence:
                related.append(pred)

        # Sort by confidence
        related.sort(key=lambda x: -x.confidence)
        return related


class RelationshipDiscoveryEngine:
    """High-level engine for discovering market relationships."""

    def __init__(self):
        self.classifier = RelationshipClassifier()
        self.discovered: Dict[str, RelationshipPrediction] = {}

    def _pair_key(self, id_a: str, id_b: str) -> str:
        """Create unique key for market pair."""
        return ":".join(sorted([id_a, id_b]))

    def discover_relationships(
        self,
        markets: List[Dict],
        min_similarity: float = 0.4,
        min_confidence: float = 0.5
    ) -> List[RelationshipPrediction]:
        """
        Discover all significant relationships in a set of markets.

        Returns relationships that are not independent with sufficient confidence.
        """
        predictions = self.classifier.predict_batch(markets, min_similarity)

        # Filter and dedupe
        results = []
        for pred in predictions:
            if pred.confidence < min_confidence:
                continue

            key = self._pair_key(pred.market_a_id, pred.market_b_id)
            if key not in self.discovered:
                self.discovered[key] = pred
                results.append(pred)

        logger.info(f"Discovered {len(results)} new relationships")
        return results

    def get_relationship(self, market_a_id: str, market_b_id: str) -> Optional[RelationshipPrediction]:
        """Get previously discovered relationship."""
        key = self._pair_key(market_a_id, market_b_id)
        return self.discovered.get(key)

    def get_all_relationships(self) -> List[RelationshipPrediction]:
        """Get all discovered relationships."""
        return list(self.discovered.values())

    def get_cross_platform_relationships(self) -> List[RelationshipPrediction]:
        """Get relationships between markets on different platforms."""
        return [
            p for p in self.discovered.values()
            if p.features.get("cross_platform", 0) > 0.5
        ]

    def get_same_event_pairs(self) -> List[RelationshipPrediction]:
        """Get market pairs that are likely the same event."""
        return [
            p for p in self.discovered.values()
            if p.relationship_type == "same_event"
        ]

    def get_correlated_pairs(self) -> List[RelationshipPrediction]:
        """Get correlated market pairs."""
        return [
            p for p in self.discovered.values()
            if p.relationship_type in ["correlated", "same_event"]
        ]

    def get_anti_correlated_pairs(self) -> List[RelationshipPrediction]:
        """Get anti-correlated market pairs."""
        return [
            p for p in self.discovered.values()
            if p.relationship_type == "anti_correlated"
        ]


# Convenience functions
_engine: Optional[RelationshipDiscoveryEngine] = None


def get_discovery_engine() -> RelationshipDiscoveryEngine:
    """Get or create singleton discovery engine."""
    global _engine
    if _engine is None:
        _engine = RelationshipDiscoveryEngine()
    return _engine


def discover_relationships(markets: List[Dict], min_confidence: float = 0.5) -> List[RelationshipPrediction]:
    """Discover relationships between markets."""
    engine = get_discovery_engine()
    return engine.discover_relationships(markets, min_confidence=min_confidence)


def predict_relationship(market_a: Dict, market_b: Dict) -> RelationshipPrediction:
    """Predict relationship between two markets."""
    engine = get_discovery_engine()
    return engine.classifier.predict(market_a, market_b)
