"""Specialized models for different arbitrage types.

Each model is optimized for a specific type of arbitrage opportunity.
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)

# Model storage
MODEL_DIR = Path(__file__).resolve().parents[3] / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SpecializedPrediction:
    """Prediction from specialized model."""
    opportunity_type: str
    is_opportunity: bool
    probability: float
    confidence: float
    edge_estimate: float
    recommended_action: str
    explanation: str


class SpecializedModel(ABC):
    """Base class for specialized arbitrage models."""

    OPPORTUNITY_TYPE: str = "base"
    MODEL_FILE: str = "base_model.pkl"

    # Feature subsets for this model type
    FEATURE_SUBSET: List[str] = []

    def __init__(self):
        self.model = None
        self.is_fitted = False
        self._load()

    def _get_model_path(self) -> Path:
        return MODEL_DIR / self.MODEL_FILE

    def _save(self) -> None:
        try:
            with open(self._get_model_path(), 'wb') as f:
                pickle.dump({"model": self.model, "is_fitted": self.is_fitted}, f)
        except Exception as e:
            logger.error(f"Error saving {self.OPPORTUNITY_TYPE} model: {e}")

    def _load(self) -> None:
        path = self._get_model_path()
        if path.exists():
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data.get("model")
                    self.is_fitted = data.get("is_fitted", False)
            except Exception as e:
                logger.debug(f"Error loading {self.OPPORTUNITY_TYPE} model: {e}")

    @abstractmethod
    def extract_features(self, item: Dict) -> np.ndarray:
        """Extract model-specific features."""
        pass

    @abstractmethod
    def predict(self, item: Dict) -> SpecializedPrediction:
        """Predict if item is an opportunity."""
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train the model."""
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import cross_val_score

            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
            self.model.fit(X, y)
            self.is_fitted = True

            cv_scores = cross_val_score(self.model, X, y, cv=3, scoring='roc_auc')
            self._save()

            return {
                "train_score": self.model.score(X, y),
                "cv_score": cv_scores.mean(),
                "cv_std": cv_scores.std()
            }
        except ImportError:
            logger.warning("sklearn not available")
            return {"error": "sklearn_not_available"}


class IntraMarketModel(SpecializedModel):
    """
    Model for detecting intra-market arbitrage (overround detection).

    Detects when a single market's prices don't sum correctly.
    """

    OPPORTUNITY_TYPE = "intra_market"
    MODEL_FILE = "intra_market_model.pkl"

    FEATURE_SUBSET = [
        "yes_price", "no_price", "price_sum", "overround",
        "log_volume", "log_liquidity", "time_to_expiry_days"
    ]

    def extract_features(self, item: Dict) -> np.ndarray:
        """Extract features for intra-market analysis."""
        yes_price = float(item.get("yes_price", 0.5))
        no_price = float(item.get("no_price", 0.5))

        price_sum = yes_price + no_price
        overround = price_sum - 1.0

        volume = float(item.get("volume", 0))
        liquidity = float(item.get("liquidity", 0))
        tte = float(item.get("time_to_expiry", 720))

        import math
        return np.array([
            yes_price,
            no_price,
            price_sum,
            overround,
            math.log1p(volume),
            math.log1p(liquidity),
            tte / 24.0  # Convert to days
        ])

    def predict(self, item: Dict) -> SpecializedPrediction:
        """Detect intra-market arbitrage opportunity."""
        features = self.extract_features(item)

        yes_price = features[0]
        no_price = features[1]
        price_sum = features[2]
        overround = features[3]

        # Rule-based detection
        is_opportunity = abs(overround) > 0.03
        edge = abs(overround)

        if overround > 0.03:
            # Overround - prices too high
            action = "SELL_BOTH" if edge > 0.1 else "WATCH"
            explanation = f"Market overround {overround:.1%} - prices sum to {price_sum:.1%}"
        elif overround < -0.03:
            # Underround - prices too low
            action = "BUY_BOTH" if edge > 0.05 else "WATCH"
            explanation = f"Market underround {abs(overround):.1%} - opportunity to buy both sides"
        else:
            action = "PASS"
            explanation = "No significant mispricing detected"

        # Use ML model if available
        probability = 0.5
        if self.is_fitted and self.model is not None:
            try:
                proba = self.model.predict_proba(features.reshape(1, -1))
                probability = float(proba[0, 1]) if proba.ndim > 1 else float(proba[0])
            except:
                pass

        confidence = min(0.5 + edge * 2, 0.95)  # Higher edge = higher confidence

        return SpecializedPrediction(
            opportunity_type=self.OPPORTUNITY_TYPE,
            is_opportunity=is_opportunity,
            probability=probability,
            confidence=confidence,
            edge_estimate=edge * 100,
            recommended_action=action,
            explanation=explanation
        )


class CrossPlatformModel(SpecializedModel):
    """
    Model for detecting cross-platform arbitrage.

    Detects same event priced differently on Polymarket vs Kalshi.
    """

    OPPORTUNITY_TYPE = "cross_platform"
    MODEL_FILE = "cross_platform_model.pkl"

    FEATURE_SUBSET = [
        "price_diff", "similarity_score", "settlement_similarity",
        "volume_ratio", "time_overlap", "category_match"
    ]

    def extract_features(self, item: Dict) -> np.ndarray:
        """Extract features for cross-platform analysis."""
        # Expects item to have poly_* and kalshi_* fields
        poly_price = float(item.get("poly_yes_price") or item.get("yes_price_a", 0.5))
        kalshi_price = float(item.get("kalshi_yes_price") or item.get("yes_price_b", 0.5))

        price_diff = abs(poly_price - kalshi_price)
        similarity = float(item.get("similarity", 0))
        settlement_sim = float(item.get("settlement_sim", 0))

        poly_volume = float(item.get("poly_volume", 0))
        kalshi_volume = float(item.get("kalshi_volume", 0))
        volume_ratio = min(poly_volume, kalshi_volume) / max(poly_volume, kalshi_volume, 1)

        time_overlap = float(item.get("time_overlap", 0.5))
        category_match = float(item.get("category_match", 0))

        return np.array([
            price_diff,
            similarity,
            settlement_sim,
            volume_ratio,
            time_overlap,
            category_match
        ])

    def predict(self, item: Dict) -> SpecializedPrediction:
        """Detect cross-platform arbitrage opportunity."""
        features = self.extract_features(item)

        price_diff = features[0]
        similarity = features[1]
        settlement_sim = features[2]

        # Rule-based: high similarity + price diff = opportunity
        is_opportunity = similarity > 0.8 and price_diff > 0.03

        if is_opportunity:
            poly_price = float(item.get("poly_yes_price") or item.get("yes_price_a", 0.5))
            kalshi_price = float(item.get("kalshi_yes_price") or item.get("yes_price_b", 0.5))

            if poly_price < kalshi_price:
                action = "BUY_POLYMARKET"
                explanation = f"Poly={poly_price:.1%} < Kalshi={kalshi_price:.1%}, buy cheaper Poly"
            else:
                action = "BUY_KALSHI"
                explanation = f"Kalshi={kalshi_price:.1%} < Poly={poly_price:.1%}, buy cheaper Kalshi"
        else:
            action = "PASS"
            explanation = "No significant cross-platform edge"

        # ML model probability
        probability = 0.5
        if self.is_fitted and self.model is not None:
            try:
                proba = self.model.predict_proba(features.reshape(1, -1))
                probability = float(proba[0, 1]) if proba.ndim > 1 else float(proba[0])
            except:
                pass

        confidence = similarity * 0.5 + (1 - abs(0.5 - settlement_sim)) * 0.5

        return SpecializedPrediction(
            opportunity_type=self.OPPORTUNITY_TYPE,
            is_opportunity=is_opportunity,
            probability=probability,
            confidence=confidence,
            edge_estimate=price_diff * 100,
            recommended_action=action,
            explanation=explanation
        )


class RelationshipModel(SpecializedModel):
    """
    Model for detecting relationship-based arbitrage.

    Detects logical inconsistencies in correlated/anti-correlated markets.
    """

    OPPORTUNITY_TYPE = "relationship"
    MODEL_FILE = "relationship_model.pkl"

    FEATURE_SUBSET = [
        "relationship_type", "correlation_strength", "price_divergence",
        "semantic_similarity", "entity_overlap", "confidence"
    ]

    def extract_features(self, item: Dict) -> np.ndarray:
        """Extract features for relationship analysis."""
        # Relationship type encoding
        rel_type = item.get("relationship_type", "independent")
        rel_encoding = {
            "same_event": 1.0,
            "correlated": 0.8,
            "anti_correlated": -0.8,
            "subset": 0.6,
            "exclusive": -0.6,
            "independent": 0.0
        }

        rel_encoded = rel_encoding.get(rel_type, 0.0)
        correlation_strength = float(item.get("correlation_strength") or item.get("confidence", 0.5))

        # Price info
        price_a = float(item.get("price_a") or item.get("yes_price_a", 0.5))
        price_b = float(item.get("price_b") or item.get("yes_price_b", 0.5))
        price_divergence = abs(price_a - price_b)

        semantic_sim = float(item.get("semantic_similarity", 0))
        entity_overlap = float(item.get("entity_overlap", 0))
        confidence = float(item.get("confidence", 0.5))

        return np.array([
            rel_encoded,
            correlation_strength,
            price_divergence,
            semantic_sim,
            entity_overlap,
            confidence
        ])

    def predict(self, item: Dict) -> SpecializedPrediction:
        """Detect relationship-based arbitrage opportunity."""
        features = self.extract_features(item)

        rel_encoded = features[0]
        correlation_strength = features[1]
        price_divergence = features[2]

        rel_type = item.get("relationship_type", "independent")

        # Rule-based detection
        is_opportunity = False
        action = "PASS"
        explanation = "No relationship arbitrage detected"
        edge = 0.0

        if rel_type == "correlated" and price_divergence > 0.05:
            is_opportunity = True
            edge = price_divergence
            action = "BUY_CHEAPER"
            explanation = f"Correlated markets diverged by {price_divergence:.1%}"

        elif rel_type == "anti_correlated":
            price_a = float(item.get("price_a", 0.5))
            price_b = float(item.get("price_b", 0.5))
            sum_deviation = abs((price_a + price_b) - 1.0)

            if sum_deviation > 0.1:
                is_opportunity = True
                edge = sum_deviation
                action = "BUY_BOTH" if (price_a + price_b) < 1.0 else "SELL_HIGHER"
                explanation = f"Anti-correlated sum deviation: {sum_deviation:.1%}"

        elif rel_type == "same_event" and price_divergence > 0.03:
            is_opportunity = True
            edge = price_divergence
            action = "CROSS_PLATFORM_ARB"
            explanation = f"Same event, price diff: {price_divergence:.1%}"

        # ML model probability
        probability = 0.5
        if self.is_fitted and self.model is not None:
            try:
                proba = self.model.predict_proba(features.reshape(1, -1))
                probability = float(proba[0, 1]) if proba.ndim > 1 else float(proba[0])
            except:
                pass

        return SpecializedPrediction(
            opportunity_type=self.OPPORTUNITY_TYPE,
            is_opportunity=is_opportunity,
            probability=probability,
            confidence=correlation_strength,
            edge_estimate=edge * 100,
            recommended_action=action,
            explanation=explanation
        )


class ClusterModel(SpecializedModel):
    """
    Model for detecting cluster-based arbitrage.

    Detects mispricing within event clusters.
    """

    OPPORTUNITY_TYPE = "cluster"
    MODEL_FILE = "cluster_model.pkl"

    FEATURE_SUBSET = [
        "cluster_size", "cluster_mean", "deviation_from_mean",
        "cluster_std", "volume_rank", "is_outlier"
    ]

    def extract_features(self, item: Dict) -> np.ndarray:
        """Extract features for cluster analysis."""
        cluster_size = float(item.get("cluster_size", 1))
        cluster_mean = float(item.get("cluster_price_mean") or item.get("cluster_mean", 0.5))
        cluster_std = float(item.get("cluster_price_std") or item.get("cluster_std", 0))

        yes_price = float(item.get("yes_price", 0.5))
        deviation = abs(yes_price - cluster_mean)

        volume_rank = float(item.get("volume_rank", 0.5))

        # Is outlier (>2 std from mean)
        is_outlier = 1.0 if cluster_std > 0 and deviation > 2 * cluster_std else 0.0

        return np.array([
            min(cluster_size / 20, 1.0),  # Normalized cluster size
            cluster_mean,
            deviation,
            cluster_std,
            volume_rank,
            is_outlier
        ])

    def predict(self, item: Dict) -> SpecializedPrediction:
        """Detect cluster-based arbitrage opportunity."""
        features = self.extract_features(item)

        cluster_size = features[0] * 20  # Denormalize
        cluster_mean = features[1]
        deviation = features[2]
        is_outlier = features[5]

        yes_price = float(item.get("yes_price", 0.5))

        # Rule-based: outlier in cluster = opportunity
        is_opportunity = cluster_size >= 3 and deviation > 0.1

        if is_opportunity:
            if yes_price < cluster_mean:
                action = "BUY_YES"
                explanation = f"Price {yes_price:.1%} below cluster mean {cluster_mean:.1%}"
            else:
                action = "SELL_YES"
                explanation = f"Price {yes_price:.1%} above cluster mean {cluster_mean:.1%}"
            edge = deviation
        else:
            action = "PASS"
            explanation = "No significant cluster deviation"
            edge = 0.0

        # ML model probability
        probability = 0.5
        if self.is_fitted and self.model is not None:
            try:
                proba = self.model.predict_proba(features.reshape(1, -1))
                probability = float(proba[0, 1]) if proba.ndim > 1 else float(proba[0])
            except:
                pass

        confidence = min(0.5 + deviation, 0.9) if cluster_size >= 3 else 0.3

        return SpecializedPrediction(
            opportunity_type=self.OPPORTUNITY_TYPE,
            is_opportunity=is_opportunity,
            probability=probability,
            confidence=confidence,
            edge_estimate=edge * 100,
            recommended_action=action,
            explanation=explanation
        )


# Model registry
SPECIALIZED_MODELS = {
    "intra_market": IntraMarketModel,
    "cross_platform": CrossPlatformModel,
    "relationship": RelationshipModel,
    "cluster": ClusterModel,
}


def get_specialized_model(opportunity_type: str) -> Optional[SpecializedModel]:
    """Get specialized model by opportunity type."""
    model_class = SPECIALIZED_MODELS.get(opportunity_type)
    if model_class:
        return model_class()
    return None


def predict_all_types(item: Dict) -> List[SpecializedPrediction]:
    """Run all specialized models on an item."""
    predictions = []
    for name, model_class in SPECIALIZED_MODELS.items():
        try:
            model = model_class()
            pred = model.predict(item)
            predictions.append(pred)
        except Exception as e:
            logger.debug(f"Error in {name} model: {e}")
    return predictions
