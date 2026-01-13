"""Advanced feature builder for ML models.

Expands from 12 basic features to 50+ features across multiple categories,
based on arXiv paper 2512.02436 "Semantic Trading".
"""

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Complete feature set for a market or opportunity."""
    features: Dict[str, float] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)

    def to_vector(self) -> np.ndarray:
        """Convert to numpy array in consistent order."""
        return np.array([self.features.get(fn, 0.0) for fn in self.feature_names])

    def to_dict(self) -> Dict[str, float]:
        return self.features.copy()


class AdvancedFeatureBuilder:
    """Builds comprehensive feature vectors for ML models."""

    # All feature names organized by category
    PRICE_FEATURES = [
        "yes_price",
        "no_price",
        "price_sum",  # yes + no (should be ~1.0)
        "price_spread",  # abs(yes - no)
        "price_mid",  # (yes + no) / 2
        "price_skew",  # yes - 0.5 (distance from 50%)
        "price_extreme",  # distance from nearest boundary (0 or 1)
        "implied_prob",  # yes_price normalized
        "overround",  # price_sum - 1.0
    ]

    VOLUME_FEATURES = [
        "log_volume",
        "log_volume_24h",
        "volume_momentum",  # volume change ratio
        "volume_rank",  # percentile rank
        "volume_to_liquidity",  # volume / liquidity ratio
        "dollar_volume",  # volume in USD
    ]

    LIQUIDITY_FEATURES = [
        "liquidity",
        "log_liquidity",
        "liquidity_depth",  # estimated depth
        "spread_estimate",  # bid-ask spread estimate
    ]

    TEMPORAL_FEATURES = [
        "time_to_expiry_days",
        "time_to_expiry_hours",
        "log_time_to_expiry",
        "days_since_creation",
        "is_expiring_soon",  # < 24h
        "is_long_term",  # > 30 days
        "hour_of_day",  # 0-23 normalized
        "day_of_week",  # 0-6 normalized
        "is_weekend",
    ]

    SEMANTIC_FEATURES = [
        "category_politics",
        "category_crypto",
        "category_sports",
        "category_economy",
        "category_tech",
        "category_other",
        "entity_count",
        "specificity_score",
        "question_length",
        "has_numeric_threshold",
        "has_date_constraint",
    ]

    RELATIONSHIP_FEATURES = [
        "num_related_markets",
        "num_correlated",
        "num_anti_correlated",
        "avg_relationship_confidence",
        "cross_platform_match",
        "cluster_size",
        "cluster_price_mean",
        "cluster_price_std",
        "deviation_from_cluster",
    ]

    SENTIMENT_FEATURES = [
        "news_mentions",
        "news_sentiment",
        "trend_score",
        "social_volume",
        "sentiment_momentum",
    ]

    ARBITRAGE_FEATURES = [
        "edge_pct",
        "similarity_score",
        "settlement_similarity",
        "confidence_score",
        "relationship_type_encoded",
        "inconsistency_severity",
        "recommended_action_encoded",
    ]

    # Combined feature order
    ALL_FEATURES = (
        PRICE_FEATURES +
        VOLUME_FEATURES +
        LIQUIDITY_FEATURES +
        TEMPORAL_FEATURES +
        SEMANTIC_FEATURES +
        RELATIONSHIP_FEATURES +
        SENTIMENT_FEATURES +
        ARBITRAGE_FEATURES
    )

    def __init__(self):
        self.feature_names = self.ALL_FEATURES.copy()
        self._category_map = {
            "politics": "category_politics",
            "crypto": "category_crypto",
            "sports": "category_sports",
            "economy": "category_economy",
            "tech": "category_tech",
        }

    def _safe_get(self, d: Dict, key: str, default: float = 0.0) -> float:
        """Safely get numeric value from dict."""
        val = d.get(key, default)
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _safe_log(self, val: float) -> float:
        """Safe log transform."""
        return math.log1p(max(val, 0))

    def _parse_prices(self, item: Dict) -> Tuple[float, float]:
        """Extract yes and no prices from various formats."""
        yes_price = self._safe_get(item, "yes_price")
        no_price = self._safe_get(item, "no_price")

        if yes_price == 0 and no_price == 0:
            # Try outcomePrices format
            prices = item.get("outcomePrices")
            if prices:
                if isinstance(prices, str):
                    import json
                    try:
                        prices = json.loads(prices)
                    except:
                        prices = [0.5, 0.5]
                if isinstance(prices, list) and len(prices) >= 2:
                    yes_price = float(prices[0])
                    no_price = float(prices[1])

        if yes_price == 0 and no_price == 0:
            # Try Kalshi format (cents)
            yes_ask = self._safe_get(item, "yes_ask") or self._safe_get(item, "last_price")
            if yes_ask > 1:
                yes_price = yes_ask / 100.0
                no_price = 1.0 - yes_price

        # Ensure valid range
        yes_price = max(0.0, min(1.0, yes_price)) if yes_price else 0.5
        no_price = max(0.0, min(1.0, no_price)) if no_price else 0.5

        return yes_price, no_price

    def _build_price_features(self, item: Dict) -> Dict[str, float]:
        """Build price-related features."""
        yes_price, no_price = self._parse_prices(item)

        price_sum = yes_price + no_price
        price_mid = (yes_price + no_price) / 2

        return {
            "yes_price": yes_price,
            "no_price": no_price,
            "price_sum": price_sum,
            "price_spread": abs(yes_price - no_price),
            "price_mid": price_mid,
            "price_skew": yes_price - 0.5,
            "price_extreme": min(yes_price, 1.0 - yes_price, no_price, 1.0 - no_price),
            "implied_prob": yes_price / price_sum if price_sum > 0 else 0.5,
            "overround": price_sum - 1.0,
        }

    def _build_volume_features(self, item: Dict) -> Dict[str, float]:
        """Build volume-related features."""
        volume = (
            self._safe_get(item, "volume") or
            self._safe_get(item, "poly_volume") or
            self._safe_get(item, "kalshi_volume") or
            self._safe_get(item, "dollar_volume") or
            0
        )
        volume_24h = self._safe_get(item, "volume_24h", volume * 0.1)
        liquidity = self._safe_get(item, "liquidity", 1.0)

        return {
            "log_volume": self._safe_log(volume),
            "log_volume_24h": self._safe_log(volume_24h),
            "volume_momentum": volume_24h / max(volume * 0.1, 1) if volume > 0 else 1.0,
            "volume_rank": min(self._safe_log(volume) / 15, 1.0),  # Normalized log rank
            "volume_to_liquidity": volume / max(liquidity, 1) if liquidity > 0 else 0,
            "dollar_volume": volume,
        }

    def _build_liquidity_features(self, item: Dict) -> Dict[str, float]:
        """Build liquidity-related features."""
        liquidity = self._safe_get(item, "liquidity") or self._safe_get(item, "open_interest", 0)

        return {
            "liquidity": liquidity,
            "log_liquidity": self._safe_log(liquidity),
            "liquidity_depth": min(liquidity / 10000, 1.0),  # Normalized depth
            "spread_estimate": 0.02 if liquidity > 1000 else 0.05,  # Rough spread estimate
        }

    def _build_temporal_features(self, item: Dict) -> Dict[str, float]:
        """Build time-related features."""
        now = time.time()

        # Time to expiry
        close_time = item.get("close_time") or item.get("endDate") or item.get("expiration_time")
        if isinstance(close_time, str):
            try:
                from datetime import datetime
                close_time = datetime.fromisoformat(close_time.replace('Z', '+00:00')).timestamp()
            except:
                close_time = None

        if close_time:
            tte_seconds = max(close_time - now, 0)
            tte_hours = tte_seconds / 3600
            tte_days = tte_seconds / 86400
        else:
            tte_hours = self._safe_get(item, "time_to_expiry", 720)  # Default 30 days
            tte_days = tte_hours / 24
            tte_seconds = tte_hours * 3600

        # Days since creation
        created = item.get("created_at") or item.get("createdAt") or item.get("open_time")
        if isinstance(created, str):
            try:
                from datetime import datetime
                created = datetime.fromisoformat(created.replace('Z', '+00:00')).timestamp()
            except:
                created = None

        days_since = (now - created) / 86400 if created else 30

        # Time of day features
        from datetime import datetime
        dt = datetime.now()

        return {
            "time_to_expiry_days": tte_days,
            "time_to_expiry_hours": tte_hours,
            "log_time_to_expiry": self._safe_log(tte_hours),
            "days_since_creation": days_since,
            "is_expiring_soon": 1.0 if tte_hours < 24 else 0.0,
            "is_long_term": 1.0 if tte_days > 30 else 0.0,
            "hour_of_day": dt.hour / 23.0,
            "day_of_week": dt.weekday() / 6.0,
            "is_weekend": 1.0 if dt.weekday() >= 5 else 0.0,
        }

    def _build_semantic_features(self, item: Dict, analysis: Optional[Any] = None) -> Dict[str, float]:
        """Build semantic/content features."""
        features = {
            "category_politics": 0.0,
            "category_crypto": 0.0,
            "category_sports": 0.0,
            "category_economy": 0.0,
            "category_tech": 0.0,
            "category_other": 0.0,
            "entity_count": 0.0,
            "specificity_score": 0.5,
            "question_length": 0.0,
            "has_numeric_threshold": 0.0,
            "has_date_constraint": 0.0,
        }

        question = item.get("question") or item.get("title") or ""
        features["question_length"] = min(len(question) / 200, 1.0)

        # Check for numeric threshold
        import re
        if re.search(r'\d+[%$]|\$\d+|\d+\s*(percent|%)', question.lower()):
            features["has_numeric_threshold"] = 1.0

        # Check for date constraint
        if re.search(r'\b(by|before|on|in)\s+(january|february|march|april|may|june|july|august|september|october|november|december|\d{4})', question.lower()):
            features["has_date_constraint"] = 1.0

        # Use analysis if provided
        if analysis:
            category = getattr(analysis, 'category', '') or ''
            cat_feature = self._category_map.get(category.lower(), "category_other")
            features[cat_feature] = 1.0

            features["entity_count"] = len(getattr(analysis, 'entities', [])) / 10.0
            features["specificity_score"] = getattr(analysis, 'specificity_score', 0.5)
        else:
            # Quick category detection
            q_lower = question.lower()
            if any(w in q_lower for w in ["election", "trump", "biden", "president", "vote"]):
                features["category_politics"] = 1.0
            elif any(w in q_lower for w in ["bitcoin", "btc", "eth", "crypto", "token"]):
                features["category_crypto"] = 1.0
            elif any(w in q_lower for w in ["nfl", "nba", "super bowl", "championship"]):
                features["category_sports"] = 1.0
            elif any(w in q_lower for w in ["gdp", "inflation", "fed", "interest rate"]):
                features["category_economy"] = 1.0
            elif any(w in q_lower for w in ["ai", "openai", "google", "apple", "tech"]):
                features["category_tech"] = 1.0
            else:
                features["category_other"] = 1.0

        return features

    def _build_relationship_features(self, item: Dict, cluster_info: Optional[Dict] = None) -> Dict[str, float]:
        """Build relationship and cluster features."""
        features = {
            "num_related_markets": self._safe_get(item, "num_related", 0),
            "num_correlated": self._safe_get(item, "num_correlated", 0),
            "num_anti_correlated": self._safe_get(item, "num_anti_correlated", 0),
            "avg_relationship_confidence": self._safe_get(item, "relationship_confidence", 0.5),
            "cross_platform_match": 1.0 if item.get("cross_platform") else 0.0,
            "cluster_size": 1.0,
            "cluster_price_mean": 0.5,
            "cluster_price_std": 0.0,
            "deviation_from_cluster": 0.0,
        }

        if cluster_info:
            features["cluster_size"] = min(cluster_info.get("size", 1) / 20, 1.0)
            features["cluster_price_mean"] = cluster_info.get("price_mean", 0.5)
            features["cluster_price_std"] = cluster_info.get("price_std", 0.0)

            yes_price = self._parse_prices(item)[0]
            features["deviation_from_cluster"] = abs(yes_price - features["cluster_price_mean"])

        return features

    def _build_sentiment_features(self, item: Dict) -> Dict[str, float]:
        """Build sentiment and news features."""
        return {
            "news_mentions": self._safe_get(item, "news_mentions", 0) / 100.0,  # Normalized
            "news_sentiment": self._safe_get(item, "sentiment", 0.5),
            "trend_score": self._safe_get(item, "trend_score", 0.5),
            "social_volume": self._safe_get(item, "social_volume", 0) / 1000.0,
            "sentiment_momentum": self._safe_get(item, "sentiment_momentum", 0),
        }

    def _build_arbitrage_features(self, item: Dict) -> Dict[str, float]:
        """Build arbitrage-specific features."""
        # Encode relationship type
        rel_type = item.get("relationship_type", "")
        rel_encoding = {
            "same_event": 1.0,
            "correlated": 0.8,
            "anti_correlated": -0.8,
            "subset": 0.6,
            "exclusive": -0.6,
            "independent": 0.0,
        }

        # Encode action
        action = item.get("recommended_action", "")
        action_encoding = {
            "BUY_YES": 1.0,
            "BUY_NO": -1.0,
            "BUY_A_YES": 0.8,
            "BUY_B_YES": 0.8,
            "SELL": -0.5,
            "WATCH": 0.0,
        }

        # Encode severity
        severity = item.get("severity", "")
        severity_encoding = {
            "critical": 1.0,
            "high": 0.75,
            "medium": 0.5,
            "low": 0.25,
        }

        return {
            "edge_pct": self._safe_get(item, "edge_pct", 0) / 100.0,  # Normalize to 0-1
            "similarity_score": self._safe_get(item, "similarity", 0),
            "settlement_similarity": self._safe_get(item, "settlement_sim", 0),
            "confidence_score": self._safe_get(item, "confidence", 0.5),
            "relationship_type_encoded": rel_encoding.get(rel_type, 0.0),
            "inconsistency_severity": severity_encoding.get(severity, 0.0),
            "recommended_action_encoded": action_encoding.get(action.split("_")[0] if action else "", 0.0),
        }

    def build_features(
        self,
        item: Dict,
        analysis: Optional[Any] = None,
        cluster_info: Optional[Dict] = None
    ) -> FeatureSet:
        """
        Build complete feature set for a market or opportunity.

        Args:
            item: Market or opportunity dict
            analysis: Optional SemanticAnalysis object
            cluster_info: Optional cluster statistics

        Returns:
            FeatureSet with all features
        """
        features = {}

        # Build all feature categories
        features.update(self._build_price_features(item))
        features.update(self._build_volume_features(item))
        features.update(self._build_liquidity_features(item))
        features.update(self._build_temporal_features(item))
        features.update(self._build_semantic_features(item, analysis))
        features.update(self._build_relationship_features(item, cluster_info))
        features.update(self._build_sentiment_features(item))
        features.update(self._build_arbitrage_features(item))

        return FeatureSet(
            features=features,
            feature_names=self.feature_names
        )

    def build_features_batch(
        self,
        items: List[Dict],
        analyses: Optional[List[Any]] = None,
        cluster_infos: Optional[List[Dict]] = None
    ) -> np.ndarray:
        """
        Build feature matrix for multiple items.

        Returns:
            numpy array of shape (n_items, n_features)
        """
        feature_sets = []

        for i, item in enumerate(items):
            analysis = analyses[i] if analyses and i < len(analyses) else None
            cluster = cluster_infos[i] if cluster_infos and i < len(cluster_infos) else None
            fs = self.build_features(item, analysis, cluster)
            feature_sets.append(fs.to_vector())

        return np.array(feature_sets)

    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names."""
        return self.feature_names.copy()

    def get_feature_categories(self) -> Dict[str, List[str]]:
        """Get features organized by category."""
        return {
            "price": self.PRICE_FEATURES,
            "volume": self.VOLUME_FEATURES,
            "liquidity": self.LIQUIDITY_FEATURES,
            "temporal": self.TEMPORAL_FEATURES,
            "semantic": self.SEMANTIC_FEATURES,
            "relationship": self.RELATIONSHIP_FEATURES,
            "sentiment": self.SENTIMENT_FEATURES,
            "arbitrage": self.ARBITRAGE_FEATURES,
        }


# Legacy compatibility wrapper
def build_features(item: Dict) -> Tuple[List[float], List[str]]:
    """
    Legacy-compatible feature builder.
    Returns (feature_vector, feature_names) tuple.
    """
    builder = AdvancedFeatureBuilder()
    fs = builder.build_features(item)
    return fs.to_vector().tolist(), fs.feature_names


# Convenience functions
_builder: Optional[AdvancedFeatureBuilder] = None


def get_feature_builder() -> AdvancedFeatureBuilder:
    """Get or create singleton feature builder."""
    global _builder
    if _builder is None:
        _builder = AdvancedFeatureBuilder()
    return _builder


def extract_features(item: Dict) -> Dict[str, float]:
    """Extract features as dictionary."""
    builder = get_feature_builder()
    return builder.build_features(item).to_dict()


def extract_features_batch(items: List[Dict]) -> np.ndarray:
    """Extract features for multiple items."""
    builder = get_feature_builder()
    return builder.build_features_batch(items)
