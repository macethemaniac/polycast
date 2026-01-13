"""Logical consistency checker for prediction markets.

Detects probability violations and inconsistencies across related markets,
which signal potential arbitrage opportunities.

Based on arXiv paper 2512.02436 "Semantic Trading".
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np

from src.ml.relationship_discovery import (
    RelationshipDiscoveryEngine,
    RelationshipPrediction,
    get_discovery_engine
)

logger = logging.getLogger(__name__)


class InconsistencyType(Enum):
    """Types of logical inconsistencies."""
    CORRELATED_DIVERGENCE = "correlated_divergence"      # Correlated markets with different prices
    ANTI_CORRELATED_SUM = "anti_correlated_sum"          # Anti-correlated sum != ~1.0
    SUBSET_VIOLATION = "subset_violation"                 # Subset has higher price than superset
    EXCLUSIVE_SUM = "exclusive_sum"                       # Exclusive events sum > 1.0
    CROSS_PLATFORM_ARBITRAGE = "cross_platform_arbitrage" # Same event, different prices
    INTRA_MARKET_OVERROUND = "intra_market_overround"    # Market prices don't sum correctly


@dataclass
class Inconsistency:
    """Detected inconsistency in market prices."""
    inconsistency_type: InconsistencyType
    severity: str  # "low", "medium", "high", "critical"
    edge_pct: float  # Potential profit edge as percentage
    confidence: float  # Confidence in the inconsistency detection

    # Markets involved
    market_a: Dict = field(default_factory=dict)
    market_b: Optional[Dict] = None

    # Prices
    price_a: float = 0.0
    price_b: float = 0.0
    expected_price_a: Optional[float] = None
    expected_price_b: Optional[float] = None

    # Relationship context
    relationship: Optional[RelationshipPrediction] = None

    # Action recommendation
    recommended_action: str = ""  # "BUY_A_YES", "BUY_B_NO", etc.
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.inconsistency_type.value,
            "severity": self.severity,
            "edge_pct": self.edge_pct,
            "confidence": self.confidence,
            "market_a_id": self.market_a.get("id") or self.market_a.get("market_id") or "",
            "market_b_id": self.market_b.get("id") or self.market_b.get("market_id") or "" if self.market_b else None,
            "price_a": self.price_a,
            "price_b": self.price_b,
            "recommended_action": self.recommended_action,
            "explanation": self.explanation
        }


class LogicalConsistencyChecker:
    """Checks logical consistency across related markets."""

    # Thresholds for different inconsistency types
    CORRELATED_THRESHOLD = 0.05       # Max allowed price diff for correlated markets
    ANTI_CORRELATED_TOLERANCE = 0.1   # Tolerance for anti-correlated sum from 1.0
    EXCLUSIVE_MAX_SUM = 1.05          # Max allowed sum for exclusive events
    SUBSET_TOLERANCE = 0.02           # Tolerance for subset price exceeding superset
    CROSS_PLATFORM_MIN_EDGE = 0.03    # Min edge for cross-platform arbitrage
    OVERROUND_THRESHOLD = 0.03        # Max overround before flagging

    def __init__(self, discovery_engine: Optional[RelationshipDiscoveryEngine] = None):
        self.discovery_engine = discovery_engine or get_discovery_engine()

    def _get_yes_price(self, market: Dict) -> float:
        """Extract YES price from market dict."""
        price = market.get("yes_price")
        if price is not None:
            return float(price)

        prices = market.get("outcomePrices")
        if prices:
            if isinstance(prices, str):
                import json
                try:
                    prices = json.loads(prices)
                except:
                    prices = [0.5]
            return float(prices[0]) if prices else 0.5

        # Kalshi format
        yes_ask = market.get("yes_ask") or market.get("last_price")
        if yes_ask:
            price = float(yes_ask)
            if price > 1:  # Cents to decimal
                price = price / 100.0
            return price

        return 0.5

    def _get_no_price(self, market: Dict) -> float:
        """Extract NO price from market dict."""
        price = market.get("no_price")
        if price is not None:
            return float(price)

        prices = market.get("outcomePrices")
        if prices:
            if isinstance(prices, str):
                import json
                try:
                    prices = json.loads(prices)
                except:
                    prices = [0.5, 0.5]
            return float(prices[1]) if len(prices) > 1 else (1.0 - float(prices[0]))

        # Kalshi format
        no_ask = market.get("no_ask")
        if no_ask:
            price = float(no_ask)
            if price > 1:
                price = price / 100.0
            return price

        # Default: 1 - yes_price
        return 1.0 - self._get_yes_price(market)

    def _severity_from_edge(self, edge: float) -> str:
        """Determine severity based on edge size."""
        if edge >= 0.15:
            return "critical"
        elif edge >= 0.10:
            return "high"
        elif edge >= 0.05:
            return "medium"
        else:
            return "low"

    def check_intra_market(self, market: Dict) -> Optional[Inconsistency]:
        """Check for overround within a single market."""
        yes_price = self._get_yes_price(market)
        no_price = self._get_no_price(market)

        total = yes_price + no_price
        overround = total - 1.0

        if abs(overround) > self.OVERROUND_THRESHOLD:
            edge = abs(overround)

            if overround > 0:
                # Overround - prices too high
                action = "SELL" if overround > 0.1 else "WATCH"
                explanation = f"Market overround of {overround:.1%} - prices sum to {total:.1%}"
            else:
                # Underround - prices too low, potential arb
                action = "BUY_BOTH" if abs(overround) > 0.05 else "WATCH"
                explanation = f"Market underround of {abs(overround):.1%} - prices sum to {total:.1%}"

            return Inconsistency(
                inconsistency_type=InconsistencyType.INTRA_MARKET_OVERROUND,
                severity=self._severity_from_edge(edge),
                edge_pct=edge * 100,
                confidence=0.95,  # High confidence for simple math
                market_a=market,
                price_a=yes_price,
                price_b=no_price,
                recommended_action=action,
                explanation=explanation
            )

        return None

    def check_correlated_pair(
        self,
        market_a: Dict,
        market_b: Dict,
        relationship: RelationshipPrediction
    ) -> Optional[Inconsistency]:
        """Check for price divergence in correlated markets."""
        price_a = self._get_yes_price(market_a)
        price_b = self._get_yes_price(market_b)

        diff = abs(price_a - price_b)

        if diff > self.CORRELATED_THRESHOLD:
            # Prices should be similar for correlated markets
            edge = diff

            # Recommend buying the cheaper one
            if price_a < price_b:
                action = "BUY_A_YES"
                explanation = f"Correlated markets diverged: A={price_a:.1%} vs B={price_b:.1%}. Buy cheaper A."
            else:
                action = "BUY_B_YES"
                explanation = f"Correlated markets diverged: A={price_a:.1%} vs B={price_b:.1%}. Buy cheaper B."

            return Inconsistency(
                inconsistency_type=InconsistencyType.CORRELATED_DIVERGENCE,
                severity=self._severity_from_edge(edge),
                edge_pct=edge * 100,
                confidence=relationship.confidence,
                market_a=market_a,
                market_b=market_b,
                price_a=price_a,
                price_b=price_b,
                expected_price_a=(price_a + price_b) / 2,
                expected_price_b=(price_a + price_b) / 2,
                relationship=relationship,
                recommended_action=action,
                explanation=explanation
            )

        return None

    def check_anti_correlated_pair(
        self,
        market_a: Dict,
        market_b: Dict,
        relationship: RelationshipPrediction
    ) -> Optional[Inconsistency]:
        """Check if anti-correlated markets sum to ~1.0."""
        price_a = self._get_yes_price(market_a)
        price_b = self._get_yes_price(market_b)

        total = price_a + price_b
        deviation = abs(total - 1.0)

        if deviation > self.ANTI_CORRELATED_TOLERANCE:
            edge = deviation

            if total > 1.0:
                # Sum too high - sell both or short the higher one
                action = "SELL_HIGHER"
                explanation = f"Anti-correlated sum {total:.1%} > 100%. Likely overpriced."
            else:
                # Sum too low - buy both
                action = "BUY_BOTH"
                explanation = f"Anti-correlated sum {total:.1%} < 100%. Both underpriced."

            return Inconsistency(
                inconsistency_type=InconsistencyType.ANTI_CORRELATED_SUM,
                severity=self._severity_from_edge(edge),
                edge_pct=edge * 100,
                confidence=relationship.confidence * 0.9,  # Slightly lower confidence
                market_a=market_a,
                market_b=market_b,
                price_a=price_a,
                price_b=price_b,
                expected_price_a=price_a * (1.0 / total) if total > 0 else 0.5,
                expected_price_b=price_b * (1.0 / total) if total > 0 else 0.5,
                relationship=relationship,
                recommended_action=action,
                explanation=explanation
            )

        return None

    def check_subset_pair(
        self,
        subset_market: Dict,
        superset_market: Dict,
        relationship: RelationshipPrediction
    ) -> Optional[Inconsistency]:
        """Check if subset probability <= superset probability."""
        subset_price = self._get_yes_price(subset_market)
        superset_price = self._get_yes_price(superset_market)

        # Subset should never be more likely than superset
        if subset_price > superset_price + self.SUBSET_TOLERANCE:
            edge = subset_price - superset_price

            return Inconsistency(
                inconsistency_type=InconsistencyType.SUBSET_VIOLATION,
                severity=self._severity_from_edge(edge),
                edge_pct=edge * 100,
                confidence=relationship.confidence * 0.85,
                market_a=subset_market,
                market_b=superset_market,
                price_a=subset_price,
                price_b=superset_price,
                expected_price_a=min(subset_price, superset_price - 0.01),
                relationship=relationship,
                recommended_action="SELL_SUBSET_BUY_SUPERSET",
                explanation=f"Subset ({subset_price:.1%}) > superset ({superset_price:.1%}). Logical violation."
            )

        return None

    def check_exclusive_group(
        self,
        markets: List[Dict],
        relationships: List[RelationshipPrediction]
    ) -> Optional[Inconsistency]:
        """Check if exclusive events sum to <= 1.0."""
        total = sum(self._get_yes_price(m) for m in markets)

        if total > self.EXCLUSIVE_MAX_SUM:
            edge = total - 1.0

            # Find the most overpriced market
            prices = [(m, self._get_yes_price(m)) for m in markets]
            prices.sort(key=lambda x: -x[1])
            highest_market, highest_price = prices[0]

            return Inconsistency(
                inconsistency_type=InconsistencyType.EXCLUSIVE_SUM,
                severity=self._severity_from_edge(edge),
                edge_pct=edge * 100,
                confidence=min(r.confidence for r in relationships) * 0.8,
                market_a=highest_market,
                price_a=highest_price,
                price_b=total,
                recommended_action="SELL_HIGHEST",
                explanation=f"Exclusive events sum to {total:.1%}. Highest at {highest_price:.1%} likely overpriced."
            )

        return None

    def check_cross_platform_pair(
        self,
        market_a: Dict,
        market_b: Dict,
        relationship: RelationshipPrediction
    ) -> Optional[Inconsistency]:
        """Check for arbitrage between same event on different platforms."""
        price_a = self._get_yes_price(market_a)
        price_b = self._get_yes_price(market_b)

        diff = abs(price_a - price_b)

        if diff >= self.CROSS_PLATFORM_MIN_EDGE:
            platform_a = market_a.get("platform", "unknown")
            platform_b = market_b.get("platform", "unknown")

            # Buy on cheaper platform, effectively short on expensive
            if price_a < price_b:
                action = f"BUY_{platform_a.upper()}_SELL_{platform_b.upper()}"
                explanation = f"Cross-platform arb: {platform_a}={price_a:.1%} vs {platform_b}={price_b:.1%}"
            else:
                action = f"BUY_{platform_b.upper()}_SELL_{platform_a.upper()}"
                explanation = f"Cross-platform arb: {platform_a}={price_a:.1%} vs {platform_b}={price_b:.1%}"

            return Inconsistency(
                inconsistency_type=InconsistencyType.CROSS_PLATFORM_ARBITRAGE,
                severity=self._severity_from_edge(diff),
                edge_pct=diff * 100,
                confidence=relationship.confidence,
                market_a=market_a,
                market_b=market_b,
                price_a=price_a,
                price_b=price_b,
                expected_price_a=(price_a + price_b) / 2,
                expected_price_b=(price_a + price_b) / 2,
                relationship=relationship,
                recommended_action=action,
                explanation=explanation
            )

        return None

    def check_all(
        self,
        markets: List[Dict],
        relationships: Optional[List[RelationshipPrediction]] = None
    ) -> List[Inconsistency]:
        """
        Check all markets and relationships for inconsistencies.

        Args:
            markets: List of market dicts
            relationships: Pre-computed relationships, or will discover them

        Returns:
            List of detected inconsistencies, sorted by edge size
        """
        inconsistencies = []

        # Check intra-market consistency
        for market in markets:
            result = self.check_intra_market(market)
            if result:
                inconsistencies.append(result)

        # Discover relationships if not provided
        if relationships is None:
            relationships = self.discovery_engine.discover_relationships(markets)

        # Build market lookup
        market_lookup = {}
        for m in markets:
            mid = m.get("id") or m.get("market_id") or m.get("ticker") or ""
            if mid:
                market_lookup[mid] = m

        # Check each relationship
        for rel in relationships:
            market_a = market_lookup.get(rel.market_a_id)
            market_b = market_lookup.get(rel.market_b_id)

            if not market_a or not market_b:
                continue

            # Check based on relationship type
            if rel.relationship_type == "same_event":
                result = self.check_cross_platform_pair(market_a, market_b, rel)
                if result:
                    inconsistencies.append(result)

            elif rel.relationship_type == "correlated":
                result = self.check_correlated_pair(market_a, market_b, rel)
                if result:
                    inconsistencies.append(result)

            elif rel.relationship_type == "anti_correlated":
                result = self.check_anti_correlated_pair(market_a, market_b, rel)
                if result:
                    inconsistencies.append(result)

            elif rel.relationship_type == "subset":
                result = self.check_subset_pair(market_a, market_b, rel)
                if result:
                    inconsistencies.append(result)

        # Sort by edge (highest first)
        inconsistencies.sort(key=lambda x: -x.edge_pct)

        logger.info(f"Found {len(inconsistencies)} inconsistencies")
        return inconsistencies

    def get_arbitrage_opportunities(
        self,
        markets: List[Dict],
        min_edge: float = 0.03,
        min_confidence: float = 0.6
    ) -> List[Inconsistency]:
        """
        Get actionable arbitrage opportunities.

        Filters inconsistencies to only those worth trading.
        """
        all_inconsistencies = self.check_all(markets)

        opportunities = [
            inc for inc in all_inconsistencies
            if inc.edge_pct >= min_edge * 100 and inc.confidence >= min_confidence
        ]

        return opportunities


# Convenience functions
_checker: Optional[LogicalConsistencyChecker] = None


def get_consistency_checker() -> LogicalConsistencyChecker:
    """Get or create singleton checker."""
    global _checker
    if _checker is None:
        _checker = LogicalConsistencyChecker()
    return _checker


def check_consistency(markets: List[Dict]) -> List[Inconsistency]:
    """Check markets for logical inconsistencies."""
    checker = get_consistency_checker()
    return checker.check_all(markets)


def find_arbitrage(markets: List[Dict], min_edge: float = 0.03) -> List[Inconsistency]:
    """Find arbitrage opportunities in markets."""
    checker = get_consistency_checker()
    return checker.get_arbitrage_opportunities(markets, min_edge=min_edge)
