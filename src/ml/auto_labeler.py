"""Automatic labeling system for ML training.

Labels opportunities based on actual outcomes for supervised learning.
"""

import logging
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

# Data storage
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "training"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LABELED_DATA_FILE = DATA_DIR / "labeled_opportunities.jsonl"


class Label(Enum):
    """Opportunity labels."""
    PROFITABLE = 1       # Made money
    UNPROFITABLE = 0     # Lost money
    UNCERTAIN = -1       # Not enough data


@dataclass
class LabeledOpportunity:
    """Opportunity with outcome label."""
    opportunity_id: str
    timestamp: float
    features: Dict[str, float]

    # Label info
    label: int  # 1, 0, or -1
    label_source: str  # "resolution", "price_movement", "manual"

    # Context
    market_id: str = ""
    platform: str = ""
    opportunity_type: str = ""

    # Outcome tracking
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    hold_time_hours: Optional[float] = None

    # Market resolution
    resolution: Optional[str] = None  # "yes", "no", "cancelled"
    resolved_at: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "opportunity_id": self.opportunity_id,
            "timestamp": self.timestamp,
            "features": self.features,
            "label": self.label,
            "label_source": self.label_source,
            "market_id": self.market_id,
            "platform": self.platform,
            "opportunity_type": self.opportunity_type,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "hold_time_hours": self.hold_time_hours,
            "resolution": self.resolution,
            "resolved_at": self.resolved_at,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "LabeledOpportunity":
        return cls(**d)


class AutoLabeler:
    """Automatically labels opportunities based on outcomes."""

    # Thresholds for labeling
    MIN_PROFIT_PCT = 0.05        # 5% profit to be "profitable"
    MAX_LOSS_PCT = -0.05         # 5% loss to be "unprofitable"
    MIN_HOLD_HOURS = 1           # Minimum hold time to label
    MAX_HORIZON_HOURS = 168      # 1 week max horizon

    def __init__(self):
        self.labeled: Dict[str, LabeledOpportunity] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing labeled data."""
        if LABELED_DATA_FILE.exists():
            try:
                with open(LABELED_DATA_FILE, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            opp = LabeledOpportunity.from_dict(data)
                            self.labeled[opp.opportunity_id] = opp
                logger.info(f"Loaded {len(self.labeled)} labeled opportunities")
            except Exception as e:
                logger.error(f"Error loading labeled data: {e}")

    def _save_labeled(self, opp: LabeledOpportunity) -> None:
        """Save labeled opportunity."""
        try:
            with open(LABELED_DATA_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(opp.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Error saving labeled opportunity: {e}")

    def label_from_resolution(
        self,
        opportunity: Dict,
        resolution: str,
        resolved_at: float
    ) -> LabeledOpportunity:
        """
        Label opportunity based on market resolution.

        If we recommended BUY_YES and market resolved YES -> profitable
        If we recommended BUY_YES and market resolved NO -> unprofitable
        """
        opp_id = opportunity.get("opportunity_id") or opportunity.get("id") or ""
        action = opportunity.get("recommended_action") or ""
        entry_price = float(opportunity.get("entry_price") or opportunity.get("yes_price") or 0.5)

        # Determine if profitable based on action and resolution
        if "YES" in action.upper():
            # Bought YES
            if resolution.lower() == "yes":
                pnl = 1.0 - entry_price  # Won
                label = Label.PROFITABLE.value if pnl >= self.MIN_PROFIT_PCT else Label.UNCERTAIN.value
            elif resolution.lower() == "no":
                pnl = -entry_price  # Lost
                label = Label.UNPROFITABLE.value
            else:
                pnl = 0
                label = Label.UNCERTAIN.value
        elif "NO" in action.upper():
            # Bought NO
            if resolution.lower() == "no":
                pnl = 1.0 - (1.0 - entry_price)
                label = Label.PROFITABLE.value if pnl >= self.MIN_PROFIT_PCT else Label.UNCERTAIN.value
            elif resolution.lower() == "yes":
                pnl = -(1.0 - entry_price)
                label = Label.UNPROFITABLE.value
            else:
                pnl = 0
                label = Label.UNCERTAIN.value
        else:
            # Unknown action
            pnl = None
            label = Label.UNCERTAIN.value

        # Calculate hold time
        opp_time = opportunity.get("timestamp") or time.time()
        hold_hours = (resolved_at - opp_time) / 3600 if resolved_at > opp_time else None

        labeled_opp = LabeledOpportunity(
            opportunity_id=opp_id,
            timestamp=opp_time,
            features=opportunity.get("features", {}),
            label=label,
            label_source="resolution",
            market_id=opportunity.get("market_id", ""),
            platform=opportunity.get("platform", ""),
            opportunity_type=opportunity.get("opportunity_type", ""),
            entry_price=entry_price,
            exit_price=1.0 if label == Label.PROFITABLE.value else 0.0,
            pnl=pnl,
            hold_time_hours=hold_hours,
            resolution=resolution,
            resolved_at=resolved_at,
        )

        self.labeled[opp_id] = labeled_opp
        self._save_labeled(labeled_opp)

        return labeled_opp

    def label_from_price_movement(
        self,
        opportunity: Dict,
        future_prices: List[Tuple[float, float]],  # [(timestamp, price), ...]
        horizon_hours: int = 24
    ) -> LabeledOpportunity:
        """
        Label opportunity based on price movement within horizon.

        Profitable if price moved in predicted direction by >= 5%
        """
        opp_id = opportunity.get("opportunity_id") or opportunity.get("id") or ""
        action = opportunity.get("recommended_action") or ""
        entry_price = float(opportunity.get("entry_price") or opportunity.get("yes_price") or 0.5)
        opp_time = opportunity.get("timestamp") or time.time()

        # Filter to horizon
        horizon_end = opp_time + horizon_hours * 3600
        valid_prices = [(t, p) for t, p in future_prices if opp_time < t <= horizon_end]

        if not valid_prices:
            label = Label.UNCERTAIN.value
            pnl = None
            exit_price = None
            hold_hours = None
        else:
            # Get best exit price within horizon
            if "YES" in action.upper():
                # Want price to go up
                best_exit = max(valid_prices, key=lambda x: x[1])
                exit_price = best_exit[1]
                pnl = exit_price - entry_price
            elif "NO" in action.upper():
                # Want YES price to go down (NO to go up)
                best_exit = min(valid_prices, key=lambda x: x[1])
                exit_price = best_exit[1]
                pnl = entry_price - exit_price
            else:
                exit_price = valid_prices[-1][1]
                pnl = abs(exit_price - entry_price) - 0.02  # Assume 2% fees

            hold_hours = (best_exit[0] - opp_time) / 3600

            # Determine label
            if pnl >= self.MIN_PROFIT_PCT:
                label = Label.PROFITABLE.value
            elif pnl <= self.MAX_LOSS_PCT:
                label = Label.UNPROFITABLE.value
            else:
                label = Label.UNCERTAIN.value

        labeled_opp = LabeledOpportunity(
            opportunity_id=opp_id,
            timestamp=opp_time,
            features=opportunity.get("features", {}),
            label=label,
            label_source="price_movement",
            market_id=opportunity.get("market_id", ""),
            platform=opportunity.get("platform", ""),
            opportunity_type=opportunity.get("opportunity_type", ""),
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            hold_time_hours=hold_hours,
        )

        self.labeled[opp_id] = labeled_opp
        self._save_labeled(labeled_opp)

        return labeled_opp

    def label_cross_platform(
        self,
        poly_opportunity: Dict,
        kalshi_opportunity: Dict,
        poly_resolution: Optional[str] = None,
        kalshi_resolution: Optional[str] = None
    ) -> LabeledOpportunity:
        """
        Label cross-platform arbitrage based on resolutions.

        Profitable if both markets resolved same way (true arbitrage).
        """
        opp_id = f"cross_{poly_opportunity.get('market_id')}_{kalshi_opportunity.get('market_id')}"

        poly_price = float(poly_opportunity.get("yes_price", 0.5))
        kalshi_price = float(kalshi_opportunity.get("yes_price", 0.5))
        edge = abs(poly_price - kalshi_price)

        # If both resolved, check if arbitrage would have worked
        if poly_resolution and kalshi_resolution:
            if poly_resolution.lower() == kalshi_resolution.lower():
                # Same resolution - arb would have worked
                label = Label.PROFITABLE.value
                pnl = edge  # Captured the edge
            else:
                # Different resolution - not same event!
                label = Label.UNPROFITABLE.value
                pnl = -max(poly_price, kalshi_price)  # Lost on one side
        else:
            label = Label.UNCERTAIN.value
            pnl = None

        labeled_opp = LabeledOpportunity(
            opportunity_id=opp_id,
            timestamp=time.time(),
            features={
                "poly_price": poly_price,
                "kalshi_price": kalshi_price,
                "edge": edge,
                "poly_resolution": poly_resolution or "",
                "kalshi_resolution": kalshi_resolution or "",
            },
            label=label,
            label_source="cross_platform_resolution",
            market_id=poly_opportunity.get("market_id", ""),
            platform="cross_platform",
            opportunity_type="cross_platform",
            entry_price=min(poly_price, kalshi_price),
            pnl=pnl,
        )

        self.labeled[opp_id] = labeled_opp
        self._save_labeled(labeled_opp)

        return labeled_opp

    def get_training_data(
        self,
        min_confidence: float = 0.0,
        exclude_uncertain: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Get labeled data for training.

        Returns:
            X: Feature matrix
            y: Labels
            feature_names: Feature names
        """
        X = []
        y = []
        feature_names = None

        for opp in self.labeled.values():
            if exclude_uncertain and opp.label == Label.UNCERTAIN.value:
                continue

            if not opp.features:
                continue

            if feature_names is None:
                feature_names = list(opp.features.keys())

            feature_vec = [opp.features.get(fn, 0.0) for fn in feature_names]
            X.append(feature_vec)
            y.append(opp.label)

        if not X:
            return np.array([]), np.array([]), []

        return np.array(X), np.array(y), feature_names or []

    def get_stats(self) -> Dict[str, Any]:
        """Get labeling statistics."""
        labels = [opp.label for opp in self.labeled.values()]

        return {
            "total_labeled": len(self.labeled),
            "profitable": labels.count(Label.PROFITABLE.value),
            "unprofitable": labels.count(Label.UNPROFITABLE.value),
            "uncertain": labels.count(Label.UNCERTAIN.value),
            "by_source": {
                "resolution": len([o for o in self.labeled.values() if o.label_source == "resolution"]),
                "price_movement": len([o for o in self.labeled.values() if o.label_source == "price_movement"]),
                "cross_platform": len([o for o in self.labeled.values() if o.label_source == "cross_platform_resolution"]),
            },
            "avg_pnl": np.mean([o.pnl for o in self.labeled.values() if o.pnl is not None]) if self.labeled else 0,
        }


# Convenience functions
_labeler: Optional[AutoLabeler] = None


def get_auto_labeler() -> AutoLabeler:
    """Get or create singleton labeler."""
    global _labeler
    if _labeler is None:
        _labeler = AutoLabeler()
    return _labeler


def label_opportunity(opportunity: Dict, resolution: str, resolved_at: float) -> LabeledOpportunity:
    """Label an opportunity from resolution."""
    labeler = get_auto_labeler()
    return labeler.label_from_resolution(opportunity, resolution, resolved_at)


def get_training_data() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Get training data from labeled opportunities."""
    labeler = get_auto_labeler()
    return labeler.get_training_data()
