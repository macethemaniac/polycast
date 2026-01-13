"""Data schemas for ML training data.

Defines structured schemas for markets, relationships, and training data.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import time
import json
import hashlib


class Platform(Enum):
    POLYMARKET = "polymarket"
    KALSHI = "kalshi"


class RelationshipType(Enum):
    CORRELATED = "correlated"           # If A happens, B more likely
    ANTI_CORRELATED = "anti_correlated" # If A happens, B less likely
    SUBSET = "subset"                   # A is subset of B (A implies B)
    EXCLUSIVE = "exclusive"             # A and B cannot both happen
    INDEPENDENT = "independent"         # No relationship
    SEQUENTIAL = "sequential"           # A must happen before B
    SAME_EVENT = "same_event"           # Same event, different platforms


class Resolution(Enum):
    YES = "yes"
    NO = "no"
    CANCELLED = "cancelled"
    PENDING = "pending"


@dataclass
class PricePoint:
    """Single price observation."""
    timestamp: float
    yes_price: float
    no_price: float
    volume: float = 0.0
    liquidity: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "PricePoint":
        return cls(**d)


@dataclass
class MarketSnapshot:
    """Complete market state at a point in time."""
    market_id: str
    platform: str
    question: str
    description: str = ""
    outcomes: List[str] = field(default_factory=lambda: ["YES", "NO"])
    yes_price: float = 0.5
    no_price: float = 0.5
    volume: float = 0.0
    liquidity: float = 0.0
    created_at: Optional[float] = None
    close_time: Optional[float] = None
    resolution: str = "pending"
    resolved_at: Optional[float] = None
    category: str = ""
    tags: List[str] = field(default_factory=list)
    snapshot_ts: float = field(default_factory=time.time)
    embedding: Optional[List[float]] = None
    raw: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        # Don't store embedding in JSON (too large)
        d.pop('embedding', None)
        d.pop('raw', None)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "MarketSnapshot":
        d.pop('embedding', None)
        d.pop('raw', None)
        return cls(**d)

    def get_hash(self) -> str:
        """Get content hash for deduplication."""
        content = f"{self.market_id}:{self.question}:{self.yes_price}:{self.no_price}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class HistoricalMarket:
    """Complete historical record of a market."""
    market_id: str
    platform: str
    question: str
    description: str = ""
    outcomes: List[str] = field(default_factory=lambda: ["YES", "NO"])
    price_history: List[PricePoint] = field(default_factory=list)
    resolution: str = "pending"
    resolved_at: Optional[float] = None
    created_at: Optional[float] = None
    close_time: Optional[float] = None
    category: str = ""
    tags: List[str] = field(default_factory=list)
    final_yes_price: float = 0.0
    final_no_price: float = 0.0
    total_volume: float = 0.0
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['price_history'] = [p.to_dict() if isinstance(p, PricePoint) else p for p in self.price_history]
        d.pop('embedding', None)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "HistoricalMarket":
        d['price_history'] = [PricePoint.from_dict(p) if isinstance(p, dict) else p for p in d.get('price_history', [])]
        d.pop('embedding', None)
        return cls(**d)


@dataclass
class MarketRelationship:
    """Relationship between two markets."""
    market_a_id: str
    market_b_id: str
    platform_a: str
    platform_b: str
    relationship_type: str
    confidence: float = 0.0
    discovered_at: float = field(default_factory=time.time)
    verified: bool = False
    verified_by: str = ""  # "auto", "llm", "human"
    notes: str = ""

    # Evidence for relationship
    semantic_similarity: float = 0.0
    entity_overlap: float = 0.0
    time_overlap: float = 0.0
    historical_correlation: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "MarketRelationship":
        return cls(**d)

    def get_pair_id(self) -> str:
        """Unique ID for this market pair (order-independent)."""
        ids = sorted([f"{self.platform_a}:{self.market_a_id}", f"{self.platform_b}:{self.market_b_id}"])
        return hashlib.sha256(":".join(ids).encode()).hexdigest()[:16]


@dataclass
class TrainingExample:
    """Single training example for ML models."""
    example_id: str
    timestamp: float
    features: Dict[str, float]
    label: Optional[int] = None  # 1=profitable, 0=not profitable, None=unknown
    label_source: str = ""  # "auto", "human", "resolution"

    # Context
    market_id: str = ""
    platform: str = ""
    opportunity_type: str = ""  # "intra_market", "cross_platform", "relationship"

    # Outcome tracking
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    hold_time_hours: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "TrainingExample":
        return cls(**d)


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity."""
    opportunity_id: str
    opportunity_type: str  # "intra_market", "cross_platform", "relationship", "cluster"
    timestamp: float = field(default_factory=time.time)

    # Markets involved
    markets: List[Dict] = field(default_factory=list)

    # Pricing
    edge_pct: float = 0.0
    expected_profit: float = 0.0

    # Confidence
    model_score: float = 0.0
    relationship_confidence: float = 0.0
    consistency_score: float = 0.0

    # Action
    recommended_action: str = ""  # "BUY_YES", "BUY_NO", "SELL", "WATCH"
    position_size: float = 0.0

    # Features used
    features: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "ArbitrageOpportunity":
        return cls(**d)


def generate_id(*args) -> str:
    """Generate unique ID from arguments."""
    content = ":".join(str(a) for a in args)
    return hashlib.sha256(content.encode()).hexdigest()[:16]
