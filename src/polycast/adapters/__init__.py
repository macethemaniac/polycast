from dataclasses import dataclass
from typing import List, Optional, Any


@dataclass
class Outcome:
    name: str
    best_bid: Optional[float] = None
    bid_size: Optional[float] = None
    best_ask: Optional[float] = None
    ask_size: Optional[float] = None
    raw: Optional[Any] = None


@dataclass
class Market:
    id: str
    title: str
    outcomes: List[Outcome]
    updated_ts: Optional[float] = None
    raw: Optional[Any] = None


@dataclass
class OrderBook:
    best_bid: Optional[float]
    bid_size: Optional[float]
    best_ask: Optional[float]
    ask_size: Optional[float]
    ts: Optional[float] = None
