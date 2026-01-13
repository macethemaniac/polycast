"""Streaming processor for real-time market updates.

Handles continuous price updates and triggers opportunity detection.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Awaitable
from enum import Enum

import numpy as np

from src.ml.inference_engine import InferenceEngine, ScoredOpportunity, get_inference_engine

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PriceUpdate:
    """Single price update event."""
    market_id: str
    platform: str
    timestamp: float
    yes_price: float
    no_price: float
    volume: float = 0.0


@dataclass
class OpportunityAlert:
    """Alert for detected opportunity."""
    alert_id: str
    opportunity: ScoredOpportunity
    severity: AlertSeverity
    timestamp: float = field(default_factory=time.time)
    notified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "opportunity_id": self.opportunity.opportunity_id,
            "opportunity_type": self.opportunity.opportunity_type,
            "severity": self.severity.value,
            "score": self.opportunity.final_score,
            "edge_pct": self.opportunity.edge_pct,
            "action": self.opportunity.recommended_action,
            "timestamp": self.timestamp,
        }


class PriceBuffer:
    """Circular buffer for price history."""

    def __init__(self, maxlen: int = 1000):
        self.buffer: deque = deque(maxlen=maxlen)
        self.by_market: Dict[str, deque] = {}
        self.maxlen_per_market = 100

    def add(self, update: PriceUpdate) -> None:
        """Add price update to buffer."""
        self.buffer.append(update)

        if update.market_id not in self.by_market:
            self.by_market[update.market_id] = deque(maxlen=self.maxlen_per_market)
        self.by_market[update.market_id].append(update)

    def get_recent(self, n: int = 100) -> List[PriceUpdate]:
        """Get most recent updates."""
        return list(self.buffer)[-n:]

    def get_market_history(self, market_id: str) -> List[PriceUpdate]:
        """Get price history for specific market."""
        return list(self.by_market.get(market_id, []))

    def get_price_change(self, market_id: str, hours: float = 1.0) -> Optional[float]:
        """Calculate price change over time period."""
        history = self.get_market_history(market_id)
        if len(history) < 2:
            return None

        cutoff = time.time() - hours * 3600
        old_prices = [u.yes_price for u in history if u.timestamp < cutoff]
        new_prices = [u.yes_price for u in history if u.timestamp >= cutoff]

        if not old_prices or not new_prices:
            return None

        return new_prices[-1] - old_prices[0]


class MarketStateManager:
    """Manages current state of all tracked markets."""

    def __init__(self):
        self.markets: Dict[str, Dict] = {}
        self.last_update: Dict[str, float] = {}

    def update(self, market_id: str, market_data: Dict) -> None:
        """Update market state."""
        self.markets[market_id] = market_data
        self.last_update[market_id] = time.time()

    def update_price(self, update: PriceUpdate) -> None:
        """Update just the price for a market."""
        if update.market_id in self.markets:
            self.markets[update.market_id]["yes_price"] = update.yes_price
            self.markets[update.market_id]["no_price"] = update.no_price
            self.markets[update.market_id]["volume"] = update.volume
            self.last_update[update.market_id] = update.timestamp

    def get_market(self, market_id: str) -> Optional[Dict]:
        """Get current market state."""
        return self.markets.get(market_id)

    def get_all_markets(self) -> List[Dict]:
        """Get all tracked markets."""
        return list(self.markets.values())

    def get_stale_markets(self, max_age_seconds: int = 300) -> List[str]:
        """Get markets that haven't been updated recently."""
        cutoff = time.time() - max_age_seconds
        return [mid for mid, ts in self.last_update.items() if ts < cutoff]

    def remove_market(self, market_id: str) -> None:
        """Remove market from tracking."""
        self.markets.pop(market_id, None)
        self.last_update.pop(market_id, None)


class StreamingProcessor:
    """
    Real-time streaming processor for market updates.

    Monitors price changes and triggers opportunity detection.
    """

    def __init__(
        self,
        inference_engine: Optional[InferenceEngine] = None,
        scan_interval: float = 5.0,
        alert_cooldown: float = 60.0,
        min_score: float = 0.6,
        min_edge: float = 0.03
    ):
        """
        Initialize streaming processor.

        Args:
            inference_engine: Inference engine for scoring
            scan_interval: Seconds between opportunity scans
            alert_cooldown: Minimum seconds between alerts for same market
            min_score: Minimum score to alert
            min_edge: Minimum edge % to alert
        """
        self.engine = inference_engine or get_inference_engine()
        self.scan_interval = scan_interval
        self.alert_cooldown = alert_cooldown
        self.min_score = min_score
        self.min_edge = min_edge

        # State
        self.price_buffer = PriceBuffer()
        self.market_state = MarketStateManager()

        # Alerts
        self.pending_alerts: List[OpportunityAlert] = []
        self.alert_history: deque = deque(maxlen=1000)
        self.last_alert_time: Dict[str, float] = {}

        # Callbacks
        self.alert_callbacks: List[Callable[[OpportunityAlert], Awaitable[None]]] = []

        # Control
        self._running = False
        self._scan_task: Optional[asyncio.Task] = None

    def register_alert_callback(self, callback: Callable[[OpportunityAlert], Awaitable[None]]) -> None:
        """Register callback for new alerts."""
        self.alert_callbacks.append(callback)

    def process_price_update(self, update: PriceUpdate) -> None:
        """Process incoming price update."""
        self.price_buffer.add(update)
        self.market_state.update_price(update)

    def process_market_update(self, market_data: Dict) -> None:
        """Process full market data update."""
        market_id = market_data.get("id") or market_data.get("market_id") or ""
        if market_id:
            self.market_state.update(market_id, market_data)

    def process_batch_update(self, markets: List[Dict]) -> None:
        """Process batch of market updates."""
        for market in markets:
            self.process_market_update(market)

    def _should_alert(self, opportunity: ScoredOpportunity) -> bool:
        """Check if we should create alert for opportunity."""
        # Check thresholds
        if opportunity.final_score < self.min_score:
            return False
        if opportunity.edge_pct / 100.0 < self.min_edge:
            return False

        # Check cooldown
        for mid in opportunity.market_ids:
            last_alert = self.last_alert_time.get(mid, 0)
            if time.time() - last_alert < self.alert_cooldown:
                return False

        return True

    def _determine_severity(self, opportunity: ScoredOpportunity) -> AlertSeverity:
        """Determine alert severity from opportunity."""
        if opportunity.edge_pct >= 15 or opportunity.final_score >= 0.9:
            return AlertSeverity.CRITICAL
        elif opportunity.edge_pct >= 10 or opportunity.final_score >= 0.8:
            return AlertSeverity.HIGH
        elif opportunity.edge_pct >= 5 or opportunity.final_score >= 0.7:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW

    def _create_alert(self, opportunity: ScoredOpportunity) -> OpportunityAlert:
        """Create alert from opportunity."""
        alert = OpportunityAlert(
            alert_id=f"alert_{opportunity.opportunity_id}",
            opportunity=opportunity,
            severity=self._determine_severity(opportunity)
        )

        # Update cooldown
        for mid in opportunity.market_ids:
            self.last_alert_time[mid] = time.time()

        return alert

    async def _notify_alert(self, alert: OpportunityAlert) -> None:
        """Send alert to registered callbacks."""
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
        alert.notified = True

    def scan_for_opportunities(self) -> List[ScoredOpportunity]:
        """Scan current markets for opportunities."""
        markets = self.market_state.get_all_markets()
        if not markets:
            return []

        return self.engine.score_all_opportunities(markets)

    async def _scan_loop(self) -> None:
        """Main scanning loop."""
        while self._running:
            try:
                # Scan for opportunities
                opportunities = self.scan_for_opportunities()

                # Create alerts for qualifying opportunities
                for opp in opportunities:
                    if self._should_alert(opp):
                        alert = self._create_alert(opp)
                        self.pending_alerts.append(alert)
                        self.alert_history.append(alert)
                        await self._notify_alert(alert)

            except Exception as e:
                logger.error(f"Scan loop error: {e}")

            await asyncio.sleep(self.scan_interval)

    async def start(self) -> None:
        """Start the streaming processor."""
        if self._running:
            return

        self._running = True
        self._scan_task = asyncio.create_task(self._scan_loop())
        logger.info("Streaming processor started")

    async def stop(self) -> None:
        """Stop the streaming processor."""
        self._running = False
        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
        logger.info("Streaming processor stopped")

    def get_pending_alerts(self) -> List[OpportunityAlert]:
        """Get pending (unnotified) alerts."""
        return [a for a in self.pending_alerts if not a.notified]

    def get_recent_alerts(self, n: int = 10) -> List[OpportunityAlert]:
        """Get most recent alerts."""
        return list(self.alert_history)[-n:]

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            "running": self._running,
            "tracked_markets": len(self.market_state.markets),
            "price_buffer_size": len(self.price_buffer.buffer),
            "total_alerts": len(self.alert_history),
            "pending_alerts": len(self.get_pending_alerts()),
            "scan_interval": self.scan_interval,
        }


class OpportunityMonitor:
    """
    High-level opportunity monitor.

    Simplified interface for monitoring and alerting.
    """

    def __init__(self):
        self.processor = StreamingProcessor()
        self._alert_handlers: List[Callable[[Dict], None]] = []

    def add_alert_handler(self, handler: Callable[[Dict], None]) -> None:
        """Add handler for alerts (non-async)."""
        async def async_wrapper(alert: OpportunityAlert):
            handler(alert.to_dict())

        self.processor.register_alert_callback(async_wrapper)
        self._alert_handlers.append(handler)

    def update_markets(self, markets: List[Dict]) -> None:
        """Update tracked markets."""
        self.processor.process_batch_update(markets)

    def get_opportunities(self, n: int = 10) -> List[Dict]:
        """Get current top opportunities."""
        opps = self.processor.scan_for_opportunities()
        return [o.to_dict() for o in opps[:n]]

    async def start_monitoring(self) -> None:
        """Start continuous monitoring."""
        await self.processor.start()

    async def stop_monitoring(self) -> None:
        """Stop monitoring."""
        await self.processor.stop()


# Convenience functions
_processor: Optional[StreamingProcessor] = None


def get_streaming_processor() -> StreamingProcessor:
    """Get or create singleton streaming processor."""
    global _processor
    if _processor is None:
        _processor = StreamingProcessor()
    return _processor


def update_market_prices(updates: List[Dict]) -> None:
    """Update market prices."""
    processor = get_streaming_processor()
    for update in updates:
        price_update = PriceUpdate(
            market_id=update.get("market_id") or update.get("id") or "",
            platform=update.get("platform", ""),
            timestamp=update.get("timestamp") or time.time(),
            yes_price=float(update.get("yes_price", 0.5)),
            no_price=float(update.get("no_price", 0.5)),
            volume=float(update.get("volume", 0))
        )
        processor.process_price_update(price_update)


def scan_opportunities() -> List[Dict]:
    """Scan for current opportunities."""
    processor = get_streaming_processor()
    opps = processor.scan_for_opportunities()
    return [o.to_dict() for o in opps]
