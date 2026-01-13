"""Backtesting engine for arbitrage strategies.

Simulates trading on historical data to evaluate strategy performance.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Single trade record."""
    trade_id: str
    market_id: str
    platform: str
    timestamp: float

    # Trade details
    action: str  # "BUY_YES", "BUY_NO", "SELL_YES", "SELL_NO"
    entry_price: float
    position_size: float  # In dollars

    # Exit (filled when closed)
    exit_price: Optional[float] = None
    exit_timestamp: Optional[float] = None
    exit_reason: str = ""  # "resolution", "stop_loss", "take_profit", "timeout"

    # P&L
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None

    # Context
    signal_score: float = 0.0
    edge_pct: float = 0.0
    opportunity_type: str = ""

    @property
    def is_open(self) -> bool:
        return self.exit_price is None

    @property
    def hold_time_hours(self) -> Optional[float]:
        if self.exit_timestamp is None:
            return None
        return (self.exit_timestamp - self.timestamp) / 3600

    def close(
        self,
        exit_price: float,
        exit_timestamp: float,
        exit_reason: str = ""
    ) -> None:
        """Close the trade and calculate P&L."""
        self.exit_price = exit_price
        self.exit_timestamp = exit_timestamp
        self.exit_reason = exit_reason

        # Calculate P&L based on action
        if "YES" in self.action:
            if "BUY" in self.action:
                # Bought YES, profit if price goes up or resolves YES
                self.pnl = (exit_price - self.entry_price) * self.position_size
            else:
                # Sold YES, profit if price goes down
                self.pnl = (self.entry_price - exit_price) * self.position_size
        else:
            if "BUY" in self.action:
                # Bought NO, profit if YES price goes down
                self.pnl = (self.entry_price - exit_price) * self.position_size
            else:
                self.pnl = (exit_price - self.entry_price) * self.position_size

        # Percentage P&L
        cost = self.entry_price * self.position_size
        self.pnl_pct = self.pnl / cost if cost > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "market_id": self.market_id,
            "action": self.action,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "exit_reason": self.exit_reason,
            "hold_time_hours": self.hold_time_hours,
        }


@dataclass
class Position:
    """Current position in a market."""
    market_id: str
    platform: str
    side: str  # "YES" or "NO"
    size: float  # In contracts/shares
    avg_price: float
    timestamp: float


@dataclass
class BacktestResult:
    """Complete backtest results."""
    # Basic stats
    initial_capital: float
    final_capital: float
    total_pnl: float
    total_return: float

    # Trade stats
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # Risk metrics
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float

    # Profit metrics
    avg_profit: float
    avg_loss: float
    profit_factor: float
    avg_trade_pnl: float

    # Time metrics
    avg_hold_time_hours: float
    total_days: float

    # All trades
    trades: List[Trade] = field(default_factory=list)

    # Equity curve
    equity_curve: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "total_pnl": self.total_pnl,
            "total_return": self.total_return,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "profit_factor": self.profit_factor,
            "avg_hold_time_hours": self.avg_hold_time_hours,
        }

    def summary(self) -> str:
        return f"""
Backtest Results:
================
Initial Capital: ${self.initial_capital:,.2f}
Final Capital: ${self.final_capital:,.2f}
Total Return: {self.total_return:.1%}

Trades: {self.total_trades} ({self.winning_trades}W / {self.losing_trades}L)
Win Rate: {self.win_rate:.1%}
Profit Factor: {self.profit_factor:.2f}

Max Drawdown: {self.max_drawdown_pct:.1%}
Sharpe Ratio: {self.sharpe_ratio:.2f}
Avg Hold Time: {self.avg_hold_time_hours:.1f}h
"""


class Strategy:
    """Base strategy class for backtesting."""

    def __init__(self, name: str = "base"):
        self.name = name

    def generate_signals(
        self,
        timestamp: float,
        markets: List[Dict],
        positions: Dict[str, Position]
    ) -> List[Dict]:
        """
        Generate trading signals.

        Returns list of signal dicts with:
        - market_id: str
        - action: "BUY_YES", "BUY_NO", "SELL_YES", "SELL_NO"
        - size: float (position size in dollars)
        - score: float (signal strength)
        """
        raise NotImplementedError


class MLStrategy(Strategy):
    """Strategy using ML model for signals."""

    def __init__(
        self,
        min_score: float = 0.6,
        min_edge: float = 0.03,
        position_size: float = 100.0,
        max_positions: int = 10
    ):
        super().__init__("ml_strategy")
        self.min_score = min_score
        self.min_edge = min_edge
        self.position_size = position_size
        self.max_positions = max_positions

    def generate_signals(
        self,
        timestamp: float,
        markets: List[Dict],
        positions: Dict[str, Position]
    ) -> List[Dict]:
        """Generate signals based on ML scores."""
        from src.ml.inference_engine import get_inference_engine

        engine = get_inference_engine()
        signals = []

        # Don't open more than max_positions
        if len(positions) >= self.max_positions:
            return signals

        # Score opportunities
        try:
            opportunities = engine.score_all_opportunities(markets)
        except Exception as e:
            logger.debug(f"Scoring failed: {e}")
            return signals

        for opp in opportunities:
            # Skip if already in position
            if any(mid in positions for mid in opp.market_ids):
                continue

            # Check thresholds
            if opp.final_score < self.min_score:
                continue
            if opp.edge_pct / 100.0 < self.min_edge:
                continue

            # Generate signal
            action = opp.recommended_action
            if not action or action == "PASS" or action == "WATCH":
                continue

            # Map action to trade action
            if "BUY" in action.upper() and "KALSHI" in action.upper():
                trade_action = "BUY_YES"
                market_id = opp.market_ids[1] if len(opp.market_ids) > 1 else opp.market_ids[0]
            elif "BUY" in action.upper():
                trade_action = "BUY_YES"
                market_id = opp.market_ids[0]
            elif "SELL" in action.upper():
                trade_action = "SELL_YES"
                market_id = opp.market_ids[0]
            else:
                continue

            signals.append({
                "market_id": market_id,
                "action": trade_action,
                "size": self.position_size,
                "score": opp.final_score,
                "edge_pct": opp.edge_pct,
                "opportunity_type": opp.opportunity_type,
            })

            if len(signals) + len(positions) >= self.max_positions:
                break

        return signals


class BacktestEngine:
    """
    Backtesting engine for trading strategies.

    Simulates trading on historical data.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_pct: float = 0.01,
        max_position_pct: float = 0.1,
        stop_loss_pct: float = 0.2,
        take_profit_pct: float = 0.5
    ):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital in dollars
            commission_pct: Commission per trade (as decimal)
            max_position_pct: Max position size as % of capital
            stop_loss_pct: Stop loss threshold
            take_profit_pct: Take profit threshold
        """
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # State
        self.capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[float] = []

        # Metrics
        self.peak_equity = initial_capital
        self.max_drawdown = 0.0

    def reset(self) -> None:
        """Reset engine state."""
        self.capital = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.equity_curve.clear()
        self.timestamps.clear()
        self.peak_equity = self.initial_capital
        self.max_drawdown = 0.0

    def _calculate_equity(self, current_prices: Dict[str, float]) -> float:
        """Calculate current total equity."""
        equity = self.capital

        for market_id, pos in self.positions.items():
            current_price = current_prices.get(market_id, pos.avg_price)
            if pos.side == "YES":
                equity += pos.size * current_price
            else:
                equity += pos.size * (1.0 - current_price)

        return equity

    def _update_drawdown(self, equity: float) -> None:
        """Update max drawdown."""
        if equity > self.peak_equity:
            self.peak_equity = equity

        drawdown = self.peak_equity - equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

    def _open_position(
        self,
        signal: Dict,
        timestamp: float,
        market_data: Dict
    ) -> Optional[Trade]:
        """Open a new position."""
        market_id = signal["market_id"]

        # Check if already in position
        if market_id in self.positions:
            return None

        # Get entry price
        action = signal["action"]
        if "YES" in action:
            entry_price = float(market_data.get("yes_price", 0.5))
        else:
            entry_price = float(market_data.get("no_price", 0.5))

        # Calculate position size
        requested_size = signal.get("size", 100.0)
        max_size = self.capital * self.max_position_pct
        position_size = min(requested_size, max_size)

        # Check capital
        cost = position_size * entry_price * (1 + self.commission_pct)
        if cost > self.capital:
            return None

        # Deduct capital
        self.capital -= cost

        # Create position
        side = "YES" if "YES" in action else "NO"
        self.positions[market_id] = Position(
            market_id=market_id,
            platform=market_data.get("platform", ""),
            side=side,
            size=position_size / entry_price,  # Convert to shares
            avg_price=entry_price,
            timestamp=timestamp
        )

        # Create trade
        trade = Trade(
            trade_id=f"trade_{len(self.trades)}",
            market_id=market_id,
            platform=market_data.get("platform", ""),
            timestamp=timestamp,
            action=action,
            entry_price=entry_price,
            position_size=position_size,
            signal_score=signal.get("score", 0),
            edge_pct=signal.get("edge_pct", 0),
            opportunity_type=signal.get("opportunity_type", ""),
        )

        self.trades.append(trade)
        return trade

    def _close_position(
        self,
        market_id: str,
        exit_price: float,
        timestamp: float,
        reason: str = ""
    ) -> Optional[Trade]:
        """Close an existing position."""
        if market_id not in self.positions:
            return None

        pos = self.positions[market_id]

        # Find corresponding trade
        trade = None
        for t in reversed(self.trades):
            if t.market_id == market_id and t.is_open:
                trade = t
                break

        if trade is None:
            return None

        # Close trade
        trade.close(exit_price, timestamp, reason)

        # Return capital
        proceeds = pos.size * exit_price * (1 - self.commission_pct)
        self.capital += proceeds

        # Remove position
        del self.positions[market_id]

        return trade

    def _check_exits(
        self,
        timestamp: float,
        market_prices: Dict[str, Dict],
        resolutions: Dict[str, str]
    ) -> None:
        """Check for exit conditions."""
        markets_to_close = []

        for market_id, pos in self.positions.items():
            # Check resolution
            if market_id in resolutions:
                resolution = resolutions[market_id]
                exit_price = 1.0 if resolution.lower() == "yes" else 0.0
                markets_to_close.append((market_id, exit_price, "resolution"))
                continue

            # Get current price
            market_data = market_prices.get(market_id, {})
            current_price = float(market_data.get("yes_price", pos.avg_price))

            # Check stop loss
            if pos.side == "YES":
                pnl_pct = (current_price - pos.avg_price) / pos.avg_price
            else:
                pnl_pct = (pos.avg_price - current_price) / pos.avg_price

            if pnl_pct <= -self.stop_loss_pct:
                markets_to_close.append((market_id, current_price, "stop_loss"))
            elif pnl_pct >= self.take_profit_pct:
                markets_to_close.append((market_id, current_price, "take_profit"))

        # Close positions
        for market_id, exit_price, reason in markets_to_close:
            self._close_position(market_id, exit_price, timestamp, reason)

    def run(
        self,
        historical_data: List[Dict],
        strategy: Strategy
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            historical_data: List of snapshots, each with:
                - timestamp: float
                - markets: List[Dict] with market data
                - resolutions: Dict[str, str] of resolved markets

            strategy: Strategy to test

        Returns:
            BacktestResult with all metrics
        """
        self.reset()
        logger.info(f"Running backtest with {len(historical_data)} snapshots...")

        for snapshot in historical_data:
            timestamp = snapshot.get("timestamp", 0)
            markets = snapshot.get("markets", [])
            resolutions = snapshot.get("resolutions", {})

            # Build price lookup
            market_prices = {}
            for m in markets:
                mid = m.get("id") or m.get("market_id") or ""
                if mid:
                    market_prices[mid] = m

            # Check exits
            self._check_exits(timestamp, market_prices, resolutions)

            # Generate signals
            signals = strategy.generate_signals(timestamp, markets, self.positions)

            # Execute signals
            for signal in signals:
                market_data = market_prices.get(signal["market_id"], {})
                if market_data:
                    self._open_position(signal, timestamp, market_data)

            # Record equity
            current_prices = {mid: m.get("yes_price", 0.5) for mid, m in market_prices.items()}
            equity = self._calculate_equity(current_prices)
            self.equity_curve.append(equity)
            self.timestamps.append(timestamp)
            self._update_drawdown(equity)

        # Close remaining positions at last known prices
        final_snapshot = historical_data[-1] if historical_data else {}
        final_timestamp = final_snapshot.get("timestamp", 0)
        for market_id in list(self.positions.keys()):
            market_data = {m.get("id") or m.get("market_id"): m for m in final_snapshot.get("markets", [])}
            exit_price = market_data.get(market_id, {}).get("yes_price", 0.5)
            self._close_position(market_id, exit_price, final_timestamp, "backtest_end")

        # Calculate results
        return self._calculate_results()

    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest results."""
        final_capital = self.capital
        total_pnl = final_capital - self.initial_capital
        total_return = total_pnl / self.initial_capital if self.initial_capital > 0 else 0

        # Trade stats
        closed_trades = [t for t in self.trades if not t.is_open]
        winning = [t for t in closed_trades if t.pnl and t.pnl > 0]
        losing = [t for t in closed_trades if t.pnl and t.pnl <= 0]

        win_rate = len(winning) / len(closed_trades) if closed_trades else 0

        # Profit metrics
        profits = [t.pnl for t in winning if t.pnl]
        losses = [abs(t.pnl) for t in losing if t.pnl]

        avg_profit = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = sum(profits) / sum(losses) if losses and sum(losses) > 0 else float('inf')

        avg_trade_pnl = np.mean([t.pnl for t in closed_trades if t.pnl]) if closed_trades else 0

        # Time metrics
        hold_times = [t.hold_time_hours for t in closed_trades if t.hold_time_hours]
        avg_hold_time = np.mean(hold_times) if hold_times else 0

        total_days = (self.timestamps[-1] - self.timestamps[0]) / 86400 if len(self.timestamps) > 1 else 0

        # Risk metrics
        max_dd_pct = self.max_drawdown / self.peak_equity if self.peak_equity > 0 else 0

        # Sharpe and Sortino
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1
            sortino = np.mean(returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
        else:
            sharpe = 0
            sortino = 0

        return BacktestResult(
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_pnl=total_pnl,
            total_return=total_return,
            total_trades=len(closed_trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            max_drawdown=self.max_drawdown,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_trade_pnl,
            avg_hold_time_hours=avg_hold_time,
            total_days=total_days,
            trades=self.trades,
            equity_curve=self.equity_curve,
            timestamps=self.timestamps,
        )


# Convenience functions
def run_backtest(
    historical_data: List[Dict],
    initial_capital: float = 10000.0,
    min_score: float = 0.6,
    min_edge: float = 0.03
) -> BacktestResult:
    """Run backtest with ML strategy."""
    engine = BacktestEngine(initial_capital=initial_capital)
    strategy = MLStrategy(min_score=min_score, min_edge=min_edge)
    return engine.run(historical_data, strategy)
