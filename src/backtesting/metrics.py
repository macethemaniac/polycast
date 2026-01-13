"""Performance metrics for backtesting.

Calculates comprehensive trading performance metrics.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive trading performance metrics."""

    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    daily_returns_mean: float = 0.0
    daily_returns_std: float = 0.0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_duration_days: float = 0.0
    avg_drawdown: float = 0.0

    # Win/Loss
    win_rate: float = 0.0
    profit_factor: float = 0.0
    payoff_ratio: float = 0.0  # avg_win / avg_loss
    expectancy: float = 0.0  # Expected profit per trade

    # Trade stats
    total_trades: int = 0
    avg_trade_pnl: float = 0.0
    avg_winning_trade: float = 0.0
    avg_losing_trade: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Time
    avg_hold_time_hours: float = 0.0
    avg_winning_hold_time: float = 0.0
    avg_losing_hold_time: float = 0.0

    # By category
    metrics_by_type: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "expectancy": self.expectancy,
        }

    def summary(self) -> str:
        return f"""
Performance Metrics
==================
Total Return: {self.total_return:.1%}
Annualized Return: {self.annualized_return:.1%}

Risk-Adjusted:
  Sharpe Ratio: {self.sharpe_ratio:.2f}
  Sortino Ratio: {self.sortino_ratio:.2f}
  Calmar Ratio: {self.calmar_ratio:.2f}

Drawdown:
  Max Drawdown: {self.max_drawdown:.1%}
  Max DD Duration: {self.max_drawdown_duration_days:.1f} days

Trade Stats:
  Total Trades: {self.total_trades}
  Win Rate: {self.win_rate:.1%}
  Profit Factor: {self.profit_factor:.2f}
  Expectancy: ${self.expectancy:.2f}

  Avg Win: ${self.avg_winning_trade:.2f}
  Avg Loss: ${self.avg_losing_trade:.2f}
  Largest Win: ${self.largest_win:.2f}
  Largest Loss: ${self.largest_loss:.2f}
"""


def calculate_returns(equity_curve: List[float]) -> np.ndarray:
    """Calculate period returns from equity curve."""
    if len(equity_curve) < 2:
        return np.array([])

    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]
    return returns


def calculate_drawdowns(equity_curve: List[float]) -> tuple:
    """
    Calculate drawdown series and max drawdown.

    Returns:
        (drawdowns, max_drawdown, max_dd_duration)
    """
    if len(equity_curve) < 2:
        return np.array([]), 0.0, 0

    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdowns = (peak - equity) / peak

    max_dd = float(np.max(drawdowns))

    # Calculate max duration
    in_drawdown = drawdowns > 0
    max_duration = 0
    current_duration = 0

    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    return drawdowns, max_dd, max_duration


def calculate_sharpe(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    if np.std(excess_returns) == 0:
        return 0.0

    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    return float(sharpe * np.sqrt(periods_per_year))


def calculate_sortino(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """Calculate annualized Sortino ratio."""
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return float('inf') if np.mean(excess_returns) > 0 else 0.0

    downside_std = np.std(downside_returns)
    if downside_std == 0:
        return 0.0

    sortino = np.mean(excess_returns) / downside_std
    return float(sortino * np.sqrt(periods_per_year))


def calculate_calmar(
    total_return: float,
    max_drawdown: float,
    years: float
) -> float:
    """Calculate Calmar ratio (annualized return / max drawdown)."""
    if max_drawdown == 0 or years == 0:
        return 0.0

    annualized_return = (1 + total_return) ** (1 / years) - 1
    return annualized_return / max_drawdown


def calculate_consecutive_stats(pnls: List[float]) -> tuple:
    """Calculate max consecutive wins and losses."""
    if not pnls:
        return 0, 0

    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0

    for pnl in pnls:
        if pnl > 0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        elif pnl < 0:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)
        else:
            current_wins = 0
            current_losses = 0

    return max_wins, max_losses


def calculate_metrics(
    equity_curve: List[float],
    trades: List[Any],
    timestamps: Optional[List[float]] = None,
    risk_free_rate: float = 0.0
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics.

    Args:
        equity_curve: List of equity values over time
        trades: List of Trade objects with pnl, hold_time, etc.
        timestamps: Optional timestamps for equity curve
        risk_free_rate: Annual risk-free rate for Sharpe calculation

    Returns:
        PerformanceMetrics object
    """
    metrics = PerformanceMetrics()

    if not equity_curve or len(equity_curve) < 2:
        return metrics

    # Calculate returns
    returns = calculate_returns(equity_curve)

    # Total return
    metrics.total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]

    # Time period
    if timestamps and len(timestamps) >= 2:
        days = (timestamps[-1] - timestamps[0]) / 86400
        years = days / 365
    else:
        days = len(equity_curve)
        years = days / 365

    # Annualized return
    if years > 0:
        metrics.annualized_return = (1 + metrics.total_return) ** (1 / years) - 1

    # Daily returns stats
    metrics.daily_returns_mean = float(np.mean(returns)) if len(returns) > 0 else 0
    metrics.daily_returns_std = float(np.std(returns)) if len(returns) > 0 else 0

    # Risk-adjusted metrics
    metrics.sharpe_ratio = calculate_sharpe(returns, risk_free_rate)
    metrics.sortino_ratio = calculate_sortino(returns, risk_free_rate)

    # Drawdown
    drawdowns, max_dd, max_duration = calculate_drawdowns(equity_curve)
    metrics.max_drawdown = max_dd
    metrics.max_drawdown_duration_days = max_duration
    metrics.avg_drawdown = float(np.mean(drawdowns)) if len(drawdowns) > 0 else 0

    # Calmar ratio
    metrics.calmar_ratio = calculate_calmar(metrics.total_return, max_dd, years) if years > 0 else 0

    # Trade metrics
    closed_trades = [t for t in trades if hasattr(t, 'pnl') and t.pnl is not None]
    metrics.total_trades = len(closed_trades)

    if closed_trades:
        pnls = [t.pnl for t in closed_trades]
        winning = [t for t in closed_trades if t.pnl > 0]
        losing = [t for t in closed_trades if t.pnl <= 0]

        # Win rate
        metrics.win_rate = len(winning) / len(closed_trades)

        # P&L stats
        metrics.avg_trade_pnl = float(np.mean(pnls))
        metrics.avg_winning_trade = float(np.mean([t.pnl for t in winning])) if winning else 0
        metrics.avg_losing_trade = float(np.mean([abs(t.pnl) for t in losing])) if losing else 0

        metrics.largest_win = float(max(pnls)) if pnls else 0
        metrics.largest_loss = float(min(pnls)) if pnls else 0

        # Profit factor
        total_profit = sum(t.pnl for t in winning) if winning else 0
        total_loss = sum(abs(t.pnl) for t in losing) if losing else 0
        metrics.profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # Payoff ratio
        if metrics.avg_losing_trade > 0:
            metrics.payoff_ratio = metrics.avg_winning_trade / metrics.avg_losing_trade

        # Expectancy
        metrics.expectancy = (
            metrics.win_rate * metrics.avg_winning_trade -
            (1 - metrics.win_rate) * metrics.avg_losing_trade
        )

        # Consecutive wins/losses
        metrics.max_consecutive_wins, metrics.max_consecutive_losses = calculate_consecutive_stats(pnls)

        # Hold times
        hold_times = [t.hold_time_hours for t in closed_trades if hasattr(t, 'hold_time_hours') and t.hold_time_hours]
        if hold_times:
            metrics.avg_hold_time_hours = float(np.mean(hold_times))

        winning_times = [t.hold_time_hours for t in winning if hasattr(t, 'hold_time_hours') and t.hold_time_hours]
        if winning_times:
            metrics.avg_winning_hold_time = float(np.mean(winning_times))

        losing_times = [t.hold_time_hours for t in losing if hasattr(t, 'hold_time_hours') and t.hold_time_hours]
        if losing_times:
            metrics.avg_losing_hold_time = float(np.mean(losing_times))

        # Metrics by opportunity type
        by_type: Dict[str, List] = {}
        for t in closed_trades:
            opp_type = getattr(t, 'opportunity_type', 'unknown') or 'unknown'
            if opp_type not in by_type:
                by_type[opp_type] = []
            by_type[opp_type].append(t)

        for opp_type, type_trades in by_type.items():
            type_pnls = [t.pnl for t in type_trades]
            type_winning = [t for t in type_trades if t.pnl > 0]

            metrics.metrics_by_type[opp_type] = {
                "count": len(type_trades),
                "win_rate": len(type_winning) / len(type_trades) if type_trades else 0,
                "avg_pnl": float(np.mean(type_pnls)) if type_pnls else 0,
                "total_pnl": sum(type_pnls),
            }

    return metrics


def compare_strategies(
    results: Dict[str, "BacktestResult"]
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple strategy backtest results.

    Args:
        results: Dict of strategy_name -> BacktestResult

    Returns:
        Comparison table as dict
    """
    comparison = {}

    for name, result in results.items():
        comparison[name] = {
            "total_return": result.total_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown_pct,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "total_trades": result.total_trades,
        }

    return comparison


def generate_report(metrics: PerformanceMetrics) -> str:
    """Generate text report from metrics."""
    report = [
        "=" * 50,
        "BACKTEST PERFORMANCE REPORT",
        "=" * 50,
        "",
        "RETURNS",
        "-" * 30,
        f"Total Return:      {metrics.total_return:>10.1%}",
        f"Annualized Return: {metrics.annualized_return:>10.1%}",
        "",
        "RISK METRICS",
        "-" * 30,
        f"Sharpe Ratio:      {metrics.sharpe_ratio:>10.2f}",
        f"Sortino Ratio:     {metrics.sortino_ratio:>10.2f}",
        f"Calmar Ratio:      {metrics.calmar_ratio:>10.2f}",
        f"Max Drawdown:      {metrics.max_drawdown:>10.1%}",
        "",
        "TRADE STATISTICS",
        "-" * 30,
        f"Total Trades:      {metrics.total_trades:>10d}",
        f"Win Rate:          {metrics.win_rate:>10.1%}",
        f"Profit Factor:     {metrics.profit_factor:>10.2f}",
        f"Expectancy:        ${metrics.expectancy:>9.2f}",
        "",
        f"Avg Win:           ${metrics.avg_winning_trade:>9.2f}",
        f"Avg Loss:          ${metrics.avg_losing_trade:>9.2f}",
        f"Largest Win:       ${metrics.largest_win:>9.2f}",
        f"Largest Loss:      ${metrics.largest_loss:>9.2f}",
        "",
        f"Max Consecutive Wins:   {metrics.max_consecutive_wins}",
        f"Max Consecutive Losses: {metrics.max_consecutive_losses}",
        "",
        "=" * 50,
    ]

    if metrics.metrics_by_type:
        report.extend([
            "",
            "BY OPPORTUNITY TYPE",
            "-" * 30,
        ])
        for opp_type, type_metrics in metrics.metrics_by_type.items():
            report.append(f"  {opp_type}:")
            report.append(f"    Trades: {type_metrics['count']}, Win Rate: {type_metrics['win_rate']:.1%}, Total P&L: ${type_metrics['total_pnl']:.2f}")

    return "\n".join(report)
