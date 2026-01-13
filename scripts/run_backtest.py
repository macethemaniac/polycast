"""Run backtest to find cross-market arbitrage opportunities.

Uses the trained ML model to identify opportunities and simulates trading.
"""

import sys
sys.path.insert(0, '.')

import json
import pickle
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np

from src.ml.feature_builder_v2 import AdvancedFeatureBuilder
from src.ml.market_clustering_v2 import HierarchicalClusterer
from src.ml.relationship_discovery import RelationshipClassifier, RelationshipType
from src.ml.logical_consistency import LogicalConsistencyChecker
from src.backtesting.engine import BacktestEngine, Trade, BacktestResult
from src.backtesting.metrics import calculate_metrics, generate_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path('data')
MODEL_PATH = DATA_DIR / 'models' / 'ensemble_model.pkl'
HISTORICAL_FILE = DATA_DIR / 'historical' / 'polymarket_markets.jsonl'


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity."""
    opportunity_id: str
    opportunity_type: str  # "cross_platform", "intra_cluster", "relationship"
    markets: List[Dict]
    edge_pct: float
    confidence: float
    recommended_action: str
    details: str


class MLArbitrageStrategy:
    """ML-based arbitrage detection strategy."""

    def __init__(
        self,
        model_path: Path,
        min_score: float = 0.5,
        min_edge: float = 3.0,  # Minimum 3% edge
        position_size: float = 100.0,  # $100 per trade
    ):
        self.min_score = min_score
        self.min_edge = min_edge
        self.position_size = position_size

        # Load model
        logger.info("Loading ML model...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.ensemble = model_data['ensemble']
        self.feature_names = model_data['feature_names']

        # Initialize components
        self.feature_builder = AdvancedFeatureBuilder()
        self.clusterer = HierarchicalClusterer()
        self.relationship_classifier = RelationshipClassifier()
        self.consistency_checker = LogicalConsistencyChecker()

        logger.info(f"Strategy initialized: min_score={min_score}, min_edge={min_edge}%")

    def score_market(self, market: Dict) -> float:
        """Score a single market for opportunity."""
        try:
            fs = self.feature_builder.build_features(market)
            features = fs.to_vector().reshape(1, -1)
            return float(self.ensemble.predict_proba(features)[0])
        except Exception as e:
            logger.debug(f"Error scoring market: {e}")
            return 0.0

    def find_opportunities(self, markets: List[Dict], timestamp: float) -> List[ArbitrageOpportunity]:
        """Find all arbitrage opportunities in current markets."""
        opportunities = []

        # 1. Score individual markets for mispricing
        for market in markets:
            score = self.score_market(market)
            yes_price = market.get('yes_price', market.get('final_yes_price', 0.5))

            if score >= self.min_score:
                # Calculate edge based on price distance from 0.5
                price_extreme = max(yes_price, 1 - yes_price)
                edge = (price_extreme - 0.5) * 100 * 2  # Simple edge estimate

                if edge >= self.min_edge:
                    action = "BUY_YES" if yes_price < 0.5 else "BUY_NO"
                    opp = ArbitrageOpportunity(
                        opportunity_id=f"single_{market.get('market_id', '')}_{int(timestamp)}",
                        opportunity_type="mispricing",
                        markets=[market],
                        edge_pct=edge,
                        confidence=score,
                        recommended_action=action,
                        details=f"ML score {score:.2f}, price {yes_price:.2f}"
                    )
                    opportunities.append(opp)

        # 2. Cluster markets and find intra-cluster arbitrage
        if len(markets) >= 5:
            try:
                cluster_tree = self.clusterer.cluster_markets(markets)
                divergences = self.clusterer.find_price_divergences(cluster_tree, min_divergence=0.05)

                for div in divergences:
                    edge = div.get('divergence', 0) * 100
                    if edge >= self.min_edge:
                        opp = ArbitrageOpportunity(
                            opportunity_id=f"cluster_{div.get('market_id', '')}_{int(timestamp)}",
                            opportunity_type="cluster_divergence",
                            markets=[div.get('market', {})],
                            edge_pct=edge,
                            confidence=0.7,  # Medium confidence for cluster arb
                            recommended_action="BUY_YES" if div.get('direction') == 'underpriced' else "BUY_NO",
                            details=f"Diverges {edge:.1f}% from cluster mean"
                        )
                        opportunities.append(opp)
            except Exception as e:
                logger.debug(f"Clustering error: {e}")

        # 3. Find relationship-based arbitrage
        if len(markets) >= 2:
            try:
                # Sample pairs for relationship analysis
                for i, m1 in enumerate(markets[:20]):
                    for m2 in markets[i+1:min(i+10, len(markets))]:
                        rel = self.relationship_classifier.predict(m1, m2)

                        if rel.confidence > 0.7 and rel.relationship_type in [
                            RelationshipType.CORRELATED,
                            RelationshipType.ANTI_CORRELATED,
                            RelationshipType.SAME_EVENT
                        ]:
                            p1 = m1.get('yes_price', m1.get('final_yes_price', 0.5))
                            p2 = m2.get('yes_price', m2.get('final_yes_price', 0.5))

                            # Check for pricing inconsistency
                            if rel.relationship_type == RelationshipType.CORRELATED:
                                # Correlated markets should have similar prices
                                price_diff = abs(p1 - p2)
                                if price_diff > 0.1:
                                    edge = price_diff * 100
                                    opp = ArbitrageOpportunity(
                                        opportunity_id=f"rel_{m1.get('market_id', '')}_{m2.get('market_id', '')}_{int(timestamp)}",
                                        opportunity_type="relationship",
                                        markets=[m1, m2],
                                        edge_pct=edge,
                                        confidence=rel.confidence,
                                        recommended_action=f"BUY_YES on {m1.get('market_id')} if underpriced",
                                        details=f"Correlated markets diverged by {price_diff:.1%}"
                                    )
                                    opportunities.append(opp)

                            elif rel.relationship_type == RelationshipType.ANTI_CORRELATED:
                                # Anti-correlated should sum to ~1.0
                                price_sum = p1 + p2
                                if abs(price_sum - 1.0) > 0.1:
                                    edge = abs(price_sum - 1.0) * 100
                                    opp = ArbitrageOpportunity(
                                        opportunity_id=f"anti_{m1.get('market_id', '')}_{m2.get('market_id', '')}_{int(timestamp)}",
                                        opportunity_type="anti_correlated",
                                        markets=[m1, m2],
                                        edge_pct=edge,
                                        confidence=rel.confidence,
                                        recommended_action="Arbitrage: prices should sum to 1.0",
                                        details=f"Anti-correlated prices sum to {price_sum:.2f}"
                                    )
                                    opportunities.append(opp)
            except Exception as e:
                logger.debug(f"Relationship analysis error: {e}")

        # Sort by confidence * edge
        opportunities.sort(key=lambda x: x.confidence * x.edge_pct, reverse=True)
        return opportunities

    def generate_signals(
        self,
        timestamp: float,
        markets: List[Dict],
        current_positions: Dict[str, any]
    ) -> List[Dict]:
        """Generate trading signals for backtester."""
        signals = []
        opportunities = self.find_opportunities(markets, timestamp)

        for opp in opportunities[:5]:  # Limit to top 5 opportunities
            market = opp.markets[0]
            market_id = market.get('market_id', market.get('id', ''))

            # Skip if already have position
            if market_id in current_positions:
                continue

            signal = {
                'market_id': market_id,
                'action': opp.recommended_action,
                'size': self.position_size,
                'price': market.get('yes_price', market.get('final_yes_price', 0.5)),
                'timestamp': timestamp,
                'score': opp.confidence,
                'edge': opp.edge_pct,
                'opportunity_type': opp.opportunity_type,
            }
            signals.append(signal)

        return signals


def load_historical_data() -> List[Dict]:
    """Load historical market data."""
    markets = []
    with open(HISTORICAL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                markets.append(json.loads(line))
    return markets


def infer_resolution(market: Dict) -> Optional[str]:
    """Infer market resolution from final price."""
    yes_price = market.get('final_yes_price')
    if yes_price is None:
        return None
    if yes_price > 0.95:
        return "yes"
    elif yes_price < 0.05:
        return "no"
    return None


def run_backtest(
    markets: List[Dict],
    strategy: MLArbitrageStrategy,
    initial_capital: float = 10000.0,
) -> BacktestResult:
    """Run backtest simulation."""

    # Filter to resolved markets with known outcomes
    resolved_markets = []
    for m in markets:
        res = infer_resolution(m)
        if res:
            m['_resolution'] = res
            m['yes_price'] = m.get('final_yes_price', 0.5)
            m['no_price'] = 1.0 - m['yes_price']
            resolved_markets.append(m)

    logger.info(f"Running backtest on {len(resolved_markets)} resolved markets")

    # Sort by close_time to simulate temporal order
    resolved_markets.sort(key=lambda x: x.get('close_time') or 0)

    # Simulation state
    capital = initial_capital
    positions = {}
    trades = []
    equity_curve = [initial_capital]
    timestamps = [(resolved_markets[0].get('close_time') or time.time()) - 86400 * 30]

    # Simulate market-by-market
    trade_counter = 0

    # Group markets by time period for batch processing
    time_batches = defaultdict(list)
    for m in resolved_markets:
        # Group by week
        close_time = m.get('close_time') or 0
        week = int(close_time // (86400 * 7))
        time_batches[week].append(m)

    for week in sorted(time_batches.keys()):
        batch_markets = time_batches[week]
        batch_timestamp = (batch_markets[0].get('close_time') or time.time()) - 86400 * 7

        # Get signals from strategy
        signals = strategy.generate_signals(batch_timestamp, batch_markets, positions)

        # Execute trades
        for signal in signals:
            market_id = signal['market_id']

            # Find the market
            market = next((m for m in batch_markets if m.get('market_id') == market_id), None)
            if not market:
                continue

            # Skip if not enough capital
            if signal['size'] > capital * 0.2:  # Max 20% per trade
                signal['size'] = capital * 0.2

            if signal['size'] < 10:  # Minimum $10
                continue

            # Create trade
            trade_counter += 1
            trade = Trade(
                trade_id=f"trade_{trade_counter}",
                market_id=market_id,
                platform="polymarket",
                timestamp=batch_timestamp,
                action=signal['action'],
                entry_price=signal['price'],
                position_size=signal['size'],
                signal_score=signal['score'],
                edge_pct=signal['edge'],
                opportunity_type=signal['opportunity_type'],
            )

            # Deduct capital
            capital -= signal['size']
            positions[market_id] = trade

        # Resolve trades based on actual outcomes
        for market in batch_markets:
            market_id = market.get('market_id')
            if market_id in positions:
                trade = positions[market_id]
                resolution = market.get('_resolution')

                # Determine exit price based on resolution
                if resolution == 'yes':
                    exit_price = 1.0
                elif resolution == 'no':
                    exit_price = 0.0
                else:
                    exit_price = 0.5  # Unknown

                # Close trade
                trade.close(
                    exit_price=exit_price,
                    exit_timestamp=market.get('close_time') or (batch_timestamp + 86400),
                    exit_reason="resolution"
                )

                # Update capital
                capital += trade.position_size + (trade.pnl or 0)
                trades.append(trade)
                del positions[market_id]

        # Record equity
        equity_curve.append(capital)
        timestamps.append(batch_timestamp)

    # Close any remaining positions at current price
    for market_id, trade in positions.items():
        trade.close(
            exit_price=trade.entry_price,  # No change
            exit_timestamp=timestamps[-1],
            exit_reason="timeout"
        )
        capital += trade.position_size
        trades.append(trade)

    # Calculate results
    winning = [t for t in trades if (t.pnl or 0) > 0]
    losing = [t for t in trades if (t.pnl or 0) <= 0]

    total_pnl = sum(t.pnl or 0 for t in trades)
    total_profit = sum(t.pnl for t in winning) if winning else 0
    total_loss = abs(sum(t.pnl for t in losing)) if losing else 0

    # Drawdown calculation
    peak = initial_capital
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd

    # Build result
    result = BacktestResult(
        initial_capital=initial_capital,
        final_capital=capital,
        total_pnl=total_pnl,
        total_return=(capital - initial_capital) / initial_capital,
        total_trades=len(trades),
        winning_trades=len(winning),
        losing_trades=len(losing),
        win_rate=len(winning) / len(trades) if trades else 0,
        max_drawdown=max_dd * initial_capital,
        max_drawdown_pct=max_dd,
        sharpe_ratio=0,  # Would need daily returns
        sortino_ratio=0,
        avg_profit=np.mean([t.pnl for t in winning]) if winning else 0,
        avg_loss=np.mean([abs(t.pnl) for t in losing]) if losing else 0,
        profit_factor=total_profit / total_loss if total_loss > 0 else float('inf'),
        avg_trade_pnl=total_pnl / len(trades) if trades else 0,
        avg_hold_time_hours=np.mean([t.hold_time_hours or 0 for t in trades]) if trades else 0,
        total_days=(timestamps[-1] - timestamps[0]) / 86400 if len(timestamps) > 1 else 0,
        trades=trades,
        equity_curve=equity_curve,
        timestamps=timestamps,
    )

    return result


def print_results(result: BacktestResult):
    """Print backtest results."""
    print()
    print("=" * 60)
    print("BACKTEST RESULTS: CROSS-MARKET ARBITRAGE")
    print("=" * 60)
    print()
    print("CAPITAL")
    print("-" * 40)
    print(f"Initial Capital:     ${result.initial_capital:,.2f}")
    print(f"Final Capital:       ${result.final_capital:,.2f}")
    print(f"Total P&L:           ${result.total_pnl:,.2f}")
    print(f"Total Return:        {result.total_return:.1%}")
    print()
    print("TRADES")
    print("-" * 40)
    print(f"Total Trades:        {result.total_trades}")
    print(f"Winning Trades:      {result.winning_trades}")
    print(f"Losing Trades:       {result.losing_trades}")
    print(f"Win Rate:            {result.win_rate:.1%}")
    print()
    print("PROFITABILITY")
    print("-" * 40)
    print(f"Avg Winning Trade:   ${result.avg_profit:,.2f}")
    print(f"Avg Losing Trade:    ${result.avg_loss:,.2f}")
    print(f"Profit Factor:       {result.profit_factor:.2f}")
    print(f"Avg Trade P&L:       ${result.avg_trade_pnl:,.2f}")
    print()
    print("RISK")
    print("-" * 40)
    print(f"Max Drawdown:        ${result.max_drawdown:,.2f}")
    print(f"Max Drawdown %:      {result.max_drawdown_pct:.1%}")
    print()
    print("TIME")
    print("-" * 40)
    print(f"Total Days:          {result.total_days:.0f}")
    print(f"Avg Hold Time:       {result.avg_hold_time_hours:.1f} hours")
    print()

    # Sample trades
    if result.trades:
        print("SAMPLE TRADES")
        print("-" * 40)
        for trade in result.trades[:5]:
            pnl_str = f"${trade.pnl:+.2f}" if trade.pnl else "N/A"
            print(f"  {trade.action} @ {trade.entry_price:.2f} -> {trade.exit_price:.2f} | P&L: {pnl_str} | {trade.opportunity_type}")

    # Opportunities by type
    type_stats = defaultdict(lambda: {'count': 0, 'pnl': 0, 'wins': 0})
    for trade in result.trades:
        t = trade.opportunity_type or 'unknown'
        type_stats[t]['count'] += 1
        type_stats[t]['pnl'] += trade.pnl or 0
        if (trade.pnl or 0) > 0:
            type_stats[t]['wins'] += 1

    if type_stats:
        print()
        print("BY OPPORTUNITY TYPE")
        print("-" * 40)
        for opp_type, stats in sorted(type_stats.items(), key=lambda x: -x[1]['pnl']):
            wr = stats['wins'] / stats['count'] if stats['count'] > 0 else 0
            print(f"  {opp_type}:")
            print(f"    Trades: {stats['count']}, Win Rate: {wr:.1%}, P&L: ${stats['pnl']:+,.2f}")

    print()
    print("=" * 60)


def main():
    print("=" * 60)
    print("CROSS-MARKET ARBITRAGE BACKTEST")
    print("=" * 60)
    print()

    # Load data
    print("Loading historical data...")
    markets = load_historical_data()
    print(f"Loaded {len(markets)} markets")

    # Initialize strategy
    print()
    print("Initializing ML strategy...")
    strategy = MLArbitrageStrategy(
        model_path=MODEL_PATH,
        min_score=0.45,  # Lower threshold to find more opportunities
        min_edge=2.0,    # 2% minimum edge
        position_size=100.0,
    )

    # Run backtest
    print()
    print("Running backtest...")
    result = run_backtest(markets, strategy, initial_capital=10000.0)

    # Print results
    print_results(result)


if __name__ == "__main__":
    main()
