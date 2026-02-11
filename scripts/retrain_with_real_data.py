"""Retrain ML model with real collected snapshot data.

Run this after collecting sufficient snapshots (recommended: 1000+ price changes).
"""

import sys
sys.path.insert(0, '.')

import json
import pickle
import logging
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import random

import numpy as np

from src.ml.feature_builder_v2 import AdvancedFeatureBuilder
from src.ml.ensemble import ArbitrageEnsemble

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_all_snapshots() -> List[Dict]:
    """Load all collected snapshots."""
    snapshots = []
    files = sorted(SNAPSHOTS_DIR.glob("snapshots_*.jsonl"))

    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            for line in fp:
                if line.strip():
                    snapshots.append(json.loads(line))

    return snapshots


def build_price_history(snapshots: List[Dict]) -> Dict[str, List[Dict]]:
    """Build price history per market from snapshots."""
    history = defaultdict(list)

    for s in snapshots:
        market_id = s.get("market_id")
        if market_id:
            history[market_id].append({
                "timestamp": s.get("timestamp"),
                "yes_price": s.get("yes_price", 0.5),
                "no_price": s.get("no_price", 0.5),
                "volume": s.get("volume", 0),
                "platform": s.get("platform"),
                "question": s.get("question", ""),
                "category": s.get("category", ""),
            })

    # Sort by timestamp
    for market_id in history:
        history[market_id].sort(key=lambda x: x.get("timestamp", 0))

    return dict(history)


def find_price_movements(history: Dict[str, List[Dict]], min_change: float = 0.05) -> List[Dict]:
    """Find significant price movements for training."""
    movements = []

    for market_id, prices in history.items():
        if len(prices) < 2:
            continue

        for i in range(len(prices) - 1):
            p1 = prices[i]
            p2 = prices[i + 1]

            price_change = p2["yes_price"] - p1["yes_price"]

            if abs(price_change) >= min_change:
                # This is a significant price movement
                movements.append({
                    "market_id": market_id,
                    "platform": p1.get("platform"),
                    "question": p1.get("question", ""),
                    "category": p1.get("category", ""),
                    "timestamp_before": p1["timestamp"],
                    "timestamp_after": p2["timestamp"],
                    "price_before": p1["yes_price"],
                    "price_after": p2["yes_price"],
                    "price_change": price_change,
                    "volume": p1.get("volume", 0),
                })

    return movements


def find_cross_platform_pairs(history: Dict[str, List[Dict]]) -> List[Dict]:
    """Find potential cross-platform arbitrage pairs."""
    pairs = []

    # Group by approximate question similarity (simple approach)
    poly_markets = {}
    kalshi_markets = {}

    for market_id, prices in history.items():
        if not prices:
            continue
        latest = prices[-1]
        platform = latest.get("platform")
        question = latest.get("question", "").lower()

        if platform == "polymarket":
            poly_markets[market_id] = {"question": question, "prices": prices}
        elif platform == "kalshi":
            kalshi_markets[market_id] = {"question": question, "prices": prices}

    # Simple matching by common words
    for poly_id, poly_data in poly_markets.items():
        poly_words = set(poly_data["question"].split())

        for kalshi_id, kalshi_data in kalshi_markets.items():
            kalshi_words = set(kalshi_data["question"].split())

            # Check word overlap
            common = poly_words & kalshi_words
            if len(common) >= 3:  # At least 3 common words
                # Get concurrent prices
                poly_prices = poly_data["prices"]
                kalshi_prices = kalshi_data["prices"]

                if poly_prices and kalshi_prices:
                    poly_latest = poly_prices[-1]
                    kalshi_latest = kalshi_prices[-1]

                    price_diff = abs(poly_latest["yes_price"] - kalshi_latest["yes_price"])

                    if price_diff >= 0.03:  # 3% or more difference
                        pairs.append({
                            "poly_id": poly_id,
                            "kalshi_id": kalshi_id,
                            "poly_question": poly_data["question"][:100],
                            "kalshi_question": kalshi_data["question"][:100],
                            "poly_price": poly_latest["yes_price"],
                            "kalshi_price": kalshi_latest["yes_price"],
                            "price_diff": price_diff,
                            "common_words": len(common),
                        })

    return pairs


def generate_training_data(
    movements: List[Dict],
    pairs: List[Dict],
    history: Dict[str, List[Dict]],
    min_samples: int = 500
) -> Tuple[List[Dict], List[int]]:
    """Generate labeled training data from real price movements."""
    opportunities = []
    labels = []

    # From price movements: if price moved toward 0 or 1, the earlier price was an opportunity
    for m in movements:
        price_before = m["price_before"]
        price_after = m["price_after"]
        change = m["price_change"]

        # Create opportunity record
        opp = {
            "market_id": m["market_id"],
            "platform": m["platform"],
            "question": m["question"],
            "category": m["category"],
            "yes_price": price_before,
            "no_price": 1.0 - price_before,
            "volume": m["volume"],
            "timestamp": m["timestamp_before"],
            "opportunity_type": "price_movement",
            "price_change": change,
        }

        # Label: 1 if betting in direction of movement would have been profitable
        # Lower threshold to 5% for positive labels to capture more opportunities
        if abs(change) >= 0.05:
            # Significant movement - this was a real opportunity
            opportunities.append(opp)
            labels.append(1)

    # Generate negative samples from stable markets (little to no movement)
    stable_markets = []
    for market_id, prices in history.items():
        if len(prices) < 2:
            continue
        # Check max price swing across all snapshots
        yes_prices = [p["yes_price"] for p in prices]
        max_swing = max(yes_prices) - min(yes_prices)
        if max_swing < 0.03:  # Stable market - no significant movement
            latest = prices[-1]
            stable_markets.append({
                "market_id": market_id,
                "platform": latest.get("platform"),
                "question": latest.get("question", ""),
                "category": latest.get("category", ""),
                "yes_price": latest["yes_price"],
                "no_price": 1.0 - latest["yes_price"],
                "volume": latest.get("volume", 0),
                "timestamp": latest.get("timestamp", time.time()),
                "opportunity_type": "stable_market",
                "price_change": 0,
            })

    # Sample negatives to roughly balance with positives
    num_positives = len(opportunities)
    num_negatives_needed = min(len(stable_markets), num_positives * 2)
    if stable_markets and num_negatives_needed > 0:
        sampled_negatives = random.sample(stable_markets, num_negatives_needed)
        for neg in sampled_negatives:
            opportunities.append(neg)
            labels.append(0)

    # From cross-platform pairs
    for pair in pairs:
        # The platform with lower YES price has the arbitrage opportunity
        if pair["price_diff"] >= 0.05:
            # Create opportunity for both sides
            opp_poly = {
                "market_id": pair["poly_id"],
                "platform": "polymarket",
                "question": pair["poly_question"],
                "category": "",
                "yes_price": pair["poly_price"],
                "no_price": 1.0 - pair["poly_price"],
                "volume": 0,
                "timestamp": time.time(),
                "opportunity_type": "cross_platform",
                "price_diff": pair["price_diff"],
            }

            opportunities.append(opp_poly)
            labels.append(1)  # Cross-platform divergence is an opportunity

    # Balance dataset if needed
    pos_idx = [i for i, l in enumerate(labels) if l == 1]
    neg_idx = [i for i, l in enumerate(labels) if l == 0]

    logger.info(f"Raw data: {len(pos_idx)} positive, {len(neg_idx)} negative")

    # If very imbalanced, downsample majority class
    if len(neg_idx) > 2 * len(pos_idx) and len(pos_idx) > 0:
        neg_idx = random.sample(neg_idx, min(2 * len(pos_idx), len(neg_idx)))

    selected_idx = pos_idx + neg_idx
    random.shuffle(selected_idx)

    opportunities = [opportunities[i] for i in selected_idx]
    labels = [labels[i] for i in selected_idx]

    return opportunities, labels


def build_features(opportunities: List[Dict]) -> Tuple[np.ndarray, List[str]]:
    """Build feature matrix from opportunities."""
    builder = AdvancedFeatureBuilder()

    X_list = []
    valid_idx = []

    for i, opp in enumerate(opportunities):
        market = {
            "id": opp.get("market_id", ""),
            "question": opp.get("question", ""),
            "description": "",
            "yes_price": opp.get("yes_price", 0.5),
            "no_price": opp.get("no_price", 0.5),
            "volume": opp.get("volume", 0),
            "category": opp.get("category", ""),
            "close_time": opp.get("timestamp", time.time()) + 86400 * 7,
            "created_at": opp.get("timestamp", time.time()) - 86400 * 30,
        }

        try:
            fs = builder.build_features(market)
            X_list.append(fs.to_vector())
            valid_idx.append(i)
        except Exception as e:
            logger.debug(f"Feature error: {e}")
            continue

    return np.array(X_list), valid_idx, builder.get_feature_names()


def train_model(X: np.ndarray, y: np.ndarray) -> Tuple[ArbitrageEnsemble, Dict]:
    """Train ensemble model."""
    logger.info(f"Training on {len(X)} samples...")

    ensemble = ArbitrageEnsemble(use_stacking=True)
    results = ensemble.fit(X, y, validation_split=0.2)

    return ensemble, results


def evaluate_model(ensemble: ArbitrageEnsemble, X: np.ndarray, y: np.ndarray) -> Dict:
    """Evaluate model performance."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    y_pred_proba = ensemble.predict_proba(X)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "auc": roc_auc_score(y, y_pred_proba) if len(np.unique(y)) > 1 else 0.5,
    }


def main():
    print("=" * 60)
    print("RETRAIN MODEL WITH REAL DATA")
    print("=" * 60)
    print()

    # Load snapshots
    print("Loading snapshots...")
    snapshots = load_all_snapshots()
    print(f"Loaded {len(snapshots)} snapshots")

    if len(snapshots) < 100:
        print("\nWARNING: Not enough snapshots yet!")
        print("Need at least 100 snapshots (ideally 1000+)")
        print("Run: python scripts/collect_snapshots.py --interval 30")
        print("Wait a few hours for data to accumulate")
        return

    # Build price history
    print("\nBuilding price history...")
    history = build_price_history(snapshots)
    print(f"Markets with history: {len(history)}")

    markets_with_changes = sum(1 for h in history.values() if len(h) > 1)
    print(f"Markets with multiple snapshots: {markets_with_changes}")

    if markets_with_changes < 50:
        print("\nWARNING: Not enough price changes yet!")
        print("Need markets with multiple snapshots to track price movements")
        print("Continue running collector and try again later")
        return

    # Find price movements
    print("\nFinding price movements...")
    movements = find_price_movements(history, min_change=0.03)
    print(f"Significant price movements: {len(movements)}")

    # Find cross-platform pairs
    print("\nFinding cross-platform pairs...")
    pairs = find_cross_platform_pairs(history)
    print(f"Cross-platform arbitrage pairs: {len(pairs)}")

    # Generate training data
    print("\nGenerating training data...")
    opportunities, labels = generate_training_data(movements, pairs, history)
    print(f"Training samples: {len(opportunities)}")
    print(f"Positive: {sum(labels)}, Negative: {len(labels) - sum(labels)}")

    if len(opportunities) < 100:
        print("\nWARNING: Not enough training data!")
        print("Need at least 100 samples (ideally 500+)")
        return

    # Build features
    print("\nBuilding features...")
    X, valid_idx, feature_names = build_features(opportunities)
    y = np.array([labels[i] for i in valid_idx])
    print(f"Feature matrix: {X.shape}")

    # Train model
    print("\nTraining model...")
    ensemble, train_results = train_model(X, y)

    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate_model(ensemble, X, y)

    print()
    print("RESULTS")
    print("-" * 40)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")

    # Save model
    print("\nSaving model...")
    model_path = MODEL_DIR / "ensemble_model_real.pkl"
    model_data = {
        "ensemble": ensemble,
        "feature_names": feature_names,
        "timestamp": time.time(),
        "version": "2.0-real-data",
        "metrics": metrics,
        "training_samples": len(X),
        "snapshot_count": len(snapshots),
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"Model saved to {model_path}")

    # Also update main model if this one is better
    main_model_path = MODEL_DIR / "ensemble_model.pkl"
    if main_model_path.exists():
        with open(main_model_path, "rb") as f:
            old_model = pickle.load(f)
        old_auc = old_model.get("metrics", {}).get("auc", 0)

        if metrics["auc"] > old_auc:
            print(f"\nNew model is better! (AUC {metrics['auc']:.4f} > {old_auc:.4f})")
            print("Updating main model...")
            with open(main_model_path, "wb") as f:
                pickle.dump(model_data, f)

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
