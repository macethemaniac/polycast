"""Training script for ML models.

Generates training data from historical markets and trains the ensemble.
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle

import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
HISTORICAL_FILE = DATA_DIR / "historical" / "polymarket_markets.jsonl"
TRAINING_FILE = DATA_DIR / "training" / "generated_opportunities.jsonl"
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_historical_markets() -> List[Dict]:
    """Load historical markets from file."""
    markets = []
    if HISTORICAL_FILE.exists():
        with open(HISTORICAL_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    markets.append(json.loads(line))
    logger.info(f"Loaded {len(markets)} historical markets")
    return markets


def infer_resolution(market: Dict) -> Optional[str]:
    """Infer resolution from final price."""
    close_time = market.get('close_time')
    if not close_time or close_time > time.time():
        return None

    yes_price = market.get('final_yes_price')
    if yes_price is None:
        return None

    if yes_price > 0.95:
        return "yes"
    elif yes_price < 0.05:
        return "no"
    return None


def generate_training_opportunities(markets: List[Dict]) -> List[Dict]:
    """Generate training opportunities from historical markets."""
    opportunities = []

    # Filter to resolved markets
    resolved = []
    for m in markets:
        resolution = infer_resolution(m)
        if resolution:
            m['_inferred_resolution'] = resolution
            resolved.append(m)

    logger.info(f"Found {len(resolved)} resolved markets")

    # Group by category for finding related markets
    by_category = {}
    for m in resolved:
        cat = m.get('category', 'unknown') or 'unknown'
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(m)

    # Generate positive opportunities (correct arbitrage signals)
    # These are cases where price diverged from true outcome
    for m in resolved:
        resolution = m['_inferred_resolution']
        yes_price = m.get('final_yes_price', 0.5)

        # Get historical prices if available
        price_history = m.get('price_history', [])
        if price_history:
            for ph in price_history:
                hist_yes = ph.get('yes_price', 0.5)

                # Create opportunity based on mispricing
                if resolution == "yes" and hist_yes < 0.7:
                    # Market underpriced YES - this was a good buy
                    edge = (1.0 - hist_yes) * 100  # Edge if we bought YES
                    opp = create_opportunity(m, hist_yes, edge, label=1, action="BUY_YES")
                    opportunities.append(opp)

                elif resolution == "no" and hist_yes > 0.3:
                    # Market overpriced YES - this was a good sell/buy NO
                    edge = hist_yes * 100  # Edge if we bought NO
                    opp = create_opportunity(m, hist_yes, edge, label=1, action="BUY_NO")
                    opportunities.append(opp)

        # Also create from final prices if there was mispricing opportunity
        if resolution == "yes" and yes_price < 0.8:
            edge = (1.0 - yes_price) * 100
            opp = create_opportunity(m, yes_price, edge, label=1, action="BUY_YES")
            opportunities.append(opp)
        elif resolution == "no" and yes_price > 0.2:
            edge = yes_price * 100
            opp = create_opportunity(m, yes_price, edge, label=1, action="BUY_NO")
            opportunities.append(opp)

    # Generate negative opportunities (false signals)
    # These are cases where the signal was wrong
    for m in resolved:
        resolution = m['_inferred_resolution']
        yes_price = m.get('final_yes_price', 0.5)

        # Create false signal - suggesting wrong action
        if resolution == "yes" and yes_price > 0.5:
            # Market correctly priced high, but simulate false sell signal
            opp = create_opportunity(m, yes_price, 5.0, label=0, action="BUY_NO")
            opportunities.append(opp)
        elif resolution == "no" and yes_price < 0.5:
            # Market correctly priced low, but simulate false buy signal
            opp = create_opportunity(m, yes_price, 5.0, label=0, action="BUY_YES")
            opportunities.append(opp)

    # Generate cross-category comparisons (related market arbitrage)
    categories = list(by_category.keys())
    for cat in categories:
        cat_markets = by_category[cat]
        if len(cat_markets) < 2:
            continue

        # Create pairs within category
        for i, m1 in enumerate(cat_markets[:20]):  # Limit for speed
            for m2 in cat_markets[i+1:i+5]:  # Sample pairs
                # Check if resolutions aligned (correlated markets)
                res1 = m1['_inferred_resolution']
                res2 = m2['_inferred_resolution']

                price1 = m1.get('final_yes_price', 0.5)
                price2 = m2.get('final_yes_price', 0.5)

                # Calculate similarity
                q1 = m1.get('question', '').lower()
                q2 = m2.get('question', '').lower()
                common_words = len(set(q1.split()) & set(q2.split()))
                similarity = common_words / max(len(q1.split()), len(q2.split()), 1)

                if similarity > 0.3:  # Related markets
                    if res1 == res2:
                        # Correlated markets that resolved same - opportunity if prices diverged
                        price_diff = abs(price1 - price2)
                        if price_diff > 0.1:
                            # This was an arbitrage opportunity
                            edge = price_diff * 50
                            opp = create_pair_opportunity(m1, m2, edge, label=1, opp_type="correlated")
                            opportunities.append(opp)
                    else:
                        # Markets looked similar but resolved differently
                        # Betting on correlation would have been wrong
                        edge = abs(price1 - price2) * 50
                        opp = create_pair_opportunity(m1, m2, edge, label=0, opp_type="false_correlation")
                        opportunities.append(opp)

    logger.info(f"Generated {len(opportunities)} training opportunities")
    return opportunities


def create_opportunity(market: Dict, yes_price: float, edge: float, label: int, action: str) -> Dict:
    """Create single-market opportunity."""
    return {
        "opportunity_id": f"gen_{market['market_id']}_{random.randint(1000, 9999)}",
        "timestamp": market.get('close_time', time.time()) - 86400,  # Day before close
        "market_id": market['market_id'],
        "platform": "polymarket",
        "question": market.get('question', ''),
        "description": market.get('description', ''),
        "category": market.get('category', ''),
        "yes_price": yes_price,
        "no_price": 1.0 - yes_price,
        "volume": market.get('total_volume', 0),
        "edge_pct": edge,
        "action": action,
        "opportunity_type": "single_market",
        "label": label,
        "label_source": "resolution",
        "resolution": market.get('_inferred_resolution'),
        "features": {
            "yes_price": yes_price,
            "no_price": 1.0 - yes_price,
            "edge_pct": edge,
            "volume": market.get('total_volume', 0),
        }
    }


def create_pair_opportunity(m1: Dict, m2: Dict, edge: float, label: int, opp_type: str) -> Dict:
    """Create pair-market opportunity."""
    return {
        "opportunity_id": f"pair_{m1['market_id']}_{m2['market_id']}_{random.randint(1000, 9999)}",
        "timestamp": min(m1.get('close_time', time.time()), m2.get('close_time', time.time())) - 86400,
        "market_id": f"{m1['market_id']}_{m2['market_id']}",
        "platform": "polymarket",
        "question": f"{m1.get('question', '')[:50]} vs {m2.get('question', '')[:50]}",
        "description": "",
        "category": m1.get('category', ''),
        "yes_price": m1.get('final_yes_price', 0.5),
        "no_price": m2.get('final_yes_price', 0.5),
        "volume": (m1.get('total_volume', 0) + m2.get('total_volume', 0)) / 2,
        "edge_pct": edge,
        "action": "PAIR_TRADE",
        "opportunity_type": opp_type,
        "label": label,
        "label_source": "resolution",
        "resolution": f"{m1.get('_inferred_resolution')}_{m2.get('_inferred_resolution')}",
        "features": {
            "yes_price": m1.get('final_yes_price', 0.5),
            "no_price": m2.get('final_yes_price', 0.5),
            "price_diff": abs(m1.get('final_yes_price', 0.5) - m2.get('final_yes_price', 0.5)),
            "edge_pct": edge,
            "volume": (m1.get('total_volume', 0) + m2.get('total_volume', 0)) / 2,
        }
    }


def build_features(opportunities: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Build feature matrix from opportunities."""
    from src.ml.feature_builder_v2 import AdvancedFeatureBuilder

    builder = AdvancedFeatureBuilder()

    X_list = []
    y_list = []

    for opp in opportunities:
        # Build market dict for feature builder
        market = {
            "id": opp.get("market_id", ""),
            "question": opp.get("question", ""),
            "description": opp.get("description", ""),
            "yes_price": opp.get("yes_price", 0.5),
            "no_price": opp.get("no_price", 0.5),
            "volume": opp.get("volume", 0),
            "category": opp.get("category", ""),
            "close_time": opp.get("timestamp", time.time()) + 86400 * 7,  # Week from timestamp
            "created_at": opp.get("timestamp", time.time()) - 86400 * 30,  # Month before
        }

        try:
            features = builder.build_features(market)
            X_list.append(features)
            y_list.append(opp.get("label", 0))
        except Exception as e:
            logger.debug(f"Error building features: {e}")
            continue

    X = np.array(X_list)
    y = np.array(y_list)

    logger.info(f"Built features: X shape={X.shape}, y shape={y.shape}")
    logger.info(f"Label distribution: {sum(y)} positive, {len(y) - sum(y)} negative")

    return X, y


def train_ensemble(X: np.ndarray, y: np.ndarray) -> Dict:
    """Train the ensemble model."""
    from src.ml.ensemble import ArbitrageEnsemble
    from src.ml.training import TimeSeriesSplit

    logger.info("Training ensemble model...")

    # Create ensemble
    ensemble = ArbitrageEnsemble(use_stacking=True)

    # Train with validation
    results = ensemble.fit(X, y, validation_split=0.2)

    logger.info(f"Training complete!")
    logger.info(f"  Train AUC: {results.get('train_auc', 0):.4f}")
    logger.info(f"  Validation AUC: {results.get('val_auc', 0):.4f}")
    logger.info(f"  Validation Accuracy: {results.get('val_accuracy', 0):.4f}")

    return {"ensemble": ensemble, "results": results}


def evaluate_model(ensemble, X: np.ndarray, y: np.ndarray) -> Dict:
    """Evaluate model performance."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report
    )

    # Get predictions
    y_pred_proba = ensemble.predict_proba(X)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "auc": roc_auc_score(y, y_pred_proba) if len(np.unique(y)) > 1 else 0.5,
    }

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)

    logger.info("\n=== Model Evaluation ===")
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"AUC:       {metrics['auc']:.4f}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    logger.info(f"  FN={cm[1,0]}, TP={cm[1,1]}")

    return metrics


def save_model(ensemble, feature_names: List[str]) -> str:
    """Save trained model to disk."""
    model_path = MODEL_DIR / "ensemble_model.pkl"

    model_data = {
        "ensemble": ensemble,
        "feature_names": feature_names,
        "timestamp": time.time(),
        "version": "1.0"
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    logger.info(f"Model saved to {model_path}")
    return str(model_path)


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("ML MODEL TRAINING PIPELINE")
    print("=" * 60)
    print()

    # Step 1: Load historical data
    print("Step 1: Loading historical markets...")
    markets = load_historical_markets()

    if len(markets) < 100:
        print("ERROR: Not enough historical data. Run data collection first.")
        return

    # Step 2: Generate training opportunities
    print("\nStep 2: Generating training opportunities...")
    opportunities = generate_training_opportunities(markets)

    if len(opportunities) < 100:
        print("ERROR: Not enough training opportunities generated.")
        return

    # Save generated opportunities
    TRAINING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_FILE, 'w', encoding='utf-8') as f:
        for opp in opportunities:
            f.write(json.dumps(opp) + '\n')
    print(f"Saved {len(opportunities)} opportunities to {TRAINING_FILE}")

    # Step 3: Build features
    print("\nStep 3: Building features...")
    X, y = build_features(opportunities)

    if len(X) < 100:
        print("ERROR: Not enough valid training samples.")
        return

    # Step 4: Train ensemble
    print("\nStep 4: Training ensemble model...")
    training_result = train_ensemble(X, y)
    ensemble = training_result["ensemble"]

    # Step 5: Evaluate
    print("\nStep 5: Evaluating model...")
    metrics = evaluate_model(ensemble, X, y)

    # Step 6: Save model
    print("\nStep 6: Saving model...")
    from src.ml.feature_builder_v2 import AdvancedFeatureBuilder
    builder = AdvancedFeatureBuilder()
    model_path = save_model(ensemble, builder.get_feature_names())

    # Summary
    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Training samples:  {len(X)}")
    print(f"Positive samples:  {sum(y)} ({100*sum(y)/len(y):.1f}%)")
    print(f"Negative samples:  {len(y) - sum(y)} ({100*(len(y)-sum(y))/len(y):.1f}%)")
    print(f"Final AUC:         {metrics['auc']:.4f}")
    print(f"Final Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Model saved to:    {model_path}")
    print()


if __name__ == "__main__":
    main()
