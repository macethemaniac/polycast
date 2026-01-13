"""Generate training data from historical markets."""

import json
import time
import random
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
HISTORICAL_FILE = DATA_DIR / "historical" / "polymarket_markets.jsonl"
TRAINING_FILE = DATA_DIR / "training" / "generated_opportunities.jsonl"


def infer_resolution(market):
    """Infer resolution from final price."""
    close_time = market.get("close_time")
    if not close_time or close_time > time.time():
        return None
    yes_price = market.get("final_yes_price")
    if yes_price is None:
        return None
    if yes_price > 0.95:
        return "yes"
    elif yes_price < 0.05:
        return "no"
    return None


def main():
    # Load historical markets
    print("Loading historical markets...")
    markets = []
    with open(HISTORICAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                markets.append(json.loads(line))
    print(f"Loaded {len(markets)} markets")

    # Filter to resolved markets
    resolved = []
    for m in markets:
        res = infer_resolution(m)
        if res:
            m["_resolution"] = res
            resolved.append(m)
    print(f"Resolved markets: {len(resolved)}")

    # Generate opportunities
    opportunities = []

    for m in resolved:
        resolution = m["_resolution"]
        final_yes = m.get("final_yes_price", 0.5)
        volume = m.get("total_volume", 0)
        base_opp = {
            "market_id": m["market_id"],
            "question": m.get("question", ""),
            "description": m.get("description", ""),
            "category": m.get("category", ""),
            "volume": volume,
            "timestamp": m.get("close_time", time.time()) - 86400,
        }

        # Generate synthetic "earlier" prices to simulate opportunities
        # POSITIVE examples: Simulate prices that were mispriced before resolution
        if resolution == "yes":
            # Simulate earlier prices when YES was underpriced
            for sim_price in [0.3, 0.4, 0.5, 0.6, 0.7]:
                opp = base_opp.copy()
                opp.update({
                    "opportunity_id": f"pos_{m['market_id']}_{random.randint(1000,9999)}",
                    "yes_price": sim_price,
                    "no_price": 1.0 - sim_price,
                    "edge_pct": (1.0 - sim_price) * 100,
                    "action": "BUY_YES",
                    "opportunity_type": "mispriced",
                    "label": 1,
                    "resolution": resolution,
                })
                opportunities.append(opp)

        if resolution == "no":
            # Simulate earlier prices when NO was underpriced (YES overpriced)
            for sim_price in [0.3, 0.4, 0.5, 0.6, 0.7]:
                opp = base_opp.copy()
                opp.update({
                    "opportunity_id": f"pos_{m['market_id']}_{random.randint(1000,9999)}",
                    "yes_price": sim_price,
                    "no_price": 1.0 - sim_price,
                    "edge_pct": sim_price * 100,
                    "action": "BUY_NO",
                    "opportunity_type": "mispriced",
                    "label": 1,
                    "resolution": resolution,
                })
                opportunities.append(opp)

        # NEGATIVE examples: Simulate cases where there was no edge
        if resolution == "yes":
            # Already priced correctly high - no edge
            for sim_price in [0.85, 0.9, 0.95]:
                opp = base_opp.copy()
                opp.update({
                    "opportunity_id": f"neg_{m['market_id']}_{random.randint(1000,9999)}",
                    "yes_price": sim_price,
                    "no_price": 1.0 - sim_price,
                    "edge_pct": (1.0 - sim_price) * 100,
                    "action": "BUY_YES",
                    "opportunity_type": "no_edge",
                    "label": 0,
                    "resolution": resolution,
                })
                opportunities.append(opp)

            # FALSE: Betting NO when YES resolved
            for sim_price in [0.5, 0.6, 0.7]:
                opp = base_opp.copy()
                opp.update({
                    "opportunity_id": f"false_{m['market_id']}_{random.randint(1000,9999)}",
                    "yes_price": sim_price,
                    "no_price": 1.0 - sim_price,
                    "edge_pct": sim_price * 100,
                    "action": "BUY_NO",
                    "opportunity_type": "false_signal",
                    "label": 0,
                    "resolution": resolution,
                })
                opportunities.append(opp)

        if resolution == "no":
            # Already priced correctly low - no edge
            for sim_price in [0.05, 0.1, 0.15]:
                opp = base_opp.copy()
                opp.update({
                    "opportunity_id": f"neg_{m['market_id']}_{random.randint(1000,9999)}",
                    "yes_price": sim_price,
                    "no_price": 1.0 - sim_price,
                    "edge_pct": sim_price * 100,
                    "action": "BUY_NO",
                    "opportunity_type": "no_edge",
                    "label": 0,
                    "resolution": resolution,
                })
                opportunities.append(opp)

            # FALSE: Betting YES when NO resolved
            for sim_price in [0.3, 0.4, 0.5]:
                opp = base_opp.copy()
                opp.update({
                    "opportunity_id": f"false_{m['market_id']}_{random.randint(1000,9999)}",
                    "yes_price": sim_price,
                    "no_price": 1.0 - sim_price,
                    "edge_pct": (1.0 - sim_price) * 100,
                    "action": "BUY_YES",
                    "opportunity_type": "false_signal",
                    "label": 0,
                    "resolution": resolution,
                })
                opportunities.append(opp)

    # Balance dataset
    positive = [o for o in opportunities if o["label"] == 1]
    negative = [o for o in opportunities if o["label"] == 0]
    print(f"Raw: {len(positive)} positive, {len(negative)} negative")

    if len(negative) > 2 * len(positive):
        negative = random.sample(negative, 2 * len(positive))

    opportunities = positive + negative
    random.shuffle(opportunities)

    # Save
    TRAINING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_FILE, "w", encoding="utf-8") as f:
        for opp in opportunities:
            f.write(json.dumps(opp) + "\n")

    labels = [o["label"] for o in opportunities]
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    print(f"Saved {len(opportunities)} opportunities")
    print(f"Distribution: {pos_count} positive ({100*pos_count/len(labels):.1f}%), {neg_count} negative ({100*neg_count/len(labels):.1f}%)")


if __name__ == "__main__":
    main()
