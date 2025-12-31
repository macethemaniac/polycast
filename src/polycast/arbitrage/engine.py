from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import time

from polycast.adapters import polymarket as pm_adapter
from polycast.adapters import kalshi as kal_adapter
from polycast.adapters import OrderBook


@dataclass
class ArbConfig:
    min_edge: float = 0.01  # minimum edge after costs
    min_size: float = 0.0   # minimum available size
    max_staleness: int = 3600  # seconds
    fee_buy: float = 0.005
    fee_sell: float = 0.005
    slippage: float = 0.01


def _ob_stale(ob: Optional[OrderBook], max_age: int) -> bool:
    if ob is None or ob.ts is None:
        return False
    try:
        now = time.time()
        return (now - float(ob.ts)) > max_age
    except Exception:
        return False


def _available_size(ob_buy: Optional[OrderBook], ob_sell: Optional[OrderBook]) -> float:
    sizes = []
    if ob_buy and ob_buy.ask_size is not None:
        sizes.append(ob_buy.ask_size)
    if ob_sell and ob_sell.bid_size is not None:
        sizes.append(ob_sell.bid_size)
    return min(sizes) if sizes else 0.0


def _edge_after_costs(buy_ask: float, sell_bid: float, cfg: ArbConfig) -> float:
    sell_eff = sell_bid * (1 - cfg.slippage) - cfg.fee_sell
    buy_eff = buy_ask * (1 + cfg.slippage) + cfg.fee_buy
    return sell_eff - buy_eff


def compute_opportunities(
    match_results: Dict[str, Dict[str, Any]],
    cfg: ArbConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Compute arbitrage opps for matched PM/Kalshi pairs."""
    rejections: Dict[str, int] = {}
    opportunities: List[Dict[str, Any]] = []

    pm_markets = pm_adapter.list_markets(return_debug=False)
    kal_markets = kal_adapter.list_markets(return_debug=False)
    pm_map = {m.id: m for m in pm_markets}
    kal_map = {m.id: m for m in kal_markets}

    for pm_id, data in match_results.items():
        pm_market = pm_map.get(str(pm_id))
        if not pm_market:
            continue
        confidence_matches = data.get("matches", [])
        for m in confidence_matches:
            kal_id = m.get("kal_id")
            confidence = float(m.get("score", 0.0))
            kal_market = kal_map.get(str(kal_id))
            if not kal_market:
                continue

            ob_pm_yes = pm_adapter.get_orderbook(pm_id, "YES")
            ob_pm_no = pm_adapter.get_orderbook(pm_id, "NO")
            ob_kal_yes = kal_adapter.get_orderbook(kal_id, "YES")
            ob_kal_no = kal_adapter.get_orderbook(kal_id, "NO")

            combos = [
                ("pol_yes+kal_no", ob_pm_yes, ob_kal_no),
                ("pol_no+kal_yes", ob_pm_no, ob_kal_yes),
            ]
            for combo, buy_ob, sell_ob in combos:
                reasons = []
                if buy_ob is None or sell_ob is None:
                    reasons.append("NO_LIQUIDITY")
                buy_ask = buy_ob.best_ask if buy_ob else None
                sell_bid = sell_ob.best_bid if sell_ob else None
                if buy_ask is None or sell_bid is None:
                    reasons.append("NO_LIQUIDITY")
                if _ob_stale(buy_ob, cfg.max_staleness) or _ob_stale(sell_ob, cfg.max_staleness):
                    reasons.append("STALE")

                if reasons:
                    for r in reasons:
                        rejections[r] = rejections.get(r, 0) + 1
                    continue

                edge_before = sell_bid - buy_ask
                edge_after = _edge_after_costs(buy_ask, sell_bid, cfg)
                size = _available_size(buy_ob, sell_ob)
                if size < cfg.min_size:
                    rejections["LOW_SIZE"] = rejections.get("LOW_SIZE", 0) + 1
                    continue
                if edge_after < cfg.min_edge:
                    rejections["LOW_EDGE"] = rejections.get("LOW_EDGE", 0) + 1
                    continue

                rank_score = edge_after * size * confidence
                opportunities.append(
                    {
                        "pm_id": pm_id,
                        "pm_title": pm_market.title,
                        "kal_id": kal_id,
                        "kal_title": kal_market.title,
                        "combo": combo,
                        "buy_ask": buy_ask,
                        "sell_bid": sell_bid,
                        "edge_before": edge_before,
                        "edge_after": edge_after,
                        "size": size,
                        "confidence": confidence,
                        "rank_score": rank_score,
                    }
                )

    opportunities.sort(key=lambda x: x["rank_score"], reverse=True)
    return opportunities, rejections


def save_opportunities(opps: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"opportunities": opps}, f, indent=2, ensure_ascii=False)


def format_console_table(opps: List[Dict[str, Any]], top_n: int = 20) -> str:
    lines = []
    headers = ["pm_id", "kal_id", "combo", "edge_after", "size", "conf", "score"]
    lines.append(" | ".join(headers))
    lines.append("-" * 72)
    for o in opps[:top_n]:
        lines.append(
            f"{o['pm_id']} | {o['kal_id']} | {o['combo']} | "
            f"{o['edge_after']:.4f} | {o['size']:.2f} | {o['confidence']:.2f} | {o['rank_score']:.4f}"
        )
    return "\n".join(lines)
