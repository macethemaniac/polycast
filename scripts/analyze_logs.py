"""Simple analyzer for opportunity logs."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

LOG_FILE = Path(__file__).resolve().parents[1] / "data" / "logs" / "opportunities.jsonl"


def load_logs() -> List[Dict]:
    if not LOG_FILE.exists():
        return []
    out = []
    with LOG_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def summarize():
    logs = load_logs()
    print(f"Total entries: {len(logs)}")
    if not logs:
        return
    by_cmd = Counter([l.get("command") for l in logs])
    print("By command:", by_cmd)

    scores = []
    markets = Counter()
    for entry in logs:
        items = entry.get("items") or []
        for it in items:
            if "score" in it:
                scores.append(float(it.get("score", 0.0)))
            mid = it.get("market_id") or it.get("poly_market_id") or it.get("kalshi_market_id")
            if mid:
                markets[mid] += 1
    if scores:
        print(f"Scores: count={len(scores)}, avg={sum(scores)/len(scores):.2f}, max={max(scores):.2f}")
    print("Top recurring markets:")
    for mid, cnt in markets.most_common(10):
        print(f"  {mid}: {cnt}")


if __name__ == "__main__":
    summarize()
