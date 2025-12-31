import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "polycast"))

from polycast.adapters import polymarket as pm_adapter  # noqa: E402
from polycast.adapters import kalshi as kal_adapter  # noqa: E402
from polycast.matching.engine import match_markets  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Export Polymarket↔Kalshi match candidates")
    parser.add_argument(
        "--export-candidates",
        type=str,
        default="outputs/match_candidates.json",
        help="Path to write match candidates JSON",
    )
    parser.add_argument("--limit-pol", type=int, default=100)
    parser.add_argument("--limit-kal", type=int, default=100)
    parser.add_argument("--min-confidence", type=float, default=0.25)
    args = parser.parse_args()

    pm_markets = pm_adapter.list_markets(limit=args.limit_pol, return_debug=False)
    kal_markets = kal_adapter.list_markets(limit=args.limit_kal, return_debug=False)

    results = match_markets(pm_markets, kal_markets, top_k=5, min_confidence=args.min_confidence)

    out_path = Path(args.export_candidates)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"candidates": results}
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote match candidates to {out_path}")


if __name__ == "__main__":
    main()
