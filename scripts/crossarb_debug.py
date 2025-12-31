import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "polycast"))

from cross_arb import find_cross_market_arbitrage  # noqa: E402


def _print_attempts(name: str, attempts: list[dict[str, Any]]):
    print(f"{name} attempts:")
    if not attempts:
        print("  (no attempts recorded)")
        return
    for a in attempts:
        status = a.get("status_code")
        latency = a.get("latency_ms")
        ok = a.get("ok")
        err = a.get("error")
        url = a.get("url")
        print(f"  - {url} | ok={ok} status={status} latency_ms={latency} error={err}")


def _print_candidate(idx: int, cand: dict[str, Any]):
    print(f"[{idx}] combo={cand.get('combo')} similarity={cand.get('similarity',{}).get('combined')}")
    print(f"  pol: {cand.get('pol_question')}")
    print(
        f"    best_bid={cand.get('pol',{}).get('best_bid')} "
        f"best_ask={cand.get('pol',{}).get('best_ask')} "
        f"bid_size={cand.get('pol',{}).get('bid_size')} "
        f"ask_size={cand.get('pol',{}).get('ask_size')} "
        f"prob_bid={cand.get('pol',{}).get('prob_bid')} "
        f"prob_ask={cand.get('pol',{}).get('prob_ask')}"
    )
    print(f"  kal: {cand.get('kal_question')}")
    print(
        f"    best_bid={cand.get('kal',{}).get('best_bid')} "
        f"best_ask={cand.get('kal',{}).get('best_ask')} "
        f"bid_size={cand.get('kal',{}).get('bid_size')} "
        f"ask_size={cand.get('kal',{}).get('ask_size')} "
        f"prob_bid={cand.get('kal',{}).get('prob_bid')} "
        f"prob_ask={cand.get('kal',{}).get('prob_ask')}"
    )
    print(
        f"  edge={cand.get('edge')} total={cand.get('total')} "
        f"reasons={','.join(cand.get('reasons', []))}"
    )
    sim = cand.get("similarity", {})
    print(
        f"  sims: combined={sim.get('combined')} seq={sim.get('seq')} "
        f"jaccard={sim.get('jaccard')} date={sim.get('date')}"
    )


def main():
    parser = argparse.ArgumentParser(description="Cross-market arbitrage debug runner")
    parser.add_argument("--limit-pol", type=int, default=100)
    parser.add_argument("--limit-kal", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=1.02)
    parser.add_argument("--min-similarity", type=float, default=0.4)
    parser.add_argument("--verbose", action="store_true", help="Print verbose diagnostics")
    parser.add_argument(
        "--dump-debug",
        action="store_true",
        help="Write debug/run_<timestamp>.json with diagnostics",
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        default=None,
        help="Optional directory for debug dumps (default: ./debug)",
    )
    args = parser.parse_args()

    result = find_cross_market_arbitrage(
        limit_pol=args.limit_pol,
        limit_kal=args.limit_kal,
        threshold=args.threshold,
        min_similarity=args.min_similarity,
        collect_debug=True,
        verbose=args.verbose,
        dump_debug=args.dump_debug,
        debug_dir=args.debug_dir,
    )
    opportunities, debug = result

    print(f"Polymarket markets fetched: {debug.get('polymarket',{}).get('count')}")
    print(f"Kalshi markets fetched: {debug.get('kalshi',{}).get('count')}")
    print(f"Pairs compared: {debug.get('pairs_compared')}")
    _print_attempts("Polymarket", debug.get("polymarket", {}).get("attempts", []))
    _print_attempts("Kalshi", debug.get("kalshi", {}).get("attempts", []))

    print(f"Opportunities found: {len(opportunities)}")

    candidates = debug.get("candidates", [])
    top_n = min(20, len(candidates))
    print(f"Showing diagnostics for {top_n} candidate pairs (or fewer):")
    for idx, cand in enumerate(candidates[:top_n], start=1):
        _print_candidate(idx, cand)

    if args.dump_debug:
        dbg_file = debug.get("debug_file")
        if dbg_file:
            print(f"Debug dump written to: {dbg_file}")
        else:
            print(f"Debug dump requested but failed: {debug.get('debug_file_error')}")

    if args.verbose:
        print("Full debug payload (truncated to first 2 entries per attempt list):")
        trimmed = debug.copy()
        for key in ("polymarket", "kalshi"):
            if key in trimmed and "attempts" in trimmed[key]:
                trimmed[key] = trimmed[key].copy()
                trimmed[key]["attempts"] = trimmed[key]["attempts"][:2]
        print(json.dumps(trimmed, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
