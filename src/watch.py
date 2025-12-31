import argparse
import datetime
import time
import traceback
from pathlib import Path

from polycast.cross_arb import find_cross_market_arbitrage
from polycast.arbitrage.engine import save_opportunities, format_console_table


def run_scan_once(interval: int, limit_pol: int = 50, limit_kal: int = 50) -> None:
    attempts = 3
    backoff = 2
    for attempt in range(attempts):
        try:
            opps, debug = find_cross_market_arbitrage(
                limit_pol=limit_pol,
                limit_kal=limit_kal,
                collect_debug=True,
            )
            ts = datetime.datetime.now(datetime.UTC).isoformat()
            pm_count = debug.get("polymarket", {}).get("count", 0)
            kal_count = debug.get("kalshi", {}).get("count", 0)
            pairs_compared = debug.get("pairs_compared", 0)
            opp_count = len(opps)
            best_edge = max((o.get("edge_after", 0.0) for o in opps), default=0.0)
            print(
                f"[{ts}] markets_fetched={pm_count + kal_count} comparisons={pairs_compared} "
                f"opps={opp_count} best_edge={best_edge:.4f}"
            )
            if opps:
                print(format_console_table(opps))
                out_path = Path("outputs") / f"opps_{int(time.time())}.json"
                save_opportunities(opps, out_path)
                print(f"Saved opportunities to {out_path}")
            else:
                rej = debug.get("rejection_counts", {})
                if rej:
                    sorted_rej = sorted(rej.items(), key=lambda x: x[1], reverse=True)
                    print("Top rejection reasons:", ", ".join(f"{k}:{v}" for k, v in sorted_rej[:5]))
                else:
                    print("No opportunities; no rejection data available.")
            return
        except Exception as exc:
            print(f"Scan attempt {attempt+1} failed: {exc}")
            traceback.print_exc()
            if attempt < attempts - 1:
                time.sleep(backoff)
                backoff *= 2
            else:
                print("All attempts failed; will try again after interval.")


def watch_loop(interval: int, limit_pol: int, limit_kal: int, once: bool = False) -> None:
    next_run = time.time()
    while True:
        now = time.time()
        if now < next_run:
            time.sleep(max(0, next_run - now))
        start = time.time()
        run_scan_once(interval, limit_pol=limit_pol, limit_kal=limit_kal)
        next_run = start + interval
        if once:
            break


def main():
    parser = argparse.ArgumentParser(description="Run periodic arbitrage scans.")
    parser.add_argument("--interval", type=int, default=300, help="Interval in seconds between scans (default 300)")
    parser.add_argument("--limit-pol", type=int, default=50, help="Polymarket limit per scan")
    parser.add_argument("--limit-kal", type=int, default=50, help="Kalshi limit per scan")
    parser.add_argument("--once", action="store_true", help="Run a single scan and exit")
    args = parser.parse_args()
    watch_loop(args.interval, limit_pol=args.limit_pol, limit_kal=args.limit_kal, once=args.once)


if __name__ == "__main__":
    main()
