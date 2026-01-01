"""
Telegram bot integration for the arbitrage scanner.

Provides Telegram commands to scan spot exchanges via CCXT and check
Polymarket/Kalshi arbitrage, with optional scheduled watch and deduped alerts.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple, List

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from scanner import scan_arbitrage
from polymarket_api import find_polymarket_arbitrage
from cross_arb import find_cross_market_arbitrage

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SEEN_OPPS_FILE = DATA_DIR / "seen_opps.json"


def load_seen(prune_older_sec: int = 7 * 24 * 3600) -> Dict[str, Dict[str, float]]:
    """Load seen opportunities, pruning entries older than prune_older_sec."""
    try:
        if not SEEN_OPPS_FILE.exists():
            return {}
        data = json.loads(SEEN_OPPS_FILE.read_text(encoding="utf-8"))
        now_ts = time.time()
        pruned = {k: v for k, v in data.items() if now_ts - float(v.get("ts", 0)) <= prune_older_sec}
        if len(pruned) != len(data):
            save_seen(pruned)
        return pruned
    except Exception:
        return {}


def save_seen(d: Dict[str, Dict[str, float]]) -> None:
    try:
        SEEN_OPPS_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except Exception:
        pass


def build_fingerprint(buy_src: str, sell_src: str, market_id: str, buy_price: float, sell_price: float) -> str:
    """Stable fingerprint for an opportunity."""
    rounded_buy = round(float(buy_price), 4)
    rounded_sell = round(float(sell_price), 4)
    payload = f"{buy_src}|{sell_src}|{market_id}|{rounded_buy}|{rounded_sell}"
    return str(hash(payload))


def should_alert(seen: Dict[str, Dict[str, float]], fp: str, edge: float, cooldown_sec: int, improve_threshold: float, now_ts: float) -> Tuple[bool, str, float]:
    """
    Decide whether to alert.
    Returns (ok, tag, prev_edge) with tag in {"NEW","IMPROVED"}.
    """
    prev = seen.get(fp)
    if not prev:
        return True, "NEW", 0.0
    prev_ts = prev.get("ts", 0)
    prev_edge = float(prev.get("edge", 0.0))
    if now_ts - prev_ts < cooldown_sec:
        if edge - prev_edge >= improve_threshold:
            return True, "IMPROVED", prev_edge
        return False, "COOLDOWN", prev_edge
    if edge - prev_edge >= improve_threshold:
        return True, "IMPROVED", prev_edge
    return False, "COOLDOWN", prev_edge


def format_arbitrage_message(pair: str, prices: Dict[str, float], arbitrage_result: Dict) -> str:
    """Build a human-readable arbitrage message (ASCII-safe) showing only best spread."""
    lines = [
        "<b>Arbitrage Scanner</b>",
        f"Pair: <code>{pair}</code>",
        "",
        "<b>Best Spread:</b>",
        f"- Buy on: <b>{arbitrage_result['buy_exchange'].upper()}</b> @ <code>${arbitrage_result['buy_price']:,.2f}</code>",
        f"- Sell on: <b>{arbitrage_result['sell_exchange'].upper()}</b> @ <code>${arbitrage_result['sell_price']:,.2f}</code>",
        f"- Spread: <code>${arbitrage_result['spread']:,.2f}</code> (<code>{arbitrage_result['spread_percent']:.4f}%</code>)",
        "",
    ]
    if arbitrage_result["spread_percent"] > 0:
        lines.append("<b>Arbitrage opportunity detected!</b>")
    else:
        lines.append("No arbitrage opportunity (prices are equal)")
    return "\n".join(lines)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    welcome_message = (
        "<b>Welcome to Arbitrage Scanner Bot!</b>\n\n"
        "I can help you discover arbitrage opportunities and fetch market prices.\n\n"
        "<b>Quick Commands</b>\n"
        "- /scan [pair] - Scan a pair (default BTC/USDT) via CCXT\n"
        "- /price &lt;pair&gt; - Get prices for any pair via CCXT\n"
        "- /polyarb - Check Polymarket for YES/NO arbitrage\n"
        "- /crossarb [min_similarity] - Cross-market scan (Polymarket &lt;-&gt; Kalshi)\n"
        "- /help - Full help and examples"
    )
    await update.message.reply_text(welcome_message, parse_mode="HTML")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_message = (
        "<b>Help - Arbitrage Scanner Bot</b>\n\n"
        "Use these commands to query prices and find arbitrage:\n\n"
        "- <code>/start</code> - Show the welcome message\n"
        "- <code>/scan [pair]</code> - Scan a pair (default BTC/USDT) via CCXT\n"
        "- <code>/price &lt;pair&gt;</code> - Get prices for any <code>BASE/QUOTE</code>\n"
        "- <code>/polyarb</code> - Detect internal Polymarket binary arbitrage\n"
        "- <code>/crossarb [min_similarity]</code> - Cross-market scan between Polymarket and Kalshi. Optional similarity (0-1).\n\n"
        "Examples:\n"
        "<code>/scan</code>\n"
        "<code>/scan ETH/USDC</code>\n"
        "<code>/price BTC/USDT</code>\n"
        "<code>/crossarb 0.25</code> - run cross-arb with similarity 0.25"
    )
    await update.message.reply_text(help_message, parse_mode="HTML")


async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pair = "BTC/USDT"
    min_similarity = 0.4
    for arg in context.args:
        if "/" in arg:
            pair = arg.upper()
            continue
        try:
            min_similarity = float(arg)
        except Exception:
            continue

    processing_msg = await update.message.reply_text(
        f"Fetching {pair} prices from data sources...",
        parse_mode="HTML",
    )

    try:
        arbitrage_result, error = scan_arbitrage(pair)
        if error:
            spot_message = f"<b>Spot scan error:</b> {error}"
            logger.error("Error in scan_command: %s", error)
        else:
            spot_message = format_arbitrage_message(pair, arbitrage_result["prices"], arbitrage_result)

        cross_lines = ["<b>Cross-market Arbitrage (Polymarket &lt;-&gt; Kalshi)</b>\n"]
        try:
            cross_results = find_cross_market_arbitrage(limit_pol=100, limit_kal=100, min_similarity=min_similarity)
            if not cross_results:
                cross_lines.append("No cross-market arbitrage found right now.")
            else:
                for r in cross_results[:5]:
                    cross_lines.append(
                        f"Type: {r['type']}\n"
                        f"Polymarket: {r['pol_question']}\n"
                        f"Kalshi: {r['kal_question']}\n"
                        f"Total: {r['total']:.2f} Profit: {r['profit_pct']:.2f}%\n"
                    )
        except Exception as exc:
            cross_lines.append(f"<b>Cross-market error:</b> {exc}")
            logger.error("Error in scan_command (crossarb): %s", exc, exc_info=True)

        message = "\n\n".join([spot_message, "\n".join(cross_lines)])
        await processing_msg.edit_text(message, parse_mode="HTML")

    except Exception as exc:
        error_message = f"<b>Error:</b> {exc}"
        await processing_msg.edit_text(error_message, parse_mode="HTML")
        logger.error("Error in scan_command: %s", exc, exc_info=True)


async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            "Please specify a trading pair.\n\n"
            "<b>Example:</b> <code>/price BTC/USDT</code>",
            parse_mode="HTML",
        )
        return

    pair = context.args[0].upper()
    if "/" not in pair:
        await update.message.reply_text(
            "Invalid pair format. Please use format: <code>BASE/QUOTE</code>\n\n"
            "<b>Example:</b> <code>/price BTC/USDT</code>",
            parse_mode="HTML",
        )
        return

    processing_msg = await update.message.reply_text(
        f"Fetching {pair} prices from data sources...",
        parse_mode="HTML",
    )

    try:
        arbitrage_result, error = scan_arbitrage(pair)
        if error:
            error_message = f"<b>Error:</b> {error}"
            await processing_msg.edit_text(error_message, parse_mode="HTML")
            logger.error("Error in price_command: %s", error)
            return

        message = format_arbitrage_message(pair, arbitrage_result["prices"], arbitrage_result)
        await processing_msg.edit_text(message, parse_mode="HTML")
    except Exception as exc:
        error_message = f"<b>Error:</b> {exc}"
        await processing_msg.edit_text(error_message, parse_mode="HTML")
        logger.error("Error in price_command: %s", exc, exc_info=True)


async def polyarb_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    processing_msg = await update.message.reply_text(
        "Checking Polymarket for arbitrage...",
        parse_mode="HTML",
    )
    try:
        results = find_polymarket_arbitrage()
        if not results:
            await processing_msg.edit_text("No Polymarket arbitrage found right now.", parse_mode="HTML")
            return
        out_lines = ["<b>Polymarket Arbitrage</b>\n"]
        for r in results[:5]:
            q = r.get("question", "")
            yes = r.get("yes_price", 0.0)
            no = r.get("no_price", 0.0)
            total = r.get("total", 0.0)
            profit = r.get("profit_pct", 0.0)
            out_lines.append(
                f"Question: {q}\nYES: {yes:.2f} NO: {no:.2f} Total: {total:.2f} Profit: {profit:.2f}%\n"
            )
        message = "\n".join(out_lines)
        await processing_msg.edit_text(message, parse_mode="HTML")
    except Exception as exc:
        await processing_msg.edit_text(f"<b>Error:</b> {exc}", parse_mode="HTML")
        logger.error("Error in polyarb_command: %s", exc, exc_info=True)


async def crossarb_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    processing_msg = await update.message.reply_text(
        "Checking cross-market arbitrage (Polymarket &lt;-&gt; Kalshi)...",
        parse_mode="HTML",
    )
    try:
        min_sim = 0.4
        if context.args:
            try:
                min_sim = float(context.args[0])
            except Exception:
                pass
        results = find_cross_market_arbitrage(limit_pol=100, limit_kal=100, min_similarity=min_sim)
        if not results:
            await processing_msg.edit_text("No cross-market arbitrage found right now.", parse_mode="HTML")
            return
        out_lines = ["<b>Cross-market Arbitrage (Polymarket &lt;-&gt; Kalshi)</b>\n"]
        for r in results[:5]:
            out_lines.append(
                f"Type: {r['type']}\n"
                f"Polymarket: {r['pol_question']}\n"
                f"Kalshi: {r['kal_question']}\n"
                f"Pol YES: {r['pol_yes']:.2f} Pol NO: {r['pol_no']:.2f}\n"
                f"Kal YES: {r['kal_yes']:.2f} Kal NO: {r['kal_no']:.2f}\n"
                f"Total: {r['total']:.2f} Profit: {r['profit_pct']:.2f}%\n\n"
            )
        await processing_msg.edit_text("\n".join(out_lines), parse_mode="HTML")
    except Exception as exc:
        await processing_msg.edit_text(f"<b>Error:</b> {exc}", parse_mode="HTML")
        logger.error("Error in crossarb_command: %s", exc, exc_info=True)


async def alerts_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Enable/disable scheduled alerts for this chat."""
    chat_id = str(update.effective_chat.id)
    enabled_file = DATA_DIR / "alerts_chats.json"
    enabled_chats: Dict[str, bool] = {}
    try:
        if enabled_file.exists():
            enabled_chats = json.loads(enabled_file.read_text(encoding="utf-8"))
    except Exception:
        enabled_chats = {}

    if not context.args:
        enabled = bool(enabled_chats.get(chat_id, False))
        msg = "Alerts are ENABLED for this chat." if enabled else "Alerts are DISABLED for this chat."
        await update.message.reply_text(msg)
        return

    cmd = context.args[0].lower()
    if cmd in ("enable", "on"):
        enabled_chats[chat_id] = True
        enabled_file.write_text(json.dumps(enabled_chats), encoding="utf-8")
        await update.message.reply_text("Alerts enabled for this chat.")
        return
    if cmd in ("disable", "off"):
        enabled_chats[chat_id] = False
        enabled_file.write_text(json.dumps(enabled_chats), encoding="utf-8")
        await update.message.reply_text("Alerts disabled for this chat.")
        return
    if cmd == "status":
        enabled = bool(enabled_chats.get(chat_id, False))
        msg = "Alerts are ENABLED for this chat." if enabled else "Alerts are DISABLED for this chat."
        await update.message.reply_text(msg)
        return
    await update.message.reply_text("Usage: /alerts enable|disable|status")


async def watch_on_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Enable scheduled cross-arb watch for this chat."""
    chat_id = str(update.effective_chat.id)
    interval = context.bot_data.get("watch_interval", 300)
    if context.args:
        try:
            interval = int(context.args[0])
        except Exception:
            interval = context.bot_data.get("watch_interval", 300)

    cooldown_sec = int(context.bot_data.get("alert_cooldown_sec", 30 * 60))
    improve_pct = float(context.bot_data.get("alert_improve_pct", 0.5))

    jq = context.job_queue
    job_name = f"watch_{chat_id}"
    for job in jq.get_jobs_by_name(job_name):
        job.schedule_removal()

    async def _watch_job(context: ContextTypes.DEFAULT_TYPE):
        try:
            opps, debug = find_cross_market_arbitrage(limit_pol=100, limit_kal=100, min_similarity=0.3, collect_debug=True)
            if not opps:
                return
            seen = load_seen()
            now_ts = int(time.time())
            new_out: List[Dict[str, Any]] = []
            for o in opps[:10]:
                combo = o.get("combo", "")
                buy_src = "polymarket" if "pol" in combo else "kalshi"
                sell_src = "kalshi" if buy_src == "polymarket" else "polymarket"
                market_id = f"{o.get('pm_id','')}|{o.get('kal_id','')}|{combo}"
                buy_price = o.get("buy_ask", 0.0)
                sell_price = o.get("sell_bid", 0.0)
                fp = build_fingerprint(buy_src, sell_src, market_id, buy_price, sell_price)
                edge_val = float(o.get("edge_after", 0.0)) * 100.0
                ok, tag, prev_edge = should_alert(seen, fp, edge_val, cooldown_sec, improve_pct, now_ts)
                if not ok:
                    continue
                seen[fp] = {"ts": now_ts, "edge": edge_val}
                o["alert_tag"] = tag
                o["prev_edge"] = prev_edge
                new_out.append(o)
            if new_out:
                lines = ["<b>Scheduled Cross-market Arbitrage</b>\n"]
                for r in new_out[:5]:
                    tag = r.get("alert_tag", "NEW")
                    prev_edge = r.get("prev_edge", 0.0)
                    line = (
                        f"[{tag}] Combo: {r.get('combo','')}\n"
                        f"Polymarket ID: {r.get('pm_id','')}\nKalshi ID: {r.get('kal_id','')}\n"
                        f"Edge: {r.get('edge_after',0.0):.4f}"
                    )
                    if tag == "IMPROVED":
                        line += f" (prev {prev_edge:.2f}bp)"
                    lines.append(line + "\n")
                try:
                    await context.bot.send_message(chat_id=chat_id, text="\n".join(lines), parse_mode="HTML")
                except Exception:
                    pass
                save_seen(seen)
        except Exception:
            return

    jq.run_repeating(_watch_job, interval=interval, first=0, name=job_name)
    await update.message.reply_text(f"Watch enabled every {interval}s.")


async def watch_off_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = str(update.effective_chat.id)
    jq = context.job_queue
    job_name = f"watch_{chat_id}"
    removed = False
    for job in jq.get_jobs_by_name(job_name):
        job.schedule_removal()
        removed = True
    if removed:
        await update.message.reply_text("Watch disabled.")
    else:
        await update.message.reply_text("No watch was active.")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = str(update.effective_chat.id)
    jq = context.job_queue
    job_name = f"watch_{chat_id}"
    jobs = jq.get_jobs_by_name(job_name)
    if jobs:
        await update.message.reply_text("Watch is ON.")
    else:
        await update.message.reply_text("Watch is OFF.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alert-cooldown", type=int, default=int(os.getenv("ALERT_COOLDOWN", "30")), help="Cooldown in minutes for repeating alerts")
    parser.add_argument("--alert-improve", type=float, default=float(os.getenv("ALERT_IMPROVE", "0.005")), help="Required improvement in edge to resend within cooldown (fraction, e.g., 0.005 = 0.5%)")
    parser.add_argument("--watch-interval", type=int, default=int(os.getenv("WATCH_INTERVAL", "300")), help="Watch interval seconds")
    args, _ = parser.parse_known_args()
    alert_cooldown_min = args.alert_cooldown
    alert_improve = args.alert_improve
    watch_interval = args.watch_interval

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        logger.error("TELEGRAM_BOT_TOKEN environment variable not set!")
        print("ERROR: TELEGRAM_BOT_TOKEN environment variable is required.")
        return

    application = Application.builder().token(bot_token).build()
    application.bot_data["alert_cooldown_sec"] = alert_cooldown_min * 60
    application.bot_data["alert_improve_pct"] = alert_improve * 100.0
    application.bot_data["watch_interval"] = watch_interval

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("scan", scan_command))
    application.add_handler(CommandHandler("price", price_command))
    application.add_handler(CommandHandler("polyarb", polyarb_command))
    application.add_handler(CommandHandler("crossarb", crossarb_command))
    application.add_handler(CommandHandler("alerts", alerts_command))
    application.add_handler(CommandHandler("watch_on", watch_on_command))
    application.add_handler(CommandHandler("watch_off", watch_off_command))
    application.add_handler(CommandHandler("status", status_command))

    alert_chat = os.getenv("TELEGRAM_ALERT_CHAT_ID")
    if alert_chat:
        jq = application.job_queue
        cooldown_sec = alert_cooldown_min * 60
        improve_pct = alert_improve * 100.0

        async def _cross_arb_job(context: ContextTypes.DEFAULT_TYPE):
            try:
                results = find_cross_market_arbitrage(limit_pol=100, limit_kal=100, min_similarity=0.3)
                if not results:
                    return
                seen = load_seen()
                new_out = []
                now_ts = int(time.time())
                for r in results[:20]:
                    buy_src = "polymarket" if "pol" in r.get("type", "") else "kalshi"
                    sell_src = "kalshi" if buy_src == "polymarket" else "polymarket"
                    market_id = r.get("pol_question", "") or r.get("kal_question", "")
                    buy_price = r.get("pol_yes") if "yes" in r.get("type", "") else r.get("pol_no")
                    sell_price = r.get("kal_no") if "kal_no" in r.get("type", "") else r.get("kal_yes")
                    fp = build_fingerprint(buy_src, sell_src, market_id, buy_price or 0.0, sell_price or 0.0)
                    edge = float(r.get("profit_pct", 0.0))
                    ok, tag, prev_edge = should_alert(seen, fp, edge, cooldown_sec, improve_pct, now_ts)
                    if not ok:
                        continue
                    seen[fp] = {"ts": now_ts, "edge": edge}
                    r["alert_tag"] = tag
                    r["prev_edge"] = prev_edge
                    new_out.append(r)
                if not new_out:
                    save_seen(seen)
                    return
                out_lines = ["<b>Automated Cross-market Arbitrage Alert</b>\n"]
                for r in new_out[:5]:
                    tag = r.get("alert_tag", "NEW")
                    prev_edge = r.get("prev_edge", 0.0)
                    line = (
                        f"[{tag}] Type: {r['type']}\n"
                        f"Polymarket: {r['pol_question']}\n"
                        f"Kalshi: {r['kal_question']}\n"
                        f"Total: {r['total']:.2f} Profit: {r['profit_pct']:.2f}%"
                    )
                    if tag == "IMPROVED":
                        line += f" (prev {prev_edge:.2f}%)"
                    line += "\n"
                    out_lines.append(line)
                # send alerts
                try:
                    recipients = {str(alert_chat)}
                    enabled_file = DATA_DIR / "alerts_chats.json"
                    enabled_chats = {}
                    try:
                        if enabled_file.exists():
                            enabled_chats = json.loads(enabled_file.read_text(encoding="utf-8"))
                    except Exception:
                        enabled_chats = {}
                    for cid, val in enabled_chats.items():
                        try:
                            if val:
                                recipients.add(str(cid))
                        except Exception:
                            continue
                    for cid in recipients:
                        try:
                            await context.bot.send_message(chat_id=cid, text="\n".join(out_lines), parse_mode="HTML")
                        except Exception:
                            continue
                finally:
                    save_seen(seen)
            except Exception:
                return

        jq.run_repeating(_cross_arb_job, interval=180, first=10)

    logger.info("Bot starting...")
    print("Bot is running. Press Ctrl+C to stop.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
