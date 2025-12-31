"""
Telegram bot integration for the arbitrage scanner.

Provides Telegram commands to scan spot exchanges via CCXT and check
Polymarket/Kalshi arbitrage.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict

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


def format_arbitrage_message(pair: str, prices: Dict[str, float], arbitrage_result: Dict) -> str:
    """Build a human-readable arbitrage message (ASCII-safe)."""
    lines = [
        "<b>Arbitrage Scanner</b>",
        f"Pair: <code>{pair}</code>",
        "",
        "<b>Current Prices:</b>",
    ]
    for exchange, price in prices.items():
        lines.append(f"- {exchange}: <code>${price:,.2f}</code>")

    lines.append("")
    lines.append("<b>Arbitrage Analysis:</b>")
    lines.append(f"- Buy on: <b>{arbitrage_result['buy_exchange'].upper()}</b> @ <code>${arbitrage_result['buy_price']:,.2f}</code>")
    lines.append(f"- Sell on: <b>{arbitrage_result['sell_exchange'].upper()}</b> @ <code>${arbitrage_result['sell_price']:,.2f}</code>")
    lines.append(f"- Spread: <code>${arbitrage_result['spread']:,.2f}</code> (<code>{arbitrage_result['spread_percent']:.4f}%</code>)")
    lines.append("")
    if arbitrage_result["spread_percent"] > 0:
        lines.append("<b>Arbitrage opportunity detected!</b>")
    else:
        lines.append("No arbitrage opportunity (prices are equal)")
    return "\n".join(lines)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /start command."""
    welcome_message = (
        "<b>Welcome to Arbitrage Scanner Bot!</b>\n\n"
        "I can help you discover arbitrage opportunities and fetch market prices.\n\n"
        "<b>Quick Commands</b>\n"
        "- /scan [pair] - Scan a pair (default BTC/USDT) via CCXT\n"
        "- /price <pair> - Get prices for any pair via CCXT\n"
        "- /polyarb - Check Polymarket for YES/NO arbitrage\n"
        "- /crossarb [min_similarity] - Cross-market scan (Polymarket <-> Kalshi)\n"
        "- /help - Full help and examples"
    )
    await update.message.reply_text(welcome_message, parse_mode="HTML")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /help command."""
    help_message = (
        "<b>Help - Arbitrage Scanner Bot</b>\n\n"
        "Use these commands to query prices and find arbitrage:\n\n"
        "- <code>/start</code> - Show the welcome message\n"
        "- <code>/scan [pair]</code> - Scan a pair (default BTC/USDT) via CCXT\n"
        "- <code>/price <pair></code> - Get prices for any <code>BASE/QUOTE</code>\n"
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
    """Handle the /scan command - scans a pair and cross-market opportunities."""
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
            spot_message = format_arbitrage_message(
                pair,
                arbitrage_result["prices"],
                arbitrage_result,
            )

        cross_lines = ["<b>Cross-market Arbitrage (Polymarket <-> Kalshi)</b>\n"]
        try:
            cross_results = find_cross_market_arbitrage(
                limit_pol=100,
                limit_kal=100,
                min_similarity=min_similarity,
            )
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
    """Handle the /price command - gets prices for a specified pair."""
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

        message = format_arbitrage_message(
            pair,
            arbitrage_result["prices"],
            arbitrage_result,
        )

        await processing_msg.edit_text(message, parse_mode="HTML")

    except Exception as exc:
        error_message = f"<b>Error:</b> {exc}"
        await processing_msg.edit_text(error_message, parse_mode="HTML")
        logger.error("Error in price_command: %s", exc, exc_info=True)


async def polyarb_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /polyarb command - find Polymarket binary arbitrage."""
    processing_msg = await update.message.reply_text(
        "Checking Polymarket for arbitrage...",
        parse_mode="HTML",
    )

    try:
        results = find_polymarket_arbitrage()

        if not results:
            await processing_msg.edit_text(
                "No Polymarket arbitrage found right now.",
                parse_mode="HTML",
            )
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
    """Handle the /crossarb command - find cross-market arbitrage Polymarket vs Kalshi."""
    processing_msg = await update.message.reply_text(
        "Checking cross-market arbitrage (Polymarket <-> Kalshi)...",
        parse_mode="HTML",
    )

    try:
        min_sim = 0.4
        if context.args:
            try:
                min_sim = float(context.args[0])
            except Exception:
                pass

        results = find_cross_market_arbitrage(
            limit_pol=100,
            limit_kal=100,
            min_similarity=min_sim,
        )

        if not results:
            await processing_msg.edit_text(
                "No cross-market arbitrage found right now.",
                parse_mode="HTML",
            )
            return

        out_lines = ["<b>Cross-market Arbitrage (Polymarket <-> Kalshi)</b>\n"]
        for r in results[:5]:
            out_lines.append(
                f"Type: {r['type']}\n"
                f"Polymarket: {r['pol_question']}\n"
                f"Kalshi: {r['kal_question']}\n"
                f"Pol YES: {r['pol_yes']:.2f} Pol NO: {r['pol_no']:.2f}\n"
                f"Kal YES: {r['kal_yes']:.2f} Kal NO: {r['kal_no']:.2f}\n"
                f"Total: {r['total']:.2f} Profit: {r['profit_pct']:.2f}%\n\n"
            )

        message = "\n".join(out_lines)
        await processing_msg.edit_text(message, parse_mode="HTML")

    except Exception as exc:
        await processing_msg.edit_text(f"<b>Error:</b> {exc}", parse_mode="HTML")
        logger.error("Error in crossarb_command: %s", exc, exc_info=True)


async def alerts_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """In-chat command to enable/disable scheduled cross-arb alerts."""
    chat_id = str(update.effective_chat.id)
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    enabled_file = data_dir / "alerts_chats.json"

    enabled_chats = {}
    try:
        if enabled_file.exists():
            with enabled_file.open("r", encoding="utf-8") as f:
                enabled_chats = json.load(f)
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
        try:
            with enabled_file.open("w", encoding="utf-8") as f:
                json.dump(enabled_chats, f)
        except Exception:
            pass
        await update.message.reply_text("Alerts enabled for this chat.")
        return

    if cmd in ("disable", "off"):
        enabled_chats[chat_id] = False
        try:
            with enabled_file.open("w", encoding="utf-8") as f:
                json.dump(enabled_chats, f)
        except Exception:
            pass
        await update.message.reply_text("Alerts disabled for this chat.")
        return

    if cmd == "status":
        enabled = bool(enabled_chats.get(chat_id, False))
        msg = "Alerts are ENABLED for this chat." if enabled else "Alerts are DISABLED for this chat."
        await update.message.reply_text(msg)
        return

    await update.message.reply_text("Usage: /alerts enable|disable|status")


def main() -> None:
    """Start the Telegram bot."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not bot_token:
        logger.error("TELEGRAM_BOT_TOKEN environment variable not set!")
        print("ERROR: TELEGRAM_BOT_TOKEN environment variable is required.")
        print("Set it using: export TELEGRAM_BOT_TOKEN='your_token_here'")
        print("Or create a .env file with: TELEGRAM_BOT_TOKEN=your_token_here")
        return

    application = Application.builder().token(bot_token).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("scan", scan_command))
    application.add_handler(CommandHandler("price", price_command))
    application.add_handler(CommandHandler("polyarb", polyarb_command))
    application.add_handler(CommandHandler("crossarb", crossarb_command))
    application.add_handler(CommandHandler("alerts", alerts_command))

    alert_chat = os.getenv("TELEGRAM_ALERT_CHAT_ID")
    if alert_chat:
        try:
            jq = application.job_queue

            data_dir = Path(__file__).resolve().parents[1] / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            seen_file = data_dir / "seen_opportunities.json"
            dedupe_ttl = int(os.getenv("TELEGRAM_ALERT_DEDUPE_TTL", "3600"))

            def _load_seen() -> Dict[str, Dict[str, int]]:
                try:
                    if seen_file.exists():
                        with seen_file.open("r", encoding="utf-8") as f:
                            return json.load(f)
                except Exception:
                    pass
                return {}

            def _save_seen(d: Dict[str, Dict[str, int]]):
                try:
                    with seen_file.open("w", encoding="utf-8") as f:
                        json.dump(d, f)
                except Exception:
                    pass

            async def _reminder_job(context: ContextTypes.DEFAULT_TYPE):
                try:
                    data = context.job.data or {}
                    recipients = data.get("recipients", [])
                    keys = data.get("keys", [])
                    text = data.get("text", "")

                    seen = _load_seen()
                    now_ts = int(time.time())

                    for key in keys:
                        meta = seen.get(key, {})
                        if meta.get("reminder_sent"):
                            continue
                        for cid in recipients:
                            try:
                                await context.bot.send_message(chat_id=cid, text=text, parse_mode="HTML")
                            except Exception:
                                continue
                        meta["reminder_sent"] = True
                        meta["last_seen"] = now_ts
                        meta["count"] = int(meta.get("count", 1)) + 1
                        seen[key] = meta

                    _save_seen(seen)
                except Exception:
                    return

            async def _cross_arb_job(context: ContextTypes.DEFAULT_TYPE):
                try:
                    results = find_cross_market_arbitrage(limit_pol=100, limit_kal=100, min_similarity=0.3)
                    if not results:
                        return

                    seen = _load_seen()
                    new_out = []
                    now_ts = int(time.time())

                    for r in results[:20]:
                        key = f"{r.get('type')}|{r.get('pol_question','')}|{r.get('kal_question','')}|{round(r.get('total',0),4)}"
                        meta = seen.get(key, {})
                        last_seen = int(meta.get("last_seen", 0))
                        count = int(meta.get("count", 0))

                        if count == 0:
                            seen[key] = {"last_seen": now_ts, "count": 1}
                            new_out.append(r)
                            continue

                        if count == 1 and (now_ts - last_seen) >= dedupe_ttl:
                            seen[key] = {"last_seen": now_ts, "count": 2}
                            new_out.append(r)
                            continue

                        continue

                    if not new_out:
                        _save_seen(seen)
                        return

                    out_lines = ["<b>Automated Cross-market Arbitrage Alert</b>\n"]
                    for r in new_out[:5]:
                        out_lines.append(
                            f"Type: {r['type']}\n"
                            f"Polymarket: {r['pol_question']}\n"
                            f"Kalshi: {r['kal_question']}\n"
                            f"Total: {r['total']:.2f} Profit: {r['profit_pct']:.2f}%\n"
                        )

                    enabled_file = data_dir / "alerts_chats.json"
                    enabled_chats = {}
                    try:
                        if enabled_file.exists():
                            with enabled_file.open("r", encoding="utf-8") as f:
                                enabled_chats = json.load(f)
                    except Exception:
                        enabled_chats = {}

                    recipients = set()
                    recipients.add(str(alert_chat))
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

                    try:
                        reminder_text = "\n".join(["<b>Reminder:</b>\n"] + out_lines[1:])
                        scheduled_keys = []
                        for r in new_out:
                            k = f"{r.get('type')}|{r.get('pol_question','')}|{r.get('kal_question','')}|{round(r.get('total',0),4)}"
                            scheduled_keys.append(k)
                            meta = seen.get(k, {})
                            meta.setdefault("reminder_sent", False)
                            seen[k] = meta

                        job_data = {"recipients": list(recipients), "keys": scheduled_keys, "text": reminder_text}
                        jq.run_once(_reminder_job, when=60, data=job_data)
                    except Exception:
                        pass

                    _save_seen(seen)
                except Exception:
                    return

            jq.run_repeating(_cross_arb_job, interval=180, first=10)
        except Exception:
            logger.exception("Failed to schedule cross-arb job")

    logger.info("Bot starting...")
    print("Bot is running. Press Ctrl+C to stop.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
