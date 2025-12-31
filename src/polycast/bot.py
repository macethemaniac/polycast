"""
Telegram bot integration for the arbitrage scanner.

This module provides a Telegram bot interface for the arbitrage scanner,
allowing users to check arbitrage opportunities via Telegram commands.
"""

import os
import logging
import json
import time
from pathlib import Path
from typing import Dict
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.ext import JobQueue
from scanner import scan_arbitrage
from polymarket_api import find_polymarket_arbitrage
from cross_arb import find_cross_market_arbitrage

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def format_arbitrage_message(pair: str, coingecko_price: float, 
                            defillama_price: float, 
                            arbitrage_result: Dict) -> str:
    """
    Format arbitrage results as a Telegram message.
    
    Args:
        pair: Trading pair (e.g., 'BTC/USDT')
        coingecko_price: Price from CoinGecko
        defillama_price: Price from DeFiLlama
        arbitrage_result: Result dictionary from check_arbitrage
        
    Returns:
        Formatted message string
    """
    message = f"🔍 <b>Arbitrage Scanner</b>\n"
    message += f"📊 Pair: <code>{pair}</code>\n\n"
    
    message += f"💰 <b>Current Prices:</b>\n"
    message += f"• CoinGecko: <code>${coingecko_price:,.2f}</code>\n"
    message += f"• DeFiLlama: <code>${defillama_price:,.2f}</code>\n\n"
    
    message += f"📈 <b>Arbitrage Analysis:</b>\n"
    message += f"• Buy on: <b>{arbitrage_result['buy_exchange'].upper()}</b> "
    message += f"@ <code>${arbitrage_result['buy_price']:,.2f}</code>\n"
    message += f"• Sell on: <b>{arbitrage_result['sell_exchange'].upper()}</b> "
    message += f"@ <code>${arbitrage_result['sell_price']:,.2f}</code>\n"
    message += f"• Spread: <code>${arbitrage_result['spread']:,.2f}</code> "
    message += f"(<code>{arbitrage_result['spread_percent']:.4f}%</code>)\n\n"
    
    if arbitrage_result['spread_percent'] > 0:
        message += "✅ <b>Arbitrage opportunity detected!</b>"
    else:
        message += "❌ No arbitrage opportunity (prices are equal)"
    
    return message


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /start command."""
    welcome_message = (
        "👋 <b>Welcome to Arbitrage Scanner Bot!</b>\n\n"
        "I can help you discover arbitrage opportunities and fetch market prices.\n\n"
        "<b>Quick Commands</b>\n"
        "• /scan — Scan BTC/USDT for arbitrage\n"
        "• /price &lt;pair&gt; — Get prices for a pair (e.g., /price BTC/USDT)\n"
        "• /polyarb — Check Polymarket for YES/NO arbitrage\n"
        "• /crossarb [min_similarity] — Cross-market scan (Polymarket ⇄ Kalshi)\n"
        "• /help — Full help and examples"
    )
    await update.message.reply_text(welcome_message, parse_mode='HTML')


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /help command."""
    help_message = (
        "📚 <b>Help - Arbitrage Scanner Bot</b>\n\n"
        "Use these commands to query prices and find arbitrage:\n\n"
        "• <code>/start</code> — Show the welcome message\n"
        "• <code>/scan</code> — Scan BTC/USDT for arbitrage (CoinGecko ⇄ DeFiLlama)\n"
        "• <code>/price &lt;pair&gt;</code> — Get prices for <code>BASE/QUOTE</code>\n"
        "• <code>/polyarb</code> — Detect internal Polymarket binary arbitrage\n"
        "• <code>/crossarb [min_similarity]</code> — Cross-market scan between Polymarket and Kalshi. Optional similarity (0-1).\n\n"
        "Examples:\n"
        "<code>/scan</code>\n"
        "<code>/price BTC/USDT</code>\n"
        "<code>/crossarb 0.25</code> — run cross-arb with similarity 0.25"
    )
    await update.message.reply_text(help_message, parse_mode='HTML')


async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /scan command - scans BTC/USDT for arbitrage."""
    pair = 'BTC/USDT'
    
    # Send a "processing" message
    processing_msg = await update.message.reply_text(
        f"⏳ Fetching {pair} prices from data sources...",
        parse_mode='HTML'
    )
    
    try:
        # Scan for arbitrage opportunity
        arbitrage_result, error = scan_arbitrage(pair)
        
        if error:
            error_message = f"❌ <b>Error:</b> {error}"
            await processing_msg.edit_text(error_message, parse_mode='HTML')
            logger.error(f"Error in scan_command: {error}")
            return
        
        # Format and send the result
        message = format_arbitrage_message(
            pair, 
            arbitrage_result['coingecko_price'], 
            arbitrage_result['defillama_price'], 
            arbitrage_result
        )
        
        await processing_msg.edit_text(message, parse_mode='HTML')
        
    except Exception as e:
        error_message = f"❌ <b>Error:</b> {str(e)}"
        await processing_msg.edit_text(error_message, parse_mode='HTML')
        logger.error(f"Error in scan_command: {str(e)}", exc_info=True)


async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /price command - gets prices for a specified pair."""
    if not context.args:
        await update.message.reply_text(
            "⚠️ Please specify a trading pair.\n\n"
            "<b>Example:</b> <code>/price BTC/USDT</code>",
            parse_mode='HTML'
        )
        return
    
    pair = context.args[0].upper()
    
    # Validate pair format
    if '/' not in pair:
        await update.message.reply_text(
            "⚠️ Invalid pair format. Please use format: <code>BASE/QUOTE</code>\n\n"
            "<b>Example:</b> <code>/price BTC/USDT</code>",
            parse_mode='HTML'
        )
        return
    
    # Send a "processing" message
    processing_msg = await update.message.reply_text(
        f"⏳ Fetching {pair} prices from data sources...",
        parse_mode='HTML'
    )
    
    try:
        # Scan for arbitrage opportunity
        arbitrage_result, error = scan_arbitrage(pair)
        
        if error:
            error_message = f"❌ <b>Error:</b> {error}"
            await processing_msg.edit_text(error_message, parse_mode='HTML')
            logger.error(f"Error in price_command: {error}")
            return
        
        # Format and send the result
        message = format_arbitrage_message(
            pair,
            arbitrage_result['coingecko_price'],
            arbitrage_result['defillama_price'],
            arbitrage_result
        )
        
        await processing_msg.edit_text(message, parse_mode='HTML')
        
    except Exception as e:
        error_message = f"❌ <b>Error:</b> {str(e)}"
        await processing_msg.edit_text(error_message, parse_mode='HTML')
        logger.error(f"Error in price_command: {str(e)}", exc_info=True)


async def polyarb_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /polyarb command - find Polymarket binary arbitrage."""
    processing_msg = await update.message.reply_text(
        "⏳ Checking Polymarket for arbitrage...",
        parse_mode='HTML'
    )

    try:
        results = find_polymarket_arbitrage()

        if not results:
            await processing_msg.edit_text("No Polymarket arbitrage found right now.", parse_mode='HTML')
            return

        # Format top results (limit to 5)
        out_lines = ["🔥 <b>Polymarket Arbitrage</b>\n"]
        for r in results[:5]:
            q = r.get('question', '')
            yes = r.get('yes_price', 0.0)
            no = r.get('no_price', 0.0)
            total = r.get('total', 0.0)
            profit = r.get('profit_pct', 0.0)
            out_lines.append(f"Question: {q}\nYES: {yes:.2f} NO: {no:.2f} Total: {total:.2f} Profit: {profit:.2f}%\n")

        message = "\n".join(out_lines)
        await processing_msg.edit_text(message, parse_mode='HTML')

    except Exception as e:
        await processing_msg.edit_text(f"❌ <b>Error:</b> {str(e)}", parse_mode='HTML')
        logger.error(f"Error in polyarb_command: {str(e)}", exc_info=True)


async def crossarb_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /crossarb command - find cross-market arbitrage Polymarket vs Kalshi."""
    processing_msg = await update.message.reply_text(
        "⏳ Checking cross-market arbitrage (Polymarket <> Kalshi)...",
        parse_mode='HTML'
    )

    try:
        # allow optional args: /crossarb [min_similarity]
        min_sim = 0.4
        if context.args:
            try:
                min_sim = float(context.args[0])
            except Exception:
                pass

        results = find_cross_market_arbitrage(limit_pol=100, limit_kal=100, min_similarity=min_sim)

        if not results:
            await processing_msg.edit_text("No cross-market arbitrage found right now.", parse_mode='HTML')
            return

        out_lines = ["🔗 <b>Cross-market Arbitrage (Polymarket ⇄ Kalshi)</b>\n"]
        for r in results[:5]:
            out_lines.append(
                f"Type: {r['type']}\nPolymarket: {r['pol_question']}\nKalshi: {r['kal_question']}\n"
                f"Pol YES: {r['pol_yes']:.2f} Pol NO: {r['pol_no']:.2f}\nKal YES: {r['kal_yes']:.2f} Kal NO: {r['kal_no']:.2f}\n"
                f"Total: {r['total']:.2f} Profit: {r['profit_pct']:.2f}%\n\n"
            )

        message = "\n".join(out_lines)
        await processing_msg.edit_text(message, parse_mode='HTML')

    except Exception as e:
        await processing_msg.edit_text(f"❌ <b>Error:</b> {str(e)}", parse_mode='HTML')
        logger.error(f"Error in crossarb_command: {str(e)}", exc_info=True)
    except Exception as e:
        error_message = f"❌ <b>Error:</b> {str(e)}"
        await processing_msg.edit_text(error_message, parse_mode='HTML')
        logger.error(f"Error in price_command: {str(e)}", exc_info=True)


async def alerts_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """In-chat command to enable/disable scheduled cross-arb alerts.

    Usage: /alerts enable  OR /alerts disable  OR /alerts status
    """
    chat_id = str(update.effective_chat.id)
    data_dir = Path(__file__).resolve().parents[1] / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    enabled_file = data_dir / 'alerts_chats.json'

    # load
    enabled_chats = {}
    try:
        if enabled_file.exists():
            with enabled_file.open('r', encoding='utf-8') as f:
                enabled_chats = json.load(f)
    except Exception:
        enabled_chats = {}

    if not context.args:
        # show status
        enabled = bool(enabled_chats.get(chat_id, False))
        msg = "Alerts are ENABLED for this chat." if enabled else "Alerts are DISABLED for this chat."
        await update.message.reply_text(msg)
        return

    cmd = context.args[0].lower()
    if cmd in ('enable', 'on'):
        enabled_chats[chat_id] = True
        try:
            with enabled_file.open('w', encoding='utf-8') as f:
                json.dump(enabled_chats, f)
        except Exception:
            pass
        await update.message.reply_text('✅ Alerts enabled for this chat.')
        return

    if cmd in ('disable', 'off'):
        enabled_chats[chat_id] = False
        try:
            with enabled_file.open('w', encoding='utf-8') as f:
                json.dump(enabled_chats, f)
        except Exception:
            pass
        await update.message.reply_text('⛔ Alerts disabled for this chat.')
        return

    if cmd == 'status':
        enabled = bool(enabled_chats.get(chat_id, False))
        msg = "Alerts are ENABLED for this chat." if enabled else "Alerts are DISABLED for this chat."
        await update.message.reply_text(msg)
        return

    await update.message.reply_text('Usage: /alerts enable|disable|status')


def main() -> None:
    """Start the Telegram bot."""
    # Get bot token from environment variable
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    
    if not bot_token:
        logger.error("TELEGRAM_BOT_TOKEN environment variable not set!")
        print("ERROR: TELEGRAM_BOT_TOKEN environment variable is required.")
        print("Set it using: export TELEGRAM_BOT_TOKEN='your_token_here'")
        print("Or create a .env file with: TELEGRAM_BOT_TOKEN=your_token_here")
        return
    
    # Create the Application
    application = Application.builder().token(bot_token).build()
    
    # Register command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("scan", scan_command))
    application.add_handler(CommandHandler("price", price_command))
    application.add_handler(CommandHandler("polyarb", polyarb_command))
    application.add_handler(CommandHandler("crossarb", crossarb_command))
    application.add_handler(CommandHandler("alerts", alerts_command))
    # Schedule periodic cross-arb alerts if chat id set
    alert_chat = os.getenv('TELEGRAM_ALERT_CHAT_ID')
    if alert_chat:
        try:
            jq = application.job_queue

            async def _cross_arb_job(context: ContextTypes.DEFAULT_TYPE):
                try:
                    results = find_cross_market_arbitrage(limit_pol=100, limit_kal=100, min_similarity=0.3)
                    if not results:
                        return
                    out_lines = ["🔔 <b>Automated Cross-market Arbitrage Alert</b>\n"]
                    for r in results[:5]:
                        out_lines.append(
                            f"Type: {r['type']}\nPolymarket: {r['pol_question']}\nKalshi: {r['kal_question']}\n"
                            f"Total: {r['total']:.2f} Profit: {r['profit_pct']:.2f}%\n"
                        )
                    await context.bot.send_message(chat_id=alert_chat, text="\n".join(out_lines), parse_mode='HTML')
                except Exception:
                    return

            # File-backed seen-opportunities to avoid duplicate alerts
            data_dir = Path(__file__).resolve().parents[1] / 'data'
            data_dir.mkdir(parents=True, exist_ok=True)
            seen_file = data_dir / 'seen_opportunities.json'
            dedupe_ttl = int(os.getenv('TELEGRAM_ALERT_DEDUPE_TTL', '3600'))

            def _load_seen() -> Dict[str, Dict[str, int]]:
                try:
                    if seen_file.exists():
                        with seen_file.open('r', encoding='utf-8') as f:
                            return json.load(f)
                except Exception:
                    pass
                return {}

            def _save_seen(d: Dict[str, Dict[str, int]]):
                try:
                    with seen_file.open('w', encoding='utf-8') as f:
                        json.dump(d, f)
                except Exception:
                    pass

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
                        last_seen = int(meta.get('last_seen', 0))
                        count = int(meta.get('count', 0))

                        # If never seen, send and set count=1
                        if count == 0:
                            seen[key] = {'last_seen': now_ts, 'count': 1}
                            new_out.append(r)
                            continue

                        # If seen once, allow exactly one repeat after dedupe_ttl
                        if count == 1 and (now_ts - last_seen) >= dedupe_ttl:
                            seen[key] = {'last_seen': now_ts, 'count': 2}
                            new_out.append(r)
                            continue

                        # If count >=2, skip (already repeated once)
                        continue

                    if not new_out:
                        _save_seen(seen)
                        return

                    out_lines = ["🔔 <b>Automated Cross-market Arbitrage Alert</b>\n"]
                    for r in new_out[:5]:
                        out_lines.append(
                            f"Type: {r['type']}\nPolymarket: {r['pol_question']}\nKalshi: {r['kal_question']}\n"
                            f"Total: {r['total']:.2f} Profit: {r['profit_pct']:.2f}%\n"
                        )

                    # Determine recipients: env var chat and enabled chats file
                    enabled_file = data_dir / 'alerts_chats.json'
                    enabled_chats = {}
                    try:
                        if enabled_file.exists():
                            with enabled_file.open('r', encoding='utf-8') as f:
                                enabled_chats = json.load(f)
                    except Exception:
                        enabled_chats = {}

                    recipients = set()
                    if alert_chat:
                        recipients.add(str(alert_chat))
                    for cid, v in enabled_chats.items():
                        try:
                            if v:
                                recipients.add(str(cid))
                        except Exception:
                            continue

                    for cid in recipients:
                        try:
                            await context.bot.send_message(chat_id=cid, text="\n".join(out_lines), parse_mode='HTML')
                        except Exception:
                            continue

                    # schedule a single reminder in 60 seconds for these new alerts
                    try:
                        # prepare reminder payload
                        reminder_text = "\n".join(["🔔 <b>Reminder:</b>\n"] + out_lines[1:])
                        # keys for which reminders were scheduled
                        scheduled_keys = []
                        for r in new_out:
                            k = f"{r.get('type')}|{r.get('pol_question','')}|{r.get('kal_question','')}|{round(r.get('total',0),4)}"
                            scheduled_keys.append(k)
                            # ensure reminder_sent flag exists
                            meta = seen.get(k, {})
                            meta.setdefault('reminder_sent', False)
                            seen[k] = meta

                        # job data contains recipients and keys and reminder text
                        job_data = {'recipients': list(recipients), 'keys': scheduled_keys, 'text': reminder_text}
                        # run once after 60 seconds
                        jq.run_once(_reminder_job, when=60, data=job_data)
                    except Exception:
                        pass

                    _save_seen(seen)
                except Exception:
                    return

            # run every 3 minutes
            jq.run_repeating(_cross_arb_job, interval=180, first=10)
        except Exception:
            logger.exception('Failed to schedule cross-arb job')

            async def _reminder_job(context: ContextTypes.DEFAULT_TYPE):
                """Job to send a one-time reminder for previously sent alerts."""
                try:
                    data = context.job.data or {}
                    recipients = data.get('recipients', [])
                    keys = data.get('keys', [])
                    text = data.get('text', '')

                    # load and update seen metadata
                    seen = _load_seen()
                    now_ts = int(time.time())

                    for k in keys:
                        meta = seen.get(k, {})
                        # send reminder only if not already sent
                        if meta.get('reminder_sent'):
                            continue
                        for cid in recipients:
                            try:
                                await context.bot.send_message(chat_id=cid, text=text, parse_mode='HTML')
                            except Exception:
                                continue
                        # mark reminder as sent and increment count to 2
                        meta['reminder_sent'] = True
                        meta['last_seen'] = now_ts
                        meta['count'] = int(meta.get('count', 1)) + 1
                        seen[k] = meta

                    _save_seen(seen)
                except Exception:
                    return
    
    
    # Start the bot
    logger.info("Bot starting...")
    print("Bot is running. Press Ctrl+C to stop.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()

