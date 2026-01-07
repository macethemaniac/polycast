"""
Telegram bot integration for the arbitrage scanner.

Provides Telegram commands to scan spot exchanges via CCXT and check
Polymarket/Kalshi arbitrage, with optional scheduled watch and deduped alerts.
"""

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple, List

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
)

from polycast.scanner import (
    scan_arbitrage,
    scan_polymarket_raw,
    scan_polymarket_ml,
    scan_polymarket_trending,
    scan_polymarket_clusters,
    scan_cross_market_mismatches,
    scan_social_trending,
    scan_best_opportunities,
    scan_market_details,
    scan_trending_news,
)
from src.engines.market_search import format_market_analysis
from src.engines.watchlist import (
    add_to_watchlist,
    remove_from_watchlist,
    list_watchlist,
    scan_watchlist,
)
from src.opportunity_logger import log_opportunities
from src.ml.label_store import save_label, load_labels
from polycast.cross_arb import find_cross_market_arbitrage

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SEEN_OPPS_FILE = DATA_DIR / "seen_opps.json"
XARB_ALERTS_FILE = DATA_DIR / "xarb_alerts.json"
XARB_SEEN_FILE = DATA_DIR / "xarb_seen.json"
LABELS_CACHE_FILE = DATA_DIR / "labels_cache.json"
PRICE_ALERTS_FILE = DATA_DIR / "price_alerts.json"
DIGEST_SETTINGS_FILE = DATA_DIR / "digest_settings.json"


def load_price_alerts() -> Dict[str, List[Dict]]:
    """Load price alerts. Structure: {chat_id: [{query, target_price, direction, created}]}"""
    try:
        if not PRICE_ALERTS_FILE.exists():
            return {}
        return json.loads(PRICE_ALERTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_price_alerts(alerts: Dict[str, List[Dict]]) -> None:
    try:
        PRICE_ALERTS_FILE.write_text(json.dumps(alerts, indent=2), encoding="utf-8")
    except Exception:
        pass


def load_digest_settings() -> Dict[str, Dict]:
    """Load daily digest settings. Structure: {chat_id: {enabled, hour}}"""
    try:
        if not DIGEST_SETTINGS_FILE.exists():
            return {}
        return json.loads(DIGEST_SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_digest_settings(settings: Dict[str, Dict]) -> None:
    try:
        DIGEST_SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except Exception:
        pass


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


def load_xarb_seen(prune_older_sec: int = 6 * 3600) -> Dict[str, Dict[str, float]]:
    """Load seen cross-arb alerts."""
    try:
        if not XARB_SEEN_FILE.exists():
            return {}
        data = json.loads(XARB_SEEN_FILE.read_text(encoding="utf-8"))
        now_ts = time.time()
        pruned = {k: v for k, v in data.items() if now_ts - float(v.get("ts", 0)) <= prune_older_sec}
        if len(pruned) != len(data):
            save_xarb_seen(pruned)
        return pruned
    except Exception:
        return {}


def save_xarb_seen(d: Dict[str, Dict[str, float]]) -> None:
    try:
        XARB_SEEN_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except Exception:
        pass


def load_xarb_alert_flag() -> bool:
    try:
        if not XARB_ALERTS_FILE.exists():
            return False
        data = json.loads(XARB_ALERTS_FILE.read_text(encoding="utf-8"))
        return bool(data.get("enabled"))
    except Exception:
        return False


def save_xarb_alert_flag(enabled: bool) -> None:
    try:
        XARB_ALERTS_FILE.write_text(json.dumps({"enabled": enabled}, indent=2), encoding="utf-8")
    except Exception:
        pass


def build_fingerprint(buy_src: str, sell_src: str, market_id: str, buy_price: float, sell_price: float) -> str:
    """Stable fingerprint for an opportunity."""
    rounded_buy = round(float(buy_price), 4)
    rounded_sell = round(float(sell_price), 4)
    payload = f"{buy_src}|{sell_src}|{market_id}|{rounded_buy}|{rounded_sell}"
    return str(hash(payload))


def _is_admin(chat_id: str) -> bool:
    admin_id = os.getenv("TELEGRAM_ADMIN_CHAT_ID") or os.getenv("TELEGRAM_ALERT_CHAT_ID")
    return admin_id and str(chat_id) == str(admin_id)


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
        "<b>Welcome to PolyCAS Trading Assistant!</b>\n\n"
        "Find high-confidence prediction market opportunities.\n\n"
        "<b>Main Commands</b>\n"
        "/discover [category] - Best opportunities\n"
        "/market &lt;query&gt; - Deep-dive analysis\n"
        "/alert &lt;query&gt; &lt;price&gt; - Price alert\n"
        "/alerts - Toggle notifications\n"
        "/digest - Daily summary\n\n"
        "<b>Categories</b>\n"
        "politics | crypto | sports | entertainment\n\n"
        "/help - Full help and examples"
    )
    await update.message.reply_text(welcome_message, parse_mode="HTML")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_message = (
        "<b>Help - PolyCAS Trading Assistant</b>\n\n"
        "<b>Main Commands</b>\n"
        "- <code>/discover [category]</code> - Find best opportunities (fast)\n"
        "- <code>/trending</code> - Markets with active news signals (slower)\n"
        "- <code>/market &lt;query&gt;</code> - Deep-dive into any market\n"
        "- <code>/alerts</code> - Manage notifications (on/off)\n"
        "- <code>/alert &lt;query&gt; &lt;price&gt;</code> - Set price alert\n"
        "- <code>/digest</code> - Daily summary (on/off)\n\n"
        "<b>Categories for /discover</b>\n"
        "politics, crypto, sports, entertainment, all (default)\n\n"
        "<b>Quick Scans</b>\n"
        "- <code>/scan [pair]</code> - Spot crypto arbitrage\n"
        "- <code>/polyarb</code> - YES/NO mispricing\n"
        "- <code>/xarb</code> - Cross-market mismatches\n\n"
        "<b>Examples</b>\n"
        "<code>/discover</code> - All categories\n"
        "<code>/discover crypto</code> - Crypto only\n"
        "<code>/trending</code> - News-driven markets\n"
        "<code>/market bitcoin</code> - Search for Bitcoin\n"
        "<code>/market politics</code> - Browse politics markets\n"
        "<code>/alert trump 0.60</code> - Price alert\n"
        "<code>/digest on</code> - Daily summary at 9 AM UTC\n\n"
        "<b>Categories</b>\n"
        "politics, crypto, sports, entertainment, tech, economy, world, climate"
    )
    await update.message.reply_text(help_message, parse_mode="HTML")


CATEGORY_KEYWORDS = {
    "politics": ["election", "president", "trump", "biden", "vote", "congress", "senate", "governor", "political", "government", "democrat", "republican", "poll"],
    "crypto": ["bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain", "token", "coin", "defi", "nft", "solana", "ripple", "xrp"],
    "sports": ["nfl", "nba", "mlb", "football", "basketball", "soccer", "tennis", "golf", "super bowl", "world cup", "championship", "playoff", "game", "match"],
    "entertainment": ["movie", "oscar", "grammy", "emmy", "album", "celebrity", "music", "film", "show", "tv", "netflix", "disney", "taylor swift"],
}


def _filter_by_category(markets: list, category: str) -> list:
    """Filter markets by category keywords."""
    if not category or category == "all":
        return markets

    keywords = CATEGORY_KEYWORDS.get(category.lower(), [])
    if not keywords:
        return markets

    filtered = []
    for m in markets:
        question = m.get("question", "").lower()
        if any(kw in question for kw in keywords):
            filtered.append(m)
    return filtered


async def discover_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Find best opportunities using unified confidence scoring."""
    # Check for category filter
    category = None
    if context.args:
        cat_arg = context.args[0].lower()
        if cat_arg in CATEGORY_KEYWORDS or cat_arg == "all":
            category = cat_arg

    cat_label = f" ({category})" if category and category != "all" else ""
    processing_msg = await update.message.reply_text(
        f"Scanning for best opportunities{cat_label}...",
        parse_mode="HTML",
    )
    try:
        # Fetch more results when filtering by category
        fetch_count = 100 if category and category != "all" else 20
        results = scan_best_opportunities(limit=200, top_n=fetch_count, min_confidence=35)
        logger.info(f"discover_command: category={category}, total_results={len(results)}")

        # Apply category filter
        if category and category != "all":
            before_filter = len(results)
            results = _filter_by_category(results, category)
            logger.info(f"discover_command: filtered {before_filter} -> {len(results)} for category={category}")
            if not results:
                # If no matches, show message instead of empty results
                await processing_msg.edit_text(
                    f"No markets found in <b>{category}</b> category.\n\n"
                    f"Try: /discover (all categories)",
                    parse_mode="HTML"
                )
                return
            results = results[:5]
        else:
            results = results[:5]
        if not results:
            await processing_msg.edit_text(
                "No high-confidence opportunities found right now.\n"
                "<i>Try again later or lower your standards.</i>",
                parse_mode="HTML"
            )
            return

        def _esc(text: str) -> str:
            return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        cat_header = f" ({category.upper()})" if category and category != "all" else ""
        out_lines = [
            f"<b>BEST OPPORTUNITIES{cat_header}</b>",
            "<i>Unified Confidence Scoring</i>",
            "━━━━━━━━━━━━━━━━━━━━━━"
        ]

        for i, r in enumerate(results, 1):
            q = _esc(r.get("question", ""))
            if len(q) > 100:
                q = q[:100] + "..."
            yes = r.get("yes_price", 0.0)
            no = r.get("no_price", 0.0)
            vol = r.get("volume", 0.0)
            confidence = r.get("confidence", 0.0)
            action = r.get("action", "WATCH")
            action_icon = r.get("action_icon", "[~]")
            signals = r.get("signals", [])

            # Format volume
            if vol >= 1_000_000:
                vol_str = f"${vol/1_000_000:.1f}M"
            elif vol >= 1_000:
                vol_str = f"${vol/1_000:.0f}K"
            else:
                vol_str = f"${vol:.0f}"

            # Confidence label
            if confidence >= 80:
                conf_label = "STRONG"
            elif confidence >= 60:
                conf_label = "GOOD"
            else:
                conf_label = ""

            # Format signals
            signal_text = ", ".join([s.get("name", "")[:30] for s in signals[:2]]) if signals else ""

            # Format price change
            change_24h = r.get("change_24h")
            change_str = ""
            if change_24h is not None:
                pct = change_24h * 100
                if pct >= 0:
                    change_str = f" | 24h: +{pct:.1f}%"
                else:
                    change_str = f" | 24h: {pct:.1f}%"

            out_lines.append(f"\n<b>#{i} {action_icon} {conf_label} {action}</b> ({confidence:.0f}/100)")
            out_lines.append(f"<b>Q:</b> {q}")
            out_lines.append(f"YES ${yes:.2f} | NO ${no:.2f} | Vol: {vol_str}{change_str}")
            if signal_text:
                out_lines.append(f"<i>Signals: {signal_text}</i>")

        out_lines.append("\n━━━━━━━━━━━━━━━━━━━━━━")

        # Build grouped keyboard - two rows per market with emojis
        number_emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
        keyboard = []
        for i, r in enumerate(results):
            market_id = r.get("market_id", "")[:20]
            market_url = r.get("market_url", "")
            num_emoji = number_emojis[i] if i < len(number_emojis) else f"#{i+1}"

            # Row 1: View Market button (if URL available)
            if market_url:
                keyboard.append([
                    InlineKeyboardButton(f"{num_emoji} View Market", url=market_url)
                ])
            # Row 2: Details and Alert buttons
            keyboard.append([
                InlineKeyboardButton("📊 Details", callback_data=f"details:{market_id}"),
                InlineKeyboardButton("🔔 Set Alert", callback_data=f"alert:{market_id}"),
            ])
        keyboard.append([
            InlineKeyboardButton("🔄 Refresh", callback_data="refresh:discover"),
        ])
        reply_markup = InlineKeyboardMarkup(keyboard)

        await processing_msg.edit_text("\n".join(out_lines), parse_mode="HTML", reply_markup=reply_markup)
    except Exception as exc:
        await processing_msg.edit_text(f"<b>Error:</b> {exc}", parse_mode="HTML")
        logger.error("Error in discover_command: %s", exc, exc_info=True)


async def trending_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Find markets with active news signals - uses full scoring with news API."""
    processing_msg = await update.message.reply_text(
        "Scanning for markets with trending news (this may take ~30s)...",
        parse_mode="HTML",
    )
    try:
        results = scan_trending_news(limit=100, top_n=10)

        if not results:
            await processing_msg.edit_text(
                "No markets with significant news signals found right now.\n"
                "<i>This scan checks GDELT for actual news mentions.</i>",
                parse_mode="HTML"
            )
            return

        def _esc(text: str) -> str:
            return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        out_lines = [
            "<b>📰 TRENDING NEWS MARKETS</b>",
            "<i>Markets with recent news coverage</i>",
            "━━━━━━━━━━━━━━━━━━━━━━"
        ]

        for i, r in enumerate(results[:5], 1):
            q = _esc(r.get("question", ""))
            if len(q) > 100:
                q = q[:100] + "..."
            yes = r.get("yes_price", 0.0)
            no = r.get("no_price", 0.0)
            vol = r.get("volume", 0.0)
            confidence = r.get("confidence", 0.0)
            action = r.get("action", "WATCH")
            action_icon = r.get("action_icon", "[~]")
            news_mentions = r.get("news_mentions", 0)
            sentiment = r.get("sentiment", 0.0)
            headlines = r.get("headlines", [])

            # Format volume
            if vol >= 1_000_000:
                vol_str = f"${vol/1_000_000:.1f}M"
            elif vol >= 1_000:
                vol_str = f"${vol/1_000:.0f}K"
            else:
                vol_str = f"${vol:.0f}"

            # Sentiment indicator
            if sentiment > 0.2:
                sent_str = "📈 Positive"
            elif sentiment < -0.2:
                sent_str = "📉 Negative"
            else:
                sent_str = "➡️ Neutral"

            out_lines.append(f"\n<b>#{i} {action_icon} {action}</b> ({confidence:.0f}/100)")
            out_lines.append(f"<b>Q:</b> {q}")
            out_lines.append(f"YES ${yes:.2f} | NO ${no:.2f} | Vol: {vol_str}")
            out_lines.append(f"📰 {news_mentions} news mentions | {sent_str}")
            if headlines:
                headline = _esc(headlines[0][:80])
                out_lines.append(f"<i>» {headline}...</i>")

        out_lines.append("\n━━━━━━━━━━━━━━━━━━━━━━")
        out_lines.append("<i>Use /discover for fast scan (no news API)</i>")

        # Build keyboard
        number_emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
        keyboard = []
        for i, r in enumerate(results[:5]):
            market_id = r.get("market_id", "")[:20]
            market_url = r.get("market_url", "")
            num_emoji = number_emojis[i] if i < len(number_emojis) else f"#{i+1}"

            if market_url:
                keyboard.append([
                    InlineKeyboardButton(f"{num_emoji} View Market", url=market_url)
                ])
            keyboard.append([
                InlineKeyboardButton("📊 Details", callback_data=f"details:{market_id}"),
                InlineKeyboardButton("🔔 Set Alert", callback_data=f"alert:{market_id}"),
            ])
        keyboard.append([
            InlineKeyboardButton("🔄 Refresh", callback_data="refresh:trending"),
        ])
        reply_markup = InlineKeyboardMarkup(keyboard)

        await processing_msg.edit_text("\n".join(out_lines), parse_mode="HTML", reply_markup=reply_markup)
    except Exception as exc:
        await processing_msg.edit_text(f"<b>Error:</b> {exc}", parse_mode="HTML")
        logger.error("Error in trending_command: %s", exc, exc_info=True)


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button presses from /discover and /market commands."""
    query = update.callback_query
    await query.answer()  # Acknowledge the callback

    data = query.data
    if not data:
        return

    try:
        action, payload = data.split(":", 1)
    except ValueError:
        return

    if action == "details":
        # Show market details
        market_id = payload
        from src.engines.market_search import get_market_by_id, get_market_details

        market = get_market_by_id(market_id)
        if not market:
            await query.edit_message_text(
                f"Market not found: {market_id}\n\nTry /market &lt;keyword&gt; to search.",
                parse_mode="HTML"
            )
            return

        details = get_market_details(market)
        analysis_text = format_market_analysis(details)

        # Add buttons with emojis
        market_url = details.get("market_url", "")
        keyboard = []
        if market_url:
            keyboard.append([InlineKeyboardButton("🔗 View on Polymarket", url=market_url)])
        keyboard.append([
            InlineKeyboardButton("🔔 Set Alert", callback_data=f"alert:{market_id}"),
            InlineKeyboardButton("⬅️ Back", callback_data="refresh:discover"),
        ])
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(analysis_text, parse_mode="HTML", reply_markup=reply_markup)

    elif action == "alert":
        # Create a price alert for the market
        market_id = payload
        from src.engines.market_search import get_market_by_id

        market = get_market_by_id(market_id)
        if not market:
            await query.edit_message_text(
                f"Market not found: {market_id}",
                parse_mode="HTML"
            )
            return

        question = market.get("question", "Unknown")[:50]
        yes_price = float(market.get("yes_price", 0))

        # Create alert at +10% movement
        target_price = min(0.99, yes_price + 0.10)
        chat_id = str(query.message.chat_id)

        alerts = load_price_alerts()
        if chat_id not in alerts:
            alerts[chat_id] = []

        alerts[chat_id].append({
            "query": question[:30],
            "market_id": market_id,
            "target_price": target_price,
            "direction": "above",
            "baseline_price": yes_price,
            "created": time.time(),
        })
        save_price_alerts(alerts)

        await query.answer(f"Alert set: YES price crosses ${target_price:.2f}", show_alert=True)

    elif action == "refresh":
        # Refresh discover results
        if payload == "discover":
            await query.edit_message_text("Refreshing opportunities...", parse_mode="HTML")

            results = scan_best_opportunities(limit=200, top_n=5, min_confidence=35, use_cache=False)

            if not results:
                await query.edit_message_text(
                    "No high-confidence opportunities found.\n<i>Try again later.</i>",
                    parse_mode="HTML"
                )
                return

            def _esc(text: str) -> str:
                return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            out_lines = [
                "<b>BEST OPPORTUNITIES</b>",
                "<i>Unified Confidence Scoring</i>",
                "━━━━━━━━━━━━━━━━━━━━━━"
            ]

            for i, r in enumerate(results, 1):
                q = _esc(r.get("question", ""))
                if len(q) > 100:
                    q = q[:100] + "..."
                yes = r.get("yes_price", 0.0)
                no = r.get("no_price", 0.0)
                vol = r.get("volume", 0.0)
                confidence = r.get("confidence", 0.0)
                action_str = r.get("action", "WATCH")
                action_icon = r.get("action_icon", "[~]")
                signals = r.get("signals", [])

                if vol >= 1_000_000:
                    vol_str = f"${vol/1_000_000:.1f}M"
                elif vol >= 1_000:
                    vol_str = f"${vol/1_000:.0f}K"
                else:
                    vol_str = f"${vol:.0f}"

                if confidence >= 80:
                    conf_label = "STRONG"
                elif confidence >= 60:
                    conf_label = "GOOD"
                else:
                    conf_label = ""

                signal_text = ", ".join([s.get("name", "")[:30] for s in signals[:2]]) if signals else ""

                # Format price change
                change_24h = r.get("change_24h")
                change_str = ""
                if change_24h is not None:
                    pct = change_24h * 100
                    if pct >= 0:
                        change_str = f" | 24h: +{pct:.1f}%"
                    else:
                        change_str = f" | 24h: {pct:.1f}%"

                out_lines.append(f"\n<b>#{i} {action_icon} {conf_label} {action_str}</b> ({confidence:.0f}/100)")
                out_lines.append(f"<b>Q:</b> {q}")
                out_lines.append(f"YES ${yes:.2f} | NO ${no:.2f} | Vol: {vol_str}{change_str}")
                if signal_text:
                    out_lines.append(f"<i>Signals: {signal_text}</i>")

            out_lines.append("\n━━━━━━━━━━━━━━━━━━━━━━")

            # Build grouped keyboard - two rows per market with emojis
            number_emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
            keyboard = []
            for i, r in enumerate(results):
                market_id = r.get("market_id", "")[:20]
                market_url = r.get("market_url", "")
                num_emoji = number_emojis[i] if i < len(number_emojis) else f"#{i+1}"

                if market_url:
                    keyboard.append([
                        InlineKeyboardButton(f"{num_emoji} View Market", url=market_url)
                    ])
                keyboard.append([
                    InlineKeyboardButton("📊 Details", callback_data=f"details:{market_id}"),
                    InlineKeyboardButton("🔔 Set Alert", callback_data=f"alert:{market_id}"),
                ])
            keyboard.append([
                InlineKeyboardButton("🔄 Refresh", callback_data="refresh:discover"),
            ])
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text("\n".join(out_lines), parse_mode="HTML", reply_markup=reply_markup)

        elif payload == "trending":
            await query.edit_message_text("Refreshing trending news markets...", parse_mode="HTML")

            results = scan_trending_news(limit=100, top_n=10)

            if not results:
                await query.edit_message_text(
                    "No markets with significant news signals found.\n<i>Try again later.</i>",
                    parse_mode="HTML"
                )
                return

            def _esc(text: str) -> str:
                return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            out_lines = [
                "<b>📰 TRENDING NEWS MARKETS</b>",
                "<i>Markets with recent news coverage</i>",
                "━━━━━━━━━━━━━━━━━━━━━━"
            ]

            for i, r in enumerate(results[:5], 1):
                q = _esc(r.get("question", ""))
                if len(q) > 100:
                    q = q[:100] + "..."
                yes = r.get("yes_price", 0.0)
                no = r.get("no_price", 0.0)
                vol = r.get("volume", 0.0)
                confidence = r.get("confidence", 0.0)
                action_str = r.get("action", "WATCH")
                action_icon = r.get("action_icon", "[~]")
                news_mentions = r.get("news_mentions", 0)
                sentiment = r.get("sentiment", 0.0)
                headlines = r.get("headlines", [])

                if vol >= 1_000_000:
                    vol_str = f"${vol/1_000_000:.1f}M"
                elif vol >= 1_000:
                    vol_str = f"${vol/1_000:.0f}K"
                else:
                    vol_str = f"${vol:.0f}"

                if sentiment > 0.2:
                    sent_str = "📈 Positive"
                elif sentiment < -0.2:
                    sent_str = "📉 Negative"
                else:
                    sent_str = "➡️ Neutral"

                out_lines.append(f"\n<b>#{i} {action_icon} {action_str}</b> ({confidence:.0f}/100)")
                out_lines.append(f"<b>Q:</b> {q}")
                out_lines.append(f"YES ${yes:.2f} | NO ${no:.2f} | Vol: {vol_str}")
                out_lines.append(f"📰 {news_mentions} news mentions | {sent_str}")
                if headlines:
                    headline = _esc(headlines[0][:80])
                    out_lines.append(f"<i>» {headline}...</i>")

            out_lines.append("\n━━━━━━━━━━━━━━━━━━━━━━")

            number_emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
            keyboard = []
            for i, r in enumerate(results[:5]):
                market_id = r.get("market_id", "")[:20]
                market_url = r.get("market_url", "")
                num_emoji = number_emojis[i] if i < len(number_emojis) else f"#{i+1}"

                if market_url:
                    keyboard.append([
                        InlineKeyboardButton(f"{num_emoji} View Market", url=market_url)
                    ])
                keyboard.append([
                    InlineKeyboardButton("📊 Details", callback_data=f"details:{market_id}"),
                    InlineKeyboardButton("🔔 Set Alert", callback_data=f"alert:{market_id}"),
                ])
            keyboard.append([
                InlineKeyboardButton("🔄 Refresh", callback_data="refresh:trending"),
            ])
            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text("\n".join(out_lines), parse_mode="HTML", reply_markup=reply_markup)


async def market_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Deep-dive into a specific market by keyword search."""
    if not context.args:
        await update.message.reply_text(
            "Please provide a search term.\n\n"
            "<b>Example:</b>\n"
            "<code>/market bitcoin</code>\n"
            "<code>/market trump election</code>",
            parse_mode="HTML",
        )
        return

    query = " ".join(context.args)
    processing_msg = await update.message.reply_text(
        f"Searching markets for '{query}'...",
        parse_mode="HTML",
    )

    try:
        details, error = scan_market_details(query)
        if error:
            await processing_msg.edit_text(f"<b>Error:</b> {error}", parse_mode="HTML")
            return

        # Format the analysis
        analysis_text = format_market_analysis(details)

        # Add inline buttons with emojis
        market_id = details.get("market_id", "")[:20]
        market_url = details.get("market_url", "")
        keyboard = []
        if market_url:
            keyboard.append([InlineKeyboardButton("🔗 View on Polymarket", url=market_url)])
        keyboard.append([
            InlineKeyboardButton("🔔 Set Alert", callback_data=f"alert:{market_id}"),
            InlineKeyboardButton("🔍 Discover More", callback_data="refresh:discover"),
        ])
        reply_markup = InlineKeyboardMarkup(keyboard)

        await processing_msg.edit_text(analysis_text, parse_mode="HTML", reply_markup=reply_markup)

    except Exception as exc:
        await processing_msg.edit_text(f"<b>Error:</b> {exc}", parse_mode="HTML")
        logger.error("Error in market_command: %s", exc, exc_info=True)


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
        results = scan_polymarket_raw(limit=200, threshold=0.02)
        if not results:
            await processing_msg.edit_text("No Polymarket arbitrage found right now.", parse_mode="HTML")
            return

        def _esc(text: str) -> str:
            return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        out_lines = ["<b>Polymarket Arbitrage</b> (top 5)\n"]
        for r in results[:5]:
            q = _esc(r.get("question", ""))
            if len(q) > 120:
                q = q[:120] + "..."
            yes = r.get("yes_price", 0.0)
            no = r.get("no_price", 0.0)
            total = r.get("total", 0.0)
            profit = r.get("profit_pct", 0.0)
            vol = r.get("volume", 0.0)
            out_lines.append(
                f"{q}\n"
                f"YES: {yes:.3f}  NO: {no:.3f}  TOTAL: {total:.3f}  Profit: {profit:.2f}%  Volume: {vol:.0f}\n"
            )
        await processing_msg.edit_text("\n".join(out_lines), parse_mode="HTML")
    except Exception as exc:
        await processing_msg.edit_text(f"<b>Error:</b> {exc}", parse_mode="HTML")
        logger.error("Error in polyarb_command: %s", exc, exc_info=True)


# LEGACY COMMANDS REMOVED - Use /discover instead


async def xarb_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    processing_msg = await update.message.reply_text(
        "Scanning cross-market mismatches (Polymarket vs Kalshi)...",
        parse_mode="HTML",
    )
    try:
        results, error = scan_cross_market_mismatches(limit=200, top_n=5)
        if error:
            await processing_msg.edit_text(error, parse_mode="HTML")
            return
        if not results:
            await processing_msg.edit_text("No cross-market mismatches found right now.", parse_mode="HTML")
            return

        def _esc(text: str) -> str:
            return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        out_lines = ["<b>Cross-market Mismatches</b>\n<i>Warning: different settlement rules; this is not risk-free.</i>\n"]
        for r in results:
            pq = _esc(r.get("question_poly", ""))
            kq = _esc(r.get("question_kalshi", ""))
            if len(pq) > 120:
                pq = pq[:120] + "..."
            if len(kq) > 120:
                kq = kq[:120] + "..."
            sim = r.get("similarity", 0.0)
            edge = r.get("edge_pct", 0.0)
            py = r.get("poly_yes", 0.0)
            ky = r.get("kalshi_yes", 0.0)
            pv = r.get("poly_volume", 0.0)
            kv = r.get("kalshi_volume", 0.0)
            conf = r.get("confidence", 0.0)
            conf_label = r.get("confidence_label", "LOW")
            settle_sim = r.get("settlement_sim", 0.0)
            out_lines.append(
                f"Sim: {sim:.2f}  Settle: {settle_sim:.2f}  Edge: {edge:.2f}%\n"
                f"Conf: {conf:.1f} ({conf_label})\n"
                f"Poly YES: {py:.3f}  Kalshi YES: {ky:.3f}\n"
                f"Poly Vol: {pv:.0f}  Kalshi Vol: {kv:.0f}\n"
                f"P: {pq}\n"
                f"K: {kq}\n"
            )
        sent = await processing_msg.edit_text("\n".join(out_lines), parse_mode="HTML")
        try:
            enriched = log_opportunities("xarb", results)
            if enriched:
                short_ids = [it.get("short_id") for it in enriched if it.get("short_id")]
                if short_ids:
                    await sent.reply_text(f"IDs: {', '.join(short_ids)}", parse_mode="HTML")
        except Exception:
            pass
    except Exception as exc:
        await processing_msg.edit_text(f"<b>Error:</b> {exc}", parse_mode="HTML")
        logger.error("Error in xarb_command: %s", exc, exc_info=True)


def _match_short_id(short_id: str, items: list[dict]) -> str | None:
    for it in items:
        if short_id == (it.get("short_id") or ""):
            return it.get("opportunity_id")
    return None


async def toggle_xarb_alert(update: Update, context: ContextTypes.DEFAULT_TYPE, enabled: bool) -> None:
    save_xarb_alert_flag(enabled)
    status = "enabled" if enabled else "disabled"
    note = ""
    if enabled and context.job_queue is None:
        note = " (job queue unavailable; alerts won't run until enabled)"
    await update.message.reply_text(f"Cross-market alerting {status}.{note}", parse_mode="HTML")


async def watch_add_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if len(context.args) < 2:
        await update.message.reply_text("Usage: /watch_add <platform> <market_id>", parse_mode="HTML")
        return
    platform, market_id = context.args[0].lower(), context.args[1]
    add_to_watchlist(platform, market_id)
    await update.message.reply_text(f"Added to watchlist: {platform} {market_id}", parse_mode="HTML")


async def watch_rm_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if len(context.args) < 2:
        await update.message.reply_text("Usage: /watch_rm <platform> <market_id>", parse_mode="HTML")
        return
    platform, market_id = context.args[0].lower(), context.args[1]
    remove_from_watchlist(platform, market_id)
    await update.message.reply_text(f"Removed from watchlist: {platform} {market_id}", parse_mode="HTML")


async def watch_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    wl = list_watchlist()
    pm = wl.get("polymarket", [])
    kal = wl.get("kalshi", [])
    if not pm and not kal:
        await update.message.reply_text("Watchlist is empty.", parse_mode="HTML")
        return
    lines = ["<b>Watchlist</b>"]
    if pm:
        lines.append("Polymarket:")
        lines.append(", ".join(pm))
    if kal:
        lines.append("Kalshi:")
        lines.append(", ".join(kal))
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def watch_scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    processing_msg = await update.message.reply_text("Scanning watchlist...", parse_mode="HTML")
    results, err = scan_watchlist(top_n=5)
    if err:
        await processing_msg.edit_text(err, parse_mode="HTML")
        return
    if not results:
        await processing_msg.edit_text("No watchlist opportunities right now.", parse_mode="HTML")
        return
    def _esc(text: str) -> str:
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    out = ["<b>Watchlist Cross-Market Mismatches</b>"]
    for r in results:
        pq = _esc(r.get("question_poly", ""))
        kq = _esc(r.get("question_kalshi", ""))
        if len(pq) > 120:
            pq = pq[:120] + "..."
        if len(kq) > 120:
            kq = kq[:120] + "..."
        out.append(
            f"Score: {r.get('score',0.0):.1f} Conf: {r.get('confidence',0.0):.1f}\n"
            f"P: {pq}\nK: {kq}\n"
        )
    await processing_msg.edit_text("\n".join(out), parse_mode="HTML")


async def label_short(update: Update, context: ContextTypes.DEFAULT_TYPE, label: str) -> None:
    chat_id = str(update.effective_chat.id)
    if not _is_admin(chat_id):
        await update.message.reply_text("Not authorized to label.", parse_mode="HTML")
        return
    if not context.args:
        await update.message.reply_text("Provide short_id", parse_mode="HTML")
        return
    short_id = context.args[0]
    # search recent log entries for matching short_id
    recent = []
    try:
        log_file = DATA_DIR / "logs" / "opportunities.jsonl"
        if log_file.exists():
            lines = log_file.read_text(encoding="utf-8").splitlines()
            for line in reversed(lines[-200:]):
                import json
                obj = json.loads(line)
                items = obj.get("items") or []
                oid = _match_short_id(short_id, items)
                if oid:
                    save_label(oid, label)
                    await update.message.reply_text(f"Labeled {short_id} as {label}", parse_mode="HTML")
                    return
    except Exception:
        pass
    await update.message.reply_text("No matching opportunity found.", parse_mode="HTML")


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


async def alert_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set a price alert for a market.
    Usage: /alert <query> <target_price>
    Example: /alert trump 0.60
    """
    chat_id = str(update.effective_chat.id)

    if len(context.args) < 2:
        # Show existing alerts
        alerts = load_price_alerts()
        user_alerts = alerts.get(chat_id, [])
        if not user_alerts:
            await update.message.reply_text(
                "<b>No price alerts set.</b>\n\n"
                "Set an alert:\n"
                "<code>/alert &lt;query&gt; &lt;price&gt;</code>\n\n"
                "Examples:\n"
                "<code>/alert trump 0.60</code> - Alert when Trump YES hits $0.60\n"
                "<code>/alert bitcoin 0.40</code> - Alert when Bitcoin YES hits $0.40",
                parse_mode="HTML"
            )
        else:
            lines = ["<b>Your Price Alerts</b>", "━━━━━━━━━━━━━━━━━━━━"]
            for i, a in enumerate(user_alerts, 1):
                lines.append(f"{i}. {a['query']} → ${a['target_price']:.2f}")
            lines.append("\nTo clear: <code>/alert clear</code>")
            await update.message.reply_text("\n".join(lines), parse_mode="HTML")
        return

    if context.args[0].lower() == "clear":
        alerts = load_price_alerts()
        alerts[chat_id] = []
        save_price_alerts(alerts)
        await update.message.reply_text("All price alerts cleared.")
        return

    # Parse: /alert <query words...> <price>
    try:
        target_price = float(context.args[-1])
        query = " ".join(context.args[:-1])
    except ValueError:
        await update.message.reply_text(
            "Invalid format. Use: <code>/alert &lt;query&gt; &lt;price&gt;</code>\n"
            "Example: <code>/alert trump 0.60</code>",
            parse_mode="HTML"
        )
        return

    if not query:
        await update.message.reply_text("Please provide a search query.")
        return

    if not 0.01 <= target_price <= 0.99:
        await update.message.reply_text("Price must be between 0.01 and 0.99")
        return

    # Save the alert
    alerts = load_price_alerts()
    if chat_id not in alerts:
        alerts[chat_id] = []

    # Limit to 10 alerts per user
    if len(alerts[chat_id]) >= 10:
        await update.message.reply_text("Maximum 10 alerts. Use <code>/alert clear</code> to remove.", parse_mode="HTML")
        return

    alerts[chat_id].append({
        "query": query,
        "target_price": target_price,
        "created": time.time(),
    })
    save_price_alerts(alerts)

    await update.message.reply_text(
        f"Price alert set:\n"
        f"<b>{query}</b> → ${target_price:.2f}\n\n"
        f"You'll be notified when YES price crosses this level.",
        parse_mode="HTML"
    )


async def digest_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Enable/disable daily digest.
    Usage: /digest on|off
    """
    chat_id = str(update.effective_chat.id)

    if not context.args:
        settings = load_digest_settings()
        user_settings = settings.get(chat_id, {})
        enabled = user_settings.get("enabled", False)
        msg = "Daily digest is ENABLED." if enabled else "Daily digest is DISABLED."
        msg += "\n\nUse <code>/digest on</code> or <code>/digest off</code>"
        await update.message.reply_text(msg, parse_mode="HTML")
        return

    cmd = context.args[0].lower()
    settings = load_digest_settings()

    if cmd in ("on", "enable"):
        settings[chat_id] = {"enabled": True, "hour": 9}  # 9 AM UTC
        save_digest_settings(settings)
        await update.message.reply_text("Daily digest ENABLED. You'll receive top opportunities at 9 AM UTC.")
        return

    if cmd in ("off", "disable"):
        settings[chat_id] = {"enabled": False}
        save_digest_settings(settings)
        await update.message.reply_text("Daily digest DISABLED.")
        return

    await update.message.reply_text("Usage: /digest on|off")


async def watch_on_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Enable scheduled cross-arb watch for this chat."""
    if context.job_queue is None:
        await update.message.reply_text("Scheduling is unavailable (job queue not configured).")
        return
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
    parser.add_argument(
        "--alert-improve",
        type=float,
        default=float(os.getenv("ALERT_IMPROVE", "0.005")),
        help="Required improvement in edge to resend within cooldown (fraction, e.g., 0.005 = 0.5%%)",
    )
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

    application = ApplicationBuilder().token(bot_token).build()
    application.bot_data["alert_cooldown_sec"] = alert_cooldown_min * 60
    application.bot_data["alert_improve_pct"] = alert_improve * 100.0
    application.bot_data["watch_interval"] = watch_interval
    application.bot_data["xarb_alert_interval"] = int(os.getenv("XARB_ALERT_INTERVAL", "600"))
    application.bot_data["xarb_alert_score_min"] = float(os.getenv("XARB_ALERT_SCORE_MIN", "20"))
    application.bot_data["xarb_alert_cooldown_sec"] = int(os.getenv("XARB_ALERT_COOLDOWN", "60")) * 60
    application.bot_data["xarb_alert_improve"] = float(os.getenv("XARB_ALERT_IMPROVE", "1.0"))

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    # Main commands
    application.add_handler(CommandHandler("discover", discover_command))
    application.add_handler(CommandHandler("trending", trending_command))
    application.add_handler(CommandHandler("market", market_command))
    # Quick scans
    application.add_handler(CommandHandler("scan", scan_command))
    application.add_handler(CommandHandler("price", price_command))
    application.add_handler(CommandHandler("polyarb", polyarb_command))
    application.add_handler(CommandHandler("xarb", xarb_command))
    # Legacy commands removed - only /discover now
    application.add_handler(CommandHandler("watch_add", watch_add_command))
    application.add_handler(CommandHandler("watch_rm", watch_rm_command))
    application.add_handler(CommandHandler("watch", watch_command))
    application.add_handler(CommandHandler("watch_scan", watch_scan_command))
    application.add_handler(CommandHandler("xarb_alert_on", lambda u, c: toggle_xarb_alert(u, c, True)))
    application.add_handler(CommandHandler("xarb_alert_off", lambda u, c: toggle_xarb_alert(u, c, False)))
    application.add_handler(CommandHandler("label_good", lambda u, c: label_short(u, c, "good")))
    application.add_handler(CommandHandler("label_bad", lambda u, c: label_short(u, c, "bad")))
    application.add_handler(CommandHandler("label_unknown", lambda u, c: label_short(u, c, "unknown")))
    application.add_handler(CommandHandler("crossarb", crossarb_command))
    application.add_handler(CommandHandler("alerts", alerts_command))
    application.add_handler(CommandHandler("alert", alert_command))
    application.add_handler(CommandHandler("digest", digest_command))
    application.add_handler(CommandHandler("watch_on", watch_on_command))
    application.add_handler(CommandHandler("watch_off", watch_off_command))
    application.add_handler(CommandHandler("status", status_command))
    # Inline button callback handler
    application.add_handler(CallbackQueryHandler(button_callback))

    alert_chat = os.getenv("TELEGRAM_ALERT_CHAT_ID")
    if alert_chat and application.job_queue is not None:
        jq = application.job_queue
        cooldown_sec = alert_cooldown_min * 60
        improve_pct = alert_improve * 100.0
        xarb_interval = application.bot_data["xarb_alert_interval"]
        xarb_score_min = application.bot_data["xarb_alert_score_min"]
        xarb_cooldown = application.bot_data["xarb_alert_cooldown_sec"]
        xarb_improve = application.bot_data["xarb_alert_improve"]

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

        async def _xarb_job(context: ContextTypes.DEFAULT_TYPE):
            try:
                if not load_xarb_alert_flag():
                    return
                results, error = scan_cross_market_mismatches(limit=200, top_n=5)
                if error or not results:
                    return
                seen = load_xarb_seen()
                new_out = []
                now_ts = int(time.time())
                for r in results:
                    score = float(r.get("score", 0.0) or 0.0)
                    if score < xarb_score_min:
                        continue
                    fp_payload = f"{r.get('question_poly','')}|{r.get('question_kalshi','')}"
                    fp = str(hash(fp_payload))
                    ok, tag, prev_val = should_alert(seen, fp, score, xarb_cooldown, xarb_improve, now_ts)
                    if not ok:
                        continue
                    seen[fp] = {"ts": now_ts, "edge": score}
                    r["alert_tag"] = tag
                    r["prev_score"] = prev_val
                    new_out.append(r)
                if not new_out:
                    save_xarb_seen(seen)
                    return
                out_lines = ["<b>Cross-market Mismatch Alert</b>\n<i>Not risk-free; settlement may differ.</i>\n"]
                for r in new_out[:3]:
                    tag = r.get("alert_tag", "NEW")
                    prev_score = r.get("prev_score", 0.0)
                    out_lines.append(
                        f"[{tag}] Sim: {r.get('similarity',0.0):.2f} Score: {r.get('score',0.0):.1f} (prev {prev_score:.1f})\n"
                        f"Edge: {r.get('edge_pct',0.0):.2f}%  Poly YES: {r.get('poly_yes',0.0):.3f}  Kalshi YES: {r.get('kalshi_yes',0.0):.3f}\n"
                        f"P: {r.get('question_poly','')}\n"
                        f"K: {r.get('question_kalshi','')}\n"
                    )
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
                    save_xarb_seen(seen)
            except Exception:
                return

        jq.run_repeating(_xarb_job, interval=xarb_interval, first=15)

        # Price alert checking job (every 5 minutes)
        async def _price_alert_job(context: ContextTypes.DEFAULT_TYPE):
            try:
                from src.engines.market_search import search_markets
                alerts = load_price_alerts()
                if not alerts:
                    return

                triggered = []
                for chat_id, user_alerts in list(alerts.items()):
                    remaining = []
                    for a in user_alerts:
                        query = a.get("query", "")
                        target = a.get("target_price", 0.5)
                        try:
                            results = search_markets(query, limit=1)
                            if results:
                                market = results[0]
                                yes_price = market.get("yes_price", 0.5)
                                question = market.get("question", "")[:80]

                                # Check if price crossed target
                                if yes_price >= target:
                                    triggered.append((chat_id, query, target, yes_price, question))
                                else:
                                    remaining.append(a)
                            else:
                                remaining.append(a)
                        except Exception:
                            remaining.append(a)

                    alerts[chat_id] = remaining

                # Send triggered alerts
                for chat_id, query, target, current, question in triggered:
                    try:
                        msg = (
                            f"<b>PRICE ALERT TRIGGERED</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n\n"
                            f"<b>{query}</b>\n"
                            f"Target: ${target:.2f}\n"
                            f"Current: ${current:.2f}\n\n"
                            f"<i>{question}...</i>"
                        )
                        await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode="HTML")
                    except Exception:
                        pass

                save_price_alerts(alerts)
            except Exception:
                return

        jq.run_repeating(_price_alert_job, interval=300, first=60)

        # Daily digest job (runs every hour, checks if it's digest time)
        async def _daily_digest_job(context: ContextTypes.DEFAULT_TYPE):
            try:
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc)
                current_hour = now.hour

                settings = load_digest_settings()
                for chat_id, user_settings in settings.items():
                    if not user_settings.get("enabled"):
                        continue
                    digest_hour = user_settings.get("hour", 9)
                    if current_hour != digest_hour:
                        continue

                    # Check if already sent today
                    last_sent = user_settings.get("last_sent", "")
                    today = now.strftime("%Y-%m-%d")
                    if last_sent == today:
                        continue

                    # Send digest
                    try:
                        results = scan_best_opportunities(limit=50, top_n=5, min_confidence=50)
                        if not results:
                            continue

                        lines = [
                            "<b>DAILY DIGEST</b>",
                            f"<i>{today}</i>",
                            "━━━━━━━━━━━━━━━━━━━━"
                        ]
                        for i, r in enumerate(results, 1):
                            q = r.get("question", "")[:60]
                            conf = r.get("confidence", 0)
                            action = r.get("action", "WATCH")
                            yes = r.get("yes_price", 0)
                            lines.append(f"\n#{i} [{action}] {conf:.0f}/100")
                            lines.append(f"{q}...")
                            lines.append(f"YES ${yes:.2f}")

                        lines.append("\n━━━━━━━━━━━━━━━━━━━━")
                        lines.append("<i>Use /discover for full details</i>")

                        await context.bot.send_message(chat_id=chat_id, text="\n".join(lines), parse_mode="HTML")

                        # Mark as sent
                        settings[chat_id]["last_sent"] = today
                        save_digest_settings(settings)
                    except Exception:
                        pass
            except Exception:
                return

        jq.run_repeating(_daily_digest_job, interval=3600, first=120)

    elif alert_chat and application.job_queue is None:
        logger.warning("TELEGRAM_ALERT_CHAT_ID set but JobQueue unavailable; alerts/watch scheduling disabled. Install python-telegram-bot[job-queue] or enable job queue.")

    logger.info("Bot starting...")
    print("Bot is running. Press Ctrl+C to stop.")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
