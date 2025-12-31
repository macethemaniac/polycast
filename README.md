# Arbitrage Scanner MVP

A minimal Python application for detecting arbitrage opportunities between cryptocurrency data sources, with both console and Telegram bot interfaces.

## Features

- Fetches real-time prices from CoinGecko and DeFiLlama data sources
- Calculates arbitrage spread and percentage difference
- Console interface for direct execution
- Telegram bot interface for convenient access

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. (For Telegram Bot) Get a bot token:
   - Open Telegram and search for [@BotFather](https://t.me/botfather)
   - Send `/newbot` and follow the instructions
   - Copy the bot token you receive

3. (For Telegram Bot) Set the bot token as an environment variable:

   **Windows (PowerShell):**
   ```powershell
   $env:TELEGRAM_BOT_TOKEN="your_bot_token_here"
   ```

   **Windows (CMD):**
   ```cmd
   set TELEGRAM_BOT_TOKEN=your_bot_token_here
   ```

   **Linux/Mac:**
   ```bash
   export TELEGRAM_BOT_TOKEN="your_bot_token_here"
   ```

   Or create a `.env` file (if using python-dotenv):
   ```
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   ```

## Usage

### Console Mode

Run the scanner from the project root:

```bash
cd arbitrage_mvp
python src/main.py
```

The script will:
1. Fetch BTC/USDT prices from CoinGecko and DeFiLlama
2. Calculate the price spread and percentage difference
3. Display which data source has the lowest (buy) and highest (sell) price
4. Show the potential arbitrage opportunity

### Telegram Bot Mode

**Quick Start (using startup script):**
```powershell
cd arbitrage_mvp
.\start_bot.ps1
```

Or double-click `start_bot.bat` on Windows.

**Manual Start:**
```bash
cd arbitrage_mvp
python src/bot.py
```
(Don't forget to set `TELEGRAM_BOT_TOKEN` environment variable first)

The bot supports the following commands:
- `/start` - Show welcome message and bot info
- `/scan` - Scan BTC/USDT for arbitrage opportunity
- `/price <pair>` - Get prices for a specific pair (e.g., `/price BTC/USDT`)
- `/help` - Show help information

**Example bot interactions:**
```
/scan
/price BTC/USDT
/price ETH/USDT
```

## Project Structure

```
arbitrage_mvp/
    src/
        exchanges/
            coingecko.py    # CoinGecko API integration
            defillama.py    # DeFiLlama API integration
        analytics/
            arbitrage.py    # Arbitrage calculation logic
        main.py             # Console entry point
        bot.py              # Telegram bot entry point
        scanner.py          # Shared scanner module
    requirements.txt        # Python dependencies
    start_bot.ps1          # PowerShell startup script (gitignored)
    start_bot.bat          # Windows batch startup script (gitignored)
    .gitignore             # Git ignore rules for security
```

## Security Notes

- **Bot tokens are sensitive** - Never commit them to version control
- The `.gitignore` file excludes `start_bot.ps1`, `start_bot.bat`, and `.env` files
- If your token is compromised, revoke it in @BotFather and create a new one
- For production, use environment variables or secure secret management

## Notes

- This is an MVP focusing only on arbitrage price comparison
- No authentication required (uses public APIs)
- No database or persistence layer
- No threading or websockets (sequential execution)
- Uses free tier APIs (CoinGecko and DeFiLlama) with rate limits
- Telegram bot requires a valid bot token from @BotFather

## Matching review and manual overrides
- Export match candidates for Polymarket ↔ Kalshi:  
  `python -m bot.match_review --export-candidates outputs/match_candidates.json`
- Pin matches manually by editing `data/manual_matches.json` (overrides NLP/matcher). Example:
  ```json
  {
    "516710": { "kalshi_id": "KX12345", "kal_title": "Kalshi market title", "side": "YES" }
  }
  ```
- Learned associations are stored in `data/match_memory.json` when a candidate passes the confidence threshold; future runs boost those pairs. You can promote reviewed candidates into `manual_matches.json` to pin them.

## Auto-scan watch loop
- Run periodic scans every 5 minutes (default):  
  `python -m src.watch`
- Custom interval (seconds):  
  `python -m src.watch --interval 300`
- Each cycle logs: timestamp, markets fetched, comparisons, opportunities, best edge; saves opportunities to `outputs/opps_<timestamp>.json` when found. Errors are logged and retried with backoff; overlapping runs are avoided.
