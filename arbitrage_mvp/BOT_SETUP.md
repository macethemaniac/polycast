# Telegram Bot Setup Guide

This guide will help you set up and run the Telegram bot for the arbitrage scanner.

## Step 1: Create a Telegram Bot

1. Open Telegram and search for [@BotFather](https://t.me/botfather)
2. Send the command `/newbot`
3. Follow the instructions:
   - Choose a name for your bot (e.g., "Arbitrage Scanner Bot")
   - Choose a username for your bot (must end with `bot`, e.g., `my_arbitrage_bot`)
4. BotFather will give you a bot token (looks like: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)
5. **Save this token securely** - you'll need it in the next step

## Step 2: Configure the Bot Token

### Option A: Environment Variable (Recommended)

**Windows PowerShell:**
```powershell
$env:TELEGRAM_BOT_TOKEN="your_bot_token_here"
```

**Windows CMD:**
```cmd
set TELEGRAM_BOT_TOKEN=your_bot_token_here
```

**Linux/Mac:**
```bash
export TELEGRAM_BOT_TOKEN="your_bot_token_here"
```

**Note:** This sets the token for the current session only. For persistence:
- Windows: Add it to System Environment Variables
- Linux/Mac: Add `export TELEGRAM_BOT_TOKEN="..."` to your `~/.bashrc` or `~/.zshrc`

### Option B: Create a .env file (requires python-dotenv)

1. Install python-dotenv:
   ```bash
   pip install python-dotenv
   ```

2. Create a `.env` file in the project root:
   ```
   TELEGRAM_BOT_TOKEN=your_bot_token_here
   ```

3. Update `src/bot.py` to load from .env:
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   ```

## Step 3: Install Dependencies

Make sure you have all dependencies installed:

```bash
pip install -r requirements.txt
```

## Step 4: Run the Bot

Navigate to the project directory and run:

```bash
cd arbitrage_mvp
python src/bot.py
```

You should see:
```
Bot starting...
Bot is running. Press Ctrl+C to stop.
```

## Step 5: Test the Bot

1. Find your bot on Telegram (search for the username you chose)
2. Start a conversation with your bot
3. Send `/start` to see the welcome message
4. Try the commands:
   - `/scan` - Scan for BTC/USDT arbitrage
   - `/price BTC/USDT` - Get prices for BTC/USDT
   - `/help` - Show help

## Available Commands

- `/start` - Welcome message and bot information
- `/help` - Show help and command list
- `/scan` - Quick scan for BTC/USDT arbitrage opportunity
- `/price <pair>` - Get arbitrage analysis for any trading pair
  - Examples: `/price BTC/USDT`, `/price ETH/USDT`

## Troubleshooting

### "TELEGRAM_BOT_TOKEN environment variable not set!"
- Make sure you've set the environment variable correctly
- Check that the token is correct (no extra spaces)
- Try restarting your terminal/command prompt

### "ModuleNotFoundError: No module named 'telegram'"
- Run: `pip install -r requirements.txt`

### Bot doesn't respond to commands
- Make sure the bot is running (check the terminal)
- Make sure you're talking to the correct bot
- Try restarting the bot (Ctrl+C and run again)

### API Rate Limits
- If you see errors about rate limits, wait a few seconds and try again
- CoinGecko free tier has rate limits (50 calls/minute)
- DeFiLlama also has rate limits

## Security Notes

- **Never commit your bot token to version control**
- Keep your `.env` file in `.gitignore`
- Don't share your bot token publicly
- If your token is compromised, revoke it in BotFather and create a new one

## Next Steps

Once your bot is running, you can:
- Deploy it to a server for 24/7 operation
- Add more features (price alerts, multiple pairs, etc.)
- Customize the messages and commands
- Add user authentication if needed

