# PowerShell script to start the Telegram bot
# Sets the bot token and starts the bot

$env:TELEGRAM_BOT_TOKEN = "8404888863:AAEO1fAbnXhoJJC4MsheN7pe39colQhw3kA"

Write-Host "Starting Arbitrage Scanner Telegram Bot..." -ForegroundColor Green
Write-Host "Bot token configured." -ForegroundColor Green
Write-Host ""

python src/bot.py

