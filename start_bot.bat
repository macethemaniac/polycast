@echo off
REM Batch script to start the Telegram bot on Windows

set TELEGRAM_BOT_TOKEN=8404888863:AAEO1fAbnXhoJJC4MsheN7pe39colQhw3kA

echo Starting Arbitrage Scanner Telegram Bot...
echo Bot token configured.
echo.

python src\bot.py

pause

