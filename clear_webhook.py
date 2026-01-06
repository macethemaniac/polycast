import os
import asyncio
from telegram import Bot

token = os.getenv("TELEGRAM_BOT_TOKEN")
if not token:
    raise SystemExit("TELEGRAM_BOT_TOKEN not set")

async def main():
    bot = Bot(token=token)
    await bot.delete_webhook(drop_pending_updates=True)
    print("Webhook cleared")

asyncio.run(main())
