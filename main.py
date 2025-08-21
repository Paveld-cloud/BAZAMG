# main.py
import logging
import asyncio
from telegram.ext import ApplicationBuilder
from telegram.constants import ParseMode

from app.handlers import build_handlers
from app import data
from app.config import TELEGRAM_TOKEN, WEBHOOK_URL, WEBHOOK_PATH, PORT, WEBHOOK_SECRET_TOKEN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bot")

async def _startup(app):
    await data.ensure_fresh_data_async(force=True)
    await data.ensure_users_async(force=True)
    logger.info("Bot startup completed.")

def build_app():
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    for h in build_handlers():
        application.add_handler(h)
    application.post_init = _startup
    return application

if __name__ == "__main__":
    app = build_app()
    if WEBHOOK_URL:
        # Webhook mode
        app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=WEBHOOK_PATH.lstrip("/"),
            webhook_url=f"{WEBHOOK_URL}{WEBHOOK_PATH}",
            secret_token=WEBHOOK_SECRET_TOKEN or None,
            drop_pending_updates=True,
        )
    else:
        # Polling mode
        app.run_polling(drop_pending_updates=True)
