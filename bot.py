import logging
import os
import json
import re
import io
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import gspread
from google.oauth2.service_account import Credentials
from telegram import (
    Update, InlineKeyboardMarkup, InlineKeyboardButton, InputFile
)
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, ConversationHandler, ContextTypes, filters
)
import pandas as pd

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Константы этапов
ASK_QUANTITY, ASK_COMMENT, ASK_CONFIRM = range(3)

# Переменные окружения
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
SPREADSHEET_URL = os.getenv("SPREADSHEET_URL")
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
DATA_TTL = 300  # кэш 5 минут

# Кэш данных
DATA_CACHE = {"data": None, "timestamp": None}

# Состояния списания {user_id: {...}}
user_state = {}
suppress_next_search = set()

# Авторизация Google Sheets
def get_gs_client():
    creds_info = json.loads(os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"))
    creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    return gspread.authorize(creds)

# Загрузка данных с кэшированием
def load_data(force=False):
    now = datetime.now()
    if not force and DATA_CACHE["data"] and DATA_CACHE["timestamp"] and (now - DATA_CACHE["timestamp"]).total_seconds() < DATA_TTL:
        return DATA_CACHE["data"]

    client = get_gs_client()
    sheet = client.open_by_url(SPREADSHEET_URL).worksheet("SAP")
    data = sheet.get_all_records()
    DATA_CACHE["data"] = data
    DATA_CACHE["timestamp"] = now
    logger.info(f"✅ Загружено {len(data)} строк из Google Sheet")
    return data

# Поиск деталей
def search_parts(query):
    query = query.strip().lower()
    if not query:
        return []
    data = load_data()
    results = []
    for row in data:
        text = " ".join([str(row.get(k, "")).lower() for k in ["тип", "наименование", "код", "oem", "изготовитель"]])
        if query in text:
            results.append(row)
    return results

# Команда /start
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Отправьте запрос для поиска детали.\nДля экспорта используйте /export.")

# Экспорт в Excel
async def export_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = load_data()
    df = pd.DataFrame(data)
    bio = io.BytesIO()
    bio.name = "data.xlsx"
    df.to_excel(bio, index=False)
    bio.seek(0)
    await update.message.reply_document(InputFile(bio, filename="data.xlsx"))

# Обработка поиска
async def search_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in suppress_next_search:
        suppress_next_search.remove(user_id)
        return

    query = update.message.text.strip()
    results = search_parts(query)

    if not results:
        await update.message.reply_text("❌ Ничего не найдено.")
        return

    for part in results[:10]:
        caption = f"📦 *{part.get('наименование','')}*\nКод: `{part.get('код','')}`\nOEM: {part.get('oem','')}"
        keyboard = [[InlineKeyboardButton("📥 Взять деталь", callback_data=f"take|{part.get('код')}")]]
        await update.message.reply_text(
            caption, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="Markdown"
        )

# Обработка кнопки "Взять деталь"
async def take_part_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    part_code = query.data.split("|")[1]
    user_state[query.from_user.id] = {"part_code": part_code}
    suppress_next_search.add(query.from_user.id)
    await query.message.reply_text("Введите количество:")
    return ASK_QUANTITY

# Ввод количества
async def quantity_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        qty = int(update.message.text.strip())
    except ValueError:
        await update.message.reply_text("❌ Введите число.")
        return ASK_QUANTITY
    user_state[update.effective_user.id]["quantity"] = qty
    suppress_next_search.add(update.effective_user.id)
    await update.message.reply_text("Введите комментарий:")
    return ASK_COMMENT

# Ввод комментария и подтверждение
async def comment_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    comment = update.message.text.strip()
    st = user_state[update.effective_user.id]
    st["comment"] = comment

    # Кнопки подтверждения
    keyboard = [
        [InlineKeyboardButton("✅ Да", callback_data="confirm_yes"),
         InlineKeyboardButton("❌ Нет", callback_data="confirm_no")]
    ]
    text = (f"Вы уверены, что хотите списать деталь?\n\n"
            f"📦 Код: {st['part_code']}\n"
            f"🔢 Кол-во: {st['quantity']}\n"
            f"💬 Комментарий: {st['comment']}")
    await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
    return ASK_CONFIRM

# Обработка подтверждения
async def confirm_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    uid = query.from_user.id
    if query.data == "confirm_yes":
        st = user_state.pop(uid, None)
        if st:
            client = get_gs_client()
            ws = client.open_by_url(SPREADSHEET_URL).worksheet("История")
            ws.append_row([
                datetime.now(ZoneInfo("Asia/Tashkent")).strftime("%Y-%m-%d %H:%M:%S"),
                query.from_user.username or "",
                st["part_code"], st["quantity"], st["comment"]
            ])
            await query.message.reply_text("✅ Деталь списана и записана в историю.")
    else:
        user_state.pop(uid, None)
        await query.message.reply_text("❌ Списание отменено.")
    return ConversationHandler.END

# Отмена
async def cancel_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_state.pop(update.effective_user.id, None)
    await update.message.reply_text("❌ Действие отменено.")
    return ConversationHandler.END

# Построение приложения
def build_app():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(take_part_callback, pattern=r"^take\|")],
        states={
            ASK_QUANTITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, quantity_handler)],
            ASK_COMMENT: [MessageHandler(filters.TEXT & ~filters.COMMAND, comment_handler)],
            ASK_CONFIRM: [CallbackQueryHandler(confirm_handler, pattern=r"^confirm_")]
        },
        fallbacks=[
            CommandHandler("cancel", cancel_cmd),
            MessageHandler(filters.Regex("^Отменить$"), cancel_cmd)
        ],
    )
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("export", export_cmd))
    app.add_handler(conv)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search_handler))
    return app

if __name__ == "__main__":
    app = build_app()
    PORT = int(os.environ.get("PORT", "8080"))
    WEBHOOK_URL = os.getenv("WEBHOOK_URL")
    app.run_webhook(listen="0.0.0.0", port=PORT, url_path="", webhook_url=WEBHOOK_URL)
