import logging
import os
import pickle
import json
import gspread
import re
from google.oauth2.service_account import Credentials
from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
from pandas import DataFrame

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Токен из переменных окружения
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# Google Sheets настройки
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
SPREADSHEET_URL = os.getenv("SPREADSHEET_URL")
SHEET_NAME = "SAP"

def load_data():
    creds_info = json.loads(os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"))
    creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    client = gspread.authorize(creds)
    sheet = client.open_by_url(SPREADSHEET_URL).worksheet(SHEET_NAME)
    records = sheet.get_all_records()
    return records

# Загружаем данные из Google Sheets
raw_data = load_data()

# Преобразуем в DataFrame
df = DataFrame(raw_data)
df.columns = df.columns.str.strip().str.lower()

# Приводим поля к строковому виду
for col in ['тип', 'наименование', 'код', 'oem', 'изготовитель']:
    df[col] = df[col].astype(str).str.strip().str.lower()

user_state = {}
search_count = {}

if os.path.exists("state.pkl"):
    with open("state.pkl", "rb") as f:
        user_state = pickle.load(f)

def generate_inline_keyboard(code: str):
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📖 История", callback_data=f"history|{code}"),
            InlineKeyboardButton("🧼 Уход", callback_data=f"care|{code}"),
        ],
        [
            InlineKeyboardButton("📝 Описание", callback_data=f"description|{code}"),
            InlineKeyboardButton("🎥 Видео", callback_data=f"video|{code}"),
        ]
    ])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_state.pop(user_id, None)
    await update.message.reply_text("Привет! История поиска очищена.\nОтправь тип, код или наименование детали.")

async def more(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = user_state.get(user_id)
    if not state:
        await update.message.reply_text("Сначала выполните поиск.")
        return

    query, offset, results = state["query"], state["offset"], state["results"]
    page = results.iloc[offset: offset + 5]
    for _, row in page.iterrows():
        text = (
            f"📍 Тип: {row['тип']}\n"
            f"📦 Наименование: {row['наименование']}\n"
            f"🔢 Код: {row['код']}\n"
            f"📦 Кол-во: {row['количество']}\n"
            f"💰 Цена: {row['цена']} {row['валюта']}\n"
            f"🏭 Изготовитель: {row['изготовитель']}\n"
            f"⚙️ OEM: {row['oem']}")
        await update.message.reply_text(text, reply_markup=generate_inline_keyboard(str(row['код'])))

    new_offset = offset + 5
    user_state[user_id]["offset"] = new_offset
    if new_offset < len(results):
        await update.message.reply_text("Напишите /more для следующих результатов.")
    else:
        await update.message.reply_text("Больше результатов нет.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 Доступные команды:\n"
        "/start — начать заново\n"
        "/more — показать ещё\n"
        "/help — справка\n"
        "Просто отправьте текст для поиска по типу, коду, OEM, названию или изготовителю."
    )

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    count = search_count.get(user_id, 0)
    await update.message.reply_text(f"🔍 Вы сделали {count} поисков за сессию.")

async def export(update: Update, context: ContextTypes.DEFAULT_TYPE):
    import pandas as pd
    user_id = update.effective_user.id
    state = user_state.get(user_id)
    if not state:
        await update.message.reply_text("Сначала выполните поиск.")
        return

    filename = f"export_{user_id}.xlsx"
    state["results"].to_excel(filename, index=False)
    with open(filename, "rb") as f:
        await update.message.reply_document(InputFile(f, filename))
    os.remove(filename)

async def search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    query = update.message.text.strip().lower()
    query = re.sub(r"[()]", "", query)
    query = re.sub(r"\s{2,}", " ", query)

    mask = (
        df['тип'].str.contains(query, na=False) |
        df['наименование'].str.contains(query, na=False) |
        df['код'].str.contains(query, na=False) |
        df['oem'].str.contains(query, na=False) |
        df['изготовитель'].str.contains(query, na=False)
    )
    results = df[mask]

    if results.empty:
        await update.message.reply_text(f'По запросу "{query}" ничего не найдено.')
        return

    user_state[user_id] = {
        "query": query,
        "offset": 5,
        "results": results
    }

    search_count[user_id] = search_count.get(user_id, 0) + 1
    logger.info(f"Поиск от {user_id}: {query} (всего: {search_count[user_id]})")

    for _, row in results.head(5).iterrows():
        text = (
            f"📍 Тип: {row['тип']}\n"
            f"📦 Наименование: {row['наименование']}\n"
            f"🔢 Код: {row['код']}\n"
            f"📦 Кол-во: {row['количество']}\n"
            f"💰 Цена: {row['цена']} {row['валюта']}\n"
            f"🏭 Изготовитель: {row['изготовитель']}\n"
            f"⚙️ OEM: {row['oem']}")
        await update.message.reply_text(text, reply_markup=generate_inline_keyboard(str(row['код'])))

    if len(results) > 5:
        await update.message.reply_text("Показано 5 первых результатов. Напишите /more для продолжения.")

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    action, item_code = query.data.split("|", 1)
    responses = {
        "history": f"📖 История детали: {item_code}",
        "care": f"🧼 Уход за деталью: {item_code}",
        "description": f"📝 Описание детали: {item_code}",
        "video": f"🎥 Видеообзор: {item_code}",
    }
    await query.message.reply_text(responses.get(action, "Неизвестное действие."))

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(msg="Ошибка в Telegram обработчике", exc_info=context.error)
    if isinstance(update, Update) and update.message:
        await update.message.reply_text("Произошла ошибка. Попробуйте позже.")

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("more", more))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("export", export))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search))
    app.add_error_handler(error_handler)
    logger.info("Бот запущен")
    app.run_polling()

    with open("state.pkl", "wb") as f:
        pickle.dump(user_state, f)

if __name__ == "__main__":
    main()

