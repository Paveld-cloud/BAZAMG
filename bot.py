import logging
import os
import pickle
import json
import gspread
import re
from google.oauth2.service_account import Credentials
from telegram import Update, InputFile
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from pandas import DataFrame

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройки
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
SPREADSHEET_URL = os.getenv("SPREADSHEET_URL")
SHEET_NAME = "SAP"

# Загрузка данных
def load_data():
    creds_info = json.loads(os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"))
    creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    client = gspread.authorize(creds)
    sheet = client.open_by_url(SPREADSHEET_URL).worksheet(SHEET_NAME)
    return sheet.get_all_records()

raw_data = load_data()
df = DataFrame(raw_data)
df.columns = df.columns.str.strip().str.lower()
df["код"] = df["код"].astype(str).str.strip().str.lower()
df["image"] = df["image"].astype(str).str.strip() if "image" in df.columns else ""

# Состояние
user_state = {}
search_count = {}

if os.path.exists("state.pkl"):
    with open("state.pkl", "rb") as f:
        user_state = pickle.load(f)

def normalize(text: str) -> str:
    return re.sub(r'[\W_]+', '', text.lower())

# 🔍 Поиск фото по совпадению кода в ссылке
def find_image_url_by_code(code: str) -> str:
    code = code.lower()
    for url in df["image"]:
        if code in url.lower():
            return url
    return ""

def format_row(row):
    return (
        f"🔹 Тип: {row.get('тип', '')}\n"
        f"📦 Наименование: {row.get('наименование', '')}\n"
        f"🔢 Код: {row.get('код', '')}\n"
        f"📦 Кол-во: {row.get('количество', '')}\n"
        f"💰 Цена: {row.get('цена', '')} {row.get('валюта', '')}\n"
        f"🏭 Изготовитель: {row.get('изготовитель', '')}\n"
        f"⚙️ OEM: {row.get('oem', '')}"
    )

# Фото + текст
async def send_row_with_image(update: Update, row, text: str):
    code = row.get("код", "").strip().lower()
    image_url = find_image_url_by_code(code)
    try:
        if image_url:
            await update.message.reply_photo(photo=image_url, caption=text[:1024])
        else:
            await update.message.reply_text(text)
    except Exception as e:
        logger.error(f"[Ошибка при отправке фото] {e}")
        await update.message.reply_text(text)

# Команды
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_state.pop(user_id, None)
    await update.message.reply_text("Привет! Напиши ключевое слово, и я найду нужную деталь.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📘 Команды:\n"
        "/start — сброс поиска\n"
        "/more — показать ещё\n"
        "/help — справка\n"
        "/export — экспорт\n"
        "/stats — статистика"
    )

async def more(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = user_state.get(user_id)
    if not state:
        await update.message.reply_text("Сначала выполните поиск.")
        return
    query, offset, results = state["query"], state["offset"], state["results"]
    page = results.iloc[offset: offset + 5]
    for _, row in page.iterrows():
        text = format_row(row)
        await send_row_with_image(update, row, text)
    offset += 5
    user_state[user_id]["offset"] = offset
    if offset < len(results):
        await update.message.reply_text("Напишите /more для следующих.")
    else:
        await update.message.reply_text("Это все результаты.")

async def export(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    count = search_count.get(user_id, 0)
    await update.message.reply_text(f"🔍 Поисков за сессию: {count}")

# Поиск
async def search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    query = update.message.text.strip().lower()
    norm_query = normalize(query)

    mask = df.apply(
        lambda row: any(norm_query in normalize(str(value)) for value in [
            row.get("тип", ""), row.get("наименование", ""), row.get("код", ""),
            row.get("oem", ""), row.get("изготовитель", "")
        ]),
        axis=1
    )
    results = df[mask]

    if results.empty:
        await update.message.reply_text(f'По запросу "{query}" ничего не найдено.')
        return

    search_count[user_id] = search_count.get(user_id, 0) + 1
    user_state[user_id] = {"query": query, "offset": 5, "results": results}

    for _, row in results.head(5).iterrows():
        text = format_row(row)
        await send_row_with_image(update, row, text)

    if len(results) > 5:
        await update.message.reply_text("Показаны первые 5. Напишите /more для продолжения.")

# Глобальная ошибка
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Произошла ошибка:", exc_info=context.error)
    if isinstance(update, Update) and update.message:
        await update.message.reply_text("❌ Внутренняя ошибка. Попробуйте снова позже.")

# Запуск
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("more", more))
    app.add_handler(CommandHandler("export", export))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search))
    app.add_error_handler(error_handler)
    logger.info("Бот запущен")
    app.run_polling()
    with open("state.pkl", "wb") as f:
        pickle.dump(user_state, f)

if __name__ == "__main__":
    main()
