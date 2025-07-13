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

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Админы
ADMINS = {225177765}  # ← сюда добавьте свой Telegram user_id

# Токен и Google Sheets настройки
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
SPREADSHEET_URL = os.getenv("SPREADSHEET_URL")
SHEET_NAME = "SAP"

# Загрузка данных из Google Sheets
def load_data():
    creds_info = json.loads(os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"))
    creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    client = gspread.authorize(creds)
    sheet = client.open_by_url(SPREADSHEET_URL).worksheet(SHEET_NAME)
    return sheet.get_all_records()

raw_data = load_data()

from pandas import DataFrame

df = DataFrame(raw_data)
df.columns = df.columns.str.strip().str.lower()
df["код"] = df["код"].astype(str).str.strip().str.lower()

# Состояние пользователя
user_state = {}
search_count = {}

if os.path.exists("state.pkl"):
    with open("state.pkl", "rb") as f:
        user_state = pickle.load(f)

def normalize(text: str) -> str:
    return re.sub(r'[\W_]+', '', text.lower())

def find_image_url_by_code(code: str, df) -> str:
    code = code.lower().strip()
    for url in df["image"].dropna():
        if code in url.lower():
            return url
    return None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_state.pop(user_id, None)
    await update.message.reply_text("Привет! История поиска очищена.\nОтправь тип, код или наименование детали.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "\ud83d\udcd8 Команды:\n"
        "/start — сброс поиска\n"
        "/more — показать ещё\n"
        "/help — справка\n"
        "/export — экспорт результатов\n"
        "/stats — сколько раз искали\n"
        "Просто отправьте текст — для поиска по типу, коду, OEM, названию или изготовителю."
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
        await update.message.reply_text("Напишите /more для следующих результатов.")
    else:
        await update.message.reply_text("Больше результатов нет.")

def format_row(row):
    return (
        f"\ud83d\udd39 Тип: {row['тип']}\n"
        f"\ud83d\udce6 Наименование: {row['наименование']}\n"
        f"\ud83d\udcc2 Код: {row['код']}\n"
        f"\ud83d\udce6 Кол-во: {row['количество']}\n"
        f"\ud83d\udcb0 Цена: {row['цена']} {row['валюта']}\n"
        f"\ud83c\udfe0 Изготовитель: {row['изготовитель']}\n"
        f"\u2699\ufe0f OEM: {row['oem']}"
    )

async def send_row_with_image(update: Update, row, text: str):
    code = str(row["код"]).strip().lower()
    image_url = find_image_url_by_code(code, df)
    if image_url:
        try:
            await update.message.reply_photo(photo=image_url, caption=text[:1024])
        except Exception as e:
            logger.warning(f"Ошибка при отправке фото: {e}")
            await update.message.reply_text(text)
    else:
        await update.message.reply_text(text)

async def export(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from pandas import ExcelWriter
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
    await update.message.reply_text(f"\ud83d\udd0d Вы сделали {count} поисков за сессию.")

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
    user_state[user_id] = {
        "query": query,
        "offset": 5,
        "results": results
    }

    for _, row in results.head(5).iterrows():
        text = format_row(row)
        await send_row_with_image(update, row, text)

    if len(results) > 5:
        await update.message.reply_text("Показано 5 первых результатов. Напишите /more для продолжения.")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Ошибка", exc_info=context.error)
    if isinstance(update, Update) and update.message:
        await update.message.reply_text("Произошла ошибка. Попробуйте позже.")

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
