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

# Список разрешённых пользователей (загружается/сохраняется)
ALLOWED_USERS_FILE = "allowed_users.pkl"
if os.path.exists(ALLOWED_USERS_FILE):
    with open(ALLOWED_USERS_FILE, "rb") as f:
        allowed_users = pickle.load(f)
else:
    allowed_users = set()

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

def is_allowed(user_id: int) -> bool:
    return user_id in ADMINS or user_id in allowed_users

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_allowed(user_id):
        await update.message.reply_text("⛔️ У вас нет доступа к боту.")
        return
    user_state.pop(user_id, None)
    await update.message.reply_text("Привет! История поиска очищена.\nОтправь тип, код или наименование детали.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        await update.message.reply_text("⛔️ У вас нет доступа к боту.")
        return
    await update.message.reply_text(
        "📘 Команды:\n"
        "/start — сброс поиска\n"
        "/more — показать ещё\n"
        "/help — справка\n"
        "/export — экспорт результатов\n"
        "/stats — сколько раз искали\n"
        "/add_user <id> — добавить пользователя\n"
        "/list_users — список пользователей\n"
        "Просто отправьте текст — для поиска по типу, коду, OEM, названию или изготовителю."
    )

async def add_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMINS:
        await update.message.reply_text("⛔️ Только админ может добавлять пользователей.")
        return
    if not context.args:
        await update.message.reply_text("Укажите user_id: /add_user 123456789")
        return
    try:
        user_id = int(context.args[0])
        allowed_users.add(user_id)
        with open(ALLOWED_USERS_FILE, "wb") as f:
            pickle.dump(allowed_users, f)
        await update.message.reply_text(f"✅ Пользователь {user_id} добавлен.")
    except ValueError:
        await update.message.reply_text("❌ Неверный формат user_id.")

async def list_users(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMINS:
        await update.message.reply_text("⛔️ Только админ может просматривать список пользователей.")
        return
    if not allowed_users:
        await update.message.reply_text("Нет добавленных пользователей.")
    else:
        users = "\n".join(str(uid) for uid in allowed_users)
        await update.message.reply_text(f"👥 Разрешённые пользователи:\n{users}")

async def more(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_allowed(user_id):
        await update.message.reply_text("⛔️ У вас нет доступа к боту.")
        return
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
        f"🔹 Тип: {row['тип']}\n"
        f"📦 Наименование: {row['наименование']}\n"
        f"🔢 Код: {row['код']}\n"
        f"📦 Кол-во: {row['количество']}\n"
        f"💰 Цена: {row['цена']} {row['валюта']}\n"
        f"🏭 Изготовитель: {row['изготовитель']}\n"
        f"⚙️ OEM: {row['oem']}"
    )

async def send_row_with_image(update: Update, row, text: str):
    code = row.get("код", "").upper()
    image_url = f"https://i.ibb.co/{code}.jpg"  # или другой шаблон, если известен
    try:
        await update.message.reply_photo(photo=image_url, caption=text[:1024])
    except:
        await update.message.reply_text(text)

async def export(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_allowed(user_id):
        await update.message.reply_text("⛔️ У вас нет доступа к боту.")
        return
    from pandas import ExcelWriter
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
    if not is_allowed(user_id):
        await update.message.reply_text("⛔️ У вас нет доступа к боту.")
        return
    count = search_count.get(user_id, 0)
    await update.message.reply_text(f"🔍 Вы сделали {count} поисков за сессию.")

async def search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_allowed(user_id):
        await update.message.reply_text("⛔️ У вас нет доступа к боту.")
        return
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
    app.add_handler(CommandHandler("add_user", add_user))
    app.add_handler(CommandHandler("list_users", list_users))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search))
    app.add_error_handler(error_handler)
    logger.info("Бот запущен")
    app.run_polling()

    with open("state.pkl", "wb") as f:
        pickle.dump(user_state, f)

if __name__ == "__main__":
    main()
