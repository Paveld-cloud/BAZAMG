import logging
import os
import pickle
import json
import gspread
from google.oauth2.service_account import Credentials
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)

# 🔒 Админы
ADMINS = {123456789, 987654321}  # ← сюда добавь свой user_id

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Telegram Token и Google Sheets настройки
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

# Обработка и нормализация
from pandas import DataFrame
df = DataFrame(raw_data)
df.columns = df.columns.str.strip().str.lower()
df["код"] = df["код"].astype(str).str.strip().str.lower()
df["наименование"] = df["наименование"].astype(str).str.strip().str.lower()

# Состояния
user_state = {}
search_count = {}
users_set = set()

# Загрузка предыдущих сессий
if os.path.exists("state.pkl"):
    with open("state.pkl", "rb") as f:
        user_state = pickle.load(f)

if os.path.exists("users.pkl"):
    with open("users.pkl", "rb") as f:
        users_set = pickle.load(f)

# Генерация кнопок
def generate_inline_keyboard(code: str):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🧾 История", callback_data=f"history|{code}"),
         InlineKeyboardButton("🧰 Уход", callback_data=f"care|{code}")],
        [InlineKeyboardButton("📄 Описание", callback_data=f"description|{code}"),
         InlineKeyboardButton("🎥 Видео", callback_data=f"video|{code}")]
    ])

# Команды
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    users_set.add(user_id)
    user_state.pop(user_id, None)
    await update.message.reply_text("Привет! Отправь код, OEM или наименование детали для поиска.")

async def more(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = user_state.get(user_id)
    if not state:
        await update.message.reply_text("Сначала выполните поиск.")
        return

    query, offset, results = state["query"], state["offset"], state["results"]
    page = results.iloc[offset: offset + 5]
    for _, row in page.iterrows():
        await update.message.reply_text(
            format_result(row),
            reply_markup=generate_inline_keyboard(str(row["код"]))
        )

    user_state[user_id]["offset"] += 5
    if user_state[user_id]["offset"] < len(results):
        await update.message.reply_text("Напишите /more для следующих результатов.")
    else:
        await update.message.reply_text("Больше результатов нет.")

def format_result(row):
    return (
        f"📦 Тип: {row['тип']}\n"
        f"🏷️ Наименование: {row['наименование']}\n"
        f"🆔 Код: {row['код']}\n"
        f"📦 Кол-во: {row['количество']}\n"
        f"💵 Цена: {row['цена']} {row['валюта']}\n"
        f"🏭 Изготовитель: {row['изготовитель']}\n"
        f"⚙️ OEM: {row['oem']}"
    )

async def search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    users_set.add(user_id)
    query = update.message.text.strip().lower()

    mask = (
        df["тип"].str.contains(query, na=False) |
        df["наименование"].str.contains(query, na=False) |
        df["код"].str.contains(query, na=False) |
        df["oem"].astype(str).str.contains(query, na=False) |
        df["изготовитель"].str.contains(query, na=False)
    )
    results = df[mask]

    if results.empty:
        await update.message.reply_text(f"По запросу «{query}» ничего не найдено.")
        return

    user_state[user_id] = {"query": query, "offset": 5, "results": results}
    search_count[user_id] = search_count.get(user_id, 0) + 1

    for _, row in results.head(5).iterrows():
        await update.message.reply_text(format_result(row), reply_markup=generate_inline_keyboard(str(row["код"])))

    if len(results) > 5:
        await update.message.reply_text("Показано 5 первых результатов. Напишите /more для продолжения.")

# Обработка кнопок
async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    action, code = query.data.split("|", 1)

    text_map = {
        "history": f"🧾 История детали: {code}",
        "care": f"🧰 Уход за деталью: {code}",
        "description": f"📄 Описание детали: {code}",
        "video": f"🎥 Видеообзор: {code}"
    }
    await query.message.reply_text(text_map.get(action, "Неизвестное действие."))

# 👑 Админ-панель
async def admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in ADMINS:
        await update.message.reply_text("⛔ Доступ запрещён.")
        return

    text = (
        "🔧 Админ-панель:\n"
        f"👥 Пользователей: {len(users_set)}\n"
        f"🔍 Всего поисков: {sum(search_count.values())}\n"
        "📤 Напишите /broadcast <сообщение> для рассылки"
    )
    await update.message.reply_text(text)

async def broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in ADMINS:
        await update.message.reply_text("⛔ Доступ запрещён.")
        return

    message = " ".join(context.args)
    count = 0
    for uid in users_set:
        try:
            await context.bot.send_message(chat_id=uid, text=f"📢 {message}")
            count += 1
        except:
            continue
    await update.message.reply_text(f"✅ Разослано {count} пользователям.")

# Ошибки
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Ошибка", exc_info=context.error)

# Запуск
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("more", more))
    app.add_handler(CommandHandler("admin", admin))
    app.add_handler(CommandHandler("broadcast", broadcast))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_error_handler(error_handler)

    app.run_polling()

    # Сохраняем состояние
    with open("state.pkl", "wb") as f:
        pickle.dump(user_state, f)
    with open("users.pkl", "wb") as f:
        pickle.dump(users_set, f)

if __name__ == "__main__":
    main()



