import logging
import os
import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo
import gspread
from google.oauth2.service_account import Credentials
from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Состояния диалога списания
ASK_QUANTITY, ASK_COMMENT, ASK_CONFIRM = range(3)

# Глобальные состояния
user_state = {}
issue_state = {}  # {user_id: {"part": ..., "quantity": ..., "comment": ...}}

# Переменные окружения
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
SPREADSHEET_URL = os.getenv("SPREADSHEET_URL")
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
TIMEZONE = ZoneInfo("Asia/Tashkent")

# Авторизация Google Sheets
def get_gs_client():
    creds_info = json.loads(os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"))
    creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    return gspread.authorize(creds)

# Загрузка данных
def load_data():
    client = get_gs_client()
    sheet = client.open_by_url(SPREADSHEET_URL).worksheet("SAP")
    return sheet.get_all_records()

DATA_CACHE = load_data()
logger.info(f"✅ Загружено {len(DATA_CACHE)} строк из Google Sheet")

# ====== Поиск ======
def search_parts(query):
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    return [row for row in DATA_CACHE if any(pattern.search(str(v)) for v in row.values())]

# ====== Команды ======
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Введите название детали для поиска")

async def search_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text.strip()
    results = search_parts(query)

    if not results:
        await update.message.reply_text("❌ Ничего не найдено")
        return

    for item in results[:10]:  # Ограничим вывод
        text = f"📦 {item.get('PartName')}\nКод: {item.get('PartCode')}"
        keyboard = [[InlineKeyboardButton("📤 Взять деталь", callback_data=f"take|{item.get('PartName')}")]]
        await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

# ====== Списание ======
async def take_part_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    _, part_name = query.data.split("|", 1)
    user_state[query.from_user.id] = {"part": part_name}

    await query.message.reply_text(f"Введите количество для списания ({part_name}):")
    return ASK_QUANTITY

async def ask_comment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    qty = update.message.text.strip()
    if not qty.isdigit():
        await update.message.reply_text("Введите число!")
        return ASK_QUANTITY

    user_state[update.message.from_user.id]["quantity"] = qty
    await update.message.reply_text("Введите комментарий:")
    return ASK_COMMENT

async def confirm_writeoff(update: Update, context: ContextTypes.DEFAULT_TYPE):
    comment = update.message.text.strip()
    user_id = update.message.from_user.id

    user_state[user_id]["comment"] = comment

    part = user_state[user_id]["part"]
    qty = user_state[user_id]["quantity"]

    # Кнопки подтверждения
    keyboard = [
        [
            InlineKeyboardButton("✅ Да", callback_data="confirm_yes"),
            InlineKeyboardButton("❌ Нет", callback_data="confirm_no"),
        ]
    ]
    await update.message.reply_text(
        f"Вы уверены, что хотите списать:\n\n📦 {part}\nКоличество: {qty}\nКомментарий: {comment}",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return ASK_CONFIRM

async def confirm_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    if query.data == "confirm_yes":
        # Запись в Google Sheets
        client = get_gs_client()
        sheet = client.open_by_url(SPREADSHEET_URL).worksheet("История")
        now = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")

        sheet.append_row([
            str(user_id),
            query.from_user.first_name,
            user_state[user_id]["part"],
            user_state[user_id]["quantity"],
            user_state[user_id]["comment"],
            now
        ])
        await query.message.reply_text("✅ Деталь успешно списана")
    else:
        await query.message.reply_text("❌ Списание отменено")

    user_state.pop(user_id, None)
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_state.pop(update.message.from_user.id, None)
    await update.message.reply_text("❌ Действие отменено")
    return ConversationHandler.END

# ====== Построение приложения ======
def build_app():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(take_part_callback, pattern="^take\\|")],
        states={
            ASK_QUANTITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_comment)],
            ASK_COMMENT: [MessageHandler(filters.TEXT & ~filters.COMMAND, confirm_writeoff)],
            ASK_CONFIRM: [CallbackQueryHandler(confirm_callback, pattern="^confirm_")],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        map_to_parent={ConversationHandler.END: ConversationHandler.END},
    )

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search_handler))
    app.add_handler(conv)

    return app

if __name__ == "__main__":
    application = build_app()
    application.run_webhook(
        listen="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        url_path="webhook",
        webhook_url=f"{os.getenv('WEBHOOK_URL')}/webhook"
    )
