import logging
import os
import pickle
import json
import gspread
import re
from datetime import datetime
from google.oauth2.service_account import Credentials
from telegram import (
    Update,
    InputFile,
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
from pandas import DataFrame

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Этапы диалога списания
ASK_QUANTITY, ASK_COMMENT = range(2)
issue_state = {}

# Админы и разрешённые пользователи
ADMINS = {225177765}

# Переменные окружения
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
SPREADSHEET_URL = os.getenv("SPREADSHEET_URL")
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# Google Sheets подключение
def get_gsheet():
    creds_info = json.loads(os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"))
    creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    client = gspread.authorize(creds)
    return client.open_by_url(SPREADSHEET_URL)

# Загрузка данных
def load_data():
    sheet = get_gsheet().worksheet("SAP")
    return sheet.get_all_records()

# Сохранение истории
def save_issue_to_sheet(user, part, quantity, comment):
    sheet = get_gsheet().worksheet("История")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheet.append_row([
        now,
        user.id,
        user.full_name,
        part.get("тип", ""),
        part.get("наименование", ""),
        part.get("код", ""),
        quantity,
        comment
    ])

# Команда /adduser — добавляет user_id в таблицу "Пользователи"
async def add_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in ADMINS:
        await update.message.reply_text("⛔ У вас нет прав на выполнение этой команды.")
        return

    try:
        new_user_id = int(context.args[0])
    except (IndexError, ValueError):
        await update.message.reply_text("⚠ Использование: /adduser 123456789")
        return

    sheet = get_gsheet().worksheet("Пользователи")
    existing_ids = [row[0] for row in sheet.get_all_values()]
    if str(new_user_id) in existing_ids:
        await update.message.reply_text("✅ Пользователь уже есть в списке.")
    else:
        sheet.append_row([str(new_user_id)])
        await update.message.reply_text(f"✅ Добавлен user_id: {new_user_id}")

# Инициализация данных
raw_data = load_data()
df = DataFrame(raw_data)
df.columns = df.columns.str.strip().str.lower()
df["код"] = df["код"].astype(str).str.strip().str.lower()

# Состояние пользователя
user_state = {}
search_count = {}
if os.path.exists("state.pkl"):
    with open("state.pkl", "rb") as f:
        user_state = pickle.load(f)

# Нормализация текста
def normalize(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text.lower()).strip()

# Поиск изображения по коду
def find_image_url_by_code(code: str) -> str:
    code_norm = normalize(code)
    for url in df["image"]:
        if code_norm in normalize(str(url)):
            return url
    return ""

# Форматирование строки результата
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

# Отправка строки с изображением и кнопкой
async def send_row_with_image(update: Update, row, text: str):
    code = str(row.get("код", ""))
    image_url = find_image_url_by_code(code)
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("📦 Взять деталь", callback_data=f"issue:{code}")]
    ])
    if image_url:
        try:
            await update.message.reply_photo(photo=image_url, caption=text[:1024], reply_markup=keyboard)
        except Exception as e:
            logger.warning(f"Ошибка при отправке фото: {e}")
            await update.message.reply_text(text, reply_markup=keyboard)
    else:
        await update.message.reply_text(text, reply_markup=keyboard)

# Поиск
async def search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    query = update.message.text.strip().lower()
    norm_query = normalize(query)

    if not norm_query:
        await update.message.reply_text("Введите более осмысленный запрос.")
        return

    def matches(value: str, query: str) -> bool:
        return query in value

    mask = df.apply(
        lambda row: any(
            matches(normalize(str(value)), norm_query)
            for value in [row.get("тип", ""), row.get("наименование", ""), row.get("код", ""),
                          row.get("oem", ""), row.get("изготовитель", "")]
        ),
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

# Списание: кнопка → кол-во → комментарий
async def handle_issue_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    code = query.data.split(":")[1]
    user_id = query.from_user.id

    part = df[df["код"] == code].to_dict(orient="records")
    if not part:
        await query.edit_message_text("❗ Деталь не найдена.")
        return ConversationHandler.END

    issue_state[user_id] = {"part": part[0]}
    await query.message.reply_text("Введите количество:")
    return ASK_QUANTITY

async def handle_quantity(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text.strip()

    if not text.isdigit():
        await update.message.reply_text("Введите число.")
        return ASK_QUANTITY

    issue_state[user_id]["quantity"] = int(text)
    await update.message.reply_text("Введите комментарий (или напишите 'нет'):")
    return ASK_COMMENT

async def handle_comment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    comment = update.message.text.strip()
    data = issue_state.pop(user_id, {})

    part = data.get("part")
    quantity = data.get("quantity")

    if part and quantity:
        save_issue_to_sheet(user, part, quantity, comment)
        await update.message.reply_text("✅ Списание выполнено.")
    else:
        await update.message.reply_text("⚠ Что-то пошло не так.")
    return ConversationHandler.END

# Обработка ошибок
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Ошибка", exc_info=context.error)
    if isinstance(update, Update) and update.message:
        await update.message.reply_text("Произошла ошибка. Попробуйте позже.")

# Основной запуск
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Команда добавления пользователя
    app.add_handler(CommandHandler("adduser", add_user))

    conv_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(handle_issue_button, pattern=r"^issue:")],
        states={
            ASK_QUANTITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_quantity)],
            ASK_COMMENT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_comment)],
        },
        fallbacks=[],
    )

    app.add_handler(conv_handler)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search))
    app.add_error_handler(error_handler)
    logger.info("Бот запущен")
    app.run_polling()

    with open("state.pkl", "wb") as f:
        pickle.dump(user_state, f)

if __name__ == "__main__":
    main()
