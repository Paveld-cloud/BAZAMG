import logging
import os
import pickle
import json
import gspread
import re
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+
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
import atexit
import signal

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Этапы диалога списания
ASK_QUANTITY, ASK_COMMENT = range(2)

# Глобальные состояния
user_state = {}          # Состояние поиска
issue_state = {}         # Состояние списания
search_count = {}        # Счётчик поисков

# Админы
ADMINS = {225177765}

# Переменные окружения
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
SPREADSHEET_URL = os.getenv("SPREADSHEET_URL")
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

if not TELEGRAM_TOKEN or not SPREADSHEET_URL:
    raise EnvironmentError("Отсутствуют TELEGRAM_TOKEN или SPREADSHEET_URL в переменных окружения")

# Google Sheets подключение
def get_gsheet():
    creds_info = json.loads(os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"))
    creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    client = gspread.authorize(creds)
    return client.open_by_url(SPREADSHEET_URL)

# --- Кэширование пользователей ---
_allowed_users = None
_last_users_update = 0

def get_allowed_users():
    global _allowed_users, _last_users_update
    now = datetime.now().timestamp()
    if _allowed_users is None or now - _last_users_update > 300:
        try:
            sheet = get_gsheet().worksheet("Пользователи")
            rows = sheet.get_all_values()
            _allowed_users = {int(row[0]) for row in rows if row and len(row) > 0 and row[0].strip().isdigit()}
        except Exception as e:
            logger.error(f"Ошибка загрузки пользователей: {e}")
            _allowed_users = set()
        _last_users_update = now
    return _allowed_users

# Загрузка данных
def load_data():
    try:
        sheet = get_gsheet().worksheet("SAP")
        return sheet.get_all_records()
    except Exception as e:
        logger.error(f"Ошибка загрузки данных SAP: {e}")
        return []

# Асинхронное сохранение списания
async def save_issue_to_sheet(context: ContextTypes.DEFAULT_TYPE, user, part, quantity, comment):
    try:
        # 🔹 Время по Ташкенту
        tz = ZoneInfo("Asia/Tashkent")
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

        sheet = get_gsheet().worksheet("История")
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
    except Exception as e:
        logger.error(f"Ошибка при сохранении списания: {e}")
        for admin_id in ADMINS:
            try:
                await context.bot.send_message(admin_id, f"⚠️ Ошибка сохранения списания: {e}")
            except Exception as send_err:
                logger.error(f"Не удалось отправить админу {admin_id}: {send_err}")

# Инициализация данных
raw_data = load_data()
df = DataFrame(raw_data)
df.columns = df.columns.str.strip().str.lower()
df["код"] = df["код"].astype(str).str.strip().str.lower()

# --- 🔍 Поиск изображения: код содержится в URL (как в оригинальном коде) ---
def find_image_url_by_code(code: str) -> str:
    """
    Ищет в столбце 'image' URL, содержащий код детали.
    Пример: код 'uzcss06503' → найдёт URL, в котором встречается этот код.
    """
    code_norm = re.sub(r'[^\w\s]', '', code.lower().strip())
    image_col = df["image"].astype(str)
    for url in image_col[image_col != "nan"]:
        url_norm = re.sub(r'[^\w\s]', '', url.lower().strip())
        if code_norm in url_norm:
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
    if not update.message:
        return

    code = str(row.get("код", ""))
    image_url = find_image_url_by_code(code)
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("📦 Взять деталь", callback_data=f"issue:{code}")]
    ])

    caption = text[:1021] + "..." if len(text) > 1024 else text

    if image_url:
        try:
            await update.message.reply_photo(photo=image_url, caption=caption, reply_markup=keyboard)
            return
        except Exception as e:
            logger.warning(f"Ошибка при отправке фото: {e}")

    await update.message.reply_text(text, reply_markup=keyboard)

# --- Команды ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_state.pop(user_id, None)
    search_count.pop(user_id, None)
    await update.message.reply_text("Привет! Отправь название, код или OEM детали.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🔍 Бот для поиска и списания деталей\n\n"
        "📌 Команды:\n"
        "/start — начать\n"
        "/help — справка\n"
        "/cancel — отменить списание\n\n"
        "Админ:\n"
        "/adduser 123456789 — добавить пользователя"
    )

async def add_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in ADMINS:
        await update.message.reply_text("⛔ У вас нет прав на выполнение этой команды.")
        return

    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("⚠ Использование: /adduser 123456789")
        return

    new_user_id = int(context.args[0])
    if new_user_id <= 0:
        await update.message.reply_text("⚠ Некорректный user_id.")
        return

    try:
        sheet = get_gsheet().worksheet("Пользователи")
        existing = {int(row[0]) for row in sheet.get_all_values() if row and row[0].isdigit()}
        if new_user_id in existing:
            await update.message.reply_text("✅ Пользователь уже в списке.")
        else:
            sheet.append_row([str(new_user_id)])
            get_allowed_users()  # Обновить кэш
            await update.message.reply_text(f"✅ Пользователь {new_user_id} добавлен.")
    except Exception as e:
        logger.error(f"Ошибка добавления пользователя: {e}")
        await update.message.reply_text("❌ Ошибка при добавлении пользователя.")

# --- Поиск ---
async def search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    if user_id not in get_allowed_users():
        await update.message.reply_text("⛔ У вас нет доступа к этому боту.")
        return

    query = update.message.text.strip()
    if not query:
        await update.message.reply_text("Введите запрос.")
        return

    norm_query = re.sub(r'[^\w\s]', '', query.lower())

    mask = (
        df["тип"].astype(str).apply(lambda x: norm_query in re.sub(r'[^\w\s]', '', x.lower())) |
        df["наименование"].astype(str).apply(lambda x: norm_query in re.sub(r'[^\w\s]', '', x.lower())) |
        df["код"].astype(str).apply(lambda x: norm_query in re.sub(r'[^\w\s]', '', x.lower())) |
        df["oem"].astype(str).apply(lambda x: norm_query in re.sub(r'[^\w\s]', '', x.lower())) |
        df["изготовитель"].astype(str).apply(lambda x: norm_query in re.sub(r'[^\w\s]', '', x.lower()))
    )

    results = df[mask].copy()

    if results.empty:
        await update.message.reply_text(f'❌ По запросу "{query}" ничего не найдено.')
        return

    user_state[user_id] = {
        "query": query,
        "offset": 5,
        "results": results
    }

    for _, row in results.head(5).iterrows():
        text = format_row(row)
        await send_row_with_image(update, row, text)

    if len(results) > 5:
        await update.message.reply_text("Показано 5 результатов. Напишите /more для продолжения.")

# --- Списание ---
async def handle_issue_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id

    logger.info(f"Получен callback: {query.data} от user_id={user_id}")

    if user_id not in get_allowed_users():
        await query.answer("⛔ Доступ запрещён", show_alert=True)
        return ConversationHandler.END

    await query.answer()

    try:
        code = query.data.split(":", 1)[1]
    except IndexError:
        await query.message.reply_text("⚠ Ошибка: неверный код детали.")
        return ConversationHandler.END

    part = df[df["код"] == code.lower().strip()].to_dict(orient="records")
    if not part:
        await query.edit_message_text("❗ Деталь не найдена.")
        return ConversationHandler.END

    issue_state[user_id] = {"part": part[0]}
    await query.message.reply_text("🔢 Введите количество:")
    return ASK_QUANTITY

async def handle_quantity(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text.strip()

    if not text.isdigit() or int(text) <= 0:
        await update.message.reply_text("Введите положительное число.")
        return ASK_QUANTITY

    issue_state[user_id]["quantity"] = int(text)
    await update.message.reply_text("💬 Введите комментарий (или 'нет'):")
    return ASK_COMMENT

async def handle_comment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    comment = update.message.text.strip()
    data = issue_state.pop(user_id, {})

    part = data.get("part")
    quantity = data.get("quantity")

    logger.info(f"Сохранение списания: код={part.get('код')}, кол-во={quantity}, коммент={comment}")

    if part and quantity:
        await save_issue_to_sheet(context, user, part, quantity, comment)
        await update.message.reply_text("✅ Списание выполнено.")
    else:
        await update.message.reply_text("⚠ Ошибка при списании.")
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    issue_state.pop(user_id, None)
    await update.message.reply_text("❌ Списание отменено.")
    return ConversationHandler.END

# --- Обработка ошибок ---
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Произошла ошибка", exc_info=context.error)
    if isinstance(update, Update) and update.message:
        await update.message.reply_text("⚠ Произошла ошибка. Попробуйте позже.")

# --- Основной запуск ---
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Команды
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("adduser", add_user))
    app.add_handler(CommandHandler("cancel", cancel))

    # Списание
    conv_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(handle_issue_button, pattern=r"^issue:")],
        states={
            ASK_QUANTITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_quantity)],
            ASK_COMMENT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_comment)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        per_message=True  # Защита от старых кнопок
    )
    app.add_handler(conv_handler)

    # Поиск (важно: после ConversationHandler!)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search))

    # Ошибки
    app.add_error_handler(error_handler)

    # Сохранение состояния при выходе
    def save_state():
        with open("state.pkl", "wb") as f:
            pickle.dump(user_state, f)
        logger.info("Состояние сохранено")

    atexit.register(save_state)
    signal.signal(signal.SIGINT, lambda s, f: (save_state(), exit(0)))
    signal.signal(signal.SIGTERM, lambda s, f: (save_state(), exit(0)))

    logger.info("Бот запущен")
    app.run_polling()

if __name__ == "__main__":
    main()
