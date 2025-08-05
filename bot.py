import logging
import os
import pickle
import json
import gspread
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from google.oauth2.service_account import Credentials
from telegram import (
    Update,
    InputFile,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardMarkup,
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
user_state = {}
issue_state = {}  # {user_id: {"part": ..., "quantity": ...}}
search_count = {}

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
        logger.info(f"✅ Списание сохранено: {part['код']} x{quantity}")
    except Exception as e:
        logger.error(f"❌ Ошибка при сохранении списания: {e}")
        for admin_id in ADMINS:
            try:
                await context.bot.send_message(admin_id, f"⚠️ Ошибка сохранения списания: {e}")
            except Exception as send_err:
                logger.error(f"Не удалось отправить админу {admin_id}: {send_err}")

# Инициализация данных
raw_data = load_data()
df = DataFrame(raw_data)
if df.empty:
    logger.error("⚠️ Таблица SAP пуста или не загружена")
else:
    df.columns = df.columns.str.strip().str.lower()
    df["код"] = df["код"].astype(str).str.strip().str.lower()

# --- Поиск изображения: код содержится в URL ---
def find_image_url_by_code(code: str) -> str:
    if df.empty:
        return ""
    code_norm = re.sub(r'[^\w\s]', '', code.lower().strip())
    image_col = df["image"].dropna().astype(str)
    for url in image_col:
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
    if not update.message or df.empty:
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

# --- Клавиатура меню ---
def get_main_menu():
    keyboard = [
        ["🔍 Поиск детали", "📦 Взять деталь"],
        ["📊 Мои списания", "❓ Помощь"]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)

# --- Команды ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_state.pop(user_id, None)
    search_count.pop(user_id, None)
    await update.message.reply_text(
        f"Привет, {update.effective_user.first_name}! 👋\n"
        "Выберите действие в меню ниже:",
        reply_markup=get_main_menu()
    )

async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Выберите действие:",
        reply_markup=get_main_menu()
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📌 Основное меню:\n"
        "🔍 Поиск детали — найдите по коду, названию и т.д.\n"
        "📦 Взять деталь — только после поиска\n"
        "📊 Мои списания — ваши операции\n"
        "❓ Помощь — это сообщение"
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
            get_allowed_users()
            await update.message.reply_text(f"✅ Пользователь {new_user_id} добавлен.")
    except Exception as e:
        logger.error(f"Ошибка добавления пользователя: {e}")
        await update.message.reply_text("❌ Ошибка при добавлении пользователя.")

# --- Обработка кнопок меню (ТОЛЬКО меню, НЕ мешает диалогу) ---
async def handle_menu_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    user_id = update.effective_user.id

    # 🔁 Если идёт списание — НЕ обрабатываем меню
    if user_id in issue_state:
        return

    if user_id not in get_allowed_users():
        await update.message.reply_text("⛔ У вас нет доступа.")
        return

    if text == "🔍 Поиск детали":
        await update.message.reply_text("Введите код, тип или название детали:")

    elif text == "📦 Взять деталь":
        await update.message.reply_text("Найдите деталь через поиск, затем нажмите 'Взять деталь' под карточкой.")

    elif text == "📊 Мои списания":
        await update.message.reply_text("Пока в разработке. Скоро!")

    elif text == "❓ Помощь":
        await help_command(update, context)

# --- Поиск ---
async def search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if user_id not in get_allowed_users():
        await update.message.reply_text("⛔ У вас нет доступа.")
        return

    if df.empty:
        await update.message.reply_text("⚠ Данные не загружены.")
        return

    query = update.message.text.strip()
    if not query:
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
        await send_row_with_image(update, row, format_row(row))

    if len(results) > 5:
        await update.message.reply_text("Показано 5 результатов. Напишите ещё, чтобы увидеть больше.")

# --- Списание ---
async def handle_issue_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id

    if user_id not in get_allowed_users():
        await query.answer("⛔ Доступ запрещён", show_alert=True)
        return ConversationHandler.END

    await query.answer()

    try:
        code = query.data.split(":", 1)[1]
    except IndexError:
        await query.message.reply_text("⚠ Ошибка: неверный код детали.")
        return ConversationHandler.END

    if df.empty:
        await query.message.reply_text("⚠ Данные не загружены.")
        return ConversationHandler.END

    part = df[df["код"] == code.lower().strip()].to_dict(orient="records")
    if not part:
        await query.edit_message_text("❗ Деталь не найдена.")
        return ConversationHandler.END

    global issue_state
    issue_state[user_id] = {"part": part[0]}
    await query.message.reply_text("🔢 Введите количество:")
    return ASK_QUANTITY

async def handle_quantity(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global issue_state
    user_id = update.effective_user.id
    text = update.message.text.strip()

    # 🔽 Лог для отладки
    logger.info(f"📝 handle_quantity вызван. user_id={user_id}, ввод: '{text}'")

    if not text.isdigit() or int(text) <= 0:
        await update.message.reply_text("Введите положительное число.")
        return ASK_QUANTITY

    issue_state[user_id]["quantity"] = int(text)
    logger.info(f"✅ Количество сохранено: {text}")

    await update.message.reply_text("💬 Введите комментарий (или 'нет'):")
    return ASK_COMMENT

async def handle_comment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global issue_state
    user = update.effective_user
    user_id = user.id
    comment = update.message.text.strip()

    logger.info(f"📝 handle_comment вызван. Комментарий: '{comment}'")

    data = issue_state.pop(user_id, {})
    part = data.get("part")
    quantity = data.get("quantity")

    if part and quantity:
        await save_issue_to_sheet(context, user, part, quantity, comment)
        await update.message.reply_text("✅ Списание выполнено.")
    else:
        logger.error(f"❌ Не удалось списать: part={part}, quantity={quantity}")
        await update.message.reply_text("⚠ Ошибка: не удалось сохранить списание.")
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global issue_state
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
    app.add_handler(CommandHandler("menu", menu_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("adduser", add_user))
    app.add_handler(CommandHandler("cancel", cancel))

    # Списание — ДО всех текстовых обработчиков
    conv_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(handle_issue_button, pattern=r"^issue:")],
        states={
            ASK_QUANTITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_quantity)],
            ASK_COMMENT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_comment)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        per_message=True
    )
    app.add_handler(conv_handler)

    # Кнопки меню (ТОЛЬКО кнопки, не мешает диалогу)
    app.add_handler(MessageHandler(
        filters.Regex('^(🔍 Поиск детали|📦 Взять деталь|📊 Мои списания|❓ Помощь)$'),
        handle_menu_buttons
    ))

    # Поиск — ЛЮБОЙ другой текст (кроме команд)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search))

    # Ошибки
    app.add_error_handler(error_handler)

    # Сохранение состояния
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
