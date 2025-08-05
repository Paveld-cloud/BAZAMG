import logging
import os
import pickle
import json
import gspread
import re
from datetime import datetime
from zoneinfo import ZoneInfo  # Встроено в Python 3.9+
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
user_state = {}
issue_state = {}  # Хранит: {user_id: {"part": ..., "quantity": ...}}
search_count = {}

# Админы
ADMINS = {225177765}

# Переменные окружения
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
SPREADSHEET_URL = os.getenv("SPREADSHEET_URL")
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]  # Без пробелов

if not TELEGRAM_TOKEN or not SPREADSHEET_URL:
    raise EnvironmentError("Отсутствуют TELEGRAM_TOKEN или SPREADSHEET_URL в переменных окружения")

# Google Sheets подключение
def get_gsheet():
    creds_info = json.loads(os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"))
    creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    client = gspread.authorize(creds)
    return client.open_by_url(SPREADSHEET_URL)

# Загрузка данных
def load_data():
    try:
        sheet = get_gsheet().worksheet("SAP")
        return sheet.get_all_records()
    except Exception as e:
        logger.error(f"Ошибка загрузки данных SAP: {e}")
        return []

# Сохранение списания — с временем по Ташкенту
def save_issue_to_sheet(user, part, quantity, comment):
    try:
        # 🔹 Время по Ташкенту (UTC+5)
        tz = ZoneInfo("Asia/Tashkent")
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

        sheet = get_gsheet().worksheet("История")
        sheet.append_row([
            now,
            user.id,
            user.full_name,
            part.get("наименование", ""),
            part.get("код", ""),
            quantity,
            comment
        ])
        logger.info(f"✅ Списание сохранено (Ташкент): {now} | {part.get('код')} x{quantity}")
    except Exception as e:
        logger.error(f"❌ Ошибка при сохранении списания: {e}")
        for admin_id in ADMINS:
            try:
                # Попробуем уведомить админа (если бот ещё работает)
                import asyncio
                loop = asyncio.get_event_loop()
                loop.create_task(ContextTypes.DEFAULT_TYPE.bot.send_message(admin_id, f"⚠️ Ошибка сохранения списания: {e}"))
            except:
                pass

# Инициализация данных
raw_data = load_data()
df = DataFrame(raw_data)
df.columns = df.columns.str.strip().str.lower()
df["код"] = df["код"].astype(str).str.strip().str.lower()

# Состояние пользователя
if os.path.exists("state.pkl"):
    try:
        with open("state.pkl", "rb") as f:
            saved = pickle.load(f)
            user_state.update(saved.get("user_state", {}))
            search_count.update(saved.get("search_count", {}))
    except Exception as e:
        logger.warning(f"Не удалось загрузить state.pkl: {e}")

# Нормализация текста
def normalize(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text.lower()).strip()

# Поиск изображения по коду
def find_image_url_by_code(code: str) -> str:
    code_norm = normalize(code)
    for url in df["image"]:
        if isinstance(url, str) and code_norm in normalize(url):
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
    if image_url:
        try:
            await update.message.reply_photo(photo=image_url, caption=text[:1024], reply_markup=keyboard)
            return
        except Exception as e:
            logger.warning(f"Ошибка при отправке фото: {e}")
    await update.message.reply_text(text, reply_markup=keyboard)

# Команды
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_state.pop(user_id, None)
    search_count.pop(user_id, None)
    await update.message.reply_text("Привет! История поиска очищена.\nОтправь тип, код или наименование детали.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📌 Команды:\n"
        "/start — сброс поиска\n"
        "/more — показать ещё\n"
        "/export — экспорт результатов\n"
        "/stats — сколько раз искали\n"
        "Отправьте текст — для поиска."
    )

async def more(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = user_state.get(user_id)
    if not state:
        await update.message.reply_text("Сначала выполните поиск.")
        return
    results = state["results"]
    offset = state["offset"]
    page = results.iloc[offset: offset + 5]
    if page.empty:
        await update.message.reply_text("Больше нет результатов.")
        return
    for _, row in page.iterrows():
        text = format_row(row)
        await send_row_with_image(update, row, text)
    new_offset = offset + 5
    user_state[user_id]["offset"] = new_offset
    if new_offset < len(results):
        await update.message.reply_text("Напишите /more для следующих результатов.")
    else:
        await update.message.reply_text("Больше результатов нет.")

async def export(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = user_state.get(user_id)
    if not state or state["results"].empty:
        await update.message.reply_text("Сначала выполните поиск.")
        return
    filename = f"export_{user_id}.xlsx"
    state["results"].to_excel(filename, index=False)
    try:
        with open(filename, "rb") as f:
            await update.message.reply_document(InputFile(f, filename))
    finally:
        if os.path.exists(filename):
            os.remove(filename)

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    count = search_count.get(user_id, 0)
    await update.message.reply_text(f"🔍 Вы сделали {count} поисков за сессию.")

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

# Списание
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

    # Команды
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("more", more))
    app.add_handler(CommandHandler("export", export))
    app.add_handler(CommandHandler("stats", stats))

    # Списание
    conv_handler = ConversationHandler(
        entry_points=[CallbackQueryHandler(handle_issue_button, pattern=r"^issue:")],
        states={
            ASK_QUANTITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_quantity)],
            ASK_COMMENT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_comment)],
        },
        fallbacks=[],
    )
    app.add_handler(conv_handler)

    # Поиск
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search))
    app.add_error_handler(error_handler)

    # Сохранение состояния при выходе
    def save_state():
        with open("state.pkl", "wb") as f:
            pickle.dump({
                "user_state": user_state,
                "search_count": search_count
            }, f)
        logger.info("Состояние сохранено")

    atexit.register(save_state)
    signal.signal(signal.SIGINT, lambda s, f: (save_state(), exit(0)))
    signal.signal(signal.SIGTERM, lambda s, f: (save_state(), exit(0)))

    logger.info("Бот запущен")
    app.run_polling()

if __name__ == "__main__":
    main()
