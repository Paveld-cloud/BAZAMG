import logging
import pandas as pd
import os
from telegram import Update
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

# Токен бота
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7574993294:...")

# Читаем таблицу
df = pd.read_excel("data.xlsx")
df.columns = df.columns.str.strip().str.lower()

# Хранение состояния поиска: {user_id: {"query": ..., "offset": ..., "results": DataFrame}}
user_state = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    # Полностью очищаем состояние
    user_state.pop(user_id, None)
    await update.message.reply_text(
        "Привет! История поиска очищена.\n"
        "Отправьте тип или наименование детали для нового поиска."
    )

async def search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    query = update.message.text.strip().lower()

    # Фильтр по 'тип' и 'наименование'
    mask = (
        df['тип'].str.lower().str.contains(query, na=False) |
        df['наименование'].str.lower().str.contains(query, na=False)
    )
    results = df[mask]

    if results.empty:
        await update.message.reply_text(f'По запросу "{query}" ничего не найдено.')
        return

    # Сохраняем новое состояние: стартовая страница 0–5
    user_state[user_id] = {
        "query": query,
        "offset": 0,
        "results": results
    }

    # Отправляем первую порцию
    await send_page(update, user_id)

async def more(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = user_state.get(user_id)

    if not state:
        await update.message.reply_text("Сначала отправьте запрос для поиска (/start чтобы очистить).")
        return

    await send_page(update, user_id)

async def send_page(update: Update, user_id: int):
    state = user_state[user_id]
    results = state["results"]
    offset = state["offset"]
    chunk = results.iloc[offset : offset + 5]

    for _, row in chunk.iterrows():
        await update.message.reply_text(
            f"🔹 Тип: {row['тип']}\n"
            f"📦 Наименование: {row['наименование']}\n"
            f"🔢 Код: {row['код']}\n"
            f"📦 Кол-во: {row['количество']}\n"
            f"💰 Цена: {row['цена']} {row['валюта']}\n"
            f"🏭 Изготовитель: {row['изготовитель']}\n"
            f"⚙ OEM: {row['oem']}"
        )

    # Обновляем offset
    state["offset"] += 5

    # Если ещё есть результаты
    if state["offset"] < len(results):
        await update.message.reply_text("Напишите /more для следующих результатов.")
    else:
        await update.message.reply_text("Больше результатов нет.")

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("more", more))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search))

    logger.info("Бот запущен")
    app.run_polling()

if __name__ == "__main__":
    main()
