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

# Токен из переменных окружения
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7574993294:AAGcnWNkh_A10JSaxDi0m4KjSKtSQgIdPuk")

# Читаем таблицу и нормализуем названия столбцов
df = pd.read_excel("data.xlsx")
df.columns = df.columns.str.strip().str.lower()

# Храним состояние (последний запрос, смещение) по user_id
user_state = {}

# /start — очищаем историю и приветствуем
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    # Сбрасываем состояние
    if user_id in user_state:
        user_state.pop(user_id)
    await update.message.reply_text(
        "Привет! История поиска очищена.\nОтправь тип или наименование детали."
    )

# /more — продолжение выдачи результатов
async def more(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = user_state.get(user_id)
    if not state:
        await update.message.reply_text("Сначала выполните поиск.")
        return

    query, offset, results = state["query"], state["offset"], state["results"]
    # Показываем следующую порцию
    page = results.iloc[offset: offset + 5]
    for _, row in page.iterrows():
        text = (
            f"🔹 Тип: {row['тип']}\n"
            f"📦 Наименование: {row['наименование']}\n"
            f"🔢 Код: {row['код']}\n"
            f"📦 Кол-во: {row['количество']}\n"
            f"💰 Цена: {row['цена']} {row['валюта']}\n"
            f"🏭 Изготовитель: {row['изготовитель']}\n"
            f"⚙ OEM: {row['oem']}"
        )
        await update.message.reply_text(text)

    # Обновляем смещение
    new_offset = offset + 5
    user_state[user_id]["offset"] = new_offset
    if new_offset < len(results):
        await update.message.reply_text("Напишите /more для следующих результатов.")
    else:
        await update.message.reply_text("Больше результатов нет.")

# Обработка любых текстовых сообщений — поиск
async def search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    query = update.message.text.strip().lower()

    # Фильтр по столбцам
    mask = (
        df['тип'].str.lower().str.contains(query, na=False) |
        df['наименование'].str.lower().str.contains(query, na=False)
    )
    results = df[mask]

    if results.empty:
        await update.message.reply_text(f'По запросу "{query}" ничего не найдено.')
        return

    # Сохраняем состояние
    user_state[user_id] = {
        "query": query,
        "offset": 5,
        "results": results
    }

    # Выдаём первые 5
    for _, row in results.head(5).iterrows():
        text = (
            f"🔹 Тип: {row['тип']}\n"
            f"📦 Наименование: {row['наименование']}\n"
            f"🔢 Код: {row['код']}\n"
            f"📦 Кол-во: {row['количество']}\n"
            f"💰 Цена: {row['цена']} {row['валюта']}\n"
            f"🏭 Изготовитель: {row['изготовитель']}\n"
            f"⚙ OEM: {row['oem']}"
        )
        await update.message.reply_text(text)

    if len(results) > 5:
        await update.message.reply_text("Показано 5 первых результатов. Напишите /more для продолжения.")

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("more", more))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search))

    logger.info("Бот запущен")
    app.run_polling()

if __name__ == "__main__":
    main()

    logger.info("Бот запущен")
    app.run_polling()

if __name__ == "__main__":
    main()

