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

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Читаем токен из переменных окружения (или вставьте строкой)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "ВАШ_ТОКЕН_ЗДЕСЬ")

# Загружаем и нормализуем колонки
df = pd.read_excel("data.xlsx")
df.columns = df.columns.str.strip().str.lower()

# Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Отправь тип или наименование детали, и я верну совпадения."
    )

# Обработчик текстовых сообщений — поиск
async def search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text.strip().lower()
    mask = (
        df['тип'].astype(str).str.contains(query, case=False, na=False)
        | df['наименование'].astype(str).str.contains(query, case=False, na=False)
    )
    results = df[mask]

    if results.empty:
        await update.message.reply_text(f'По запросу "{query}" ничего не найдено.')
        return

    # Формируем ответ по первым 5 совпадениям
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

    # Если совпадений больше 5 — подсказываем
    if len(results) > 5:
        await update.message.reply_text("Показано 5 первых результатов. Если нужно больше — используйте команду /ещё",)

# Обработчик /ещё для показа следующих 5 (опционально)
user_index = {}
async def more(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = context.user_data.get("last_query")
    if not query:
        await update.message.reply_text("Сначала отправьте запрос.")
        return

    mask = (
        df['тип'].astype(str).str.contains(query, case=False, na=False)
        | df['наименование'].astype(str).str.contains(query, case=False, na=False)
    )
    results = df[mask]
    start = user_index.get(update.effective_user.id, 5)
    end = start + 5

    if start >= len(results):
        await update.message.reply_text("Больше результатов нет.")
        return

    for _, row in results.iloc[start:end].iterrows():
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

    user_index[update.effective_user.id] = end
    if end < len(results):
        await update.message.reply_text("Напишите /ещё для следующих результатов.")

def main():
    app = ApplicationBuilder().token(7574993294:AAGcnWNkh_A10JSaxDi0m4KjSKtSQgIdPuk).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ещё", more))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search))

    logger.info("Бот запущен")
    app.run_polling()

if __name__ == "__main__":
    main()
