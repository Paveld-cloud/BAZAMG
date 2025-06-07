import logging
import pandas as pd
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# Загрузка Excel-файла
df = pd.read_excel('data.xlsx')

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Обработчик команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я бот для поиска деталей по Excel‑базе.\n"
        "Отправь тип или название детали, и я покажу результаты."
    )

# Обработчик текстовых сообщений
async def search(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.message.text.lower()
    results = df[df.apply(
        lambda row: query in str(row['Тип']).lower() or query in str(row['Название']).lower(),
        axis=1
    )]

    if results.empty:
        await update.message.reply_text('Ничего не найдено.')
    else:
        for _, row in results.iterrows():
            await update.message.reply_text(
                f"Тип: {row['Тип']}\n"
                f"Название: {row['Название']}"
            )

# Основная функция
def main():
    TOKEN = '7574993294:AAGcnWNkh_A10JSaxDi0m4KjSKtSQgIdPuk'  # ← вставьте сюда реальный токен
    app = ApplicationBuilder().token(TOKEN).build()

    # Регистрируем обработчики
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search))

    # Запускаем бот
    app.run_polling()

if __name__ == '__main__':
    main()
