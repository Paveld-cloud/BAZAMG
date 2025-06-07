import logging
import pandas as pd
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

# Загрузка Excel-файла
df = pd.read_excel('data.xlsx')

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

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
            await update.message.reply_text(f"""Тип: {row['Тип']}
Название: {row['Название']}""")

# Основная функция
def main():
    TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'  # ← вставь сюда свой токен
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search))

    # Просто запускаем polling (без asyncio.run!)
    app.run_polling()

if __name__ == '__main__':
    main()
