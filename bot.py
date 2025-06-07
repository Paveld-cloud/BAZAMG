import logging
import pandas as pd
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# 1) Читаем и нормализуем колонки
df = pd.read_excel('data.xlsx')
df.columns = df.columns.str.strip().str.lower()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2) /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Отправь тип или название детали."
    )

# 3) Поиск
async def search(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.message.text.strip().lower()
    mask = (
        df['тип'].astype(str).str.contains(query, case=False, na=False)
        | df['название'].astype(str).str.contains(query, case=False, na=False)
    )
    results = df[mask]

    if results.empty:
        await update.message.reply_text('Ничего не найдено.')
        return

    for _, row in results.iterrows():
        await update.message.reply_text(
            f"Тип: {row['тип']}\n"
            f"Название: {row['название']}"
        )

def main():
    TOKEN = '7574993294:AAGcnWNkh_A10JSaxDi0m4KjSKtSQgIdPuk'
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search))
    app.run_polling()

if __name__ == '__main__':
    main()

    # Запускаем бот
    app.run_polling()

if __name__ == '__main__':
    main()
