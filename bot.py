import logging
import pandas as pd
from telegram import Update
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext

# Загрузка Excel-файла
df = pd.read_excel("data.xlsx")

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Обработчик сообщений
def search(update: Update, context: CallbackContext) -> None:
    query = update.message.text.lower()
    results = df[df.apply(lambda row: query in str(row['Тип']).lower() or query in str(row['Название']).lower(), axis=1)]

    if results.empty:
        update.message.reply_text("Ничего не найдено.")
    else:
        for _, row in results.iterrows():
            update.message.reply_text(f"Тип: {row['Тип']}
Название: {row['Название']}")

def main():
    TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
    updater = Updater(TOKEN)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, search))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()