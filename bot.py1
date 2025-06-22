import logging
import pandas as pd
import os
import pickle
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from telegram import Update, InputFile
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# 🧠 Переменные
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7574993294:AAGcnWNkh_A10JSaxDi0m4KjSKtSQgIdPuk")
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1rRo92aZ3KYsCoDt1mfgXtTdow9bAFx8g/edit"
CREDENTIALS_PATH = "credentials.json"

# 🔐 Авторизация Google Sheets
def get_df_from_google_sheets():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_PATH, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_url(SPREADSHEET_URL).sheet1
    records = sheet.get_all_records()
    df = pd.DataFrame(records)
    df.columns = df.columns.str.strip().str.lower()
    return df

# 📊 Состояние пользователей
user_state = {}
search_count = {}

if os.path.exists("state.pkl"):
    with open("state.pkl", "rb") as f:
        user_state = pickle.load(f)

# /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_state.pop(user_id, None)
    await update.message.reply_text("Привет! История поиска очищена.\nОтправь тип, код или наименование детали.")

# /more
async def more(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = user_state.get(user_id)
    if not state:
        await update.message.reply_text("Сначала выполните поиск.")
        return

    offset = state["offset"]
    results = state["results"]
    page = results.iloc[offset: offset + 5]

    for _, row in page.iterrows():
        text = format_result(row)
        await update.message.reply_text(text)

    new_offset = offset + 5
    user_state[user_id]["offset"] = new_offset

    if new_offset < len(results):
        await update.message.reply_text("Напишите /more для следующих результатов.")
    else:
        await update.message.reply_text("Больше результатов нет.")

# /help
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 Доступные команды:\n"
        "/start — начать заново\n"
        "/more — показать ещё результаты\n"
        "/help — справка\n"
        "/export — выгрузка результатов\n"
        "/stats — количество поисков\n"
        "Просто отправьте текст — для поиска по типу, коду, OEM, названию или изготовителю."
    )

# /stats
async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    count = search_count.get(user_id, 0)
    await update.message.reply_text(f"🔍 Вы сделали {count} поисков за сессию.")

# /export
async def export(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = user_state.get(user_id)
    if not state:
        await update.message.reply_text("Сначала выполните поиск.")
        return

    filename = f"export_{user_id}.xlsx"
    state["results"].to_excel(filename, index=False)
    with open(filename, "rb") as f:
        await update.message.reply_document(InputFile(f, filename))
    os.remove(filename)

# Поиск
async def search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    query = update.message.text.strip().lower()
    df = get_df_from_google_sheets()

    mask = (
        df['тип'].astype(str).str.lower().str.contains(query, na=False) |
        df['наименование'].astype(str).str.lower().str.contains(query, na=False) |
        df['код'].astype(str).str.lower().str.contains(query, na=False) |
        df['oem'].astype(str).str.lower().str.contains(query, na=False) |
        df['изготовитель'].astype(str).str.lower().str.contains(query, na=False)
    )

    results = df[mask]

    if results.empty:
        await update.message.reply_text(f'По запросу "{query}" ничего не найдено.')
        return

    user_state[user_id] = {
        "query": query,
        "offset": 5,
        "results": results
    }

    search_count[user_id] = search_count.get(user_id, 0) + 1

    for _, row in results.head(5).iterrows():
        text = format_result(row)
        await update.message.reply_text(text)

    if len(results) > 5:
        await update.message.reply_text("Показано 5 первых результатов. Напишите /more для продолжения.")

# Формат результата
def format_result(row):
    return (
        f"🔹 Тип: {row['тип']}\n"
        f"📦 Наименование: {row['наименование']}\n"
        f"🔢 Код: {row['код']}\n"
        f"📦 Кол-во: {row['количество']}\n"
        f"💰 Цена: {row['цена']} {row['валюта']}\n"
        f"🏭 Изготовитель: {row['изготовитель']}\n"
        f"⚙ OEM: {row['oem']}"
    )

# Обработка ошибок
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(msg="Ошибка в Telegram обработчике", exc_info=context.error)
    if isinstance(update, Update) and update.message:
        await update.message.reply_text("Произошла ошибка. Попробуйте позже.")

# Старт
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("more", more))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("export", export))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search))
    app.add_error_handler(error_handler)
    logger.info("Бот запущен")
    app.run_polling()

    with open("state.pkl", "wb") as f:
        pickle.dump(user_state, f)

if __name__ == "__main__":
    main()

