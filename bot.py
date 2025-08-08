import os
import re
import io
import json
import math
import time
import asyncio
import logging
from datetime import datetime

import gspread
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
import pandas as pd
from pandas import DataFrame

# ===================== ЛОГИРОВАНИЕ =====================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bot")

# ===================== НАСТРОЙКИ =====================
ADMINS = {225177765}  # замените на свои ID при необходимости

# ENV
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
SPREADSHEET_URL = os.getenv("SPREADSHEET_URL")
CREDS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
WEBHOOK_URL = (os.getenv("WEBHOOK_URL") or "").rstrip("/")
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/webhook")
PORT = int(os.getenv("PORT", "8080"))
WEBHOOK_SECRET_TOKEN = os.getenv("WEBHOOK_SECRET_TOKEN", "")  # опционально

if not TELEGRAM_TOKEN or not SPREADSHEET_URL or not CREDS_JSON or not WEBHOOK_URL:
    raise RuntimeError(
        "Нужно задать ENV: TELEGRAM_TOKEN, SPREADSHEET_URL, "
        "GOOGLE_APPLICATION_CREDENTIALS_JSON, WEBHOOK_URL"
    )

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]  # чтение/запись
DATA_TTL = 300         # сек до авто-перезагрузки данных
PAGE_SIZE = 5          # карточек на страницу

ASK_QUANTITY, ASK_COMMENT = range(2)

# ===================== ГЛОБАЛЬНЫЕ СОСТОЯНИЯ =====================
df: DataFrame | None = None
_last_load_ts = 0.0

# { user_id: { "query": str, "results": DataFrame, "page": int } }
user_state: dict[int, dict] = {}

# { user_id: {"part": dict(row), "quantity": float|int, "await_comment": bool} }
issue_state: dict[int, dict] = {}

# ===================== GOOGLE SHEETS =====================
def get_gs_client():
    creds_info = json.loads(CREDS_JSON)
    creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    return gspread.authorize(creds)

def load_data() -> list[dict]:
    client = get_gs_client()
    sheet = client.open_by_url(SPREADSHEET_URL)
    ws = sheet.sheet1
    return ws.get_all_records()

def ensure_fresh_data(force: bool = False):
    global df, _last_load_ts
    if force or (time.time() - _last_load_ts > DATA_TTL) or df is None:
        raw = load_data()
        new_df = DataFrame(raw)
        new_df.columns = new_df.columns.str.strip().str.lower()
        for col in ["код", "oem"]:
            if col in new_df.columns:
                new_df[col] = new_df[col].astype(str).str.strip().str.lower()
        df = new_df
        _last_load_ts = time.time()
        logger.info(f"✅ Загружено {len(df)} строк из Google Sheet")

# ===================== ПОИСК =====================
SEARCH_FIELDS = ["тип", "наименование", "код", "oem", "изготовитель"]

def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", (text or "")).lower().strip()

def match_row(row: dict, tokens: list[str]) -> int:
    score = 0
    for field in SEARCH_FIELDS:
        val = normalize(str(row.get(field, "")))
        if val and all(tok in val for tok in tokens):
            score += 2 if field in ("код", "oem") else 1
    return score

def val(row: dict, key: str, default: str = "—") -> str:
    v = row.get(key)
    if v is None:
        return default
    try:
        if isinstance(v, float) and pd.isna(v):
            return default
    except Exception:
        pass
    s = str(v).strip()
    return s if s else default

def format_row(row: dict) -> str:
    return (
        f"🔹 Тип: {val(row, 'тип')}\n"
        f"📦 Наименование: {val(row, 'наименование')}\n"
        f"🔢 Код: {val(row, 'код')}\n"
        f"📦 Кол-во: {val(row, 'количество')}\n"
        f"💰 Цена: {val(row, 'цена')} {val(row, 'валюта')}\n"
        f"🏭 Изготовитель: {val(row, 'изготовитель')}\n"
        f"⚙️ OEM: {val(row, 'oem')}"
    )

def get_row_image(row: dict) -> str:
    for key in ("image", "изображение", "photo", "фото"):
        url = row.get(key)
        if isinstance(url, str) and url.strip():
            return url.strip()
    return ""

async def send_row_with_image(update: Update, row: dict, text: str):
    code = str(row.get("код", "")).strip().lower()
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("📦 Взять деталь", callback_data=f"issue:{code}")]
    ])

    image_url = get_row_image(row)
    if image_url:
        try:
            await update.message.reply_photo(photo=image_url, caption=text, reply_markup=keyboard)
            return
        except Exception as e:
            logger.warning(f"Фото не отправилось, шлём текст. Причина: {e}")

    await update.message.reply_text(text, reply_markup=keyboard)

def get_user_state(user_id: int) -> dict:
    return user_state.setdefault(user_id, {"query": "", "results": DataFrame(), "page": 0})

# ===================== ИСТОРИЯ СПИСАНИЙ =====================
def save_issue_to_sheet(bot, user, part: dict, quantity, comment: str):
    try:
        client = get_gs_client()
        sh = client.open_by_url(SPREADSHEET_URL)
        try:
            ws = sh.worksheet("История")
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title="История", rows=1000, cols=12)
            ws.append_row([
                "timestamp", "user_id", "username", "name",
                "код", "наименование", "количество(списано)", "комментарий"
            ])
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user.id,
            user.username or "",
            f"{user.first_name or ''} {user.last_name or ''}".strip(),
            str(part.get("код", "")),
            str(part.get("наименование", "")),
            str(quantity),
            comment or "",
        ]
        ws.append_row(row)
        logger.info("💾 Списание записано в 'История'")
    except Exception as e:
        logger.error(f"❌ Ошибка при сохранении списания: {e}")
        async def notify_admins():
            for admin_id in ADMINS:
                try:
                    await bot.send_message(chat_id=admin_id, text=f"⚠️ Ошибка сохранения списания: {e}")
                except Exception:
                    pass
        asyncio.create_task(notify_admins())

# ===================== КОМАНДЫ =====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    issue_state.pop(user_id, None)
    await update.message.reply_text(
        "Привет! Напиши запрос (например: `фильтр масла` или `96353000`).\n"
        "Команды:\n"
        "• /help — помощь\n"
        "• /more — показать ещё\n"
        "• /export — выгрузка результатов (XLSX/CSV)\n"
        "• /cancel — отменить списание\n"
        "• /reload — перезагрузить данные (только админ)",
        parse_mode="Markdown"
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Как пользоваться:\n"
        "1) Пишите слова для поиска (можно несколько).\n"
        "2) В карточке нажмите «📦 Взять деталь» — бот спросит количество и комментарий.\n"
        "Команды:\n"
        "• /more — следующая страница\n"
        "• /export — экспорт результатов\n"
        "• /cancel — отмена списания\n"
        "• /reload — перезагрузка данных (админ)"
    )

async def reload_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMINS:
        return await update.message.reply_text("Доступ запрещён.")
    ensure_fresh_data(force=True)
    await update.message.reply_text("✅ Данные перезагружены.")

async def cancel_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if issue_state.pop(user_id, None):
        await update.message.reply_text("Операция списания отменена.")
    else:
        await update.message.reply_text("Нет активной операции для отмены.")

async def export_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = get_user_state(user_id)
    results: DataFrame = state.get("results") or DataFrame()
    if results.empty:
        return await update.message.reply_text("Сначала выполните поиск.")

    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            results.to_excel(writer, index=False)
        output.seek(0)
        await update.message.reply_document(InputFile(output, filename=f"export_{user_id}.xlsx"))
    except Exception as e:
        logger.warning(f"XLSX не удалось, отправляем CSV: {e}")
        csv_data = results.to_csv(index=False, encoding="utf-8-sig")
        await update.message.reply_document(
            InputFile(io.BytesIO(csv_data.encode("utf-8-sig")), filename=f"export_{user_id}.csv")
        )

# ===================== ПОИСК (с гейтом, чтобы не мешал диалогу списания) =====================
async def search_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_fresh_data()
    if update.message is None:
        return

    user_id = update.effective_user.id

    # 🔒 Если идёт диалог списания — отключаем поиск и подсказываем следующий шаг
    st = issue_state.get(user_id)
    if st:
        if "quantity" not in st:
            return await update.message.reply_text("Вы сейчас вводите количество для списания. Введите число или /cancel.")
        if st.get("await_comment"):
            return await update.message.reply_text("Вы сейчас вводите комментарий для списания. Напишите текст или «-», либо /cancel.")

    query = update.message.text.strip()
    if not query:
        return await update.message.reply_text("Введите запрос.")

    tokens = normalize(query).split()
    if not tokens:
        return await update.message.reply_text("Введите более конкретный запрос.")

    matches = []
    for _, row in df.iterrows():
        rdict = row.to_dict()
        s = match_row(rdict, tokens)
        if s > 0:
            matches.append((s, rdict))

    if not matches:
        return await update.message.reply_text(f"По запросу «{query}» ничего не найдено.")

    matches.sort(key=lambda x: x[0], reverse=True)
    results_df = DataFrame([r for _, r in matches])

    state = get_user_state(user_id)
    state["query"] = query
    state["results"] = results_df
    state["page"] = 0

    await send_page(update, user_id)

async def more_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = get_user_state(user_id)
    if not isinstance(state.get("results"), DataFrame) or state["results"].empty:
        return await update.message.reply_text("Сначала выполните поиск.")
    state["page"] += 1
    await send_page(update, user_id)

async def send_page(update: Update, user_id: int):
    state = get_user_state(user_id)
    results: DataFrame = state["results"]
    page = state["page"]

    total = len(results)
    pages = math.ceil(total / PAGE_SIZE)
    if page >= pages:
        state["page"] = pages - 1
        return await update.message.reply_text("Больше результатов нет.")

    start = page * PAGE_SIZE
    end = min(start + PAGE_SIZE, total)
    chunk = results.iloc[start:end]

    await update.message.reply_text(f"Найдено: {total}. Показываю {start + 1}–{end} из {total}.")

    for _, row in chunk.iterrows():
        text = format_row(row.to_dict())
        await send_row_with_image(update, row.to_dict(), text)

    if end < total:
        await update.message.reply_text("Нажмите /more, чтобы показать ещё.")

# ===================== СПИСАНИЕ (Callback + Диалог) =====================
async def on_issue_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    data = query.data
    if not data.startswith("issue:"):
        return

    code = data.split(":", 1)[1].strip().lower()
    ensure_fresh_data()

    found = None
    if df is not None and "код" in df.columns:
        hit = df[df["код"].astype(str).str.lower() == code]
        if not hit.empty:
            found = hit.iloc[0].to_dict()

    if not found:
        return await query.edit_message_text("Не удалось найти деталь по коду. Обновите список или выполните поиск заново.")

    issue_state[user_id] = {"part": found}  # quantity ещё нет
    await query.message.reply_text("Сколько списать? Укажите число (например: 1 или 2.5).")
    return ASK_QUANTITY

async def handle_quantity(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = (update.message.text or "").strip().replace(",", ".")
    try:
        qty = float(text)
        if qty <= 0:
            raise ValueError
    except Exception:
        return await update.message.reply_text("Введите положительное число, например: 1 или 2.5")

    st = issue_state.get(user_id)
    if not st or "part" not in st:
        return await update.message.reply_text("Списание неактивно. Начните заново, нажав «📦 Взять деталь» в карточке.")

    st["quantity"] = qty
    st["await_comment"] = True  # ⛳ теперь ждём комментарий — поиск игнорируется
    await update.message.reply_text("Добавьте комментарий (или напишите «-», если без комментария).")
    return ASK_COMMENT

async def handle_comment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    comment = (update.message.text or "").strip()
    st = issue_state.get(user_id)
    if not st:
        return await update.message.reply_text("Списание неактивно. Начните заново из карточки детали.")

    part = st.get("part")
    qty = st.get("quantity")
    if part is None or qty is None:
        issue_state.pop(user_id, None)
        return await update.message.reply_text("Что-то пошло не так, попробуйте ещё раз.")

    save_issue_to_sheet(context.bot, update.effective_user, part, qty, "" if comment == "-" else comment)
    issue_state.pop(user_id, None)

    await update.message.reply_text(
        f"✅ Списано: {qty}\n"
        f"🔢 Код: {val(part, 'код')}\n"
        f"📦 Наименование: {val(part, 'наименование')}\n"
        f"💬 Комментарий: {comment if comment != '-' else '—'}"
    )
    return ConversationHandler.END

async def handle_cancel_in_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cancel_cmd(update, context)
    return ConversationHandler.END

# ===================== APP/WEBHOOK =====================
def build_app():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Команды
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("more", more_cmd))
    app.add_handler(CommandHandler("export", export_cmd))
    app.add_handler(CommandHandler("reload", reload_cmd))
    app.add_handler(CommandHandler("cancel", cancel_cmd))

    # Диалог списания (важно добавить ДО поиска)
    conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(on_issue_click, pattern=r"^issue:")],
        states={
            ASK_QUANTITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_quantity)],
            ASK_COMMENT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_comment)],
        },
        fallbacks=[CommandHandler("cancel", handle_cancel_in_dialog)],
        allow_reentry=True,
        # per_message не используем — чтобы не ловить warning, порядок и гейт уже защищают
    )
    app.add_handler(conv)

    # Поиск — в группе 1, чтобы отрабатывать ПОСЛЕ диалога списания
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search_text), group=1)

    return app

if __name__ == "__main__":
    ensure_fresh_data(force=True)
    application = build_app()

    webhook_full_url = f"{WEBHOOK_URL}{WEBHOOK_PATH}"
    logger.info(f"🚀 Стартуем webhook-сервер на 0.0.0.0:{PORT}")
    logger.info(f"🌐 Устанавливаем webhook: {webhook_full_url}")

    application.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        secret_token=WEBHOOK_SECRET_TOKEN or None,
        webhook_url=webhook_full_url,
        url_path=WEBHOOK_PATH.lstrip("/"),
        drop_pending_updates=True,
        allowed_updates=None,
    )
