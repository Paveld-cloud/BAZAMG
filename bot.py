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
import pandas as pd
from google.oauth2.service_account import Credentials
from pandas import DataFrame
from telegram import (
    Update, InputFile, InlineKeyboardMarkup, InlineKeyboardButton
)
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler,
    ConversationHandler, ContextTypes, filters
)

# --------------------------- ЛОГИ ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bot")

# -------------------------- НАСТРОЙКИ -----------------------
ADMINS = {225177765}  # поменяй при необходимости

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
SPREADSHEET_URL = os.getenv("SPREADSHEET_URL")
CREDS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")

WEBHOOK_URL = (os.getenv("WEBHOOK_URL") or "").rstrip("/")
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/webhook")
PORT = int(os.getenv("PORT", "8080"))
WEBHOOK_SECRET_TOKEN = os.getenv("WEBHOOK_SECRET_TOKEN", "")

if not TELEGRAM_TOKEN or not SPREADSHEET_URL or not CREDS_JSON or not WEBHOOK_URL:
    raise RuntimeError(
        "ENV нужны: TELEGRAM_TOKEN, SPREADSHEET_URL, GOOGLE_APPLICATION_CREDENTIALS_JSON, WEBHOOK_URL"
    )

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
DATA_TTL = 300
PAGE_SIZE = 5

ASK_QUANTITY, ASK_COMMENT = range(2)

# ---------------------- ГЛОБАЛЬНЫЕ СОСТОЯНИЯ ----------------
df: DataFrame | None = None
_last_load_ts = 0.0

user_state: dict[int, dict] = {}   # { user_id: { "query": str, "results": DataFrame, "page": int } }
issue_state: dict[int, dict] = {}  # { user_id: {"part": dict, "quantity": float, "await_comment": bool} }

# ------------------------- УТИЛИТЫ --------------------------
def cancel_markup():
    return InlineKeyboardMarkup([[InlineKeyboardButton("❌ Отменить", callback_data="cancel_action")]])

def get_gs_client():
    creds_info = json.loads(CREDS_JSON)
    creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    return gspread.authorize(creds)

def load_data() -> list[dict]:
    client = get_gs_client()
    sheet = client.open_by_url(SPREADSHEET_URL)
    ws = sheet.sheet1   # если нужен конкретный лист, замени на worksheet("SAP")
    return ws.get_all_records()

def ensure_fresh_data(force: bool = False):
    global df, _last_load_ts
    if force or df is None or (time.time() - _last_load_ts > DATA_TTL):
        data = load_data()
        new_df = DataFrame(data)
        new_df.columns = new_df.columns.str.strip().str.lower()
        for col in ("код", "oem"):
            if col in new_df.columns:
                new_df[col] = new_df[col].astype(str).str.strip().str.lower()
        df = new_df
        _last_load_ts = time.time()
        logger.info(f"✅ Загружено {len(df)} строк из Google Sheet")

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
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("📦 Взять деталь", callback_data=f"issue:{code}")]])
    img = get_row_image(row)
    if img:
        try:
            await update.message.reply_photo(photo=img, caption=text, reply_markup=kb)
            return
        except Exception as e:
            logger.warning(f"Не удалось отправить фото: {e}")
    await update.message.reply_text(text, reply_markup=kb)

def get_user_state(user_id: int) -> dict:
    return user_state.setdefault(user_id, {"query": "", "results": DataFrame(), "page": 0})

# --------------------- СОХРАНЕНИЕ СПИСАНИЙ -------------------
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
        ws.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user.id,
            user.username or "",
            f"{user.first_name or ''} {user.last_name or ''}".strip(),
            str(part.get("код", "")),
            str(part.get("наименование", "")),
            str(quantity),
            comment or "",
        ])
        logger.info("💾 Списание записано в 'История'")
    except Exception as e:
        logger.error(f"Ошибка записи списания: {e}")
        async def notify():
            for admin_id in ADMINS:
                try:
                    await bot.send_message(admin_id, f"⚠️ Ошибка сохранения списания: {e}")
                except Exception:
                    pass
        asyncio.create_task(notify())

# ------------------------- КОМАНДЫ --------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    issue_state.pop(uid, None)
    await update.message.reply_text(
        "Привет! Напиши запрос (например: `фильтр масла` или `96353000`).\n"
        "Команды:\n"
        "• /help — помощь\n"
        "• /more — показать ещё\n"
        "• /export — выгрузка результатов (XLSX/CSV)\n"
        "• /cancel — отменить списание (или кнопкой «Отменить»)\n"
        "• /reload — перезагрузить данные (только админ)",
        parse_mode="Markdown"
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "1) Введите слова для поиска (можно несколько).\n"
        "2) В карточке нажмите «📦 Взять деталь» — бот спросит количество и комментарий.\n"
        "Команды: /more, /export, /cancel, /reload (для админа)."
    )

async def reload_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMINS:
        return await update.message.reply_text("Доступ запрещён.")
    ensure_fresh_data(force=True)
    await update.message.reply_text("✅ Данные перезагружены.")

async def cancel_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if issue_state.pop(uid, None):
        user_state.pop(uid, None)
        await update.message.reply_text("❌ Операция списания отменена.")
    else:
        await update.message.reply_text("Нет активной операции.")

async def export_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    st = get_user_state(uid)
    results: DataFrame = st.get("results") or DataFrame()
    if results.empty:
        return await update.message.reply_text("Сначала выполните поиск.")
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            results.to_excel(w, index=False)
        buf.seek(0)
        await update.message.reply_document(InputFile(buf, filename=f"export_{uid}.xlsx"))
    except Exception as e:
        logger.warning(f"Не удалось XLSX, шлём CSV: {e}")
        csv = results.to_csv(index=False, encoding="utf-8-sig")
        await update.message.reply_document(InputFile(io.BytesIO(csv.encode("utf-8-sig")), filename=f"export_{uid}.csv"))

# ------------------------- ПОИСК -----------------------------
SEARCH_FIELDS = ["тип", "наименование", "код", "oem", "изготовитель"]
def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", (text or "")).lower().strip()

def match_row(row: dict, tokens: list[str]) -> int:
    score = 0
    for f in SEARCH_FIELDS:
        val = normalize(str(row.get(f, "")))
        if val and all(t in val for t in tokens):
            score += 2 if f in ("код", "oem") else 1
    return score

async def search_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_fresh_data()
    if update.message is None:
        return

    # если прошлый хендлер пометил "не искать" — выходим тихо
    if context.chat_data.pop("suppress_next_search", False):
        return

    uid = update.effective_user.id
    st_issue = issue_state.get(uid)
    if st_issue:
        if "quantity" not in st_issue:
            return await update.message.reply_text(
                "Вы вводите количество. Введите число или нажмите «Отменить».",
                reply_markup=cancel_markup()
            )
        if st_issue.get("await_comment"):
            return await update.message.reply_text(
                "Вы вводите комментарий. Напишите текст или «-», либо нажмите «Отменить».",
                reply_markup=cancel_markup()
            )

    q = update.message.text.strip()
    if not q:
        return await update.message.reply_text("Введите запрос.")
    tokens = normalize(q).split()
    if not tokens:
        return await update.message.reply_text("Введите более конкретный запрос.")

    matches = []
    for _, row in df.iterrows():
        rd = row.to_dict()
        s = match_row(rd, tokens)
        if s > 0:
            matches.append((s, rd))

    if not matches:
        return await update.message.reply_text(f"По запросу «{q}» ничего не найдено.")

    matches.sort(key=lambda x: x[0], reverse=True)
    results_df = DataFrame([r for _, r in matches])

    st = get_user_state(uid)
    st["query"] = q
    st["results"] = results_df
    st["page"] = 0

    await send_page(update, uid)

async def more_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    st = get_user_state(uid)
    results: DataFrame = st.get("results") or DataFrame()
    if results.empty:
        return await update.message.reply_text("Сначала выполните поиск.")
    st["page"] += 1
    await send_page(update, uid)

async def send_page(update: Update, uid: int):
    st = get_user_state(uid)
    results: DataFrame = st["results"]
    page = st["page"]

    total = len(results)
    pages = math.ceil(total / PAGE_SIZE)
    if page >= pages:
        st["page"] = pages - 1
        return await update.message.reply_text("Больше результатов нет.")

    start = page * PAGE_SIZE
    end = min(start + PAGE_SIZE, total)
    chunk = results.iloc[start:end]

    await update.message.reply_text(f"Найдено: {total}. Показываю {start + 1}–{end} из {total}.")
    for _, row in chunk.iterrows():
        await send_row_with_image(update, row.to_dict(), format_row(row.to_dict()))
    if end < total:
        await update.message.reply_text("Нажмите /more, чтобы показать ещё.")

# ------------------ СПИСАНИЕ (Диалог) -----------------------
async def on_issue_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    uid = q.from_user.id
    code = q.data.split(":", 1)[1].strip().lower()

    ensure_fresh_data()
    found = None
    if df is not None and "код" in df.columns:
        hit = df[df["код"].astype(str).str.lower() == code]
        if not hit.empty:
            found = hit.iloc[0].to_dict()

    if not found:
        return await q.edit_message_text("Не удалось найти деталь по коду. Выполните поиск заново.")

    issue_state[uid] = {"part": found}
    await q.message.reply_text("Сколько списать? Укажите число (например: 1 или 2.5).", reply_markup=cancel_markup())
    return ASK_QUANTITY

async def handle_quantity(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # подавим поиск для этого апдейта
    context.chat_data["suppress_next_search"] = True

    uid = update.effective_user.id
    text = (update.message.text or "").strip().replace(",", ".")
    try:
        qty = float(text)
        if qty <= 0:
            raise ValueError
    except Exception:
        return await update.message.reply_text("Введите положительное число, например: 1 или 2.5", reply_markup=cancel_markup())

    st = issue_state.get(uid)
    if not st or "part" not in st:
        return await update.message.reply_text("Списание неактивно — начните заново из карточки.")

    st["quantity"] = qty
    st["await_comment"] = True
    await update.message.reply_text("Добавьте комментарий (или напишите «-», если без комментария).", reply_markup=cancel_markup())
    return ASK_COMMENT

async def handle_comment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # подавим поиск для этого апдейта
    context.chat_data["suppress_next_search"] = True

    uid = update.effective_user.id
    comment = (update.message.text or "").strip()
    st = issue_state.get(uid)
    if not st:
        return await update.message.reply_text("Списание неактивно. Начните заново из карточки.")

    part = st.get("part")
    qty = st.get("quantity")
    if part is None or qty is None:
        issue_state.pop(uid, None)
        return await update.message.reply_text("Что-то пошло не так. Попробуйте ещё раз.")

    save_issue_to_sheet(context.bot, update.effective_user, part, qty, "" if comment == "-" else comment)
    issue_state.pop(uid, None)
    user_state.pop(uid, None)

    await update.message.reply_text(
        f"✅ Списано: {qty}\n"
        f"🔢 Код: {val(part, 'код')}\n"
        f"📦 Наименование: {val(part, 'наименование')}\n"
        f"💬 Комментарий: {comment if comment != '-' else '—'}"
    )
    return ConversationHandler.END

async def cancel_action(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    uid = q.from_user.id

    if uid not in issue_state:
        return  # тихо игнорируем старые кнопки

    issue_state.pop(uid, None)
    user_state.pop(uid, None)
    context.chat_data["suppress_next_search"] = True
    await q.message.reply_text("❌ Операция списания отменена.")
    return ConversationHandler.END

async def handle_cancel_in_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cancel_cmd(update, context)
    return ConversationHandler.END

# --------------------- APP / WEBHOOK ------------------------
def build_app():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("more", more_cmd))
    app.add_handler(CommandHandler("export", export_cmd))
    app.add_handler(CommandHandler("reload", reload_cmd))
    app.add_handler(CommandHandler("cancel", cancel_cmd))

    conv = ConversationHandler(
        entry_points=[
            CallbackQueryHandler(on_issue_click, pattern=r"^issue:"),
            CallbackQueryHandler(cancel_action, pattern=r"^cancel_action$"),
        ],
        states={
            ASK_QUANTITY: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_quantity),
                CallbackQueryHandler(cancel_action, pattern=r"^cancel_action$"),
            ],
            ASK_COMMENT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_comment),
                CallbackQueryHandler(cancel_action, pattern=r"^cancel_action$"),
            ],
        },
        fallbacks=[
            CommandHandler("cancel", handle_cancel_in_dialog),
            CallbackQueryHandler(cancel_action, pattern=r"^cancel_action$"),
        ],
        allow_reentry=True,
    )
    app.add_handler(conv)

    # поиск — в группе 1, чтобы диалог «съедал» апдейты первым
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search_text), group=1)

    return app

if __name__ == "__main__":
    ensure_fresh_data(force=True)
    application = build_app()

    full_webhook = f"{WEBHOOK_URL}{WEBHOOK_PATH}"
    logger.info(f"🚀 Стартуем webhook-сервер на 0.0.0.0:{PORT}")
    logger.info(f"🌐 Устанавливаем webhook: {full_webhook}")

    application.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        secret_token=WEBHOOK_SECRET_TOKEN or None,
        webhook_url=full_webhook,
        url_path=WEBHOOK_PATH.lstrip("/"),
        drop_pending_updates=True,
        allowed_updates=None,
    )
