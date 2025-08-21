# app/handlers.py
import asyncio
import logging
import re
from html import escape
from typing import Dict, Any

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    CommandHandler, MessageHandler, CallbackQueryHandler,
    ConversationHandler, ContextTypes, filters
)

from . import data
from .config import (
    PAGE_SIZE, MAX_QTY, SUPPORT_CONTACT, WELCOME_ANIMATION_URL,
    WELCOME_PHOTO_URL, WELCOME_MEDIA_ID, ADMINS
)

logger = logging.getLogger("bot.handlers")

# Этапы диалога списания
ASK_QUANTITY, ASK_COMMENT = range(2)

# Состояния пользователя
user_state: Dict[int, Dict[str, Any]] = {}
search_offset: Dict[int, int] = {}

# ---------- Кнопки ----------
def cancel_markup():
    return InlineKeyboardMarkup([[InlineKeyboardButton("❌ Отменить", callback_data="cancel_action")]])

def confirm_markup():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Да, списать", callback_data="confirm_yes"),
         InlineKeyboardButton("❌ Нет", callback_data="confirm_no")],
        [InlineKeyboardButton("❌ Отменить", callback_data="cancel_action")]
    ])

def more_markup():
    return InlineKeyboardMarkup([[InlineKeyboardButton("⏭ Ещё", callback_data="more")]])

def main_menu_markup():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔍 Поиск", callback_data="menu_search")],
        [InlineKeyboardButton("📦 Как списать деталь", callback_data="menu_issue_help")],
        [InlineKeyboardButton("📞 Поддержка", callback_data="menu_contact")],
    ])

# ---------- Безопасная отправка HTML ----------
async def _safe_send_html_message(bot, chat_id: int, text: str, **kwargs):
    try:
        return await bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML", **kwargs)
    except Exception as e:
        logger.warning(f"HTML message parse failed, fallback to plain: {e}")
        no_tags = re.sub(r"</?(b|i|code)>", "", text)
        kwargs.pop("parse_mode", None)
        return await bot.send_message(chat_id=chat_id, text=no_tags, **kwargs)

# ---------- /start ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await data.ensure_users_async(force=False)
    await data.ensure_fresh_data_async(force=False)

    chat_id = update.effective_chat.id
    full_name = escape((update.effective_user.full_name or "").strip())
    text = (
        f"<b>Добро пожаловать, {full_name}!</b>\n\n"
        "Это бот для поиска деталей и удобного списания в Google Таблицу.\n"
        "Нажмите «🔍 Поиск», чтобы начать, или отправьте текстом запрос.\n\n"
        f"{escape(SUPPORT_CONTACT)}"
    )

    # Пытаемся отправить приветствие: file_id > фото > анимация > текст
    try:
        if WELCOME_MEDIA_ID:
            await context.bot.send_photo(chat_id=chat_id, photo=WELCOME_MEDIA_ID, caption="")
        elif WELCOME_PHOTO_URL:
            await context.bot.send_photo(chat_id=chat_id, photo=WELCOME_PHOTO_URL, caption="")
        elif WELCOME_ANIMATION_URL:
            await context.bot.send_animation(chat_id=chat_id, animation=WELCOME_ANIMATION_URL)
    except Exception as e:
        logger.warning(f"Welcome media failed: {e}")
    await _safe_send_html_message(context.bot, chat_id, text, reply_markup=main_menu_markup())

# ---------- Меню ----------
async def on_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    if q.data == "menu_search":
        await _safe_send_html_message(context.bot, q.message.chat_id,
                                      "Отправьте текст запроса (код, наименование, OEM, изготовитель)…",
                                      reply_markup=cancel_markup())
    elif q.data == "menu_issue_help":
        await _safe_send_html_message(context.bot, q.message.chat_id,
                                      "<b>Как списать деталь</b>\n1) Найдите деталь\n2) Нажмите «Взять деталь»\n"
                                      "3) Введите количество\n4) Напишите комментарий\n5) Подтвердите списание.",
                                      reply_markup=main_menu_markup())
    elif q.data == "menu_contact":
        await _safe_send_html_message(context.bot, q.message.chat_id,
                                      f"Связь: {escape(SUPPORT_CONTACT)}",
                                      reply_markup=main_menu_markup())

# ---------- Поиск ----------
def _render_item_text(item: Dict[str, Any]) -> str:
    name = str(item.get("наименование", "")).strip()
    code = str(item.get("код", "")).strip()
    typ = str(item.get("тип", "")).strip()
    oem = str(item.get("oem", "")).strip()
    man = str(item.get("изготовитель", "")).strip()

    lines = []
    if name: lines.append(f"<b>{escape(name)}</b>")
    if typ: lines.append(f"Тип: {escape(typ)}")
    if code: lines.append(f"Код: <code>{escape(code)}</code>")
    if oem: lines.append(f"OEM: <code>{escape(oem)}</code>")
    if man: lines.append(f"Изготовитель: {escape(man)}")
    return "\n".join(lines) if lines else "(пусто)"

def _issue_button(item: Dict[str, Any]):
    code = str(item.get("код","")).strip()
    encoded = code.replace("|", "/")
    return InlineKeyboardMarkup([[InlineKeyboardButton("📦 Взять деталь", callback_data=f"issue|{encoded}")]])

async def _send_item(update: Update, context: ContextTypes.DEFAULT_TYPE, item: Dict[str, Any]):
    chat_id = update.effective_chat.id if update.effective_chat else update.callback_query.message.chat_id
    text = _render_item_text(item)
    # фото
    code = str(item.get("код","")).strip()
    url = data.get_image_for_code(code)
    if url:
        try:
            url = await data.resolve_image_url_async(url)
            await context.bot.send_photo(chat_id=chat_id, photo=url, caption="")
        except Exception as e:
            logger.warning(f"Failed to send photo for code={code}: {e}")
    await _safe_send_html_message(context.bot, chat_id, text, reply_markup=_issue_button(item))

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = (update.message.text or "").strip()
    if not query:
        return
    await data.ensure_fresh_data_async(False)
    items = data.search(query, offset=0, limit=PAGE_SIZE)
    search_offset[update.effective_user.id] = PAGE_SIZE
    if not items:
        await _safe_send_html_message(context.bot, update.effective_chat.id, "Ничего не найдено.")
        return
    for it in items:
        await _send_item(update, context, it)
    if len(items) == PAGE_SIZE:
        await _safe_send_html_message(context.bot, update.effective_chat.id, "Показаны первые результаты.", reply_markup=more_markup())

async def on_more(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    uid = q.from_user.id
    off = search_offset.get(uid, 0)
    # Берём текст из предыдущего сообщения пользователя (reply_to). Если нет — просим повторить запрос
    last_query = (q.message.reply_to_message.text if q.message and q.message.reply_to_message else None)
    if not last_query:
        await _safe_send_html_message(context.bot, q.message.chat_id, "Отправьте новый запрос.")
        return
    items = data.search(last_query, offset=off, limit=PAGE_SIZE)
    search_offset[uid] = off + len(items)
    for it in items:
        await _send_item(update, context, it)

# ---------- Списание ----------
async def _ask_quantity(update: Update, context: ContextTypes.DEFAULT_TYPE, code: str):
    uid = update.effective_user.id
    user_state[uid] = {"code": code}
    await _safe_send_html_message(context.bot, update.effective_chat.id,
                                  f"Введите количество для <code>{escape(code)}</code> (до {MAX_QTY}):",
                                  reply_markup=cancel_markup())
    return ASK_QUANTITY

async def _ask_comment(update: Update, context: ContextTypes.DEFAULT_TYPE, quantity: float):
    uid = update.effective_user.id
    user_state[uid]["quantity"] = quantity
    await _safe_send_html_message(context.bot, update.effective_chat.id,
                                  "Добавьте комментарий (цель, место установки, ОС и т.п.):",
                                  reply_markup=cancel_markup())
    return ASK_COMMENT

async def _confirm_issue(update: Update, context: ContextTypes.DEFAULT_TYPE, comment: str):
    uid = update.effective_user.id
    st = user_state.get(uid) or {}
    st["comment"] = comment
    code = st.get("code","")
    qty = st.get("quantity", 0)
    text = f"<b>Подтвердите списание</b>\nКод: <code>{escape(code)}</code>\nКоличество: <b>{qty}</b>\nКомментарий: {escape(comment)}"
    await _safe_send_html_message(context.bot, update.effective_chat.id, text, reply_markup=confirm_markup())
    return ConversationHandler.WAITING

async def save_issue_to_sheet(bot, user, part: Dict[str, Any], quantity, comment: str):
    """Запись в лист 'История' через to_thread, чтобы не блокировать event loop"""
    from .config import SPREADSHEET_URL
    import gspread

    def _write_row():
        client = data.get_gs_client()
        sh = client.open_by_url(SPREADSHEET_URL)
        try:
            ws = sh.worksheet("История")
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title="История", rows=1000, cols=12)
            ws.append_row(["Дата","ID","Имя","Тип","Наименование","Код","Количество","Комментарий"])

        headers_raw = ws.row_values(1)
        headers = [h.strip() for h in headers_raw]
        norm = [h.lower() for h in headers]

        full_name = f"{(user.first_name or '').strip()} {(user.last_name or '').strip()}".strip()
        display_name = full_name or (f"@{user.username}" if user.username else str(user.id))
        ts = data.now_local_str()

        values_by_key = {
            "дата": ts, "timestamp": ts,
            "id": user.id, "user_id": user.id,
            "имя": display_name, "name": display_name,
            "тип": str(part.get("тип", "")), "type": str(part.get("тип", "")),
            "наименование": str(part.get("наименование", "")), "name_item": str(part.get("наименование", "")),
            "код": str(part.get("код", "")), "code": str(part.get("код", "")),
            "количество": str(quantity), "qty": str(quantity),
            "комментарий": comment or "", "comment": comment or "",
        }
        row = [values_by_key.get(h.lower(), "") for h in norm]
        ws.append_row(row, value_input_option="USER_ENTERED")

    await asyncio.to_thread(_write_row)

# ---- Callbacks ----
async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data_cb = q.data or ""
    if data_cb.startswith("issue|"):
        code = data_cb.split("|",1)[1]
        return await _ask_quantity(update, context, code)
    elif data_cb == "confirm_yes":
        uid = q.from_user.id
        st = user_state.get(uid) or {}
        code = st.get("code","")
        qty = st.get("quantity", 0)
        comment = st.get("comment","")
        # Набираем part из df
        part = None
        if data.df is not None and not data.df.empty:
            for _, row in data.df.iterrows():
                if str(row.get("код","")).strip() == code:
                    part = row.to_dict()
                    break
        if not part:
            await _safe_send_html_message(context.bot, q.message.chat_id, "Не удалось найти деталь по коду.")
            return ConversationHandler.END
        await save_issue_to_sheet(context.bot, q.from_user, part, qty, comment)
        await _safe_send_html_message(context.bot, q.message.chat_id, "✅ Списание сохранено.", reply_markup=main_menu_markup())
        user_state.pop(uid, None)
        return ConversationHandler.END
    elif data_cb in {"confirm_no","cancel_action"}:
        await _safe_send_html_message(context.bot, q.message.chat_id, "Отменено.", reply_markup=main_menu_markup())
        user_state.pop(q.from_user.id, None)
        return ConversationHandler.END
    elif data_cb == "more":
        await on_more(update, context)
    elif data_cb.startswith("menu_"):
        await on_menu(update, context)

# ---- Conversation steps ----
async def on_ask_quantity(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip().replace(",", ".")
    try:
        qty = float(text)
        if qty <= 0 or qty > MAX_QTY:
            raise ValueError
    except Exception:
        await _safe_send_html_message(context.bot, update.effective_chat.id, f"Введите число от 0 до {MAX_QTY}.", reply_markup=cancel_markup())
        return ASK_QUANTITY
    return await _ask_comment(update, context, qty)

async def on_ask_comment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    comment = (update.message.text or "").strip()
    return await _confirm_issue(update, context, comment)

# ---- Service ----
async def reload_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMINS:
        return
    await data.ensure_fresh_data_async(True)
    await _safe_send_html_message(context.bot, update.effective_chat.id, "Данные обновлены.")

async def reload_users(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMINS:
        return
    await data.ensure_users_async(True)
    await _safe_send_html_message(context.bot, update.effective_chat.id, "Пользователи обновлены.")

def build_handlers():
    conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(on_callback, pattern=r"^(issue\|.*|confirm_yes|confirm_no|cancel_action)$")],
        states={
            ASK_QUANTITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, on_ask_quantity)],
            ASK_COMMENT: [MessageHandler(filters.TEXT & ~filters.COMMAND, on_ask_comment)],
        },
        fallbacks=[CallbackQueryHandler(on_callback, pattern=r"^(cancel_action)$")],
        per_message=False,
        per_user=True,
        per_chat=True,
    )
    return [
        CommandHandler("start", start),
        CommandHandler("reload", reload_data),
        CommandHandler("users", reload_users),
        MessageHandler(filters.TEXT & ~filters.COMMAND, on_text),
        CallbackQueryHandler(on_callback)
    ] + [conv]
