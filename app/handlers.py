from __future__ import annotations

import html
import logging
from typing import Any, Dict, List

from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)

from app import data

logger = logging.getLogger("bot.handlers")

PAGE_SIZE = 5  # сколько карточек показываем за раз

# ---------- утилиты ----------
def _val(row: Dict[str, Any], *keys: str, default: str = "") -> str:
    for k in keys:
        v = row.get(k, "")
        if str(v).strip():
            return str(v).strip()
    return default

def _persist_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup([[KeyboardButton("🔎 Поиск")]], resize_keyboard=True, is_persistent=True)

def _build_caption(row: Dict[str, Any]) -> str:
    # экранируем спецсимволы, чтобы HTML не ломался
    title = html.escape(_val(row, "наименование", "name", default="(без названия)"))
    desc  = html.escape(_val(row, "описание", "description", default=""))
    price = html.escape(_val(row, "цена", "price", default=""))

    parts = [f"<b>{title}</b>"]
    if desc:
        parts.append(desc)
    if price:
        parts.append(f"Цена: {price}")
    return "\n".join(parts)

async def _send_card(update: Update, context: ContextTypes.DEFAULT_TYPE, row: Dict[str, Any]) -> None:
    chat_id = update.effective_chat.id
    caption = _build_caption(row)

    # Фото: сначала поле image, затем попытка подобрать по коду
    code = _val(row, "код", "code", "парт номер", "oem парт номер", default="")
    image_url = ""
    raw_img = _val(row, "image", default="")
    if raw_img:
        image_url = await data.resolve_image_url_async(raw_img)
    if not image_url and code:
        image_url = await data.find_image_by_code_async(code)

    try:
        if image_url:
            await context.bot.send_photo(chat_id=chat_id, photo=image_url, caption=caption, parse_mode=ParseMode.HTML)
        else:
            await context.bot.send_message(chat_id=chat_id, text=caption, parse_mode=ParseMode.HTML)
    except Exception:
        logger.exception("send_card failed; fallback to text")
        await context.bot.send_message(chat_id=chat_id, text=caption, parse_mode=ParseMode.HTML)

# ---------- команды ----------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Привет! Напиши название, код или OEM детали.\nПоказываю по {PAGE_SIZE} результатов за раз — продолжай командой /more.",
        reply_markup=_persist_menu(),
    )
    try:
        u = update.effective_user
        data.record_user(u.id, u.first_name or "", u.username or "", "START")
    except Exception:
        logger.debug("record_user on /start failed")

async def cmd_reload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        data.ensure_fresh_data(force=True)
        await update.message.reply_text(f"Кэш обновлён. В таблице: {data.sap_count()} строк.")
    except Exception as e:
        await update.message.reply_text(f"Ошибка обновления: {e}")

async def cmd_more(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = data.user_state.setdefault(chat_id, {})
    results: List[int] = st.get("results") or []
    offset: int = st.get("offset", 0)

    if not results:
        await update.message.reply_text("Нет сохранённого поиска. Введите запрос заново.")
        return

    batch = results[offset:offset + PAGE_SIZE]
    if not batch:
        await update.message.reply_text("Больше результатов нет.")
        return

    for idx in batch:
        row = data.get_row(idx)
        await _send_card(update, context, row)

    st["offset"] = offset + len(batch)

# ---------- текст (поиск) ----------
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    text = update.message.text.strip()

    if text == "🔎 Поиск":
        await update.message.reply_text("Введи название/код или OEM…")
        return

    # лог запроса
    try:
        u = update.effective_user
        data.record_user(u.id, u.first_name or "", u.username or "", text)
    except Exception:
        logger.debug("record_user failed (not critical)")

    idxs = data.find_rows(text, limit=200)
    if not idxs:
        await update.message.reply_text("Ничего не нашёл. Попробуй изменить запрос.")
        return

    chat_id = update.effective_chat.id
    st = data.user_state.setdefault(chat_id, {})
    st["results"] = idxs
    st["offset"] = 0

    # первая порция
    batch = idxs[:PAGE_SIZE]
    for idx in batch:
        row = data.get_row(idx)
        await _send_card(update, context, row)
    st["offset"] = len(batch)

    if len(idxs) > PAGE_SIZE:
        await update.message.reply_text(f"Показал {PAGE_SIZE} из {len(idxs)}. Продолжить → /more")

# ---------- регистрация ----------
def register_handlers(app: Application):
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("reload", cmd_reload))
    app.add_handler(CommandHandler("more", cmd_more))

    # один общий обработчик текста (поиск)
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), on_text))

    logger.info("Handlers registered (minimal, no inline buttons).")
