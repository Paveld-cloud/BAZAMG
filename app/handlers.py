# app/handlers.py
from __future__ import annotations

import logging
from typing import Any, Dict, List
from urllib.parse import quote_plus, unquote_plus

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)

from app import data

logger = logging.getLogger("bot.handlers")

# ===== Настройки =====
PAGE_SIZE = 5  # сколько карточек показываем за раз

# ===== Утилиты вывода =====
def _bold(s: str) -> str:
    s = str(s or "").strip()
    return f"<b>{s}</b>" if s else ""

def _val(row: Dict[str, Any], *keys: str, default: str = "") -> str:
    for k in keys:
        if k in row and str(row.get(k, "")).strip():
            return str(row.get(k)).strip()
    return default

def _build_caption(row: Dict[str, Any]) -> str:
    """
    Строгий порядок:
    1) <b>Заголовок</b>  (наименование)
    2) Описание (если есть)
    3) Цена: ...
    """
    title = _val(row, "наименование", "name", default="(без названия)")
    desc  = _val(row, "описание", "description", default="")
    price = _val(row, "цена", "price", default="")
    parts = [_bold(title)]
    if desc:
        parts.append(desc)
    if price:
        parts.append(f"Цена: {price}")
    return "\n".join(parts)

def _kb_for_row(row: Dict[str, Any], row_idx: int) -> InlineKeyboardMarkup:
    title = _val(row, "наименование", "name", default="")
    btns = [
        [InlineKeyboardButton("🛒 Взять деталь", callback_data=f"take:{row_idx}")],
        [
            InlineKeyboardButton("🪴 Уход", callback_data=f"care:{quote_plus(title)}"),
            InlineKeyboardButton("📜 История", callback_data=f"hist:{quote_plus(title)}"),
        ],
    ]
    return InlineKeyboardMarkup(btns)

def _persist_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [
            [KeyboardButton("🔎 Поиск")],
            [KeyboardButton("📦 Связаться"), KeyboardButton("📜 Заказы")],
        ],
        resize_keyboard=True,
        is_persistent=True,
    )

async def _send_card(update: Update, context: ContextTypes.DEFAULT_TYPE, row: Dict[str, Any], row_idx: int) -> None:
    chat_id = update.effective_chat.id
    caption = _build_caption(row)
    kb = _kb_for_row(row, row_idx)

    # Пробуем фото: сначала прямое поле image (с нормализацией в data.py), затем подбор по коду
    code = _val(row, "код", "code", "парт номер", "oem парт номер", default="")
    image_url = ""
    raw_img = _val(row, "image", default="")
    if raw_img:
        image_url = await data.resolve_image_url_async(raw_img)
    if not image_url and code:
        image_url = await data.find_image_by_code_async(code)

    try:
        if image_url:
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=image_url,
                caption=caption,
                parse_mode=ParseMode.HTML,
                reply_markup=kb,
            )
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text=caption,
                parse_mode=ParseMode.HTML,
                reply_markup=kb,
            )
    except Exception:
        logger.exception("send_card failed; fallback to text")
        await context.bot.send_message(
            chat_id=chat_id,
            text=caption,
            parse_mode=ParseMode.HTML,
            reply_markup=kb,
        )

def _reset_issue(chat_id: int):
    if chat_id in data.issue_state:
        data.issue_state.pop(chat_id, None)

# ===== Команды =====
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await update.message.reply_text(
        "Привет! Напиши название, код или OEM детали.\n"
        f"Показываю по {PAGE_SIZE} результатов за раз — продолжить можно командой /more.",
        reply_markup=_persist_menu(),
    )
    try:
        data.record_user(user.id, user.first_name or "", user.username or "", "START")
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
        await _send_card(update, context, row, idx)

    st["offset"] = offset + len(batch)

# ===== Основной поиск (текст) =====
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    text = update.message.text.strip()

    # сбрасываем незавершённое списание новым запросом
    _reset_issue(update.effective_chat.id)

    if text == "🔎 Поиск":
        await update.message.reply_text("Введи название/код или OEM…")
        return

    query = text
    # лог пользователя
    try:
        user = update.effective_user
        data.record_user(user.id, user.first_name or "", user.username or "", query)
    except Exception:
        logger.debug("record_user failed (not critical)")

    # используем готовый ранжированный поиск из data.py
    idxs = data.find_rows(query, limit=200)
    if not idxs:
        await update.message.reply_text("Ничего не нашёл. Попробуй изменить запрос.")
        return

    chat_id = update.effective_chat.id
    st = data.user_state.setdefault(chat_id, {})
    st["results"] = idxs
    st["offset"] = 0

    batch = idxs[:PAGE_SIZE]
    for idx in batch:
        row = data.get_row(idx)
        await _send_card(update, context, row, idx)
    st["offset"] = len(batch)

    if len(idxs) > PAGE_SIZE:
        await update.message.reply_text(f"Показал {PAGE_SIZE} из {len(idxs)}. Продолжить → /more")

# ===== Callback’и карточек =====
async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cq = update.callback_query
    if not cq or not cq.data:
        return
    data_str = cq.data
    chat_id = cq.message.chat.id

    # 1) Взять деталь → запрашиваем количество
    if data_str.startswith("take:"):
        try:
            row_idx = int(data_str.split(":", 1)[1])
        except Exception:
            await cq.answer("Некорректные данные.")
            return
        row = data.get_row(row_idx)
        title = _val(row, "наименование", "name", default="")
        code  = _val(row, "код", "code", "парт номер", "oem парт номер", default="")
        data.issue_state[chat_id] = {
            "stage": data.ASK_QUANTITY,
            "row_idx": row_idx,
            "title": title,
            "code": code,
            "qty": None,
            "comment": "",
        }
        await cq.answer()
        await cq.message.reply_text(
            f"Сколько списать для: <b>{title}</b> (код: {code})?",
            parse_mode=ParseMode.HTML,
        )
        return

    # 2) Заглушки «Уход/История» (оставлены для совместимости)
    if data_str.startswith("care:"):
        title = unquote_plus(data_str.split(":", 1)[1])
        await cq.answer()
        await cq.message.reply_text(
            f"🪴 Раздел «Уход» для: <b>{title}</b>\n(можно подключить позже)",
            parse_mode=ParseMode.HTML,
        )
        return

    if data_str.startswith("hist:"):
        title = unquote_plus(data_str.split(":", 1)[1])
        await cq.answer()
        await cq.message.reply_text(
            f"📜 Раздел «История» для: <b>{title}</b>\n(можно подключить позже)",
            parse_mode=ParseMode.HTML,
        )
        return

# ===== Диалог списания (кол-во → комментарий → подтверждение) =====
async def on_any_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Перехватываем сообщения, когда пользователь находится в процессе «Взять деталь».
    Если не в процессе — пропускаем дальше.
    """
    if not update.message or not update.message.text:
        return
    chat_id = update.effective_chat.id
    st = data.issue_state.get(chat_id)
    if not st:
        return  # не вмешиваемся — пусть обработает on_text

    text = update.message.text.strip()
    stage = st.get("stage")

    # 1) Количество
    if stage == data.ASK_QUANTITY:
        try:
            qty = float(text.replace(",", "."))
            if qty <= 0:
                raise ValueError
        except Exception:
            await update.message.reply_text("Введи положительное число. Пример: 1 или 2.5")
            return
        st["qty"] = qty
        st["stage"] = data.ASK_COMMENT
        await update.message.reply_text("Добавь комментарий (или напиши «-» для пустого):")
        return

    # 2) Комментарий → подтверждение
    if stage == data.ASK_COMMENT:
        comment = "" if text == "-" else text
        st["comment"] = comment
        st["stage"] = data.ASK_CONFIRM

        title = st.get("title") or ""
        code  = st.get("code") or ""
        qty   = st.get("qty")
        kb = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("✅ Подтвердить", callback_data="confirm_yes"),
                    InlineKeyboardButton("❌ Отмена", callback_data="confirm_no"),
                ]
            ]
        )
        await update.message.reply_text(
            f"Подтвердить списание?\n\n"
            f"<b>{title}</b>\nКод: <code>{code}</code>\nКол-во: <b>{qty}</b>\nКомментарий: {comment or '—'}",
            parse_mode=ParseMode.HTML,
            reply_markup=kb,
        )
        return

async def on_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cq = update.callback_query
    if not cq or not cq.data:
        return
    chat_id = cq.message.chat.id
    st = data.issue_state.get(chat_id)
    if not st:
        await cq.answer("Нечего подтверждать.")
        return

    if cq.data == "confirm_no":
        data.issue_state.pop(chat_id, None)
        await cq.answer("Отменено.")
        await cq.message.edit_reply_markup(reply_markup=None)
        return

    if cq.data == "confirm_yes":
        try:
            user = cq.from_user
            title_or_code = st.get("code") or st.get("title") or ""
            qty = st.get("qty")
            comment = st.get("comment") or ""
            data.record_history(user.id, title_or_code, qty, comment)
        except Exception:
            logger.exception("record_history failed")
        finally:
            data.issue_state.pop(chat_id, None)

        await cq.answer("Записано.")
        await cq.message.edit_reply_markup(reply_markup=None)
        await cq.message.reply_text("✅ Готово.")
        return

# ===== Регистрация =====
def register_handlers(app: Application):
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("more", cmd_more))
    app.add_handler(CommandHandler("reload", cmd_reload))

    # Подтверждение/отмена списания
    app.add_handler(CallbackQueryHandler(on_confirm, pattern="^confirm_(yes|no)$"))

    # Основные callback’и карточек
    app.add_handler(CallbackQueryHandler(on_callback, pattern="^(take:|care:|hist:)"))

    # Перехват сообщений в процессе списания (до общего текстового фильтра)
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), on_any_message), group=0)

    # Общий текстовый вход (поиск)
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), on_text), group=1)

    logger.info("Handlers registered.")

