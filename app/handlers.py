# handlers.py
import logging
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ContextTypes, ConversationHandler

from config import TIMEZONE
from data import (
    search_parts,
    save_issue_to_history,
    get_user_by_id,
)

# ===== Логирование =====
logger = logging.getLogger(__name__)

# ===== Этапы диалога списания =====
ASK_QUANTITY, ASK_COMMENT, ASK_CONFIRM = range(3)

# Состояния пользователей
user_state = {}   # user_id: { "part": {...}, "quantity": int, "comment": str }

# ===== Утилиты =====
def normalize_text(text: str) -> str:
    """Приводим текст к единому виду для поиска"""
    return re.sub(r'[^a-zA-Z0-9а-яА-ЯёЁ]', '', text).lower()

# ===== Обработчики =====
async def handle_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Поиск деталей по сообщению"""
    user = update.effective_user
    query = update.message.text.strip()

    # Проверяем доступ
    if not get_user_by_id(user.id):
        await update.message.reply_text("⛔ У вас нет доступа к боту.")
        return

    results = search_parts(query)
    if not results:
        await update.message.reply_text("❌ Детали не найдены.")
        return

    for part in results:
        text = (
            f"🔎 *{part.get('Description', 'Без названия')}*\n"
            f"Код: `{part.get('Ref Des', '-')}`\n"
            f"Производитель: {part.get('MFR_MGD_1', '-')}\n"
            f"MPN: {part.get('MPN_MGD_1', '-')}\n"
            f"Количество на станции: {part.get('Qty per station', '-')}"
        )

        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("📦 Взять деталь", callback_data=f"take|{part.get('Ref Des', '')}")]
        ])

        await update.message.reply_photo(
            photo=part.get("image") or "https://via.placeholder.com/300x200.png?text=No+Image",
            caption=text,
            parse_mode="Markdown",
            reply_markup=keyboard
        )

async def handle_take_detail(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Нажатие на кнопку 'Взять деталь'"""
    query = update.callback_query
    await query.answer()

    user = update.effective_user
    _, ref_des = query.data.split("|", 1)

    part = context.bot_data.get("parts_index", {}).get(ref_des)
    if not part:
        await query.edit_message_caption(caption="❌ Ошибка: деталь не найдена.")
        return

    user_state[user.id] = {"part": part}
    await query.message.reply_text("✍ Введите количество:")
    return ASK_QUANTITY

async def handle_quantity(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Получаем количество"""
    user = update.effective_user
    text = update.message.text.strip()

    if not text.isdigit() or int(text) <= 0:
        await update.message.reply_text("⚠ Введите корректное число.")
        return ASK_QUANTITY

    user_state[user.id]["quantity"] = int(text)
    await update.message.reply_text("📝 Введите комментарий (например: причина списания):")
    return ASK_COMMENT

async def handle_comment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Получаем комментарий и спрашиваем подтверждение"""
    user = update.effective_user
    comment = update.message.text.strip()

    user_state[user.id]["comment"] = comment

    part = user_state[user.id]["part"]
    qty = user_state[user.id]["quantity"]

    text = (
        f"📋 Подтвердите списание:\n\n"
        f"🔎 {part.get('Description', '-')}\n"
        f"Код: {part.get('Ref Des', '-')}\n"
        f"Количество: {qty}\n"
        f"Комментарий: {comment}"
    )

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Да", callback_data="confirm_yes"),
            InlineKeyboardButton("❌ Нет", callback_data="confirm_no")
        ]
    ])

    await update.message.reply_text(text, reply_markup=keyboard)
    return ASK_CONFIRM

async def handle_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Подтверждение списания"""
    query = update.callback_query
    await query.answer()
    user = update.effective_user

    if query.data == "confirm_yes":
        state = user_state.get(user.id)
        if not state:
            await query.edit_message_text("❌ Ошибка: данные не найдены.")
            return ConversationHandler.END

        # Сохраняем в историю
        save_issue_to_history(
            user_id=user.id,
            username=user.username,
            part=state["part"],
            quantity=state["quantity"],
            comment=state["comment"],
            date=datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S")
        )

        await query.edit_message_text("✅ Списание сохранено.")
        user_state.pop(user.id, None)

    else:
        await query.edit_message_text("❌ Отменено пользователем.")
        user_state.pop(user.id, None)

    return ConversationHandler.END
