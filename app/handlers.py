# app/handlers.py
import io
import math
import asyncio
import logging
from html import escape

from telegram import Update, InputFile, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    CommandHandler, MessageHandler, CallbackQueryHandler,
    ConversationHandler, ContextTypes, filters, ApplicationHandlerStop
)

# берем конфиг
from app.config import (
    PAGE_SIZE, MAX_QTY,
    WELCOME_ANIMATION_URL, WELCOME_PHOTO_URL, SUPPORT_CONTACT, WELCOME_MEDIA_ID,
)

# ВАЖНО: импортируем модуль целиком, чтобы всегда видеть АКТУАЛЬНЫЙ df
import app.data as data

# используем только функции/структуры (без df)
from app.data import (
    user_state, issue_state,
    ensure_fresh_data, ensure_fresh_data_async,
    format_row, normalize, squash, match_row_by_index, _safe_col, _relevance_score,
    find_image_by_code_async, resolve_image_url_async,
    val, now_local_str, get_gs_client,
    load_users_from_sheet, SHEET_ALLOWED, SHEET_ADMINS, SHEET_BLOCKED,
    ASK_QUANTITY, ASK_COMMENT, ASK_CONFIRM,
    is_admin, is_allowed, _df_to_xlsx,
)
from app.config import ADMINS  # локальные админы из конфига
from app.config import SPREADSHEET_URL  # ссылка на таблицу для истории

logger = logging.getLogger("bot.handlers")

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
        logger.warning(f"HTML parse failed, fallback to plain: {e}")
        import re
        no_tags = re.sub(r"</?(b|i|code)>", "", text)
        kwargs.pop("parse_mode", None)
        return await bot.send_message(chat_id=chat_id, text=no_tags, **kwargs)

# --------------------- Пользователи (допуски) -----------------
async def ensure_users_async(force: bool = False):
    allowed, admins, blocked = await asyncio.to_thread(load_users_from_sheet)
    SHEET_ALLOWED.clear(); SHEET_ALLOWED.update(allowed)
    SHEET_ADMINS.clear(); SHEET_ADMINS.update(admins)
    SHEET_BLOCKED.clear(); SHEET_BLOCKED.update(blocked)

def ensure_users(force: bool = False):
    asyncio.create_task(ensure_users_async(force=True))

def is_admin_local(uid: int) -> bool:
    ensure_users()
    return uid in SHEET_ADMINS or uid in ADMINS

def is_allowed_local(uid: int) -> bool:
    ensure_users()
    if uid in SHEET_BLOCKED:
        return False
    if SHEET_ALLOWED:
        return (uid in SHEET_ALLOWED) or (uid in SHEET_ADMINS) or (uid in ADMINS)
    return True

# --------------------- Гварды -----------------
async def guard_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if user and not is_allowed_local(user.id):
        try:
            await update.effective_message.reply_text("Доступ запрещён.")
        except Exception:
            pass
        raise ApplicationHandlerStop

async def guard_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if user and not is_allowed_local(user.id):
        try:
            await update.callback_query.answer("Доступ запрещён.", show_alert=True)
        except Exception:
            pass
        raise ApplicationHandlerStop

# --------------------- Приветствие -----------------
async def send_welcome_sequence(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user = update.effective_user
    first = escape((user.first_name or "").strip() or "коллега")

    # 1) анимация (если это реально gif/mp4; фото сюда не подойдёт)
    if WELCOME_ANIMATION_URL:
        try:
            await context.bot.send_animation(
                chat_id=chat_id,
                animation=WELCOME_ANIMATION_URL,
                caption=f"⚙️ Добро пожаловать, {first}!"
            )
            await asyncio.sleep(0.3)
        except Exception as e:
            logger.warning(f"Welcome animation failed: {e}")

    # 2) фото по file_id (надежнее всего)
    sent_media = False
    if WELCOME_MEDIA_ID:
        try:
            await context.bot.send_photo(chat_id=chat_id, photo=WELCOME_MEDIA_ID, disable_notification=True)
            sent_media = True
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.warning(f"Welcome photo by file_id failed: {e}")

    # 3) фото по URL
    if not sent_media and WELCOME_PHOTO_URL:
        try:
            await context.bot.send_photo(chat_id=chat_id, photo=WELCOME_PHOTO_URL, disable_notification=True)
            sent_media = True
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.warning(f"Welcome photo by URL/file_id failed: {e}")

    # 4) карточка
    card_html = (
        f"⚙️ <b>Привет, {first}!</b>\n"
        f"<i>Инженерный бот для поиска и списания деталей</i>\n"
        f"────────\n"
        f"• Введите <code>название</code>, <code>код</code> или <code>модель</code>\n"
        f"• Откройте карточку и нажмите «📦 Взять деталь»\n"
        f"• Подтвердите списание — и готово\n\n"
        f"Пример: <code>PI 8808 DRG 500</code>\n"
        f"Удачной работы! 🚀"
    )
    await _safe_send_html_message(context.bot, chat_id, card_html, reply_markup=main_menu_markup())

# callbacks из приветственного меню
async def menu_search_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    msg = "🔍 Введите запрос: <i>название</i>/<i>модель</i>/<i>код</i>.\nПример: <code>PI 8808 DRG 500</code>"
    await _safe_send_html_message(context.bot, q.message.chat_id, msg)

async def menu_issue_help_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    msg = (
        "<b>Как списать деталь</b>:\n"
        "1) Выполните поиск по названию/коду.\n"
        "2) В карточке нажмите «📦 Взять деталь».\n"
        "3) Укажите количество и комментарий.\n"
        "4) Подтвердите списание кнопкой «Да»."
    )
    await _safe_send_html_message(context.bot, q.message.chat_id, msg)

async def menu_contact_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    await q.message.reply_text(f"{SUPPORT_CONTACT}")

# --------------------- Фото карточки -----------------
async def send_row_with_image(update: Update, row: dict, text: str):
    # показываем фото ТОЛЬКО если нашли по коду (не из строки «image»)
    code = str(row.get("код", "")).strip().lower()
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("📦 Взять деталь", callback_data=f"issue:{code}")]])

    url_raw = await find_image_by_code_async(code)
    if url_raw:
        url = await resolve_image_url_async(url_raw)
        if url:
            try:
                await update.message.reply_photo(photo=url, caption=text, reply_markup=kb)
                return
            except Exception as e:
                logger.warning(f"URL фото не сработал ({url}): {e}")
    await update.message.reply_text(text, reply_markup=kb)

async def send_row_with_image_bot(bot, chat_id: int, row: dict, text: str):
    code = str(row.get("код", "")).strip().lower()
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("📦 Взять деталь", callback_data=f"issue:{code}")]])

    url_raw = await find_image_by_code_async(code)
    if url_raw:
        url = await resolve_image_url_async(url_raw)
        if url:
            try:
                await bot.send_photo(chat_id=chat_id, photo=url, caption=text, reply_markup=kb)
                return
            except Exception as e:
                logger.warning(f"URL фото не сработал ({url}): {e}")
    await bot.send_message(chat_id=chat_id, text=text, reply_markup=kb)

# --------------------- Команды -----------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    issue_state.pop(uid, None)
    user_state.pop(uid, None)
    await send_welcome_sequence(update, context)
    if update.message:
        await asyncio.sleep(0.2)
        cmds_html = (
            "<b>Команды</b>:\n"
            "• <code>/help</code> — помощь\n"
            "• <code>/more</code> — показать ещё\n"
            "• <code>/export</code> — выгрузка результатов (XLSX/CSV)\n"
            "• <code>/cancel</code> — отменить списание\n"
            "• <code>/reload</code> — перезагрузка данных и пользователей (только админ)\n"
        )
        await _safe_send_html_message(context.bot, update.effective_chat.id, cmds_html)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "<b>Как пользоваться</b>:\n"
        "1) Выполните поиск по названию/модели/коду.\n"
        "2) В карточке нажмите «📦 Взять деталь» — бот спросит количество и комментарий.\n"
        "3) Подтвердите списание (Да/Нет).\n"
        "<i>У вас всё получится!</i>"
    )
    await _safe_send_html_message(context.bot, update.effective_chat.id, msg)

async def reload_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not is_admin_local(uid):
        return await update.message.reply_text("Доступ запрещён.")
    ensure_fresh_data(force=True)
    ensure_users(force=True)
    await update.message.reply_text("✅ Данные и пользователи перезагружены (в фоне).")

async def cancel_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if issue_state.pop(uid, None):
        await update.message.reply_text("❌ Операция списания отменена.")
    else:
        await update.message.reply_text("Нет активной операции.")

async def export_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    st = user_state.get(uid, {})
    results = st.get("results")
    if results is None or results.empty:
        return await update.message.reply_text("Сначала выполните поиск.")
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        buf = await asyncio.to_thread(_df_to_xlsx, results, f"export_{timestamp}.xlsx")
        await update.message.reply_document(InputFile(buf, filename=f"export_{timestamp}.xlsx"))
    except Exception as e:
        logger.warning(f"Не удалось XLSX (fallback CSV): {e}")
        csv = results.to_csv(index=False, encoding="utf-8-sig")
        await update.message.reply_document(
            InputFile(io.BytesIO(csv.encode("utf-8-sig")), filename=f"export_{timestamp}.csv")
        )

# --------------------- Разбивка результатов на страницы -----------------
async def send_page(update: Update, uid: int):
    st = user_state.get(uid, {})
    results = st.get("results")
    page = st.get("page", 0)

    total = len(results)
    if total == 0:
        return await update.message.reply_text("Результатов больше нет.")
    pages = max(1, math.ceil(total / PAGE_SIZE))
    if page >= pages:
        st["page"] = pages - 1
        return await update.message.reply_text("Больше результатов нет.")

    start_i = page * PAGE_SIZE
    end_i = min(start_i + PAGE_SIZE, total)

    await update.message.reply_text(f"Стр. {page+1}/{pages}. Показываю {start_i + 1}–{end_i} из {total}.")
    for _, row in results.iloc[start_i:end_i].iterrows():
        await send_row_with_image(update, row.to_dict(), format_row(row.to_dict()))
    if end_i < total:
        await update.message.reply_text("Показать ещё?", reply_markup=more_markup())

async def send_page_via_bot(bot, chat_id: int, uid: int):
    st = user_state.get(uid, {})
    results = st.get("results")
    page = st.get("page", 0)
    total = len(results)
    if total == 0:
        return await bot.send_message(chat_id=chat_id, text="Результатов больше нет.")
    pages = max(1, math.ceil(total / PAGE_SIZE))
    if page >= pages:
        st["page"] = pages - 1
        return await bot.send_message(chat_id=chat_id, text="Больше результатов нет.")
    start_i = page * PAGE_SIZE
    end_i = min(start_i + PAGE_SIZE, total)
    await bot.send_message(chat_id=chat_id, text=f"Стр. {page+1}/{pages}. Показываю {start_i + 1}–{end_i} из {total}.")
    chunk = results.iloc[start_i:end_i]
    for _, row in chunk.iterrows():
        await send_row_with_image_bot(bot, chat_id, row.to_dict(), format_row(row.to_dict()))
    if end_i < total:
        await bot.send_message(chat_id=chat_id, text="Показать ещё?", reply_markup=more_markup())

# --------------------- Поиск -----------------
async def search_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is None:
        return

    # подавим ложный поиск во время диалога списания
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

    q = (update.message.text or "").strip()
    if not q:
        return await update.message.reply_text("Введите запрос.")
    tokens = normalize(q).split()
    if not tokens:
        return await update.message.reply_text("Введите более конкретный запрос.")
    q_squash = squash(q)

    # ЖИВОЙ доступ к df
    if data.df is None:
        await ensure_fresh_data_async(force=True)
        if data.df is None:
            return await update.message.reply_text("Ошибка загрузки данных.")
    cur_df = data.df

    matched_indices = match_row_by_index(tokens)

    from pandas import Series
    if not matched_indices:
        mask_any = Series(False, index=cur_df.index)
        for col in ["тип", "наименование", "код", "oem", "изготовитель"]:
            series = _safe_col(cur_df, col)
            if series is None:
                continue
            field_mask = Series(True, index=cur_df.index)
            for t in tokens:
                if t:
                    field_mask &= series.str.contains(t, na=False)
            mask_any |= field_mask
        matched_indices = set(cur_df.index[mask_any])

    if not matched_indices and q_squash:
        mask_any = Series(False, index=cur_df.index)
        for col in ["тип", "наименование", "код", "oem", "изготовитель"]:
            series = _safe_col(cur_df, col)
            if series is None:
                continue
            series_sq = series.str.replace(r'[\W_]+', '', regex=True)
            mask_any |= series_sq.str.contains(q_squash, na=False)
        matched_indices = set(cur_df.index[mask_any])

    if not matched_indices:
        return await update.message.reply_text(f"По запросу «{q}» ничего не найдено.")

    idx_list = list(matched_indices)
    results_df = cur_df.loc[idx_list].copy()

    scores = []
    for _, r in results_df.iterrows():
        scores.append(_relevance_score(r.to_dict(), tokens, q_squash))
    results_df["__score"] = scores

    if "код" in results_df.columns:
        results_df = results_df.sort_values(
            by=["__score", "код"],
            ascending=[False, True],
            key=lambda s: s if s.name != "код" else s.astype(str).str.len()
        )
    else:
        results_df = results_df.sort_values(by=["__score"], ascending=False)
    results_df = results_df.drop(columns="__score")

    st = user_state.setdefault(uid, {})
    st["query"] = q
    st["results"] = results_df
    st["page"] = 0

    await send_page(update, uid)

# ------------------ Списание (диалог) -----------------
async def on_issue_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    uid = q.from_user.id
    code = q.data.split(":", 1)[1].strip().lower()

    cur_df = data.df
    found = None
    if cur_df is not None and "код" in cur_df.columns:
        hit = cur_df[cur_df["код"].astype(str).str.lower() == code]
        if not hit.empty:
            found = hit.iloc[0].to_dict()

    if not found:
        return await q.edit_message_text("Не удалось найти деталь по коду. Выполните поиск заново.")

    issue_state[uid] = {"part": found}
    await q.message.reply_text("Сколько списать? Укажите число (например: 1 или 2.5).", reply_markup=cancel_markup())
    return ASK_QUANTITY

async def handle_quantity(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.chat_data["suppress_next_search"] = True
    uid = update.effective_user.id
    text = (update.message.text or "").strip().replace(",", ".")
    try:
        qty = float(text)
        if not math.isfinite(qty) or qty <= 0 or qty > MAX_QTY:
            raise ValueError
        qty = float(f"{qty:.3f}")
    except Exception:
        return await update.message.reply_text(
            f"Введите число > 0 и ≤ {MAX_QTY}. Пример: 1 или 2.5",
            reply_markup=cancel_markup()
        )
    st = issue_state.get(uid)
    if not st or "part" not in st:
        return await update.message.reply_text("Списание неактивно — начните заново из карточки.")
    st["quantity"] = qty
    st["await_comment"] = True
    await update.message.reply_text("Добавьте комментарий (например: Линия сборки CSS OP-1100).", reply_markup=cancel_markup())
    return ASK_COMMENT

async def handle_comment(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
    st["comment"] = "" if comment == "-" else comment
    st["await_comment"] = False

    text = (
        "Вы уверены, что хотите списать деталь?\n\n"
        f"🔢 Код: {val(part, 'код')}\n"
        f"📦 Наименование: {val(part, 'наименование')}\n"
        f"📦 Кол-во: {qty}\n"
        f"💬 Комментарий: {st['comment'] or '—'}"
    )
    await update.message.reply_text(text, reply_markup=confirm_markup())
    return ASK_CONFIRM

async def save_issue_to_sheet(bot, user, part: dict, quantity, comment: str):
    import gspread
    client = get_gs_client()
    sh = client.open_by_url(SPREADSHEET_URL)
    try:
        ws = sh.worksheet("История")
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="История", rows=1000, cols=12)
        ws.append_row(["Дата", "ID", "Имя", "Тип", "Наименование", "Код", "Количество", "Коментарий"])

    headers_raw = ws.row_values(1)
    headers = [h.strip() for h in headers_raw]
    norm = [h.lower() for h in headers]

    full_name = f"{(user.first_name or '').strip()} {(user.last_name or '').strip()}".strip()
    display_name = full_name or (f"@{user.username}" if user.username else str(user.id))
    ts = now_local_str()

    values_by_key = {
        "дата": ts, "timestamp": ts,
        "id": user.id, "user_id": user.id,
        "имя": display_name, "name": display_name,
        "тип": str(part.get("тип", "")), "type": str(part.get("тип", "")),
        "наименование": str(part.get("наименование", "")), "name_item": str(part.get("наименование", "")),
        "код": str(part.get("код", "")), "code": str(part.get("код", "")),
        "数量": str(quantity), "количество": str(quantity), "qty": str(quantity),
        "коментарий": comment or "", "комментарий": comment or "", "comment": comment or "",
    }

    row = [values_by_key.get(hn, "") for hn in norm]
    ws.append_row(row, value_input_option="USER_ENTERED")
    logger.info("💾 Списание записано в 'История'")

async def handle_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    uid = q.from_user.id

    if q.data == "confirm_yes":
        st = issue_state.get(uid)
        if not st or "part" not in st or "quantity" not in st:
            issue_state.pop(uid, None)
            return await q.message.reply_text("Данных для списания нет. Начните заново.")
        part = st["part"]
        qty = st["quantity"]
        comment = st.get("comment", "")

        await save_issue_to_sheet(context.bot, q.from_user, part, qty, comment)
        issue_state.pop(uid, None)

        await q.message.reply_text(
            f"✅ Списано: {qty}\n"
            f"🔢 Код: {val(part, 'код')}\n"
            f"📦 Наименование: {val(part, 'наименование')}\n"
            f"💬 Комментарий: {comment or '—'}"
        )
        return ConversationHandler.END

    if q.data == "confirm_no":
        issue_state.pop(uid, None)
        await q.message.reply_text("❌ Списание отменено.")
        return ConversationHandler.END

async def cancel_action(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    uid = q.from_user.id
    if uid in issue_state:
        issue_state.pop(uid, None)
        await q.message.reply_text("❌ Операция списания отменена.")
    return ConversationHandler.END

async def handle_cancel_in_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cancel_cmd(update, context)
    return ConversationHandler.END

async def on_more_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    uid = q.from_user.id
    st = user_state.get(uid, {})
    results = st.get("results")
    if results is None or results.empty:
        return await q.message.reply_text("Сначала выполните поиск.")
    st["page"] = st.get("page", 0) + 1
    chat_id = q.message.chat.id
    await send_page_via_bot(context.bot, chat_id, uid)

# --------------------- Регистрация хендлеров -----------------
def register_handlers(app):
    # гварды
    app.add_handler(MessageHandler(filters.ALL, guard_msg), group=-1)
    app.add_handler(CallbackQueryHandler(guard_cb, pattern=".*"), group=-1)

    # команды и меню
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("more", export_cmd))  # опечатка? оставим только /export и /more ниже
    app.add_handler(CommandHandler("export", export_cmd))
    app.add_handler(CommandHandler("reload", reload_cmd))
    app.add_handler(CommandHandler("cancel", cancel_cmd))

    # кнопки меню
    app.add_handler(CallbackQueryHandler(menu_search_cb, pattern=r"^menu_search$"))
    app.add_handler(CallbackQueryHandler(menu_issue_help_cb, pattern=r"^menu_issue_help$"))
    app.add_handler(CallbackQueryHandler(menu_contact_cb, pattern=r"^menu_contact$"))

    # пагинация
    app.add_handler(CallbackQueryHandler(on_more_click, pattern=r"^more$"))
    app.add_handler(CallbackQueryHandler(cancel_action, pattern=r"^cancel_action$"))

    # диалог списания
    conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(on_issue_click, pattern=r"^issue:")],
        states={
            ASK_QUANTITY: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_quantity),
                CallbackQueryHandler(cancel_action, pattern=r"^cancel_action$")
            ],
            ASK_COMMENT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_comment),
                CallbackQueryHandler(cancel_action, pattern=r"^cancel_action$")
            ],
            ASK_CONFIRM: [
                CallbackQueryHandler(handle_confirm, pattern=r"^confirm_(yes|no)$"),
                CallbackQueryHandler(cancel_action, pattern=r"^cancel_action$")
            ],
        },
        fallbacks=[CommandHandler("cancel", handle_cancel_in_dialog)],
        allow_reentry=True,
        per_chat=True,
        per_user=True,
        per_message=False,
    )
    app.add_handler(conv)

    # сам поиск — РЕГИСТРИРУЕМ ПОСЛЕ диалога
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search_text), group=1)
