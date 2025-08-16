# This is a refactor of your monolith handlers into a module.
# It keeps the same behavior: search, images by code, issue flow, export, /fileid, /imgdebug, menu, guards.

import math
import re
import logging
import asyncio
from typing import Dict, Any, List

import pandas as pd
from telegram import Update, InputFile, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    ContextTypes, CommandHandler, MessageHandler, CallbackQueryHandler,
    ConversationHandler, ApplicationHandlerStop, filters
)

from config import (
    PAGE_SIZE, MAX_QTY,
    WELCOME_ANIMATION_URL, WELCOME_PHOTO_URL, WELCOME_MEDIA_ID, SUPPORT_CONTACT
)
from utils import normalize, squash, format_row, safe_send_html_message, df_to_xlsx, to_thread, val
from data import (
    ensure_fresh_data, ensure_fresh_data_async, ensure_users, ensure_users_async,
    is_admin, is_allowed, match_row_by_index, get_df,
    find_image_by_code_async, get_admin_ids, get_image_index_copy
)
from ui import cancel_markup, confirm_markup, more_markup, main_menu_markup
from images import resolve_image_url_async, download_image_async
from gsheets import save_issue_to_sheet_blocking
from indexing import SEARCH_FIELDS

logger = logging.getLogger("handlers")

ASK_QUANTITY, ASK_COMMENT, ASK_CONFIRM = range(3)

user_state: Dict[int, Dict[str, Any]] = {}
issue_state: Dict[int, Dict[str, Any]] = {}

async def send_welcome_sequence(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user = update.effective_user
    from html import escape
    first = escape((user.first_name or "").strip() or "коллега")

    if WELCOME_ANIMATION_URL:
        try:
            await context.bot.send_animation(chat_id=chat_id, animation=WELCOME_ANIMATION_URL,
                                             caption=f"⚙️ Добро пожаловать, {first}!")
            await asyncio.sleep(0.3)
        except Exception as e:
            logger.warning(f"Welcome animation failed: {e}")

    sent_media = False
    if WELCOME_MEDIA_ID:
        try:
            await context.bot.send_photo(chat_id=chat_id, photo=WELCOME_MEDIA_ID, disable_notification=True)
            sent_media = True
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.warning(f"Welcome photo by file_id failed: {e}")

    if not sent_media and WELCOME_PHOTO_URL:
        try:
            await context.bot.send_photo(chat_id=chat_id, photo=WELCOME_PHOTO_URL, disable_notification=True)
            sent_media = True
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.warning(f"Welcome photo by URL/file_id failed: {e}")

    card_html = (
        f"⚙️ <b>Привет, {first}!</b>\n"
        f"<b>Бот для поиска и списания деталей</b>\n"
        f"────────\n"
        f"• Введите <code>название</code>, <code>код</code> или <code>модель</code>\n"
        f"• Откройте карточку и нажмите «📦 Взять деталь»\n"
        f"• Подтвердите списание — и готово\n\n"
        f"Пример: <code>PI 8808 DRG 500</code>\n"
        f"Удачной работы! 🚀"
    )
    await safe_send_html_message(context.bot, chat_id, card_html, reply_markup=main_menu_markup())

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
            "• <code>/fileid</code> — получить <i>file_id</i> из присланного медиа\n"
            "• <code>/imgdebug &lt;код&gt;</code> — диагностика поиска фото\n"
        )
        await safe_send_html_message(context.bot, update.effective_chat.id, cmds_html)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "<b>Как пользоваться</b>:\n"
        "1) Выполните поиск по названию/модели/коду.\n"
        "2) В карточке нажмите «📦 Взять деталь» — бот спросит количество и комментарий.\n"
        "3) Подтвердите списание (Да/Нет).\n"
        "<i>У вас всё получится!</i>"
    )
    await safe_send_html_message(context.bot, update.effective_chat.id, msg)

async def reload_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not is_admin(uid):
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
    results = user_state.get(uid, {}).get("results")
    import pandas as pd
    if results is None or results.empty:
        return await update.message.reply_text("Сначала выполните поиск.")
    timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        import openpyxl  # noqa: F401
        buf = df_to_xlsx(results, f"export_{timestamp}.xlsx")
        await update.message.reply_document(InputFile(buf, filename=f"export_{timestamp}.xlsx"))
    except Exception as e:
        logger.warning(f"Не удалось XLSX (fallback CSV): {e}")
        csv = results.to_csv(index=False, encoding="utf-8-sig")
        import io
        await update.message.reply_document(
            InputFile(io.BytesIO(csv.encode("utf-8-sig")), filename=f"export_{timestamp}.csv")
        )

async def fileid_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["await_fileid"] = True
    await update.message.reply_text(
        "Отправьте фото/видео/гиф — отвечу его file_id. "
        "Потом вставьте его в WELCOME_MEDIA_ID / WELCOME_ANIMATION_URL / WELCOME_PHOTO_URL."
    )

async def capture_fileid(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.user_data.get("await_fileid"):
        return
    file_id = None
    kind = None

    if update.message.animation:
        file_id = update.message.animation.file_id
        kind = "animation"
    elif update.message.video:
        file_id = update.message.video.file_id
        kind = "video"
    elif update.message.photo:
        file_id = update.message.photo[-1].file_id
        kind = "photo"

    if file_id:
        context.user_data["await_fileid"] = False
        from html import escape
        await safe_send_html_message(
            context.bot,
            update.effective_chat.id,
            f"✅ {kind} file_id:\n<code>{escape(file_id)}</code>\n\n"
            f"Скопируйте в ENV: WELCOME_MEDIA_ID / WELCOME_ANИMATION_URL / WELCOME_PHОТО_URL."
        )
    else:
        await update.message.reply_text("Это не поддерживаемое медиа. Отправьте фото/видео/гиф.")

async def imgdebug_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = (update.message.text or "").split(maxsplit=1)
    if len(args) < 2:
        return await update.message.reply_text("Использование: /imgdebug <код>")
    code = args[1].strip()
    from indexing import _norm_code
    raw, sq = _norm_code(code)

    idx = get_image_index_copy()
    idx_hit = idx.get(raw) or idx.get(sq)
    sub_hit = None
    if idx and not idx_hit:
        for k, url in idx.items():
            if (sq and sq in k) or (raw and raw in k):
                sub_hit = url
                break

    from data import _scan_images_by_code_fallback as scan_fb
    scan_hit = scan_fb(code) if not (idx_hit or sub_hit) else ""

    from html import escape
    msg = (
        f"🔎 <b>IMGDEBUG</b>\n"
        f"code: <code>{escape(code)}</code>\n"
        f"raw:  <code>{escape(raw)}</code>\n"
        f"sq:   <code>{escape(sq)}</code>\n"
        f"— index direct: {idx_hit or '—'}\n"
        f"— index substring: {sub_hit or '—'}\n"
        f"— df scan fallback: {scan_hit or '—'}"
    )
    await safe_send_html_message(context.bot, update.effective_chat.id, msg)

async def menu_search_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    msg = "🔍 Введите запрос: <i>название</i>/<i>модель</i>/<i>код</i>.\nПример: <code>PI 8808 DRG 500</code>"
    await safe_send_html_message(context.bot, q.message.chat_id, msg)

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
    await safe_send_html_message(context.bot, q.message.chat_id, msg)

async def menu_contact_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    await q.message.reply_text(f"{SUPPORT_CONTACT}")

async def send_row_with_image(update: Update, row: dict, text: str):
    code = str(row.get("код", "")).strip()
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("📦 Взять деталь", callback_data=f"issue:{code.lower()}")]])
    url_raw = await find_image_by_code_async(code)

    if url_raw:
        url = await resolve_image_url_async(url_raw)
        if url:
            try:
                await update.message.reply_photo(photo=url, caption=text, reply_markup=kb)
                return
            except Exception as e:
                logger.warning(f"URL фото не сработал ({url}): {e}")
                bio = await download_image_async(url)
                if bio:
                    try:
                        await update.message.reply_photo(photo=bio, caption=text, reply_markup=kb)
                        return
                    except Exception as e2:
                        logger.warning(f"Скачивание/отправка фото не удалось: {e2} (src: {url})")
    await update.message.reply_text(text, reply_markup=kb)

async def send_row_with_image_bot(bot, chat_id: int, row: dict, text: str):
    code = str(row.get("код", "")).strip()
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("📦 Взять деталь", callback_data=f"issue:{code.lower()}")]])
    url_raw = await find_image_by_code_async(code)

    if url_raw:
        url = await resolve_image_url_async(url_raw)
        if url:
            try:
                await bot.send_photo(chat_id=chat_id, photo=url, caption=text, reply_markup=kb)
                return
            except Exception as e:
                logger.warning(f"URL фото не сработал ({url}): {e}")
                bio = await download_image_async(url)
                if bio:
                    try:
                        await bot.send_photo(chat_id=chat_id, photo=bio, caption=text, reply_markup=kb)
                        return
                    except Exception as e2:
                        logger.warning(f"Отправка скачанного фото не удалась: {e2} (src: {url})")
    await bot.send_message(chat_id=chat_id, text=text, reply_markup=kb)

def _relevance_score(row: dict, tokens: List[str], q_squash: str) -> int:
    score = 0
    for f in SEARCH_FIELDS:
        val_s = str(row.get(f, "")).lower()
        if not val_s:
            continue
        words = set(re.findall(r'\w+', val_s))
        tok_hit = sum(1 for t in tokens if t in words)
        sub_hit = sum(1 for t in tokens if t and t in val_s)
        sq = re.sub(r'[\W_]+', '', val_s)
        squash_hit = 1 if q_squash and q_squash in sq else 0
        weight = 2 if f in ("код", "oem") else 1
        score += weight * (2 * tok_hit + sub_hit) + 3 * squash_hit * weight
    return score

async def search_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ensure_fresh_data()
    if update.message is None:
        return

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
    q_squash = squash(q)

    df = get_df()
    if df is None:
        await ensure_fresh_data_async(force=True)
        df = get_df()
        if df is None:
            return await update.message.reply_text("Ошибка загрузки данных.")

    matched_indices = match_row_by_index(tokens)

    if not matched_indices:
        mask_any = pd.Series(False, index=df.index)
        for col in SEARCH_FIELDS:
            if col not in df.columns: 
                continue
            series = df[col].astype(str).str.lower()
            field_mask = pd.Series(True, index=df.index)
            for t in tokens:
                if t:
                    field_mask &= series.str.contains(re.escape(t), na=False)
            mask_any |= field_mask
        matched_indices = set(df.index[mask_any])

    if not matched_indices and q_squash:
        mask_any = pd.Series(False, index=df.index)
        for col in SEARCH_FIELDS:
            if col not in df.columns:
                continue
            series = df[col].astype(str).str.lower()
            series_sq = series.str.replace(r'[\W_]+', '', regex=True)
            mask_any |= series_sq.str.contains(re.escape(q_squash), na=False)
        matched_indices = set(df.index[mask_any])

    if not matched_indices:
        return await update.message.reply_text(f"По запросу «{q}» ничего не найдено.")

    idx_list = list(matched_indices)
    results_df = df.loc[idx_list].copy()

    scores: List[int] = []
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

async def more_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    st = user_state.get(uid, {})
    results = st.get("results", pd.DataFrame())
    if results.empty:
        return await update.message.reply_text("Сначала выполните поиск.")
    st["page"] = st.get("page", 0) + 1
    await send_page(update, uid)

async def send_page(update: Update, uid: int):
    st = user_state.get(uid, {})
    results = st.get("results", pd.DataFrame())
    page = st.get("page", 0)

    total = len(results)
    if total == 0:
        return await update.message.reply_text("Результатов больше нет.")
    pages = max(1, math.ceil(total / PAGE_SIZE))
    if page >= pages:
        st["page"] = pages - 1
        return await update.message.reply_text("Больше результатов нет.")

    start = page * PAGE_SIZE
    end = min(start + PAGE_SIZE, total)

    await update.message.reply_text(f"Стр. {page+1}/{pages}. Показываю {start + 1}–{end} из {total}.")
    for _, row in results.iloc[start:end].iterrows():
        await send_row_with_image(update, row.to_dict(), format_row(row.to_dict()))
    if end < total:
        await update.message.reply_text("Показать ещё?", reply_markup=more_markup())

async def send_page_via_bot(bot, chat_id: int, uid: int):
    st = user_state.get(uid, {})
    results = st.get("results", pd.DataFrame())
    page = st.get("page", 0)
    total = len(results)
    if total == 0:
        return await bot.send_message(chat_id=chat_id, text="Результатов больше нет.")
    pages = max(1, math.ceil(total / PAGE_SIZE))
    if page >= pages:
        st["page"] = pages - 1
        return await bot.send_message(chat_id=chat_id, text="Больше результатов нет.")
    start = page * PAGE_SIZE
    end = min(start + PAGE_SIZE, total)
    await bot.send_message(chat_id=chat_id, text=f"Стр. {page+1}/{pages}. Показываю {start + 1}–{end} из {total}.")
    chunk = results.iloc[start:end]
    for _, row in chunk.iterrows():
        await send_row_with_image_bot(bot, chat_id, row.to_dict(), format_row(row.to_dict()))
    if end < total:
        await bot.send_message(chat_id=chat_id, text="Показать ещё?", reply_markup=more_markup())

async def on_issue_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    uid = q.from_user.id
    code = q.data.split(":", 1)[1].strip().lower()

    ensure_fresh_data()
    df = get_df()
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

        try:
            await to_thread(save_issue_to_sheet_blocking, q.from_user, part, qty, comment)
        except Exception as e:
            logger.error(f"Ошибка записи списания: {e}")
            for admin_id in get_admin_ids():
                try:
                    await context.bot.send_message(admin_id, f"⚠️ Ошибка сохранения списания: {e}")
                except Exception:
                    pass

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
    results = st.get("results", pd.DataFrame())
    if results.empty:
        return await q.message.reply_text("Сначала выполните поиск.")
    st["page"] = st.get("page", 0) + 1
    chat_id = q.message.chat.id
    await send_page_via_bot(context.bot, chat_id, uid)

async def guard_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if user and not is_allowed(user.id):
        try:
            await update.effective_message.reply_text("Доступ запрещён.")
        except Exception:
            pass
        raise ApplicationHandlerStop

async def guard_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    if user and not is_allowed(user.id):
        try:
            await update.callback_query.answer("Доступ запрещён.", show_alert=True)
        except Exception:
            pass
        raise ApplicationHandlerStop

async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Unhandled error: %s", context.error)
    msg = f"❌ Ошибка: {context.error}"
    for admin_id in get_admin_ids():
        try:
            await context.bot.send_message(admin_id, msg)
        except Exception:
            pass

def register_handlers(app):
    app.add_handler(MessageHandler(filters.ALL, guard_msg), group=-1)
    app.add_handler(CallbackQueryHandler(guard_cb, pattern=".*"), group=-1)

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("more", more_cmd))
    app.add_handler(CommandHandler("export", export_cmd))
    app.add_handler(CommandHandler("reload", reload_cmd))
    app.add_handler(CommandHandler("cancel", cancel_cmd))
    app.add_handler(CommandHandler("fileid", fileid_cmd))
    app.add_handler(CommandHandler("imgdebug", imgdebug_cmd))

    app.add_handler(MessageHandler(filters.ANIMATION | filters.VIDEO | filters.PHOTO, capture_fileid))

    app.add_handler(CallbackQueryHandler(menu_search_cb, pattern=r"^menu_search$"))
    app.add_handler(CallbackQueryHandler(menu_issue_help_cb, pattern=r"^menu_issue_help$"))
    app.add_handler(CallbackQueryHandler(menu_contact_cb, pattern=r"^menu_contact$"))

    app.add_handler(CallbackQueryHandler(on_more_click, pattern=r"^more$"))
    app.add_handler(CallbackQueryHandler(cancel_action, pattern=r"^cancel_action$"))

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

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search_text), group=1)
    app.add_error_handler(on_error)