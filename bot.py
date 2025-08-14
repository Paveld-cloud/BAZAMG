# bot.py
import os
import re
import io
import json
import math
import time
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any, Set, List, DefaultDict
from collections import defaultdict
from html import escape

import requests            # нужен для ibb.co резолва
import aiohttp
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials
from pandas import DataFrame
from telegram import (
    Update, InputFile, InlineKeyboardMarkup, InlineKeyboardButton
)
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler,
    ConversationHandler, ContextTypes, filters, ApplicationHandlerStop
)

# --------------------------- ЛОГИ ---------------------------
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bot")

# -------------------------- НАСТРОЙКИ -----------------------
ADMINS = {225177765}  # локальные админы

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
SPREADSHEET_URL = os.getenv("SPREADSHEET_URL")
CREDS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
WEBHOOK_URL = (os.getenv("WEBHOOK_URL") or "").rstrip("/")
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/webhook")
PORT = int(os.getenv("PORT", "8080"))
WEBHOOK_SECRET_TOKEN = os.getenv("WEBHOOK_SECRET_TOKEN", "")
SHEET_NAME = os.getenv("SHEET_NAME", "").strip()
MAX_QTY = float(os.getenv("MAX_QTY", "1000"))
TZ_NAME = os.getenv("TIMEZONE", "Europe/Moscow")
PAGE_SIZE = 5

# ===== Приветственный медиаконтент =====
# основное фото (Telegram file_id) — то, что ты прислал
WELCOME_MEDIA_ID = "AgACAgIAAxkBAAIPVGieF335h6r2xO6EvVxMTTatIs7VAAJg-zEbBUHwSAgsrYCCYGWiAQADAgADeQADNgQ"
# резерв: можно указать URL картинки или ещё один file_id (если захочешь поменять)
WELCOME_PHOTO_URL = os.getenv("WELCOME_PHOTO_URL", "").strip()
# резерв: .mp4/.gif/.webm URL для «зажигательного» видео (опционально)
WELCOME_ANIMATION_URL = os.getenv("WELCOME_ANIMATION_URL", "").strip()

if not all([TELEGRAM_TOKEN, SPREADSHEET_URL, CREDS_JSON, WEBHOOK_URL]):
    raise RuntimeError("ENV нужны: TELEGRAM_TOKEN, SPREADSHEET_URL, GOOGLE_APPLICATION_CREDENTIALS_JSON, WEBHOOK_URL")

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
DATA_TTL = 300
USERS_TTL = 300

def now_local_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    try:
        return datetime.now(ZoneInfo(TZ_NAME)).strftime(fmt)
    except Exception:
        return datetime.utcnow().strftime(fmt)

# ---------------------- ГЛОБАЛЬНЫЕ СОСТОЯНИЯ ----------------
df: Optional[DataFrame] = None
_last_load_ts = 0.0
_search_index: Optional[Dict[str, Set[int]]] = None
_image_index: Optional[Dict[str, str]] = None

# пользователи
SHEET_ALLOWED: Set[int] = set()
SHEET_ADMINS: Set[int] = set()
SHEET_BLOCKED: Set[int] = set()
_last_users_ts = 0.0

# состояние поиска и списания
user_state: Dict[int, Dict[str, Any]] = {}
issue_state: Dict[int, Dict[str, Any]] = {}

# флаги фоновых задач
_loading_data = False
_loading_users = False

# шаги диалога
ASK_QUANTITY, ASK_COMMENT, ASK_CONFIRM = range(3)

# ------------------------- КНОПКИ ---------------------------
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
        [InlineKeyboardButton("🔎 Поиск", callback_data="menu_search")],
        [InlineKeyboardButton("📦 Как списать деталь", callback_data="menu_howto")],
        [InlineKeyboardButton("🛟 Поддержка", callback_data="menu_support")],
    ])

# ------------------------- ВСПОМОГАТЕЛЬНОЕ -------------------
async def _to_thread(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)

# ------------------------- GOOGLE SHEETS ---------------------
def get_gs_client():
    creds_info = json.loads(CREDS_JSON)
    creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    return gspread.authorize(creds)

def _open_data_worksheet(client):
    sh = client.open_by_url(SPREADSHEET_URL)
    if SHEET_NAME:
        try:
            return sh.worksheet(SHEET_NAME)
        except gspread.WorksheetNotFound:
            logger.warning(f"Лист {SHEET_NAME!r} не найден, fallback на sheet1")
    return sh.sheet1

def load_data_blocking() -> list[dict]:
    client = get_gs_client()
    ws = _open_data_worksheet(client)
    return ws.get_all_records()

SEARCH_FIELDS = ["тип", "наименование", "код", "oem", "изготовитель"]

def build_search_index(df: DataFrame) -> Dict[str, Set[int]]:
    index: DefaultDict[str, Set[int]] = defaultdict(set)
    for col in SEARCH_FIELDS:
        if col not in df.columns:
            continue
        for idx, val in df[col].astype(str).str.lower().items():
            tokens = re.findall(r'\w+', val)
            for t in tokens:
                index[t].add(idx)
    return dict(index)

def build_image_index(df: DataFrame) -> Dict[str, str]:
    if "image" not in df.columns:
        return {}
    index = {}
    for _, row in df.iterrows():
        code = str(row.get("код", "")).strip().lower()
        if code:
            url = str(row["image"]).strip()
            if url:
                index[code] = resolve_image_url(url)
    return index

def initial_load():
    global df, _last_load_ts, _search_index, _image_index
    data = load_data_blocking()
    new_df = DataFrame(data)
    new_df.columns = new_df.columns.str.strip().str.lower()

    for col in ("код", "oem"):
        if col in new_df.columns:
            new_df[col] = new_df[col].astype(str).str.strip().str.lower()
    if "image" in new_df.columns:
        new_df["image"] = new_df["image"].astype(str).str.strip()

    df = new_df
    _search_index = build_search_index(df)
    _image_index = build_image_index(df)
    _last_load_ts = time.time()
    logger.info(f"✅ Загружено (startup) {len(df)} строк и индексы")

    allowed, admins, blocked = load_users_from_sheet()
    global SHEET_ALLOWED, SHEET_ADMINS, SHEET_BLOCKED, _last_users_ts
    SHEET_ALLOWED, SHEET_ADMINS, SHEET_BLOCKED = allowed, admins, blocked
    _last_users_ts = time.time()
    logger.info(f"👥 Пользователи (startup): allowed={len(allowed)}, admins={len(admins)}, blocked={len(blocked)}")

async def ensure_fresh_data_async(force: bool = False):
    global df, _last_load_ts, _search_index, _image_index, _loading_data
    if not force and df is not None and (time.time() - _last_load_ts <= DATA_TTL):
        return
    if _loading_data:
        return
    _loading_data = True
    try:
        data = await _to_thread(load_data_blocking)
        new_df = DataFrame(data)
        new_df.columns = new_df.columns.str.strip().str.lower()
        for col in ("код", "oem"):
            if col in new_df.columns:
                new_df[col] = new_df[col].astype(str).str.strip().str.lower()
        if "image" in new_df.columns:
            new_df["image"] = new_df["image"].astype(str).str.strip()

        df = new_df
        _search_index = build_search_index(df)
        _image_index = build_image_index(df)
        _last_load_ts = time.time()
        logger.info(f"✅ Перезагружено {len(df)} строк и индексы")
    finally:
        _loading_data = False

def ensure_fresh_data(force: bool = False):
    if not force and df is not None and (time.time() - _last_load_ts <= DATA_TTL):
        return
    asyncio.create_task(ensure_fresh_data_async(force=True))

# ------------------------- УТИЛИТЫ --------------------------
def val(row: dict, key: str, default: str = "—") -> str:
    v = row.get(key)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
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

def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", (text or "")).lower().strip()

def squash(text: str) -> str:
    return re.sub(r"[\W_]+", "", (text or "").lower(), flags=re.UNICODE)

# ---------- Работа со ссылками на изображения ----------
def normalize_drive_url(url: str) -> str:
    m = re.search(r'drive\.google\.com/(?:file/d/([-\w]{20,})|open\?id=([-\w]{20,}))', url)
    if m:
        file_id = m.group(1) or m.group(2)
        return f'https://drive.google.com/uc?export=download&id={file_id}'
    return url

def resolve_ibb_direct(url: str) -> str:
    try:
        resp = requests.get(url, timeout=12)
        resp.raise_for_status()
        html = resp.text
        m = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
        if m:
            return m.group(1)
    except Exception as e:
        logger.warning(f"resolve_ibb_direct fail: {e}")
    return url

def resolve_image_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return u
    if "drive.google.com" in u:
        return normalize_drive_url(u)
    if re.match(r"^https?://(www\.)?ibb\.co/", u, re.I):
        return resolve_ibb_direct(u)
    return u

async def find_image_by_code_async(code: str) -> str:
    if not code or _image_index is None:
        return ""
    return _image_index.get(code.strip().lower(), "")

async def _download_image_async(url: str) -> Optional[io.BytesIO]:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=12) as resp:
                if resp.status != 200:
                    return None
                data = await resp.read()
                if len(data) > 5_000_000:
                    return None
                bio = io.BytesIO(data)
                bio.name = "image"
                return bio
    except Exception as e:
        logger.warning(f"Download failed: {e}")
        return None

async def send_row_with_image(update: Update, row: dict, text: str):
    code = str(row.get("код", "")).strip()
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("📦 Взять деталь", callback_data=f"issue:{code.lower()}")]])
    url = await find_image_by_code_async(code)

    if url:
        try:
            await update.message.reply_photo(photo=url, caption=text, reply_markup=kb)
            return
        except Exception as e:
            logger.warning(f"URL фото не сработал ({url}): {e}")
            bio = await _download_image_async(url)
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
    url = await find_image_by_code_async(code)
    if url:
        try:
            await bot.send_photo(chat_id=chat_id, photo=url, caption=text, reply_markup=kb)
            return
        except Exception as e:
            logger.warning(f"URL фото не сработал ({url}): {e}")
            bio = await _download_image_async(url)
            if bio:
                try:
                    await bot.send_photo(chat_id=chat_id, photo=bio, caption=text, reply_markup=kb)
                    return
                except Exception as e2:
                    logger.warning(f"Отправка скачанного фото не удалась: {e2} (src: {url})")
    await bot.send_message(chat_id=chat_id, text=text, reply_markup=kb)

# --------------------- ПОЛЬЗОВАТЕЛИ -------------------------
def _truthy(x) -> bool:
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "да", "истина", "ok", "ок", "allowed", "разрешен", "разрешено"} or (s.isdigit() and int(s) > 0)

def _to_int_or_none(x):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        s = str(x).strip()
        if not s:
            return None
        m = re.search(r"-?\d+", s)
        return int(m.group(0)) if m else None
    except Exception:
        return None

def load_users_from_sheet():
    client = get_gs_client()
    sh = client.open_by_url(SPREADSHEET_URL)
    try:
        ws = sh.worksheet("Пользователи")
    except gspread.WorksheetNotFound:
        try:
            ws = sh.worksheet("Users")
        except gspread.WorksheetNotFound:
            logger.info("Лист 'Пользователи' не найден — доступ разрешён всем.")
            return set(), set(), set()

    rows = ws.get_all_records()
    if not rows:
        logger.info("Лист 'Пользователи' пуст — доступ разрешён всем.")
        return set(), set(), set()

    allowed, admins, blocked = set(), set(), set()
    for row in rows:
        r = {str(k).strip().lower(): v for k, v in row.items()}
        uid = (
            _to_int_or_none(r.get("user_id"))
            or _to_int_or_none(r.get("userid"))
            or _to_int_or_none(r.get("id"))
            or _to_int_or_none(r.get("uid"))
            or _to_int_or_none(r.get("телеграм id"))
            or _to_int_or_none(r.get("пользователь"))
        )
        if not uid:
            continue

        role = str(r.get("role") or r.get("роль") or "").strip().lower()
        is_admin = role in {"admin", "админ", "administrator", "администратор"} or _truthy(r.get("admin"))
        is_allowed = _truthy(r.get("allowed") or r.get("доступ") or (not role or role == "user"))
        is_blocked = _truthy(r.get("blocked") or r.get("ban") or r.get("запрет"))

        if is_blocked:
            blocked.add(uid)
        if is_admin:
            admins.add(uid)
            is_allowed = True
        if is_allowed:
            allowed.add(uid)

    return allowed, admins, blocked

async def ensure_users_async(force: bool = False):
    global SHEET_ALLOWED, SHEET_ADMINS, SHEET_BLOCKED, _last_users_ts, _loading_users
    if not force and (time.time() - _last_users_ts <= USERS_TTL):
        return
    if _loading_users:
        return
    _loading_users = True
    try:
        allowed, admins, blocked = await _to_thread(load_users_from_sheet)
        SHEET_ALLOWED, SHEET_ADMINS, SHEET_BLOCKED = allowed, admins, blocked
        _last_users_ts = time.time()
        logger.info(f"👥 Пользователи: allowed={len(allowed)}, admins={len(admins)}, blocked={len(blocked)}")
    finally:
        _loading_users = False

def ensure_users(force: bool = False):
    if not force and (time.time() - _last_users_ts <= USERS_TTL):
        return
    asyncio.create_task(ensure_users_async(force=True))

def is_admin(uid: int) -> bool:
    ensure_users()
    return uid in SHEET_ADMINS or uid in ADMINS

def is_allowed(uid: int) -> bool:
    ensure_users()
    if uid in SHEET_BLOCKED:
        return False
    if SHEET_ALLOWED:
        return (uid in SHEET_ALLOWED) or (uid in SHEET_ADMINS) or (uid in ADMINS)
    return True

# --------------------- ГВАРДЫ -----------------------
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

# --------------------- СОХРАНЕНИЕ СПИСАНИЙ -------------------
def save_issue_to_sheet_blocking(bot, user, part: dict, quantity, comment: str):
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

async def save_issue_to_sheet(bot, user, part: dict, quantity, comment: str):
    try:
        await _to_thread(save_issue_to_sheet_blocking, bot, user, part, quantity, comment)
    except Exception as e:
        logger.error(f"Ошибка записи списания: {e}")
        async def notify():
            for admin_id in (SHEET_ADMINS | ADMINS):
                try:
                    await bot.send_message(admin_id, f"⚠️ Ошибка сохранения списания: {e}")
                except Exception:
                    pass
        asyncio.create_task(notify())

# ------------------------- ПРИВЕТСТВИЕ -----------------------
async def send_welcome_sequence(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user = update.effective_user
    first = escape((user.first_name or "").strip() or "коллега")

    # единая карточка в caption, чтобы текст шёл сразу под картинкой
    caption_html = (
        f"⚙️ <b>Привет, {first}!</b>\n"
        f"<i>Бот для поиска и списания деталей</i>\n\n"
        f"• Введите <code>название</code>, <code>код</code> или <code>модель</code>\n"
        f"• Откройте карточку и нажмите «📦 Взять деталь»\n"
        f"• Подтвердите списание — и готово\n\n"
        f"Пример: <code>PI 8808 DRG 500</code>\n"
        f"Удачной работы! 🚀"
    )

    sent = False

    # 1) если есть видео/анимация — отправим его
    if WELCOME_ANIMATION_URL and not sent:
        try:
            await context.bot.send_animation(
                chat_id=chat_id,
                animation=WELCOME_ANIMATION_URL,
                caption=caption_html,
                parse_mode="HTML",
                reply_markup=main_menu_markup()
            )
            sent = True
        except Exception as e:
            logger.warning(f"Welcome animation failed: {e}")

    # 2) пробуем фото по file_id (основной сценарий)
    if not sent and WELCOME_MEDIA_ID:
        try:
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=WELCOME_MEDIA_ID,
                caption=caption_html,
                parse_mode="HTML",
                reply_markup=main_menu_markup()
            )
            sent = True
        except Exception as e:
            logger.warning(f"Welcome photo by file_id failed: {e}")

    # 3) URL картинки (резерв)
    if not sent and WELCOME_PHOTO_URL:
        try:
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=WELCOME_PHOTO_URL,
                caption=caption_html,
                parse_mode="HTML",
                reply_markup=main_menu_markup()
            )
            sent = True
        except Exception as e:
            logger.warning(f"Welcome photo by URL failed: {e}")

    # 4) если всё упало — просто текст
    if not sent:
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=caption_html,
                parse_mode="HTML",
                reply_markup=main_menu_markup()
            )
        except Exception as e:
            logger.warning(f"Welcome text failed: {e}")
            await context.bot.send_message(chat_id=chat_id, text=re.sub(r"</?(b|i|code)>", "", caption_html))

# ------------------------- КОМАНДЫ --------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    issue_state.pop(uid, None)
    user_state.pop(uid, None)

    # приветственный блок
    await send_welcome_sequence(update, context)

    # справка отдельным сообщением
    await update.message.reply_text(
        "Команды:\n"
        "• /help — помощь\n"
        "• /more — показать ещё\n"
        "• /export — выгрузка результатов (XLSX/CSV)\n"
        "• /cancel — отменить списание\n"
        "• /reload — перезагрузка данных и пользователей (только админ)"
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "1) Выполните поиск по названию/модели/коду.\n"
        "2) В карточке нажмите «📦 Взять деталь» — бот спросит количество и комментарий,\n"
        "   затем попросит подтвердить списание (Да/Нет).\n"
        "У ВАС ВСЕ ПОЛУЧИТСЯ."
    )

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
    st = user_state.get(uid, {})
    results = st.get("results", DataFrame())
    if results.empty:
        return await update.message.reply_text("Сначала выполните поиск.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        import openpyxl  # noqa
        buf = await _to_thread(_df_to_xlsx, results, f"export_{timestamp}.xlsx")
        await update.message.reply_document(InputFile(buf, filename=f"export_{timestamp}.xlsx"))
    except Exception as e:
        logger.warning(f"Не удалось XLSX (fallback CSV): {e}")
        csv = results.to_csv(index=False, encoding="utf-8-sig")
        await update.message.reply_document(
            InputFile(io.BytesIO(csv.encode("utf-8-sig")), filename=f"export_{timestamp}.csv")
        )

def _df_to_xlsx(df: DataFrame, name: str) -> io.BytesIO:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False)
    buf.seek(0)
    buf.name = name
    return buf

# ------------------------- ПОИСК -----------------------------
def match_row_by_index(tokens: List[str]) -> Set[int]:
    if not _search_index:
        return set()
    result = None
    for t in tokens:
        indices = _search_index.get(t, set())
        if result is None:
            result = indices.copy()
        else:
            result &= indices
    return result or set()

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

    if df is None:
        await ensure_fresh_data_async(force=True)
        if df is None:
            return await update.message.reply_text("Ошибка загрузки данных.")

    matched_indices = match_row_by_index(tokens)
    if not matched_indices and q_squash and "код" in df.columns:
        try:
            matched_indices = set(df[df["код"].astype(str).str.contains(q_squash, case=False, na=False)].index)
        except Exception:
            matched_indices = set()

    if not matched_indices:
        return await update.message.reply_text(f"По запросу «{q}» ничего не найдено.")

    results_df = df.loc[list(matched_indices)].copy()
    if "код" in results_df.columns:
        try:
            results_df = results_df.sort_values(by=["код"], key=lambda x: x.astype(str).str.len(), ascending=True)
        except Exception:
            pass

    st = user_state.setdefault(uid, {})
    st["query"] = q
    st["results"] = results_df
    st["page"] = 0

    await send_page(update, uid)

async def more_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    st = user_state.get(uid, {})
    results = st.get("results", DataFrame())
    if results.empty:
        return await update.message.reply_text("Сначала выполните поиск.")
    st["page"] = st.get("page", 0) + 1
    await send_page(update, uid)

async def send_page(update: Update, uid: int):
    st = user_state.get(uid, {})
    results: DataFrame = st.get("results", DataFrame())
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
    results: DataFrame = st.get("results", DataFrame())
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

# ------------------ СПИСАНИЕ (Диалог) -----------------------
async def on_issue_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    uid = q.from_user.id
    code = q.data.split(":", 1)[1].strip().lower()

    ensure_fresh_data()
    found = None
    if df is not None and "код" in df.columns:
        hit = df[df["код"] == code]
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
    except Exception:
        return await update.message.reply_text(
            f"Введите положительное число (до {MAX_QTY}), например: 1 или 2.5",
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

    st["comment"] = comment
    st["await_comment"] = False

    text = (
        "Подтвердите списание:\n\n"
        f"{format_row(part)}\n\n"
        f"Списать: {qty}\n"
        f"Комментарий: {comment or '—'}"
    )
    await update.message.reply_text(text, reply_markup=confirm_markup())
    return ASK_CONFIRM

async def handle_confirm_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    uid = q.from_user.id
    st = issue_state.get(uid)
    if not st:
        return await q.message.reply_text("Операция неактивна.")

    if q.data == "confirm_yes":
        await save_issue_to_sheet(context.bot, q.from_user, st["part"], st["quantity"], st.get("comment", ""))
        issue_state.pop(uid, None)
        return await q.message.reply_text("✅ Списание записано.")
    elif q.data == "confirm_no":
        issue_state.pop(uid, None)
        return await q.message.reply_text("Отменено.")
    elif q.data == "cancel_action":
        issue_state.pop(uid, None)
        return await q.message.reply_text("❌ Операция списания отменена.")
    elif q.data == "more":
        # проксируем в общий "more"
        await more_cmd(update, context)

# ------------- Меню кнопок (howto/support/search) -----------
async def menu_buttons_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data
    if data == "menu_howto":
        return await q.message.reply_text(
            "Как списать деталь:\n"
            "1) Введите запрос, откройте подходящую карточку.\n"
            "2) Нажмите «📦 Взять деталь».\n"
            "3) Укажите количество и комментарий.\n"
            "4) Подтвердите списание."
        )
    if data == "menu_support":
        return await q.message.reply_text("Поддержка: @your_support_username")
    if data == "menu_search":
        return await q.message.reply_text("Напишите запрос (название/код/модель).")
    if data == "more":
        return await more_cmd(update, context)

# ------------------------- BOOTSTRAP -------------------------
def build_app():
    logging.info(f"⌚ Используем часовой пояс: {TZ_NAME}")
    if not WEBHOOK_SECRET_TOKEN:
        logging.warning("WEBHOOK_SECRET_TOKEN не задан — рекомендуется включить для продакшена.")

    initial_load()

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # глобальные гварды
    app.add_handler(MessageHandler(filters.ALL, guard_msg), group=0)
    app.add_handler(CallbackQueryHandler(guard_cb), group=0)

    # диалог списания
    conv = ConversationHandler(
        entry_points=[CallbackQueryHandler(on_issue_click, pattern=r"^issue:.+")],
        states={
            ASK_QUANTITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_quantity)],
            ASK_COMMENT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_comment)],
            ASK_CONFIRM: [CallbackQueryHandler(handle_confirm_btn, pattern=r"^(confirm_yes|confirm_no|cancel_action|more)$")],
        },
        fallbacks=[CommandHandler("cancel", cancel_cmd)],
        name="issue_conv",
        persistent=False,
    )
    app.add_handler(conv)

    # меню
    app.add_handler(CallbackQueryHandler(menu_buttons_router, pattern=r"^(menu_howto|menu_support|menu_search|more)$"))

    # команды
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("reload", reload_cmd))
    app.add_handler(CommandHandler("cancel", cancel_cmd))
    app.add_handler(CommandHandler("export", export_cmd))

    # текст как поиск
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search_text))

    return app

async def run_webhook(app):
    await app.start()
    await app.bot.delete_webhook()
    await app.bot.set_webhook(
        url=f"{WEBHOOK_URL}{WEBHOOK_PATH}",
        secret_token=WEBHOOK_SECRET_TOKEN or None,
        drop_pending_updates=True,
        max_connections=40,
        allowed_updates=["message", "callback_query"]
    )
    logging.info("🚀 Стартуем webhook-сервер на 0.0.0.0:%s", PORT)
    logging.info("🌐 Устанавливаем webhook: %s%s", WEBHOOK_URL, WEBHOOK_PATH)
    await app.updater.start_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=WEBHOOK_PATH.lstrip("/"),
        secret_token=WEBHOOK_SECRET_TOKEN or None,
    )
    await app.updater.idle()

def main():
    app = build_app()
    try:
        asyncio.run(run_webhook(app))
    except (KeyboardInterrupt, SystemExit):
        pass

if __name__ == "__main__":
    main()
