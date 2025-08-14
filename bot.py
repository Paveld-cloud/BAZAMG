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

# Новые ENV для приветствия
WELCOME_ANIMATION_URL = os.getenv("WELCOME_ANIMATION_URL", "").strip()  # .gif или .mp4 (необязательно)
WELCOME_PHOTO_URL = os.getenv("WELCOME_PHOTO_URL", "").strip()          # запасная картинка (если нет анимации)
SUPPORT_CONTACT = os.getenv("SUPPORT_CONTACT", "👨‍💻 Поддержка: @your_support")

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
        [InlineKeyboardButton("🔍 Поиск", callback_data="menu_search")],
        [InlineKeyboardButton("📦 Как списать деталь", callback_data="menu_issue_help")],
        [InlineKeyboardButton("📞 Поддержка", callback_data="menu_contact")],
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
    """Инвертированный индекс: токен -> множество индексов строк."""
    index: DefaultDict[str, Set[int]] = defaultdict(set)
    for col in SEARCH_FIELDS:
        if col not in df.columns:
            continue
        for idx, val in df[col].astype(str).str.lower().items():
            for t in re.findall(r'\w+', val):
                if t:
                    index[t].add(idx)
    return dict(index)

def build_image_index(df: DataFrame) -> Dict[str, str]:
    """'код' -> сырой URL (без сетевых запросов)."""
    if "image" not in df.columns:
        return {}
    index = {}
    for _, row in df.iterrows():
        code = str(row.get("код", "")).strip().lower()
        if code:
            url = str(row.get("image", "")).strip()
            if url:
                index[code] = url
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

async def resolve_ibb_direct_async(url: str) -> str:
    """Из ibb.co/* HTML-страницы достаём og:image (i.ibb.co/...) — асинхронно."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=12) as resp:
                if resp.status != 200:
                    return url
                html = await resp.text()
        m = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
        return m.group(1) if m else url
    except Exception as e:
        logger.warning(f"resolve_ibb_direct_async fail: {e}")
        return url

async def resolve_image_url_async(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return u
    if "drive.google.com" in u:
        return normalize_drive_url(u)
    if re.match(r"^https?://(www\.)?ibb\.co/", u, re.I):
        return await resolve_ibb_direct_async(u)
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
    url_raw = await find_image_by_code_async(code)
    url = await resolve_image_url_async(url_raw)

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
    url_raw = await find_image_by_code_async(code)
    url = await resolve_image_url_async(url_raw)
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
    first = (user.first_name or "").strip() or "коллега"

    # 1) Медиа-приветствие
    sent_visual = False
    try:
        if WELCOME_ANIMATION_URL:
            await context.bot.send_animation(
                chat_id=chat_id,
                animation=WELCOME_ANIMATION_URL,
                caption=f"⚙️ Добро пожаловать, {first}! ⚙️",
                parse_mode="Markdown"
            )
            sent_visual = True
        elif WELCOME_PHOTO_URL:
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=WELCOME_PHOTO_URL,
                caption=f"⚙️ Добро пожаловать, {first}! ⚙️",
                parse_mode="Markdown"
            )
            sent_visual = True
    except Exception as e:
        logger.warning(f"Welcome media failed: {e}")

    # 2) Текст и меню
    text_intro = (
        f"⚙️ *Привет, {first}!* \n\n"
        "Это инженерный бот для поиска и списания деталей.\n"
        "— Введите *название*, *код* или *модель*.\n"
        "— Для быстрого доступа используйте кнопки ниже.\n\n"
        "Удачной работы! 🚀"
    )
    if sent_visual:
        await asyncio.sleep(0.5)

    await context.bot.send_message(
        chat_id=chat_id,
        text=text_intro,
        parse_mode="Markdown",
        reply_markup=main_menu_markup()
    )

# ------------------------- КОМАНДЫ --------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    issue_state.pop(uid, None)
    user_state.pop(uid, None)

    # Красочное приветствие
    await send_welcome_sequence(update, context)

    # Подсказка по командам
    if update.message:
        await asyncio.sleep(0.2)
        await update.message.reply_text(
            "Команды:\n"
            "• /help — помощь\n"
            "• /more — показать ещё\n"
            "• /export — выгрузка результатов (XLSX/CSV)\n"
            "• /cancel — отменить списание\n"
            "• /reload — перезагрузка данных и пользователей (только админ)",
            parse_mode="Markdown"
        )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "1) Выполните поиск по названию/модели/коду.\n"
        "2) В карточке нажмите «📦 Взять деталь» — бот спросит количество и комментарий,\n"
        "   затем попросит подтвердить списание (Да/Нет).\n"
        "У ВАС ВСЕ ПОЛУЧИТСЯ.",
        parse_mode="Markdown"
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
        import openpyxl  # noqa: F401
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

# Меню приветствия — callbacks
async def menu_search_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    await q.message.reply_text("🔍 Введите запрос: название/модель/код. Пример: `PI 8808 DRG 500`", parse_mode="Markdown")

async def menu_issue_help_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    await q.message.reply_text(
        "Как списать деталь:\n"
        "1) Выполните поиск по названию/коду.\n"
        "2) В карточке нажмите «📦 Взять деталь».\n"
        "3) Укажите количество и комментарий.\n"
        "4) Подтвердите списание кнопкой «Да».",
        parse_mode="Markdown"
    )

async def menu_contact_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    await q.message.reply_text(f"{SUPPORT_CONTACT}")

# ------------------------- ПОИСК -----------------------------
def match_row_by_index(tokens: List[str]) -> Set[int]:
    """Точный быстрый матч по индексу (все токены как слова)."""
    if not _search_index:
        return set()
    result = None
    for t in tokens:
        indices = _search_index.get(t, set())
        if result is None:
            result = indices.copy()
        else:
            result &= indices
        if not result:
            break
    return result or set()

def _safe_col(df: DataFrame, col: str) -> Optional[pd.Series]:
    return df[col].astype(str).str.lower() if col in df.columns else None

def _relevance_score(row: dict, tokens: List[str], q_squash: str) -> int:
    """Скоринг релевантности: точные токены > подстроки; boost код/oem; слитный матч."""
    score = 0
    for f in SEARCH_FIELDS:
        val = str(row.get(f, "")).lower()
        if not val:
            continue
        words = set(re.findall(r'\w+', val))
        tok_hit = sum(1 for t in tokens if t in words)
        sub_hit = sum(1 for t in tokens if t and t in val)
        sq = re.sub(r'[\W_]+', '', val)
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

    if df is None:
        await ensure_fresh_data_async(force=True)
        if df is None:
            return await update.message.reply_text("Ошибка загрузки данных.")

    # 1) быстрый точный индекс
    matched_indices = match_row_by_index(tokens)

    # 2) подстроки по любому полю (все токены должны встретиться в одном поле)
    if not matched_indices:
        mask_any = pd.Series(False, index=df.index)
        for col in SEARCH_FIELDS:
            series = _safe_col(df, col)
            if series is None:
                continue
            field_mask = pd.Series(True, index=df.index)
            for t in tokens:
                if t:
                    field_mask &= series.str.contains(re.escape(t), na=False)
            mask_any |= field_mask
        matched_indices = set(df.index[mask_any])

    # 3) «слитный» фолбэк по всем полям
    if not matched_indices and q_squash:
        mask_any = pd.Series(False, index=df.index)
        for col in SEARCH_FIELDS:
            series = _safe_col(df, col)
            if series is None:
                continue
            series_sq = series.str.replace(r'[\W_]+', '', regex=True)
            mask_any |= series_sq.str.contains(re.escape(q_squash), na=False)
        matched_indices = set(df.index[mask_any])

    if not matched_indices:
        return await update.message.reply_text(f"По запросу «{q}» ничего не найдено.")

    # Ранжируем по релевантности
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
    results = st.get("results", DataFrame())
    if results.empty:
        return await q.message.reply_text("Сначала выполните поиск.")
    st["page"] = st.get("page", 0) + 1
    chat_id = q.message.chat.id
    await send_page_via_bot(context.bot, chat_id, uid)

# --------------------- ERROR HANDLER -------------------------
async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("Unhandled error: %s", context.error)
    msg = f"❌ Ошибка: {context.error}"
    for admin_id in (SHEET_ADMINS | ADMINS):
        try:
            await context.bot.send_message(admin_id, msg)
        except Exception:
            pass

# --------------------- APP / WEBHOOK ------------------------
def build_app():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(MessageHandler(filters.ALL, guard_msg), group=-1)
    app.add_handler(CallbackQueryHandler(guard_cb, pattern=".*"), group=-1)

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("more", more_cmd))
    app.add_handler(CommandHandler("export", export_cmd))
    app.add_handler(CommandHandler("reload", reload_cmd))
    app.add_handler(CommandHandler("cancel", cancel_cmd))

    # Меню приветствия
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
        # Явно фиксируем per_* (может показать warning от PTB — это ок)
        per_chat=True,
        per_user=True,
        per_message=False,
    )
    app.add_handler(conv)

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search_text), group=1)
    app.add_error_handler(on_error)

    return app

if __name__ == "__main__":
    logger.info(f"⌚ Используем часовой пояс: {TZ_NAME}")
    if not WEBHOOK_SECRET_TOKEN:
        logger.warning("WEBHOOK_SECRET_TOKEN не задан — рекомендуется включить для продакшена.")

    initial_load()
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
