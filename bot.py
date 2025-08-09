import os
import re
import io
import json
import math
import time
import asyncio
import logging
from datetime import datetime
from io import BytesIO
from zoneinfo import ZoneInfo  # локальное время для Истории

import requests
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("bot")

# -------------------------- НАСТРОЙКИ -----------------------
ADMINS = {225177765}  # локальные админы (добавка к листу)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
SPREADSHEET_URL = os.getenv("SPREADSHEET_URL")
CREDS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")

WEBHOOK_URL = (os.getenv("WEBHOOK_URL") or "").rstrip("/")
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/webhook")
PORT = int(os.getenv("PORT", "8080"))
WEBHOOK_SECRET_TOKEN = os.getenv("WEBHOOK_SECRET_TOKEN", "")

# Часовой пояс для записи в Историю
TZ_NAME = os.getenv("TIMEZONE", "Europe/Moscow")
def now_local_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    try:
        return datetime.now(ZoneInfo(TZ_NAME)).strftime(fmt)
    except Exception:
        return datetime.utcnow().strftime(fmt)

if not TELEGRAM_TOKEN or not SPREADSHEET_URL or not CREDS_JSON or not WEBHOOK_URL:
    raise RuntimeError(
        "ENV нужны: TELEGRAM_TOKEN, SPREADSHEET_URL, GOOGLE_APPLICATION_CREDENTIALS_JSON, WEBHOOK_URL"
    )

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
DATA_TTL = 300          # TTL для данных
USERS_TTL = 300         # TTL для листа «Пользователи»
PAGE_SIZE = 5

# шаги диалога
ASK_QUANTITY, ASK_COMMENT, ASK_CONFIRM = range(3)

# ---------------------- ГЛОБАЛЬНЫЕ СОСТОЯНИЯ ----------------
df: DataFrame | None = None
_last_load_ts = 0.0

# пользователи из листа
SHEET_ALLOWED: set[int] = set()
SHEET_ADMINS: set[int] = set()
SHEET_BLOCKED: set[int] = set()
_last_users_ts = 0.0

# состояние поиска/выдачи результатов
user_state: dict[int, dict] = {}   # { user_id: { "query": str, "results": DataFrame, "page": int } }

def get_user_state(user_id: int) -> dict:
    return user_state.setdefault(user_id, {"query": "", "results": DataFrame(), "page": 0})

# состояние операции списания
issue_state: dict[int, dict] = {}  # { user_id: {"part": dict, "quantity": float, "comment": str, "await_comment": bool} }

# ------------------------- КНОПКИ ---------------------------
def cancel_markup():
    return InlineKeyboardMarkup([[InlineKeyboardButton("❌ Отменить", callback_data="cancel_action")]])

def confirm_markup():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Да, списать", callback_data="confirm_yes"),
            InlineKeyboardButton("❌ Нет", callback_data="confirm_no"),
        ],
        [InlineKeyboardButton("❌ Отменить", callback_data="cancel_action")]
    ])

def more_markup():
    return InlineKeyboardMarkup([[InlineKeyboardButton("⏭ Ещё", callback_data="more")]])

# ------------------------- GOOGLE SHEETS ---------------------
def get_gs_client():
    creds_info = json.loads(CREDS_JSON)
    creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    return gspread.authorize(creds)

def load_data() -> list[dict]:
    client = get_gs_client()
    sheet = client.open_by_url(SPREADSHEET_URL)
    ws = sheet.sheet1   # при необходимости замени на worksheet("SAP")
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
        if "image" in new_df.columns:
            new_df["image"] = new_df["image"].astype(str).str.strip()
        df = new_df
        _last_load_ts = time.time()
        logger.info(f"✅ Загружено {len(df)} строк из Google Sheet")

# ------------------------- УТИЛИТЫ --------------------------
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

# ---------- Работа со ссылками на изображения ----------
def normalize_drive_url(url: str) -> str:
    m = re.search(r'drive\.google\.com/(?:file/d/([-\w]{20,})|open\?id=([-\w]{20,}))', url)
    if m:
        file_id = m.group(1) or m.group(2)
        return f'https://drive.google.com/uc?export=download&id={file_id}'
    return url

def resolve_ibb_direct(url: str) -> str:
    """Из ibb.co/* HTML-страницы достаём og:image (i.ibb.co/...)."""
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

def resolve_image_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return u
    if "drive.google.com" in u:
        return normalize_drive_url(u)
    if re.match(r"^https?://(www\.)?ibb\.co/", u, re.I):
        return resolve_ibb_direct(u)
    return u

def find_image_by_code(code: str) -> str:
    """
    Ищем ссылку на фото по КОДУ в столбце image (по всему листу).
    1) Точнее: код как токен в URL/имени файла (/, _, -, или расширение .png/.jpg и т.п.).
    2) Если не нашли — простой contains (case-insensitive).
    """
    if df is None or "image" not in df.columns:
        return ""
    code_raw = (code or "").strip()
    if not code_raw:
        return ""

    col = df["image"].astype(str)

    # 1) «почти точное» совпадение кода как токена
    pat = r'(?i)(?:^|[\/_\-])' + re.escape(code_raw) + r'(?:\.[a-z0-9]{2,5}(?:\?.*)?$|[\/_\-?#])'
    mask_token = col.str.contains(pat, regex=True, na=False)
    if mask_token.any():
        url = str(col[mask_token].iloc[0]).strip()
        return resolve_image_url(url)

    # 2) Фолбэк: простое вхождение кода
    mask_contains = col.str.contains(re.escape(code_raw), case=False, na=False)
    if mask_contains.any():
        url = str(col[mask_contains].iloc[0]).strip()
        return resolve_image_url(url)

    return ""

async def send_row_with_image(update: Update, row: dict, text: str):
    code = str(row.get("код", "")).strip()
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("📦 Взять деталь", callback_data=f"issue:{code.lower()}")]])
    url = find_image_by_code(code)

    if url:
        try:
            await update.message.reply_photo(photo=url, caption=text, reply_markup=kb)
            return
        except Exception as e:
            logger.warning(f"URL фото не сработал ({url}): {e}")
            try:
                r = requests.get(url, timeout=15, allow_redirects=True)
                r.raise_for_status()
                bio = BytesIO(r.content)
                ctype = r.headers.get("Content-Type", "").lower()
                if "image" not in ctype:
                    logger.warning(f"Получили non-image Content-Type ({ctype}) с {url}")
                bio.name = "image"
                await update.message.reply_photo(photo=bio, caption=text, reply_markup=kb)
                return
            except Exception as e2:
                logger.warning(f"Скачивание/отправка фото не удалось: {e2} (src: {url})")

    await update.message.reply_text(text, reply_markup=kb)

async def send_row_with_image_bot(bot, chat_id: int, row: dict, text: str):
    code = str(row.get("код", "")).strip()
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("📦 Взять деталь", callback_data=f"issue:{code.lower()}")]])
    url = find_image_by_code(code)
    if url:
        try:
            await bot.send_photo(chat_id=chat_id, photo=url, caption=text, reply_markup=kb)
            return
        except Exception as e:
            logger.warning(f"URL фото не сработал ({url}): {e}")
            try:
                r = requests.get(url, timeout=15, allow_redirects=True)
                r.raise_for_status()
                bio = BytesIO(r.content); bio.name = "image"
                await bot.send_photo(chat_id=chat_id, photo=bio, caption=text, reply_markup=kb)
                return
            except Exception as e2:
                logger.warning(f"Скачивание/отправка фото не удалось: {e2} (src: {url})")
    await bot.send_message(chat_id=chat_id, text=text, reply_markup=kb)

# --------------------- ПОЛЬЗОВАТЕЛИ (лист «Пользователи») ----
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
    """Читает лист 'Пользователи' (или 'Users') и возвращает три сета: allowed, admins, blocked."""
    client = get_gs_client()
    sh = client.open_by_url(SPREADSHEET_URL)
    try:
        ws = sh.worksheet("Пользователи")
    except gspread.WorksheetNotFound:
        try:
            ws = sh.worksheet("Users")
        except gspread.WorksheetNotFound:
            logger.info("Лист 'Пользователи' не найден — ограничение по юзерам отключено (разрешаем всем).")
            return set(), set(), set()

    rows = ws.get_all_records()
    if not rows:
        logger.info("Лист 'Пользователи' пуст — ограничение по юзерам отключено (разрешаем всем).")
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
        is_admin_flag = role in {"admin", "админ", "administrator", "администратор"} or _truthy(r.get("admin"))
        is_allowed_flag = _truthy(r.get("allowed") or r.get("доступ") or (not role or role == "user"))
        is_blocked_flag = _truthy(r.get("blocked") or r.get("ban") or r.get("запрет"))

        if is_blocked_flag:
            blocked.add(uid)
        if is_admin_flag:
            admins.add(uid)
            is_allowed_flag = True  # админ всегда разрешён
        if is_allowed_flag:
            allowed.add(uid)

    return allowed, admins, blocked

def ensure_users(force: bool = False):
    """Кэшируем список пользователей из листа 'Пользователи' с TTL."""
    global SHEET_ALLOWED, SHEET_ADMINS, SHEET_BLOCKED, _last_users_ts
    if force or (time.time() - _last_users_ts > USERS_TTL):
        SHEET_ALLOWED, SHEET_ADMINS, SHEET_BLOCKED = load_users_from_sheet()
        _last_users_ts = time.time()
        logger.info(
            f"👥 Пользователи: allowed={len(SHEET_ALLOWED)}, admins={len(SHEET_ADMINS)}, blocked={len(SHEET_BLOCKED)}"
        )

def is_admin(uid: int) -> bool:
    ensure_users()
    return uid in SHEET_ADMINS or uid in ADMINS

def is_allowed(uid: int) -> bool:
    """
    Логика:
    - если лист пуст/нет — разрешаем всем (как сейчас);
    - если есть allowed — whitelist: только allowed или админ;
    - blocked всегда запрещён.
    """
    ensure_users()
    if uid in SHEET_BLOCKED:
        return False
    if SHEET_ALLOWED:
        return (uid in SHEET_ALLOWED) or (uid in SHEET_ADMINS) or (uid in ADMINS)
    return True

# --------------------- ГВАРДЫ ДО ВСЕГО -----------------------
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
def save_issue_to_sheet(bot, user, part: dict, quantity, comment: str):
    """
    Пишем в лист 'История' строго по его текущим заголовкам.
    Поддерживаем разные названия и порядок колонок.
    Ожидаемые ключи: Дата|ID|Имя|Тип|Наименование|Код|Количество|Коментарий/Комментарий/Comment
    """
    try:
        client = get_gs_client()
        sh = client.open_by_url(SPREADSHEET_URL)
        try:
            ws = sh.worksheet("История")
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title="История", rows=1000, cols=12)
            ws.append_row(["Дата", "ID", "Имя", "Тип", "Наименование", "Код", "Количество", "Коментарий"])

        # Заголовки листа (как есть)
        headers_raw = ws.row_values(1)
        headers = [h.strip() for h in headers_raw]
        norm = [h.lower() for h in headers]

        # Имя для печати
        full_name = f"{(user.first_name or '').strip()} {(user.last_name or '').strip()}".strip()
        display_name = full_name or (f"@{user.username}" if user.username else str(user.id))

        ts = now_local_str()  # локальное время по TIMEZONE

        # Маппинг значений по нормализованным ключам
        values_by_key = {
            "дата": ts,
            "timestamp": ts,

            "id": user.id,
            "user_id": user.id,

            "имя": display_name,
            "name": display_name,

            "тип": str(part.get("тип", "")),
            "type": str(part.get("тип", "")),

            "наименование": str(part.get("наименование", "")),
            "name_item": str(part.get("наименование", "")),

            "код": str(part.get("код", "")),
            "code": str(part.get("код", "")),

            "количество": str(quantity),
            "qty": str(quantity),

            # Комментарий — с одной/двумя «м», плюс английский
            "коментарий": comment or "",
            "комментарий": comment or "",
            "comment": comment or "",
        }

        # Строка по фактическому порядку колонок
        row = [values_by_key.get(hn, "") for hn in norm]

        ws.append_row(row, value_input_option="USER_ENTERED")
        logger.info("💾 Списание записано в 'История' по текущим заголовкам")
    except Exception as e:
        logger.error(f"Ошибка записи списания: {e}")
        async def notify():
            for admin_id in (SHEET_ADMINS | ADMINS):
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
        "Привет! Напиши запрос (например: `ФИЛЬТР` или по типу детали `PI8808DRG500`).\n"
        "Команды:\n"
        "• /help — помощь\n"
        "• /more — показать ещё\n"
        "• /export — выгрузка результатов (XLSX/CSV)\n"
        "• /cancel — отменить списание (или кнопкой «Отменить»)\n"
        "• /reload — перезагрузить данные и пользователей (только админ)",
        parse_mode="Markdown"
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "1) Выполните поиск по названию/модели/коду.\n"
        "2) В карточке нажмите «📦 Взять деталь» — бот спросит количество и комментарий, "
        "а затем попросит подтвердить списание (Да/Нет).\n"
        " У ВАС ВСЕ ПОЛУЧИТСЯ.",
        parse_mode="Markdown"
    )

async def reload_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not is_admin(uid):
        return await update.message.reply_text("Доступ запрещён.")
    ensure_fresh_data(force=True)
    ensure_users(force=True)
    await update.message.reply_text("✅ Данные и пользователи перезагружены.")

async def cancel_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if issue_state.pop(uid, None):
        await update.message.reply_text("❌ Операция списания отменена.")
    else:
        await update.message.reply_text("Нет активной операции.")

async def export_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    st = get_user_state(uid)
    results = st.get("results")
    if not isinstance(results, DataFrame):
        results = DataFrame()
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
SEARCH_FIELDS = ["тип", "наименование", "код", "oem", "изготовитель"]  # image НЕ ищем

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
    results = st.get("results")
    if not isinstance(results, DataFrame):
        results = DataFrame()
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
        await update.message.reply_text("Показать ещё?", reply_markup=more_markup())

async def send_page_via_bot(bot, chat_id: int, uid: int):
    st = get_user_state(uid)
    results: DataFrame = st["results"]
    page = st["page"]
    total = len(results)
    pages = math.ceil(total / PAGE_SIZE)
    if page >= pages:
        st["page"] = pages - 1
        return await bot.send_message(chat_id=chat_id, text="Больше результатов нет.")
    start = page * PAGE_SIZE
    end = min(start + PAGE_SIZE, total)
    await bot.send_message(chat_id=chat_id, text=f"Показываю {start + 1}–{end} из {total}.")
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
        if qty <= 0:
            raise ValueError
    except Exception:
        return await update.message.reply_text("Введите положительное число, например: 1 или 2.5", reply_markup=cancel_markup())

    st = issue_state.get(uid)
    if not st or "part" not in st:
        return await update.message.reply_text("Списание неактивно — начните заново из карточки.")

    st["quantity"] = qty
    st["await_comment"] = True
    await update.message.reply_text("Добавьте комментарий Пример: (Линия сборки CSS OP-1100).", reply_markup=cancel_markup())
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

        save_issue_to_sheet(context.bot, q.from_user, part, qty, comment)

        issue_state.pop(uid, None)   # user_state НЕ трогаем

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

    if uid not in issue_state:
        return
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
    st = get_user_state(uid)
    results = st.get("results")
    if not isinstance(results, DataFrame):
        results = DataFrame()
    if results.empty:
        return await q.message.reply_text("Сначала выполните поиск.")
    st["page"] += 1
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

    # Гварды до любых хендлеров
    app.add_handler(MessageHandler(filters.ALL, guard_msg), group=-1)
    app.add_handler(CallbackQueryHandler(guard_cb, pattern=".*"), group=-1)

    # Команды
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("more", more_cmd))
    app.add_handler(CommandHandler("export", export_cmd))
    app.add_handler(CommandHandler("reload", reload_cmd))
    app.add_handler(CommandHandler("cancel", cancel_cmd))

    # Кнопка «Ещё»
    app.add_handler(CallbackQueryHandler(on_more_click, pattern=r"^more$"))

    # Диалог списания
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
            ASK_CONFIRM: [
                CallbackQueryHandler(handle_confirm, pattern=r"^confirm_(yes|no)$"),
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

    # Поиск — в группе 1, чтобы диалог «съедал» апдейты первым
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, search_text), group=1)

    # Error handler
    app.add_error_handler(on_error)

    return app

if __name__ == "__main__":
    logger.info(f"⌚ Используем часовой пояс: {TZ_NAME}")
    ensure_fresh_data(force=True)
    ensure_users(force=True)
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
