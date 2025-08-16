# app/data.py
import io
import re
import time
import math
import json
import asyncio
import logging
from html import escape
from typing import Optional, Dict, Any, Set, List, DefaultDict
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import urlparse, parse_qs
import aiohttp
import gspread
import pandas as pd
from pandas import DataFrame
from google.oauth2.service_account import Credentials

# ==== конфиг ====
from app.config import (
    SPREADSHEET_URL, CREDS_JSON, SHEET_NAME,
    SCOPES, DATA_TTL, USERS_TTL, TZ_NAME
)

# SEARCH_FIELDS можно переопределить в config, иначе используем дефолт
try:
    from app.config import SEARCH_FIELDS  # type: ignore
except Exception:
    SEARCH_FIELDS = ["тип", "наименование", "код", "oem", "изготовитель"]

logger = logging.getLogger("bot.data")

# ---------------------- ГЛОБАЛЬНЫЕ СОСТОЯНИЯ ----------------
df: Optional[DataFrame] = None
_last_load_ts = 0.0
_search_index: Optional[Dict[str, Set[int]]] = None

# Кэш картинок: ключ — нормализованный код, значение — URL
_image_index: Optional[Dict[str, str]] = None

# Пользовательские допуски (из листа "Пользователи"/"Users")
SHEET_ALLOWED: Set[int] = set()
SHEET_ADMINS: Set[int] = set()
SHEET_BLOCKED: Set[int] = set()
_last_users_ts = 0.0

# Память диалогов
user_state: Dict[int, Dict[str, Any]] = {}
issue_state: Dict[int, Dict[str, Any]] = {}

# Флаги конкуренции
_loading_data = False
_loading_users = False

# Стейты диалога списания (для ConversationHandler)
ASK_QUANTITY, ASK_COMMENT, ASK_CONFIRM = range(3)

# -------------------------- ВСПОМОГАТЕЛЬНОЕ -----------------
def now_local_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    try:
        return datetime.now(ZoneInfo(TZ_NAME)).strftime(fmt)
    except Exception:
        return datetime.utcnow().strftime(fmt)

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

def _get_all_records_safe(ws) -> list[dict]:
    """
    Безопасный парсер таблицы (устойчив к дублям/пустым заголовкам).
    """
    values: List[List[Any]] = ws.get_all_values()
    if not values:
        return []
    # найдём первую непустую строку как заголовки
    header_row_idx = 0
    for i, row in enumerate(values):
        if any(str(c).strip() for c in row):
            header_row_idx = i
            break
    headers_raw = [str(h).strip() for h in values[header_row_idx]]
    # если заголовок пуст — подставим плейсхолдер
    headers = []
    used = set()
    for i, h in enumerate(headers_raw):
        name = h or f"__col_{i}"
        # дедуп
        base = name
        k = 1
        while name.lower() in used:
            k += 1
            name = f"{base}_{k}"
        used.add(name.lower())
        headers.append(name)
    data_rows = values[header_row_idx + 1:]
    out: List[dict] = []
    for r in data_rows:
        # выравниваем длину
        row = list(r) + [""] * (len(headers) - len(r))
        d = {headers[i]: row[i] for i in range(len(headers))}
        out.append(d)
    return out

def load_data_blocking() -> list[dict]:
    client = get_gs_client()
    ws = _open_data_worksheet(client)
    try:
        # пробуем быстро
        return ws.get_all_records()
    except Exception as e:
        logger.warning(f"get_all_records упал ({e}), переключаюсь на _get_all_records_safe()")
        return _get_all_records_safe(ws)

# --------------------- ПОИСК (индекс) ------------------------
def build_search_index(df: DataFrame) -> Dict[str, Set[int]]:
    index: DefaultDict[str, Set[int]] = defaultdict(set)
    for col in SEARCH_FIELDS:
        if col not in df.columns:
            continue
        for idx, val in df[col].astype(str).str.lower().items():
            for t in re.findall(r'\w+', val):
                if t:
                    index[t].add(idx)
    return dict(index)

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
        if not result:
            break
    return result or set()

def _safe_col(df_: DataFrame, col: str) -> Optional[pd.Series]:
    return df_[col].astype(str).str.lower() if col in df_.columns else None

def _relevance_score(row: dict, tokens: List[str], q_squash: str) -> int:
    score = 0
    for f in SEARCH_FIELDS:
        val_ = str(row.get(f, "")).lower()
        if not val_:
            continue
        words = set(re.findall(r'\w+', val_))
        tok_hit = sum(1 for t in tokens if t in words)
        sub_hit = sum(1 for t in tokens if t and t in val_)
        sq = re.sub(r'[\W_]+', '', val_)
        squash_hit = 1 if q_squash and q_squash in sq else 0
        weight = 2 if f in ("код", "oem") else 1
        score += weight * (2 * tok_hit + sub_hit) + 3 * squash_hit * weight
    return score

# --------------------- Картинки по КОДУ -----------------------
def _norm_code(c: str) -> tuple[str, str]:
    raw = (c or "").strip().lower()
    squash_ = re.sub(r'[\W_]+', '', raw, flags=re.UNICODE)
    return raw, squash_

def _filename_from_url(u: str) -> str:
    """
    Достаём имя файла из URL, даже если это ibb/drive.
    """
    try:
        pu = urlparse(u)
        # drive open?id=... — имени нет, но часто есть в og:image, это уже разрулим на этапе resolve_ibb/drive
        name = pu.path.rsplit("/", 1)[-1]
        if not name or "." not in name:
            # иногда имя в query (?name=xxx.jpg)
            qs = parse_qs(pu.query)
            for k in ("name", "filename", "file", "img"):
                if k in qs and qs[k]:
                    return qs[k][0]
        return name
    except Exception:
        return ""

def build_image_index(df_: DataFrame) -> Dict[str, str]:
    """
    Индекс картинок строго по КОДУ:
    - ключи: raw и squash(код)
    - значение: URL из столбца 'image' (если он не пустой)
    *никакого фолбэка на картинку из строки при отправке — это только индекс*
    """
    if "image" not in df_.columns:
        return {}
    index: Dict[str, str] = {}
    for _, row in df_.iterrows():
        code_val = str(row.get("код", "")).strip()
        url = str(row.get("image", "")).strip()
        if not code_val or not url:
            continue
        raw, sq = _norm_code(code_val)
        if raw and raw not in index:
            index[raw] = url
        if sq and sq not in index:
            index[sq] = url
    return index

async def resolve_ibb_direct_async(url: str) -> str:
    """
    Для ibb.co: превращаем в прямой URL к картинке через og:image
    """
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

def normalize_drive_url(url: str) -> str:
    m = re.search(r'drive\.google\.com/(?:file/d/([-\w]{20,})|open\?id=([-\w]{20,}))', url)
    if m:
        file_id = m.group(1) or m.group(2)
        return f'https://drive.google.com/uc?export=download&id={file_id}'
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
    """
    1) Ищем в _image_index по raw/squash(код).
    2) Если нет — пробегаем по всем image, ищем код как подстроку в имени файла (без знаков).
    Возвращаем ПУСТО, если не нашли (handlers отправят карточку без фото).
    """
    if not code:
        return ""
    raw, sq = _norm_code(code)
    if _image_index:
        url = _image_index.get(raw) or _image_index.get(sq)
        if url:
            return url

    # медленный путь: поиск в df по имени
    global df
    if df is None or "image" not in df.columns:
        return ""
    target = sq
    for _, row in df.iterrows():
        url = str(row.get("image", "")).strip()
        if not url:
            continue
        name = _filename_from_url(url)
        name_sq = squash(name)
        if target and target in name_sq:
            return url
    return ""

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

# ------------------------- ЗАГРУЗКА ДАННЫХ -------------------
def initial_load():
    """
    Синхронная загрузка при старте.
    """
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

    # сразу подгрузим пользователей
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
        data = await asyncio.to_thread(load_data_blocking)
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

# ---------------------- ВЫГРУЗКА (XLSX/CSV) ------------------
def _df_to_xlsx(df_: DataFrame, name: str) -> io.BytesIO:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df_.to_excel(w, index=False)
    buf.seek(0)
    buf.name = name
    return buf

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

def _open_users_worksheet(client):
    sh = client.open_by_url(SPREADSHEET_URL)
    try:
        return sh.worksheet("Пользователи")
    except gspread.WorksheetNotFound:
        try:
            return sh.worksheet("Users")
        except gspread.WorksheetNotFound:
            return None

def load_users_from_sheet():
    client = get_gs_client()
    ws = _open_users_worksheet(client)
    if ws is None:
        logger.info("Лист 'Пользователи' не найден — доступ разрешён всем.")
        return set(), set(), set()

    # читаем через безопасный парсер (во избежание дублей заголовков)
    try:
        rows = _get_all_records_safe(ws)
    except Exception as e:
        logger.warning(f"_get_all_records_safe(users) упал: {e}")
        rows = []
    if not rows:
        logger.info("Лист 'Пользователи' пуст — доступ разрешён всем.")
        return set(), set(), set()

    allowed, admins, blocked = set(), set(), set()

    for row in rows:
        # нормализуем ключи
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

    logger.info(f"Пользователи прочитаны: allowed={len(allowed)}, admins={len(admins)}, blocked={len(blocked)}")
    return allowed, admins, blocked
