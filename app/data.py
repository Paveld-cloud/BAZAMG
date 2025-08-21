#import os
import re
import io
import time
import json
import asyncio
import logging
from typing import Optional, Dict, Any, Set, List, DefaultDict
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

import aiohttp
import gspread
import pandas as pd
from pandas import DataFrame
from google.oauth2.service_account import Credentials

from app.config import (
    TELEGRAM_TOKEN,
    SPREADSHEET_URL,
    GOOGLE_APPLICATION_CREDENTIALS_JSON,
    TZ_NAME,
    SHEET_NAME,
    DATA_TTL,
    USERS_TTL,
    PAGE_SIZE,
    MAX_QTY,
    IMAGE_STRICT,
    SEARCH_FIELDS,
)

logger = logging.getLogger("bot.data")

# ---------------------- Глобальное состояние ----------------------
df: Optional[DataFrame] = None
_last_load_ts = 0.0

_search_index: Optional[Dict[str, Set[int]]] = None
_image_index: Optional[Dict[str, str]] = None  # key: код (lower), val: url

SHEET_ALLOWED: Set[int] = set()
SHEET_ADMINS: Set[int] = set()
SHEET_BLOCKED: Set[int] = set()
_last_users_ts = 0.0

user_state: Dict[int, Dict[str, Any]] = {}
issue_state: Dict[int, Dict[str, Any]] = {}

ASK_QUANTITY, ASK_COMMENT, ASK_CONFIRM = range(3)

# ---------------------- Время/формат ----------------------
def now_local_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    try:
        return datetime.now(ZoneInfo(TZ_NAME)).strftime(fmt)
    except Exception:
        return datetime.utcnow().strftime(fmt)

# ---------------------- Google Sheets ----------------------
def get_gs_client():
    creds_info = json.loads(GOOGLE_APPLICATION_CREDENTIALS_JSON)
    creds = Credentials.from_service_account_info(
        creds_info, scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    return gspread.authorize(creds)

def _open_data_worksheet(client):
    sh = client.open_by_url(SPREADSHEET_URL)
    if SHEET_NAME:
        try:
            return sh.worksheet(SHEET_NAME)
        except gspread.WorksheetNotFound:
            logger.warning(f"Лист {SHEET_NAME!r} не найден, fallback на sheet1")
    return sh.sheet1

def load_data_blocking() -> List[dict]:
    client = get_gs_client()
    ws = _open_data_worksheet(client)
    return ws.get_all_records()

# ---------------------- Текстовые утилиты ----------------------
def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", (text or "")).lower().strip()

def squash(text: str) -> str:
    return re.sub(r"[\W_]+", "", (text or "").lower(), flags=re.UNICODE)

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

# ---------------------- Поиск ----------------------
def build_search_index(dataframe: DataFrame) -> Dict[str, Set[int]]:
    index: DefaultDict[str, Set[int]] = defaultdict(set)
    for col in SEARCH_FIELDS:
        if col not in dataframe.columns:
            continue
        series = dataframe[col].astype(str).str.lower()
        for idx, val_ in series.items():
            for t in re.findall(r"\w+", val_):
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

def _safe_col(dataframe: DataFrame, col: str) -> Optional[pd.Series]:
    return dataframe[col].astype(str).str.lower() if col in dataframe.columns else None

def _relevance_score(row: dict, tokens: List[str], q_squash: str) -> int:
    score = 0
    for f in SEARCH_FIELDS:
        val_ = str(row.get(f, "")).lower()
        if not val_:
            continue
        words = set(re.findall(r"\w+", val_))
        tok_hit = sum(1 for t in tokens if t in words)
        sub_hit = sum(1 for t in tokens if t and t in val_)
        sq = re.sub(r"[\W_]+", "", val_)
        squash_hit = 1 if q_squash and q_squash in sq else 0
        weight = 2 if f in ("код", "oem") else 1
        score += weight * (2 * tok_hit + sub_hit) + 3 * squash_hit * weight
    return score

# ---------------------- Картинки: строгий индекс по имени файла ----------------------
def _extract_code_from_url(url: str) -> str:
    """
    Берём последний сегмент пути, удаляем query/fragment, обрезаем расширение.
    Приводим к lower.
    Примеры:
      https://i.ibb.co/abc/UZ000346.jpg -> uz000346
      https://i.ibb.co/xyz/UZCSS03703.png -> uzcss03703
    """
    try:
        u = (url or "").strip()
        if not u:
            return ""
        # убрать query/fragment
        u = u.split("#", 1)[0].split("?", 1)[0]
        # последний сегмент пути
        last = u.rsplit("/", 1)[-1]
        # отрезать расширение
        base = last.split(".", 1)[0]
        return base.strip().lower()
    except Exception:
        return ""

def build_image_index(dataframe: DataFrame) -> Dict[str, str]:
    """
    Строгий режим: ключ = код из ИМЕНИ ФАЙЛА (последнего сегмента URL без расширения).
    Если код по имени файла не извлечён — запись пропускается.
    """
    index: Dict[str, str] = {}
    if "image" not in dataframe.columns:
        logger.info("image-index: колонка 'image' отсутствует")
        return index

    total = len(dataframe)
    added = 0
    for _, row in dataframe.iterrows():
        raw_url = str(row.get("image", "")).strip()
        if not raw_url:
            continue
        code_from_url = _extract_code_from_url(raw_url)
        if not code_from_url:
            continue
        # Если хочешь, чтобы выигрывала последняя ссылка — просто перетираем.
        index[code_from_url] = raw_url
        added += 1

    logger.info(f"image-index[STRICT(FILENAME)]: добавлено {added} из {total} строк")
    return index

async def find_image_by_code_async(code: str) -> str:
    """
    Отдаём ссылку только если код найден в индексе по имени файла.
    Никаких фолбэков на значение из строки не делаем.
    """
    if not code or _image_index is None:
        return ""
    key = (code or "").strip().lower()
    return _image_index.get(key, "")

def normalize_drive_url(url: str) -> str:
    m = re.search(
        r"drive\.google\.com/(?:file/d/([-\w]{20,})|open\?id=([-\w]{20,}))", url
    )
    if m:
        file_id = m.group(1) or m.group(2)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url

async def resolve_ibb_direct_async(url: str) -> str:
    # На всякий: если пришлют страницу ibb.co/..., попытаемся достать og:image
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=12) as resp:
                if resp.status != 200:
                    return url
                html = await resp.text()
        m = re.search(
            r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
            html,
            re.I,
        )
        return m.group(1) if m else url
    except Exception:
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

# ---------------------- Загрузка/обновление данных ----------------------
async def ensure_fresh_data_async(force: bool = False):
    global df, _last_load_ts, _search_index, _image_index
    if not force and df is not None and (time.time() - _last_load_ts <= DATA_TTL):
        return

    data = await asyncio.to_thread(load_data_blocking)
    new_df = DataFrame(data)
    new_df.columns = new_df.columns.str.strip().str.lower()

    # нормализуем важные колонки
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

def ensure_fresh_data(force: bool = False):
    if not force and df is not None and (time.time() - _last_load_ts <= DATA_TTL):
        return
    # fire-and-forget
    asyncio.create_task(ensure_fresh_data_async(force=True))

def initial_load():
    """
    Синхронная загрузка для старта процесса.
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

# ---------------------- Пользователи (allowed/admins/blocked) ----------------------
def _truthy(x) -> bool:
    s = str(x).strip().lower()
    return s in {
        "1", "true", "yes", "y", "да", "истина", "ok", "ок",
        "allowed", "разрешен", "разрешено"
    } or (s.isdigit() and int(s) > 0)

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
    """
    Робастный парсер листа 'Пользователи' (или 'Users'):
    читает первую строку как заголовок, остальные как строки,
    поддерживает разные имена колонок и пустые ячейки.
    """
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

    # вручную читаем таблицу (без get_all_records, т.к. у них дубликаты хедеров случались)
    all_values = ws.get_all_values()
    if not all_values:
        logger.info("Лист 'Пользователи' пуст — доступ разрешён всем.")
        return set(), set(), set()

    headers = [str(h).strip().lower() for h in (all_values[0] or [])]
    rows = all_values[1:]

    allowed, admins, blocked = set(), set(), set()
    for r in rows:
        row_map = {}
        for i, v in enumerate(r):
            key = headers[i] if i < len(headers) else f"col{i}"
            row_map[key] = v

        # маппинг полей
        uid = (
            _to_int_or_none(row_map.get("user_id"))
            or _to_int_or_none(row_map.get("userid"))
            or _to_int_or_none(row_map.get("id"))
            or _to_int_or_none(row_map.get("uid"))
            or _to_int_or_none(row_map.get("телеграм id"))
            or _to_int_or_none(row_map.get("пользователь"))
        )
        if not uid:
            continue

        role = str(
            row_map.get("role") or row_map.get("роль") or ""
        ).strip().lower()
        is_admin = role in {"admin", "админ", "administrator", "администратор"} or _truthy(
            row_map.get("admin")
        )
        is_allowed = _truthy(row_map.get("allowed") or row_map.get("доступ") or (not role or role == "user"))
        is_blocked = _truthy(row_map.get("blocked") or row_map.get("ban") or row_map.get("запрет"))

        if is_blocked:
            blocked.add(uid)
        if is_admin:
            admins.add(uid)
            is_allowed = True
        if is_allowed:
            allowed.add(uid)

    logger.info(f"Пользователи прочитаны: allowed={len(allowed)}, admins={len(admins)}, blocked={len(blocked)}")
    return allowed, admins, blocked

# ---------------------- Экспорт ----------------------
def _df_to_xlsx(df_: DataFrame, name: str) -> io.BytesIO:
    buf = io.BytesIO()
    # openpyxl обязан быть в requirements
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df_.to_excel(w, index=False)
    buf.seek(0)
    buf.name = name
    return buf
