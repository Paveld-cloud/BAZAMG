# app/data.py
import re
import io
import json
import time
import asyncio
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any, Set, List, DefaultDict
from collections import defaultdict

import gspread
import pandas as pd
from pandas import DataFrame
from google.oauth2.service_account import Credentials

from app.config import (
    SPREADSHEET_URL,
    GOOGLE_APPLICATION_CREDENTIALS_JSON,
    SHEET_NAME,
    TZ_NAME,
    DATA_TTL,
    USERS_TTL,
    SEARCH_FIELDS,
    IMAGE_STRICT,
)

logger = logging.getLogger("bot.data")

# ======================= Глобальные состояния =======================
df: Optional[DataFrame] = None
_last_load_ts = 0.0
_search_index: Optional[Dict[str, Set[int]]] = None
_image_index: Optional[Dict[str, str]] = None

SHEET_ALLOWED: Set[int] = set()
SHEET_ADMINS: Set[int] = set()
SHEET_BLOCKED: Set[int] = set()
_last_users_ts = 0.0

user_state: Dict[int, Dict[str, Any]] = {}
issue_state: Dict[int, Dict[str, Any]] = {}

_loading_data = False
_loading_users = False

ASK_QUANTITY, ASK_COMMENT, ASK_CONFIRM = range(3)

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# ======================= Время =======================
def now_local_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    try:
        return datetime.now(ZoneInfo(TZ_NAME)).strftime(fmt)
    except Exception:
        return datetime.utcnow().strftime(fmt)

# ======================= Sheets =======================
def get_gs_client():
    creds_info = json.loads(GOOGLE_APPLICATION_CREDENTIALS_JSON)
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

# ======================= Поиск =======================
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

def _safe_col(df: DataFrame, col: str) -> Optional[pd.Series]:
    return df[col].astype(str).str.lower() if col in df.columns else None

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

# ======================= Картинки =======================
def _norm_code(c: str) -> tuple[str, str]:
    raw = (c or "").strip().lower()
    squash_ = re.sub(r'[\W_]+', '', raw, flags=re.UNICODE)
    return raw, squash_

def _url_has_code(url: str, code_raw: str, code_sq: str) -> bool:
    s = (url or "").strip().lower()
    if not s:
        return False
    s_sq = re.sub(r'[\W_]+', '', s, flags=re.UNICODE)
    return (code_raw and code_raw in s) or (code_sq and code_sq in s_sq)

def normalize_drive_url(url: str) -> str:
    m = re.search(r'drive\.google\.com/(?:file/d/([-\w]{20,})|open\?id=([-\w]{20,}))', url)
    if m:
        file_id = m.group(1) or m.group(2)
        return f'https://drive.google.com/uc?export=download&id={file_id}'
    return url

def build_image_index(df: pd.DataFrame) -> dict[str, str]:
    """
    Индекс фотографий.
    - Если IMAGE_STRICT=1: индексируем только те URL, где код встречается в самом URL.
    - Если IMAGE_STRICT=0: каждую непустую пару код → image берём из строки.
    """
    if "код" not in df.columns or "image" not in df.columns:
        return {}

    index: dict[str, str] = {}
    total = 0
    matched = 0

    for _, row in df.iterrows():
        total += 1
        code_val = str(row.get("код", "")).strip().lower()
        url = str(row.get("image", "")).strip()
        if not code_val or not url:
            continue

        raw, sq = _norm_code(code_val)

        ok = True
        if IMAGE_STRICT:
            ok = _url_has_code(url, raw, sq)

        if ok:
            url_norm = normalize_drive_url(url)
            for k in (raw, sq):
                if k and k not in index:
                    index[k] = url_norm
            matched += 1

    mode = "STRICT(URL)" if IMAGE_STRICT else "ROW-MAP"
    logger.info(f"image-index[{mode}]: добавлено {matched} из {total} строк")
    return index

async def find_image_by_code_async(code: str) -> str:
    if not code or _image_index is None:
        return ""
    raw, sq = _norm_code(code)
    return _image_index.get(raw) or _image_index.get(sq, "") or ""

async def resolve_image_url_async(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return u
    if "drive.google.com" in u:
        return normalize_drive_url(u)
    return u

# ======================= Загрузка / перезагрузка =======================
def initial_load():
    global df, _last_load_ts, _search_index, _image_index
    data = load_data_blocking()
    new_df = DataFrame(data)
    new_df.columns = new_df.columns.str.strip().str.lower()

    # нормализация
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

# ======================= Users =======================
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

def _read_users_rows(ws) -> list[dict]:
    vals = ws.get_all_values()
    if not vals:
        return []
    headers_raw = [h.strip() for h in vals[0]]
    seen = set()
    headers_norm = []
    for i, h in enumerate(headers_raw):
        key = h.strip().lower()
        if not key or key in seen:
            key = f"col_{i}"
        seen.add(key)
        headers_norm.append(key)
    rows: list[dict] = []
    for r in vals[1:]:
        if len(r) < len(headers_norm):
            r = r + [""] * (len(headers_norm) - len(r))
        elif len(r) > len(headers_norm):
            r = r[: len(headers_norm)]
        rows.append({headers_norm[i]: r[i] for i in range(len(headers_norm))})
    return rows

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

    try:
        rows = ws.get_all_records()
    except Exception as e:
        logger.warning(f"get_all_records не сработал ({e}), fallback на ручной парсинг.")
        rows = _read_users_rows(ws)

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
            or _to_int_or_none(r.get("col_0"))
        )
        if not uid:
            continue

        role = str(r.get("role") or r.get("роль") or "").strip().lower()
        is_admin = role in {"admin", "админ", "administrator", "администратор"} or _truthy(r.get("admin"))
        is_blocked = _truthy(r.get("blocked") or r.get("ban") or r.get("запрет"))
        is_allowed = _truthy(r.get("allowed") or r.get("доступ") or (not role or role == "user"))

        if is_blocked:
            blocked.add(uid)
        if is_admin:
            admins.add(uid)
            is_allowed = True
        if is_allowed:
            allowed.add(uid)

    logger.info(f"Пользователи прочитаны: allowed={len(allowed)}, admins={len(admins)}, blocked={len(blocked)}")
    return allowed, admins, blocked

async def ensure_users_async(force: bool = False):
    global SHEET_ALLOWED, SHEET_ADMINS, SHEET_BLOCKED, _last_users_ts, _loading_users
    if not force and (time.time() - _last_users_ts <= USERS_TTL):
        return
    if _loading_users:
        return
    _loading_users = True
    try:
        allowed, admins, blocked = await asyncio.to_thread(load_users_from_sheet)
        SHEET_ALLOWED, SHEET_ADMINS, SHEET_BLOCKED = allowed, admins, blocked
        _last_users_ts = time.time()
        logger.info(f"👥 Пользователи: allowed={len(allowed)}, admins={len(admins)}, blocked={len(blocked)}")
    finally:
        _loading_users = False

def ensure_users(force: bool = False):
    if not force and (time.time() - _last_users_ts <= USERS_TTL):
        return
    asyncio.create_task(ensure_users_async(force=True))

# ======================= Экспорт =======================
def _df_to_xlsx(df: DataFrame, name: str) -> io.BytesIO:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False)
    buf.seek(0)
    buf.name = name
    return buf
