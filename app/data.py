# app/data.py
import os
import re
import io
import json
import math
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
    SEARCH_FIELDS,  # ["тип","наименование","код","oem","изготовитель"]
)

logger = logging.getLogger("bot.data")

# ---------------------- Глобальные состояния ----------------------
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

# Conversation states (должны совпадать с handlers)
ASK_QUANTITY, ASK_COMMENT, ASK_CONFIRM = range(3)

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# ---------------------- Время ----------------------
def now_local_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    try:
        return datetime.now(ZoneInfo(TZ_NAME)).strftime(fmt)
    except Exception:
        return datetime.utcnow().strftime(fmt)

# ---------------------- Google Sheets ----------------------
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

# ---------------------- Поисковый индекс ----------------------
def build_search_index(df: DataFrame) -> Dict[str, Set[int]]:
    index: DefaultDict[str, Set[int]] = defaultdict(set)
    for col in SEARCH_FIELDS:
        if col not in df.columns:
            continue
        # .items() чтобы вернуть (idx, value)
        for idx, val in df[col].astype(str).str.lower().items():
            for t in re.findall(r'\w+', val):
                if t:
                    index[t].add(idx)
    return dict(index)

# ---------------------- Утилиты ----------------------
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

# ---------------------- Работа с картинками ----------------------
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

async def resolve_image_url_async(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return u
    if "drive.google.com" in u:
        return normalize_drive_url(u)
    return u

def build_image_index(df: pd.DataFrame) -> dict[str, str]:
    """
    Индексируем ТОЛЬКО те ссылки, где код встречается в самом URL (путь/квери).
    Никаких фолбэков «взять картинку из строки».
    """
    if "image" not in df.columns or "код" not in df.columns:
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
        if _url_has_code(url, raw, sq):
            url_norm = normalize_drive_url(url)
            if raw and raw not in index:
                index[raw] = url_norm
            if sq and sq not in index:
                index[sq] = url_norm
            matched += 1
    logger.info(f"image-index: совпадений по URL={matched} из {total}")
    return index

async def find_image_by_code_async(code: str) -> str:
    if not code or _image_index is None:
        return ""
    raw, sq = _norm_code(code)
    return _image_index.get(raw) or _image_index.get(sq, "") or ""

# ---------------------- Загрузка данных ----------------------
def initial_load():
    global df, _last_load_ts, _search_index, _image_index
    data = load_data_blocking()
    new_df = DataFrame(data)
    new_df.columns = new_df.columns.str.strip().str.lower()

    # нормализуем код/оем и image
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
                new_df[col] = new_df[col].astype(str).str
