# app/data.py
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

from .config import (
    SPREADSHEET_URL, GOOGLE_APPLICATION_CREDENTIALS_JSON, SHEET_NAME, TZ_NAME,
    DATA_TTL, USERS_TTL, SEARCH_FIELDS, IMAGE_STRICT
)

logger = logging.getLogger("bot.data")

# ----------------- Время / TZ -----------------
_local_tz = ZoneInfo(TZ_NAME)

def now_local_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    return datetime.now(_local_tz).strftime(fmt)

# ----------------- Google Sheets Client -----------------
_gs_client = None

def get_gs_client():
    global _gs_client
    if _gs_client is None:
        creds_dict = json.loads(GOOGLE_APPLICATION_CREDENTIALS_JSON or "{}")
        credentials = Credentials.from_service_account_info(
            creds_dict,
            scopes=["https://www.googleapis.com/auth/spreadsheets",
                    "https://www.googleapis.com/auth/drive"]
        )
        _gs_client = gspread.authorize(credentials)
    return _gs_client

# ----------------- Датасет и индексы -----------------
df: Optional[DataFrame] = None
_search_index: Dict[str, Set[int]] = {}
_image_index: Dict[str, str] = {}

_last_loaded = 0.0
_data_refresh_lock = asyncio.Lock()

def normalize_text(s: str) -> str:
    s = (s or "").lower()
    # убрать слово "роза", кавычки, круглые скобки
    s = re.sub(r"[\"'«»]", " ", s)
    s = re.sub(r"\(.*?\)", " ", s)
    s = s.replace("роза", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    return [t for t in re.split(r"[^\w]+", s) if t]

def build_search_index(df: DataFrame) -> Dict[str, Set[int]]:
    index: Dict[str, Set[int]] = defaultdict(set)
    if df is None or df.empty:
        return dict(index)
    for i, row in df.iterrows():
        buf = []
        for field in SEARCH_FIELDS:
            if field in df.columns:
                v = str(row.get(field, "")).strip()
                if v:
                    buf.append(v)
        tokens = tokenize(" ".join(buf))
        for t in tokens:
            index[t].add(i)
    return dict(index)

def build_image_index(df: DataFrame) -> Dict[str, str]:
    res: Dict[str, str] = {}
    if df is None or df.empty:
        return res
    # 1) Жёсткая привязка: столбец image (если есть)
    if "image" in df.columns:
        for _, row in df.iterrows():
            code = str(row.get("код", "")).strip()
            img = str(row.get("image", "")).strip()
            if code and img:
                res[code] = img
    return res

def normalize_drive_url(u: str) -> str:
    # https://drive.google.com/file/d/{fileId}/view?usp=drive_link -> uc?export=download&id=
    m = re.search(r"/d/([^/]+)/", u)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    m = re.search(r"[?&]id=([^&]+)", u)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    return u

async def resolve_ibb_direct_async(page_url: str) -> str:
    # получаем meta og:image
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(page_url, timeout=10) as resp:
                html = await resp.text()
        m = re.search(r'<meta\s+property=["\']og:image["\']\s+content=["\']([^"\']+)["\']', html, re.I)
        if m:
            return m.group(1)
    except Exception as e:
        logger.warning(f"resolve_ibb_direct_async failed: {e}")
    return page_url

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
    global df, _search_index, _image_index, _last_loaded
    async with _data_refresh_lock:
        now = time.time()
        if not force and (now - _last_loaded) < DATA_TTL and df is not None:
            return
        logger.info("Loading data from Google Sheets...")
        def _load():
            client = get_gs_client()
            sh = client.open_by_url(SPREADSHEET_URL)
            ws = sh.worksheet(SHEET_NAME) if SHEET_NAME else sh.sheet1
            records = ws.get_all_records()
            return pd.DataFrame(records)
        try:
            new_df = await asyncio.to_thread(_load)
        except Exception as e:
            logger.exception(f"Failed to load dataset: {e}")
            return
        df = new_df
        _search_index = build_search_index(df)
        _image_index = build_image_index(df)
        _last_loaded = now
        logger.info(f"Data loaded: {len(df) if df is not None else 0} rows")

def get_image_for_code(code: str) -> str:
    return _image_index.get(code or "", "")

# ---------------------- Поиск ----------------------
def score_row(row: pd.Series, query: str) -> float:
    text = " ".join([str(row.get(f, "")) for f in SEARCH_FIELDS])
    q = normalize_text(query)
    # простейший скоринг: подстрока + количество совпадений токенов
    score = 0.0
    if q and q in normalize_text(text):
        score += 1.0
    q_tokens = set(tokenize(q))
    t_tokens = set(tokenize(text))
    score += len(q_tokens & t_tokens) * 0.5
    return score

def search(query: str, offset: int = 0, limit: int = 5) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    if not query:
        return []
    scores = []
    for i, row in df.iterrows():
        s = score_row(row, query)
        if s > 0:
            scores.append((s, i))
    scores.sort(reverse=True)
    items = []
    for _, idx in scores[offset:offset+limit]:
        row = df.loc[idx]
        item = {k: row.get(k) for k in df.columns}
        items.append(item)
    return items

# ---------------------- Пользователи ----------------------
SHEET_ALLOWED: Set[int] = set()
SHEET_ADMINS: Set[int] = set()
SHEET_BLOCKED: Set[int] = set()

_users_refresh_lock = asyncio.Lock()
_last_users_refreshed = 0.0

def _parse_bool(v) -> bool:
    s = str(v).strip().lower()
    return s in {"1","true","yes","y","да","ok","ok.","ок","ок."}

def load_users_from_sheet():
    """Считываем лист 'Пользователи' если есть. Возвращаем (allowed, admins, blocked)"""
    allowed, admins, blocked = set(), set(), set()
    client = get_gs_client()
    sh = client.open_by_url(SPREADSHEET_URL)
    try:
        ws = sh.worksheet("Пользователи")
    except gspread.WorksheetNotFound:
        return allowed, admins, blocked
    values = ws.get_all_values()
    if not values:
        return allowed, admins, blocked
    headers = [h.strip() for h in values[0]]
    # Ищем стандартные столбцы
    colmap = {h.lower(): i for i, h in enumerate(headers)}
    for row in values[1:]:
        def get(colname, default=""):
            i = colmap.get(colname.lower())
            if i is None or i >= len(row):
                return default
            return row[i]
        id_raw = get("id", "").strip()
        if id_raw.isdigit():
            uid = int(id_raw)
        else:
            continue
        access = str(get("access", "")).strip().lower()
        role = str(get("role", "")).strip().lower()
        is_allowed = access in {"allow","allowed","доступ","ok"} or _parse_bool(get("allowed",""))
        is_blocked = access in {"block","blocked","ban","запрет"} or _parse_bool(get("blocked",""))
        is_admin = role in {"admin","админ"} or _parse_bool(get("admin",""))
        if is_admin:
            admins.add(uid)
            allowed.add(uid)
        elif is_blocked:
            blocked.add(uid)
        elif is_allowed or (not access and not is_blocked):
            allowed.add(uid)
    return allowed, admins, blocked

async def ensure_users_async(force: bool = False):
    global _last_users_refreshed, SHEET_ALLOWED, SHEET_ADMINS, SHEET_BLOCKED
    async with _users_refresh_lock:
        now = asyncio.get_event_loop().time()
        if not force and (now - _last_users_refreshed) < USERS_TTL:
            return
        allowed, admins, blocked = await asyncio.to_thread(load_users_from_sheet)
        SHEET_ALLOWED = allowed
        SHEET_ADMINS = admins
        SHEET_BLOCKED = blocked
        _last_users_refreshed = now
        logger.info(f"Users loaded: allowed={len(allowed)}, admins={len(admins)}, blocked={len(blocked)}")
