import io
import re
import math
import time
import json
import httpx
import logging
import pandas as pd
from typing import Optional, Dict, Any, Set, List, DefaultDict
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
from pandas import DataFrame
from html import escape
from urllib.parse import urlparse, parse_qs, unquote

import gspread
from google.oauth2.service_account import Credentials

from app.config import (
    SPREADSHEET_URL, SHEET_NAME, GOOGLE_APPLICATION_CREDENTIALS_JSON,
    DATA_TTL, USERS_TTL, TZ_NAME, PAGE_SIZE, MAX_QTY
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

ASK_QUANTITY, ASK_COMMENT, ASK_CONFIRM = range(3)

SEARCH_FIELDS = ["тип", "наименование", "код", "oem", "изготовитель"]

def now_local_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    try:
        return datetime.now(ZoneInfo(TZ_NAME)).strftime(fmt)
    except Exception:
        return datetime.utcnow().strftime(fmt)

# ------------------------- Google Sheets --------------------------
def get_gs_client():
    creds_info = json.loads(GOOGLE_APPLICATION_CREDENTIALS_JSON)
    creds = Credentials.from_service_account_info(creds_info, scopes=["https://www.googleapis.com/auth/spreadsheets"])
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

# -------------------- Картинки: индекс/поиск ----------------------
def _norm_code(c: str) -> tuple[str, str]:
    raw = (c or "").strip().lower()
    squash = re.sub(r'[\W_]+', '', raw, flags=re.UNICODE)
    return raw, squash

def _filename_from_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    p = urlparse(u)
    name = unquote(p.path.rsplit("/", 1)[-1] or "")
    if not name or "." not in name:
        qnames = []
        for v in parse_qs(p.query).values():
            qnames.extend(v)
        candidates = qnames + [p.fragment]
        for cand in candidates:
            if isinstance(cand, str):
                cand = unquote(cand)
                cand_name = cand.rsplit("/", 1)[-1]
                if "." in cand_name:
                    name = cand_name
                    break
    return name

def _tokens_from_filename(u: str) -> list[str]:
    name = _filename_from_url(u)
    if not name:
        return []
    base = name.rsplit(".", 1)[0]
    fused = re.sub(r"[\W_]+", "", base.lower(), flags=re.UNICODE)
    parts = re.split(r"[\W_]+", base.lower())
    parts = [p for p in parts if p]
    seen, out = set(), []
    for t in [fused] + parts:
        if t and t not in seen:
            seen.add(t); out.append(t)
    return out

def _fuse(text: str) -> str:
    return re.sub(r"[\W_]+", "", (text or "").lower(), flags=re.UNICODE)

def _scan_images_by_code_fallback(code: str) -> str:
    global df
    if df is None or "image" not in df.columns:
        return ""
    sq = _fuse(code)
    if not sq:
        return ""
    try:
        for _, row in df.iterrows():
            u = str(row.get("image", "")).strip()
            if not u:
                continue
            base = _filename_from_url(u)
            fused = _fuse(base.rsplit(".", 1)[0])
            if fused and (sq in fused):
                return u
    except Exception:
        pass
    return ""

def build_image_index(df: DataFrame) -> Dict[str, str]:
    if "image" not in df.columns:
        return {}
    index: Dict[str, str] = {}
    for _, row in df.iterrows():
        url = str(row.get("image", "")).strip()
        if not url:
            continue
        tokens = _tokens_from_filename(url)
        for t in tokens:
            rt, st = _norm_code(t)
            if rt:
                index.setdefault(rt, url)
            if st and st != rt:
                index.setdefault(st, url)

        code_val = str(row.get("код", "")).strip().lower()
        if code_val:
            raw, sq = _norm_code(code_val)
            fused = tokens[0] if tokens else ""
            if fused:
                if raw and raw in fused:
                    index.setdefault(raw, url)
                if sq and sq in fused:
                    index.setdefault(sq, url)

        oem_val = str(row.get("oem", "")).strip().lower()
        if oem_val:
            raw, sq = _norm_code(oem_val)
            fused = tokens[0] if tokens else ""
            if fused:
                if raw and raw in fused:
                    index.setdefault(raw, url)
                if sq and sq in fused:
                    index.setdefault(sq, url)
    return index

async def resolve_ibb_direct_async(url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return url
            html = resp.text
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
    if not code or _image_index is None:
        return ""
    raw, sq = _norm_code(code)
    hit = _image_index.get(raw) or _image_index.get(sq)
    if hit:
        return hit
    for k, url in _image_index.items():
        if (sq and sq in k) or (raw and raw in k):
            return url
    return _scan_images_by_code_fallback(code)

# --------------------- Загрузка и кеш ---------------------------
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
        logger.info(f"✅ Перезагружено {len(df)} строк и индексы")
    finally:
        _loading_data = False

def ensure_fresh_data(force: bool = False):
    if not force and df is not None and (time.time() - _last_load_ts <= DATA_TTL):
        return
    import asyncio
    asyncio.create_task(ensure_fresh_data_async(force=True))

# -------------------------- Утилиты -----------------------------
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

def _relevance_score(row: dict, tokens: List[str], q_squash: str) -> int:
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

# ---------------------- Пользователи ----------------------------
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

    headers_raw = ws.row_values(1) or []
    if not headers_raw:
        logger.info("В листе 'Пользователи' пустая шапка — доступ разрешён всем.")
        return set(), set(), set()

    norm_headers: List[str] = []
    seen: Set[str] = set()
    for i, h in enumerate(headers_raw, start=1):
        name = (h or "").strip()
        if not name:
            name = f"col_{i}"
        lname = re.sub(r"\s+", " ", name.lower()).strip()
        base = lname
        k = 1
        while lname in seen:
            k += 1
            lname = f"{base}_{k}"
        seen.add(lname)
        norm_headers.append(lname)

    try:
        rows = ws.get_all_records(expected_headers=norm_headers)
    except Exception as e:
        logger.warning(f"get_all_records с expected_headers не сработал ({e}), fallback на ручной парсинг.")
        values = ws.get_all_values()
        data_rows = values[1:] if len(values) > 1 else []
        rows = []
        for r in data_rows:
            padded = (r + ["" for _ in range(len(norm_headers))])[:len(norm_headers)]
            rows.append({norm_headers[i]: padded[i] for i in range(len(norm_headers))})

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
            or _to_int_or_none(r.get("user"))
        )
        if not uid:
            continue

        role = str(r.get("role") or r.get("роль") or "").strip().lower()

        def _truthy_local(x) -> bool:
            s = str(x).strip().lower()
            return (
                s in {"1", "true", "yes", "y", "да", "истина", "ok", "ок",
                      "allowed", "разрешен", "разрешено", "доступ",
                      "admin", "админ", "ban", "blocked", "запрет"}
                or (s.isdigit() and int(s) > 0)
            )

        is_admin = role in {"admin", "админ", "administrator", "администратор"} or _truthy_local(r.get("admin"))
        is_blocked = _truthy_local(r.get("blocked") or r.get("ban") or r.get("запрет"))
        is_allowed = _truthy_local(r.get("allowed") or r.get("доступ") or (not role or role == "user"))

        if is_blocked:
            blocked.add(uid)
        if is_admin:
            admins.add(uid)
            is_allowed = True
        if is_allowed and not is_blocked:
            allowed.add(uid)

    logger.info(f"Пользователи прочитаны: allowed={len(allowed)}, admins={len(admins)}, blocked={len(blocked)}")
    return allowed, admins, blocked