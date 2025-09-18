import os
import re
import io
import json
import time
import math
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, Set, List, DefaultDict
from collections import defaultdict
from zoneinfo import ZoneInfo
from telegram.ext import ConversationHandler

ASK_QUANTITY, ASK_COMMENT, CONFIRM = range(3)

import aiohttp
import gspread
import pandas as pd
from pandas import DataFrame
from google.oauth2.service_account import Credentials

# ========================= НАСТРОЙКИ / ENV =========================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
SPREADSHEET_URL = os.getenv("SPREADSHEET_URL", "")
SHEET_NAME = (os.getenv("SHEET_NAME", "").strip() or "").strip()
CREDS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "")

TZ_NAME = os.getenv("TIMEZONE", "Asia/Tashkent")
DATA_TTL = int(os.getenv("DATA_TTL", "300"))       # кеш данных (сек)
USERS_TTL = int(os.getenv("USERS_TTL", "300"))     # кеш пользователей (сек)

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# ---- Поля для поиска (ENV + дефолт) ----
DEFAULT_SEARCH_COLUMNS = [
    "тип", "наименование", "код", "oem", "изготовитель",
    "парт номер", "oem парт номер",
]

_env = os.getenv("SEARCH_COLUMNS", "").strip()
try:
    from_env = json.loads(_env) if _env else []
    if not isinstance(from_env, list):
        from_env = []
except Exception:
    from_env = []

# Итог: порядок из ENV сохраняем, но гарантированно добавляем дефолтные поля; дубликаты убираем; всё в lower()
SEARCH_COLUMNS: List[str] = list(
    dict.fromkeys(
        [
            (c.strip().lower() if isinstance(c, str) else "")
            for c in (from_env + DEFAULT_SEARCH_COLUMNS)
            if c
        ]
    )
)

# ========================= ГЛОБАЛЬНЫЕ СОСТОЯНИЯ =========================

df: Optional[DataFrame] = None
_last_load_ts = 0.0
_search_index: Optional[Dict[str, Set[int]]] = None
_image_index: Optional[Dict[str, str]] = None

SHEET_ALLOWED: Set[int] = set()
SHEET_ADMINS: Set[int] = set()
SHEET_BLOCKED: Set[int] = set()
_last_users_ts = 0.0

# ========================= УТИЛИТЫ =========================

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
        f"🔢 Парт Номер: {val(row, 'парт номер')}\n"
        f"⚙️ OEM Парт Номер: {val(row, 'oem парт номер')}\n"
        f"📦 Кол-во: {val(row, 'количество')}\n"
        f"💰 Цена: {val(row, 'цена')} {val(row, 'валюта')}\n"
        f"🏭 Изготовитель: {val(row, 'изготовитель')}\n"
        f"⚙️ OEM: {val(row, 'oem')}"
    )

def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", (text or "")).lower().strip()

def squash(text: str) -> str:
    return re.sub(r"[\W_]+", "", (text or "").lower(), flags=re.UNICODE)

# ========================= GOOGLE SHEETS =========================

def _get_gs_client():
    if not CREDS_JSON.strip():
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS_JSON is empty or not set.")
    creds_info = json.loads(CREDS_JSON)
    creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    return gspread.authorize(creds)

def _open_data_worksheet(client):
    sh = client.open_by_url(SPREADSHEET_URL)
    if SHEET_NAME:
        try:
            return sh.worksheet(SHEET_NAME)
        except gspread.WorksheetNotFound:
            pass
    return sh.sheet1

def load_data_blocking() -> List[dict]:
    client = _get_gs_client()
    ws = _open_data_worksheet(client)
    return ws.get_all_records()

def _truthy(x) -> bool:
    s = str(x).strip().lower()
    return s in {"1","true","yes","y","да","истина","ok","ок","allowed","разрешен","разрешено"} or (s.isdigit() and int(s) > 0)

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
    client = _get_gs_client()
    sh = client.open_by_url(SPREADSHEET_URL)
    try:
        ws = sh.worksheet("Пользователи")
    except gspread.WorksheetNotFound:
        try:
            ws = sh.worksheet("Users")
        except gspread.WorksheetNotFound:
            return set(), set(), set()

    headers_raw = ws.row_values(1) or []
    if not headers_raw:
        return set(), set(), set()

    norm_headers: List[str] = []
    seen: Set[str] = set()
    for i, h in enumerate(headers_raw, start=1):
        name = (h or "").strip() or f"col_{i}"
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
    except Exception:
        values = ws.get_all_values()
        data_rows = values[1:] if len(values) > 1 else []
        rows = []
        for r in data_rows:
            padded = (r + [""] * (len(norm_headers) - len(r)))[:len(norm_headers)]
            rows.append({norm_headers[i]: padded[i] for i in range(len(norm_headers))})

    if not rows:
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
        is_admin = role in {"admin","админ","administrator","администратор"} or _truthy(r.get("admin"))
        is_blocked = _truthy(r.get("blocked") or r.get("ban") or r.get("запрет"))
        is_allowed = _truthy(r.get("allowed") or r.get("доступ") or (not role or role == "user"))

        if is_blocked:
            blocked.add(uid)
        if is_admin:
            admins.add(uid)
            is_allowed = True
        if is_allowed and not is_blocked:
            allowed.add(uid)

    return allowed, admins, blocked

def save_issue_to_sheet_blocking(user, part: Dict[str, Any], quantity, comment: str):
    client = _get_gs_client()
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
    display_name = full_name or (f"@{user.username}" if getattr(user, 'username', None) else str(user.id))
    ts = now_local_str()

    values_by_key = {
        "дата": ts, "timestamp": ts,
        "id": user.id, "user_id": user.id,
        "имя": display_name, "name": display_name,
        "тип": str(part.get("тип", "")), "type": str(part.get("тип", "")),
        "наименование": str(part.get("наименование", "")), "name_item": str(part.get("наименование", "")),
        "код": str(part.get("код", "")), "code": str(part.get("код", "")),
        "количество": str(quantity), "qty": str(quantity),
        "коментарий": comment or "", "комментарий": comment or "", "comment": comment or "",
    }

    # Подгоняем под текущие заголовки листа
    row = [values_by_key.get(hn, "") for hn in norm]
    ws.append_row(row, value_input_option="USER_ENTERED")

# ========================= ИНДЕКСАЦИЯ / ПОИСК =========================

def build_search_index(df: DataFrame) -> Dict[str, Set[int]]:
    """
    Индексируем по колонкам из SEARCH_COLUMNS.
    Токены — \w+, всё в lower().
    """
    index: DefaultDict[str, Set[int]] = defaultdict(set)
    for col in SEARCH_COLUMNS:
        if col not in df.columns:
            continue
        # Series.items() для pandas 2.x
        for idx, val in df[col].astype(str).str.lower().items():
            for t in re.findall(r"\w+", val):
                if t:
                    index[t].add(idx)
    return dict(index)

def _relevance_score(row: dict, tokens: List[str], q_squash: str) -> int:
    score = 0
    for f in SEARCH_COLUMNS:
        valx = str(row.get(f, "")).lower()
        if not valx:
            continue
        words = set(re.findall(r"\w+", valx))
        tok_hit = sum(1 for t in tokens if t in words)
        sub_hit = sum(1 for t in tokens if t and t in valx)
        sq = re.sub(r"[\W_]+", "", valx)
        squash_hit = 1 if q_squash and q_squash in sq else 0
        weight = 2 if f in ("код", "oem", "парт номер", "oem парт номер") else 1
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

# ========================= КАРТИНКИ / ИНДЕКС ИМЁН ФАЙЛОВ =========================

def _filename_from_url(u: str) -> str:
    from urllib.parse import urlparse, parse_qs, unquote
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

def _tokens_from_filename(u: str) -> List[str]:
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
            index.setdefault(t, url)

        # усилим по кодам, если код встречается в имени файла
        for col in ("код", "oem", "парт номер", "oem парт номер"):
            code_val = str(row.get(col, "")).strip().lower()
            if code_val and tokens:
                fused = tokens[0]
                raw = code_val
                sq = re.sub(r"[\W_]+", "", code_val)
                if raw and raw in fused:
                    index.setdefault(raw, url)
                if sq and sq in fused:
                    index.setdefault(sq, url)
    return index

async def find_image_by_code_async(code: str) -> str:
    if not code or not _image_index:
        return ""
    raw = (code or "").strip().lower()
    sq = re.sub(r"[\W_]+", "", raw)
    # прямой
    hit = _image_index.get(raw) or _image_index.get(sq)
    if hit:
        return hit
    # подстроки
    for k, url in _image_index.items():
        if (sq and sq in k) or (raw and raw in k):
            return url
    # ничего не нашли
    return ""

# ---------- ЗАГРУЗКА / КЕШИ ----------

def initial_load():
    global df, _last_load_ts, _search_index, _image_index
    data = load_data_blocking()
    new_df = DataFrame(data)
    new_df.columns = new_df.columns.str.strip().str.lower()

    # нормализуем ключевые колонки
    for col in ("код", "oem", "парт номер", "oem парт номер"):
        if col in new_df.columns:
            new_df[col] = new_df[col].astype(str).str.strip().str.lower()
    if "image" in new_df.columns:
        new_df["image"] = new_df["image"].astype(str).str.strip()

    df = new_df
    _search_index = build_search_index(df)
    _image_index = build_image_index(df)
    _last_load_ts = time.time()

    allowed, admins, blocked = load_users_from_sheet()
    global SHEET_ALLOWED, SHEET_ADMINS, SHEET_BLOCKED, _last_users_ts
    SHEET_ALLOWED, SHEET_ADMINS, SHEET_BLOCKED = allowed, admins, blocked
    _last_users_ts = time.time()

async def ensure_fresh_data_async(force: bool = False):
    global df, _last_load_ts, _search_index, _image_index
    if not force and df is not None and (time.time() - _last_load_ts <= DATA_TTL):
        return
    data = await asyncio.to_thread(load_data_blocking)
    new_df = DataFrame(data)
    new_df.columns = new_df.columns.str.strip().str.lower()
    for col in ("код", "oem", "парт номер", "oem парт номер"):
        if col in new_df.columns:
            new_df[col] = new_df[col].astype(str).str.strip().str.lower()
    if "image" in new_df.columns:
        new_df["image"] = new_df["image"].astype(str).str.strip()

    df = new_df
    _search_index = build_search_index(df)
    _image_index = build_image_index(df)
    _last_load_ts = time.time()

def ensure_fresh_data(force: bool = False):
    # Планово запускаем обновление в фоне
    asyncio.create_task(ensure_fresh_data_async(force=force))

async def ensure_users_async(force: bool = False):
    global SHEET_ALLOWED, SHEET_ADMINS, SHEET_BLOCKED, _last_users_ts
    if not force and (time.time() - _last_users_ts <= USERS_TTL):
        return
    allowed, admins, blocked = await asyncio.to_thread(load_users_from_sheet)
    SHEET_ALLOWED, SHEET_ADMINS, SHEET_BLOCKED = allowed, admins, blocked
    _last_users_ts = time.time()

def ensure_users(force: bool = False):
    asyncio.create_task(ensure_users_async(force=force))

# ---------- ДОСТУПЫ ----------

def is_admin(uid: int) -> bool:
    ensure_users()
    return uid in SHEET_ADMINS

def is_allowed(uid: int) -> bool:
    ensure_users()
    if uid in SHEET_BLOCKED:
        return False
    if SHEET_ALLOWED:
        # ВАЖНО: правильное имя множества админов (без кириллической 'М')
        return (uid in SHEET_ALLOWED) or (uid in SHEET_ADMINS)
    return True

def get_admin_ids() -> Set[int]:
    return set(SHEET_ADMINS)

# ---------- АДАПТЕРЫ ДЛЯ ХЕНДЛЕРОВ ----------

def get_df() -> Optional[DataFrame]:
    return df

def get_image_index_copy() -> Dict[str, str]:
    return dict(_image_index or {})

