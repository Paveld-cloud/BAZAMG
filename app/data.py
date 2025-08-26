# app/data.py
import os
import io
import re
import time
import json
import math
import logging
from typing import Dict, Set, Tuple, List, Optional

import pandas as pd
import aiohttp
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger("bot.data")

# -------- Конфиг --------
try:
    from app.config import (
        SPREADSHEET_URL,
        SAP_SHEET_NAME,          # "SAP"
        IMAGES_SHEET_NAME,       # "Изображения" (опц.)
        USERS_SHEET_NAME,        # "Пользователи" (опц.)
        DATA_TTL,                # сек, например 600
        SEARCH_COLUMNS,          # ["тип","наименование","код","oem","изготовитель"]
    )
except Exception:
    SPREADSHEET_URL = os.getenv("SPREADSHEET_URL", "")
    SAP_SHEET_NAME = "SAP"
    IMAGES_SHEET_NAME = "Изображения"
    USERS_SHEET_NAME = "Пользователи"
    DATA_TTL = int(os.getenv("DATA_TTL", "600"))
    SEARCH_COLUMNS = ["тип", "наименование", "код", "oem", "изготовитель"]

GOOGLE_APPLICATION_CREDENTIALS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "")
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# -------- Глобальное состояние --------
df: Optional[pd.DataFrame] = None
_last_load_ts: float = 0.0

_search_index: Dict[str, Set[int]] = {}
_image_index: Dict[str, str] = {}

user_state: Dict[int, dict] = {}
issue_state: Dict[int, dict] = {}

SHEET_ALLOWED: Set[int] = set()
SHEET_ADMINS: Set[int] = set()
SHEET_BLOCKED: Set[int] = set()

# Шаги диалога (используются в handlers)
ASK_QUANTITY, ASK_COMMENT, ASK_CONFIRM = range(3)

# -------- Утилиты --------
def _norm_code(x: str) -> str:
    """нормализация кода: lower + без пробелов/дефисов"""
    return re.sub(r"[\s\-]+", "", str(x or "").strip().lower())

def _norm_str(x: str) -> str:
    return str(x or "").strip().lower()

def now_local_str(tz_name: str = "Asia/Tashkent") -> str:
    tz = ZoneInfo(tz_name)
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

def val(d: dict, key: str, default: str = "") -> str:
    return str(d.get(key, default) or default)

def _extract_code_from_url(url: str) -> str:
    """берём имя файла без расширения как ключ"""
    try:
        path = re.sub(r"[?#].*$", "", url)
        tail = path.rsplit("/", 1)[-1]
        name = tail.rsplit(".", 1)[0]
        return _norm_code(name)
    except Exception:
        return ""

def _safe_col(df_: pd.DataFrame, col: str) -> Optional[pd.Series]:
    if col not in df_.columns:
        return None
    return df_[col].astype(str).fillna("").str.strip().str.lower()

# -------- Карточка --------
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

# -------- Google Sheets --------
def get_gs_client():
    if not GOOGLE_APPLICATION_CREDENTIALS_JSON:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS_JSON не задан")
    try:
        info = json.loads(GOOGLE_APPLICATION_CREDENTIALS_JSON)
        creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    except json.JSONDecodeError:
        creds = Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS_JSON, scopes=SCOPES)
    return gspread.authorize(creds)

def _load_sap_dataframe() -> pd.DataFrame:
    client = get_gs_client()
    sh = client.open_by_url(SPREADSHEET_URL)
    ws = sh.worksheet(SAP_SHEET_NAME)
    records = ws.get_all_records()
    new_df = pd.DataFrame(records)

    new_df.columns = [c.strip().lower() for c in new_df.columns]

    for col in ("код", "oem"):
        if col in new_df.columns:
            new_df[col] = new_df[col].astype(str).str.strip().str.lower()
    if "image" in new_df.columns:
        new_df["image"] = new_df["image"].astype(str).str.strip()

    return new_df

def _normalize_header_name(h: str, idx: int) -> str:
    """нормализация имени столбца + уникальность"""
    name = (h or "").strip().lower()
    name = re.sub(r"[^\w]+", "_", name).strip("_")
    if not name:
        name = f"col{idx+1}"
    return name

def _dedupe_headers(headers: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for i, h in enumerate(headers):
        base = _normalize_header_name(h, i)
        if base not in seen:
            seen[base] = 1
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
    return out

def _load_images_sheet() -> Dict[str, str]:
    """Опционально: лист 'Изображения' с колонками 'код','image'"""
    mapping: Dict[str, str] = {}
    try:
        client = get_gs_client()
        sh = client.open_by_url(SPREADSHEET_URL)
        ws = sh.worksheet(IMAGES_SHEET_NAME)
    except Exception:
        return mapping

    # Читаем значения и нормализуем заголовки
    all_vals = ws.get_all_values()
    if not all_vals:
        return mapping
    headers_raw = all_vals[0]
    headers = _dedupe_headers(headers_raw)
    rows = all_vals[1:]

    for r in rows:
        row = {headers[i]: (r[i] if i < len(r) else "") for i in range(len(headers))}
        code = _norm_code(row.get("код", ""))
        url = str(row.get("image", "")).strip()
        if code and url:
            mapping[code] = url
    logger.info(f"image-index: из листа '{IMAGES_SHEET_NAME}' прочитано {len(mapping)} ссылок")
    return mapping

# -------- Индексы --------
def build_search_index(df_: pd.DataFrame) -> Dict[str, Set[int]]:
    idx: Dict[str, Set[int]] = {}
    cols = [c for c in SEARCH_COLUMNS if c in df_.columns]
    for i, row in df_.iterrows():
        for c in cols:
            tokenized = re.findall(r"[a-zA-Zа-яА-Я0-9]+", str(row.get(c, "")), flags=re.IGNORECASE)
            for t in tokenized:
                key = _norm_str(t)
                if not key:
                    continue
                idx.setdefault(key, set()).add(i)
    return idx

def build_image_index(df_: pd.DataFrame) -> Dict[str, str]:
    """
    Индекс картинок — два прохода + слияние с отдельным листом:
    A) ключ = имя файла из URL (последний сегмент без расширения)
    B) ключ = значение 'код' (если есть image и A не покрыл)
    C) слияние с листом 'Изображения'
    """
    index: Dict[str, str] = {}

    # A: по имени файла
    added_a = 0
    if "image" in df_.columns:
        for _, row in df_.iterrows():
            raw_url = str(row.get("image", "")).strip()
            if not raw_url:
                continue
            key = _extract_code_from_url(raw_url)
            if not key:
                continue
            index[key] = raw_url
            added_a += 1

    # B: по столбцу 'код'
    added_b = 0
    if "image" in df_.columns and "код" in df_.columns:
        for _, row in df_.iterrows():
            raw_url = str(row.get("image", "")).strip()
            if not raw_url:
                continue
            code_val = _norm_code(row.get("код", ""))
            if not code_val or code_val in index:
                continue
            index[code_val] = raw_url
            added_b += 1

    # C: внешний лист
    ext = _load_images_sheet()
    before = len(index)
    index.update(ext)
    merged = len(index) - before

    logger.info(
        f"image-index: A(filename)={added_a}, B(code-col)={added_b}, "
        f"C(extra-sheet)={merged}, unique_keys={len(index)}"
    )
    return index

def ensure_fresh_data(force: bool = False):
    global df, _search_index, _image_index, _last_load_ts
    need = force or df is None or (time.time() - _last_load_ts > DATA_TTL)
    if not need:
        return
    new_df = _load_sap_dataframe()
    df = new_df
    _search_index = build_search_index(df)
    _image_index = build_image_index(df)
    _last_load_ts = time.time()
    logger.info(f"✅ Перезагружено {len(df)} строк и построены индексы")

async def ensure_fresh_data_async(force: bool = False):
    await asyncio_to_thread(ensure_fresh_data, force)

# -------- Картинки --------
async def find_image_by_code_async(code: str) -> str:
    ensure_fresh_data()
    if not code or _image_index is None:
        return ""
    key = _norm_code(code)
    return _image_index.get(key, "")

def normalize_drive_url(url: str) -> str:
    m = re.search(r"drive\.google\.com/(?:file/d/([-\w]{20,})|open\?id=([-\w]{20,}))", url)
    if m:
        file_id = m.group(1) or m.group(2)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url

async def resolve_ibb_direct_async(url: str) -> str:
    try:
        if re.search(r"^https?://i\.ibb\.co/", url, re.I):
            return url
        if not re.search(r"^https?://ibb\.co/", url, re.I):
            return url
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                if resp.status != 200:
                    return url
                html = await resp.text()
        m = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
        return m.group(1) if m else url
    except Exception as e:
        logger.warning(f"resolve_ibb_direct_async error: {e}")
        return url

async def resolve_image_url_async(url_raw: str) -> str:
    if not url_raw:
        return ""
    url = normalize_drive_url(url_raw)
    url = await resolve_ibb_direct_async(url)
    return url

# -------- Поиск и скоринг --------
def match_row_by_index(tokens: List[str]) -> Set[int]:
    ensure_fresh_data()
    if not tokens:
        return set()
    tokens_norm = [_norm_str(t) for t in tokens if t]
    if not tokens_norm:
        return set()

    sets: List[Set[int]] = []
    for t in tokens_norm:
        s = _search_index.get(t, set())
        if not s:
            return set()
        sets.append(s)

    acc = sets[0].copy()
    for s in sets[1:]:
        acc &= s
        if not acc:
            break
    return acc

def squash(text: str) -> str:
    return re.sub(r"[\W_]+", "", str(text or "").lower())

def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", str(text or "").lower()).strip()

def _relevance_score(row: dict, tokens: List[str], q_squash: str) -> float:
    """
    Простой скоринг: веса по полям, бусты за точные совпадения кода/начало кода/склейку.
    """
    tkns = [_norm_str(t) for t in tokens if t]
    if not tkns:
        return 0.0

    # поля
    code = _norm_str(row.get("код", ""))
    name = _norm_str(row.get("наименование", ""))
    ttype = _norm_str(row.get("тип", ""))
    oem  = _norm_str(row.get("oem", ""))
    manuf= _norm_str(row.get("изготовитель", ""))

    weights = {
        "код": 5.0, "наименование": 3.0, "тип": 2.0, "oem": 2.0, "изготовитель": 2.0
    }
    fields = {
        "код": code, "наименование": name, "тип": ttype, "oem": oem, "изготовитель": manuf
    }

    score = 0.0
    for f, text in fields.items():
        for t in tkns:
            if not t:
                continue
            if t in text:
                score += weights[f]

    # бусты
    if q_squash:
        joined = squash(code + name + ttype + oem + manuf)
        if q_squash in joined:
            score += 10.0

    # сильный буст за код
    q_full = " ".join(tkns)
    q_full_no_ws = squash(q_full)
    if code:
        if code == q_full:       # точное совпадение
            score += 100.0
        if code.startswith(q_full) or code.startswith(q_full_no_ws):
            score += 20.0
        # также по каждому токену
        for t in tkns:
            if code.startswith(t):
                score += 5.0

    return score

# -------- Экспорт --------
def _df_to_xlsx(df_: pd.DataFrame, filename: str = "export.xlsx") -> io.BytesIO:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df_.to_excel(writer, index=False)
    buf.seek(0)
    return buf

# -------- Пользователи --------
def _parse_int(x) -> Optional[int]:
    try:
        v = int(str(x).strip())
        return v if v > 0 else None
    except Exception:
        return None

def load_users_from_sheet() -> Tuple[Set[int], Set[int], Set[int]]:
    """
    Без падения на неуникальных заголовках.
    Поддержка схем:
      - user_id + role (admin|user|blocked)
      - булевые: allowed/admin/blocked
      - просто список user_id => allowed
    """
    allowed: Set[int] = set()
    admins: Set[int] = set()
    blocked: Set[int] = set()
    try:
        client = get_gs_client()
        sh = client.open_by_url(SPREADSHEET_URL)
        ws = sh.worksheet(USERS_SHEET_NAME)
    except Exception:
        logger.info("Лист пользователей отсутствует — пускаем всех по умолчанию")
        return allowed, admins, blocked

    all_vals = ws.get_all_values()
    if not all_vals:
        logger.info("Лист пользователей пуст")
        return allowed, admins, blocked

    headers_raw = all_vals[0]
    headers = _dedupe_headers(headers_raw)
    rows = all_vals[1:]

    # Превращаем в список словарей
    recs: List[dict] = []
    for r in rows:
        recs.append({headers[i]: (r[i] if i < len(r) else "") for i in range(len(headers))})

    dfu = pd.DataFrame(recs)

    cols = [c.strip().lower() for c in dfu.columns]
    dfu.columns = cols

    has_uid = "user_id" in dfu.columns or "uid" in dfu.columns or "id" in dfu.columns
    has_role = "role" in dfu.columns
    has_allowed = "allowed" in dfu.columns
    has_admin = "admin" in dfu.columns
    has_blocked = "blocked" in dfu.columns

    for _, r in dfu.iterrows():
        uid = _parse_int(r.get("user_id") or r.get("uid") or r.get("id"))
        if not uid:
            continue

        if has_role:
            role = str(r.get("role", "")).strip().lower()
            if role in ("admin", "админ"):
                admins.add(uid); allowed.add(uid)
            elif role in ("blocked", "ban", "заблокирован"):
                blocked.add(uid)
            else:
                allowed.add(uid)
            continue

        # булевые флаги
        def truthy(v) -> bool:
            s = str(v).strip().lower()
            return s in ("1", "true", "да", "y", "yes")

        if has_blocked and truthy(r.get("blocked")):
            blocked.add(uid); continue
        if has_admin and truthy(r.get("admin")):
            admins.add(uid); allowed.add(uid); continue
        if has_allowed and truthy(r.get("allowed")):
            allowed.add(uid); continue

        # просто список
        allowed.add(uid)

    logger.info(f"Пользователи прочитаны: allowed={len(allowed)}, admins={len(admins)}, blocked={len(blocked)}")
    return allowed, admins, blocked

# -------- Async helper --------
import asyncio
async def asyncio_to_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

# -------- Backward-compat для main.py --------
def initial_load():
    try:
        ensure_fresh_data(force=True)
    except Exception as e:
        logger.exception(f"initial_load: ensure_fresh_data error: {e}")
        raise
    try:
        allowed, admins, blocked = load_users_from_sheet()
        SHEET_ALLOWED.clear(); SHEET_ALLOWED.update(allowed)
        SHEET_ADMINS.clear(); SHEET_ADMINS.update(admins)
        SHEET_BLOCKED.clear(); SHEET_BLOCKED.update(blocked)
        logger.info(f"Пользователи прочитаны: allowed={len(allowed)}, admins={len(admins)}, blocked={len(blocked)}")
    except Exception as e:
        logger.warning(f"initial_load: не удалось загрузить пользователей: {e}")

async def initial_load_async():
    try:
        await ensure_fresh_data_async(force=True)
    except Exception as e:
        logger.exception(f"initial_load_async: ensure_fresh_data_async error: {e}")
        raise
    try:
        allowed, admins, blocked = await asyncio_to_thread(load_users_from_sheet)
        SHEET_ALLOWED.clear(); SHEET_ALLOWED.update(allowed)
        SHEET_ADMINS.clear(); SHEET_ADMINS.update(admins)
        SHEET_BLOCKED.clear(); SHEET_BLOCKED.update(blocked)
        logger.info(f"(async) Пользователи прочитаны: allowed={len(allowed)}, admins={len(admins)}, blocked={len(blocked)}")
    except Exception as e:
        logger.warning(f"initial_load_async: не удалось загрузить пользователей: {e}")

