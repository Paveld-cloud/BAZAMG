import os
import io
import re
import time
import json
import logging
from typing import Dict, Set, Tuple, List, Optional

import pandas as pd
import aiohttp
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger("bot.data")

# ---------- Конфиг ----------
try:
    from app.config import (
        SPREADSHEET_URL,
        SAP_SHEET_NAME,          # "SAP"
        USERS_SHEET_NAME,        # "Пользователи" (опц.)
        DATA_TTL,                # сек, например 600
        SEARCH_COLUMNS,          # ["тип","наименование","код","oem","изготовитель"]
    )
except Exception:
    SPREADSHEET_URL = os.getenv("SPREADSHEET_URL", "")
    SAP_SHEET_NAME = os.getenv("SAP_SHEET_NAME", "SAP")
    USERS_SHEET_NAME = os.getenv("USERS_SHEET_NAME", "Пользователи")
    DATA_TTL = int(os.getenv("DATA_TTL", "600"))
    SEARCH_COLUMNS = ["тип","наименование","код","oem","изготовитель","парт номер","oem парт номер"]

GOOGLE_APPLICATION_CREDENTIALS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "")
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# ---------- Глобальное состояние ----------
df: Optional[pd.DataFrame] = None
_last_load_ts: float = 0.0

_search_index: Dict[str, Set[int]] = {}
_image_index: Dict[str, str] = {}

# Состояние пользователей (используется в handlers)
user_state: Dict[int, dict] = {}
issue_state: Dict[int, dict] = {}

# Списки пользователей
SHEET_ALLOWED: Set[int] = set()
SHEET_ADMINS: Set[int] = set()
SHEET_BLOCKED: Set[int] = set()

# Шаги диалога
ASK_QUANTITY, ASK_COMMENT, ASK_CONFIRM = range(3)

# ---------- Утилиты ----------
def _norm_code(x: str) -> str:
    return re.sub(r"[\s\-]+", "", str(x or "").strip().lower())

def _norm_str(x: str) -> str:
    return str(x or "").strip().lower()

def now_local_str(tz_name: str = "Asia/Tashkent") -> str:
    tz = ZoneInfo(tz_name)
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

def val(d: dict, key: str, default: str = "") -> str:
    return str(d.get(key, default) or default)

def _url_name_tokens(url: str) -> List[str]:
    try:
        path = re.sub(r"[?#].*$", "", str(url or ""))
        name = path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
        return re.findall(r"[a-z0-9]+", name)
    except Exception:
        return []

def squash(text: str) -> str:
    return re.sub(r"[\W_]+", "", str(text or "").lower())

def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", str(text or "").lower()).strip()

# ---------- Формат карточки ----------
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

# ---------- Google Sheets ----------
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
    for col in ("код", "oem", "парт номер", "oem парт номер"):
        if col in new_df.columns:
            new_df[col] = new_df[col].astype(str).str.strip().str.lower()
    if "image" in new_df.columns:
        new_df["image"] = new_df["image"].astype(str).str.strip()
    return new_df

# ---------- Индексы ----------
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
    index: Dict[str, str] = {}
    if "image" not in df_.columns:
        return index
    skip = {"png", "jpg", "jpeg", "gif", "webp", "svg"}
    for _, row in df_.iterrows():
        url = str(row.get("image", "")).strip()
        if not url:
            continue
        tokens = _url_name_tokens(url)
        if not tokens:
            continue
        for t in tokens:
            if t in skip or len(t) < 4:
                continue
            index.setdefault(t, url)
        name_join = "".join(tokens)
        index.setdefault(name_join, url)
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

# ---------- Поиск ----------
def _relevance_score(row: dict, tokens: List[str], q_squash: str) -> float:
    tkns = [_norm_str(t) for t in tokens if t]
    if not tkns:
        return 0.0
    code = _norm_str(row.get("код", ""))
    name = _norm_str(row.get("наименование", ""))
    weights = {"код": 5.0, "наименование": 3.0}
    fields = {"код": code, "наименование": name}
    score = 0.0
    for f, text in fields.items():
        for t in tkns:
            if t and (t in text):
                score += weights[f]
    if q_squash:
        joined = squash(code + name)
        if q_squash in joined:
            score += 10.0
    if code and any(code.startswith(t) for t in tkns):
        score += 20.0
    return score

def match_row_by_index(tokens: List[str]) -> List[int]:
    """
    Возвращает индексы строк, отсортированные по релевантности.
    Сначала пересечение, потом fallback — объединение.
    """
    ensure_fresh_data()
    if not tokens:
        return []
    tokens_norm = [_norm_str(t) for t in tokens if t]
    if not tokens_norm:
        return []
    # строгий поиск
    sets: List[Set[int]] = []
    for t in tokens_norm:
        s = _search_index.get(t, set())
        if not s:
            sets = []
            break
        sets.append(s)
    if sets:
        acc = sets[0].copy()
        for s in sets[1:]:
            acc &= s
            if not acc:
                break
        found = acc
    else:
        # мягкий поиск
        found = set()
        for t in tokens_norm:
            found |= _search_index.get(t, set())
    if not found:
        return []
    q_squash = squash(" ".join(tokens_norm))
    scored = [(idx, _relevance_score(df.iloc[idx].to_dict(), tokens_norm, q_squash)) for idx in found]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scored]

# ---------- Экспорт ----------
def _df_to_xlsx(df_: pd.DataFrame, filename: str = "export.xlsx") -> io.BytesIO:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df_.to_excel(writer, index=False)
    buf.seek(0)
    return buf

# ---------- Пользователи ----------
def load_users_from_sheet() -> Tuple[Set[int], Set[int], Set[int]]:
    allowed: Set[int] = set()
    admins: Set[int] = set()
    blocked: Set[int] = set()
    try:
        client = get_gs_client()
        sh = client.open_by_url(SPREADSHEET_URL)
        ws = sh.worksheet(USERS_SHEET_NAME)
    except Exception:
        logger.info("Лист пользователей отсутствует")
        return allowed, admins, blocked
    all_vals = ws.get_all_values()
    if not all_vals:
        return allowed, admins, blocked
    headers = [h.strip().lower() for h in all_vals[0]]
    rows = all_vals[1:]
    for r in rows:
        row = {headers[i]: (r[i] if i < len(r) else "") for i in range(len(headers))}
        uid = row.get("user_id") or row.get("id") or row.get("uid")
        try:
            uid = int(uid)
        except Exception:
            continue
        role = str(row.get("role") or "").strip().lower()
        if role in ("admin", "админ"):
            admins.add(uid); allowed.add(uid)
        elif role in ("blocked", "ban"):
            blocked.add(uid)
        else:
            allowed.add(uid)
    return allowed, admins, blocked

# ---------- Инициализация ----------
def initial_load():
    ensure_fresh_data(force=True)
    allowed, admins, blocked = load_users_from_sheet()
    SHEET_ALLOWED.clear(); SHEET_ALLOWED.update(allowed)
    SHEET_ADMINS.clear(); SHEET_ADMINS.update(admins)
    SHEET_BLOCKED.clear(); SHEET_BLOCKED.update(blocked)
    logger.info(f"Users: allowed={len(allowed)}, admins={len(admins)}, blocked={len(blocked)}")

