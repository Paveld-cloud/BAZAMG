import os
import io
import re
import time
import json
import logging
from typing import Dict, Set, Tuple, List, Optional, Any

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
        USERS_SHEET_NAME,        # "Пользователи"
        HISTORY_SHEET_NAME,      # "История"
        DATA_TTL,
        SEARCH_COLUMNS,
        TZ_NAME,
    )
except Exception:
    SPREADSHEET_URL    = os.getenv("SPREADSHEET_URL", "")
    SAP_SHEET_NAME     = os.getenv("SAP_SHEET_NAME", "SAP")
    USERS_SHEET_NAME   = os.getenv("USERS_SHEET_NAME", "Пользователи")
    HISTORY_SHEET_NAME = os.getenv("HISTORY_SHEET_NAME", "История")
    DATA_TTL           = int(os.getenv("DATA_TTL", "600"))
    TZ_NAME            = os.getenv("TIMEZONE", "Asia/Tashkent")
    SEARCH_COLUMNS     = ["тип","наименование","код","oem","изготовитель","парт номер","oem парт номер"]

GOOGLE_APPLICATION_CREDENTIALS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "")
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# ---------- Глобальное состояние ----------
df: Optional[pd.DataFrame] = None
_last_load_ts: float = 0.0

# Индексы
_search_index: Dict[str, Set[int]] = {}
_code_index: Dict[str, List[int]] = {}
_oem_index:  Dict[str, List[int]] = {}
_image_index: Dict[str, str] = {}

# Состояния (если где-то используются)
user_state: Dict[int, dict] = {}
issue_state: Dict[int, dict] = {}
SHEET_ALLOWED: Set[int] = set()
SHEET_ADMINS:  Set[int] = set()
SHEET_BLOCKED: Set[int] = set()

ASK_QUANTITY, ASK_COMMENT, ASK_CONFIRM = range(3)

# ---------- Нормализация ----------
LOOKALIKES = str.maketrans({
    "А":"A","В":"B","Е":"E","К":"K","М":"M","Н":"H","О":"O","Р":"P","С":"C","Т":"T","У":"Y","Х":"X",
    "а":"a","е":"e","о":"o","р":"p","с":"c","у":"y","х":"x","к":"k","м":"m","н":"h","т":"t",
})

def _ascii_like(s: str) -> str:
    return (s or "").translate(LOOKALIKES)

def _smart_o_to_zero(s: str) -> str:
    # Меняем 'o'→'0' только если 'o' стоит между цифрами: 12o3 -> 1203
    return re.sub(r'(?<=\d)o(?=\d)', '0', s)

def _norm_code(x: str) -> str:
    """Нормализация кодов/артикулов: кир→лат, lower, smart 'o'→'0', убрать разделители и оставить [a-z0-9]."""
    s = _ascii_like(str(x or "").strip())
    s = s.lower()
    s = _smart_o_to_zero(s)
    s = re.sub(r"[\s\-_\.\/\\]+", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s

def _norm_str(x: str) -> str:
    s = _ascii_like(str(x or "").strip())
    return s.lower()

def now_local_str(tz_name: str = TZ_NAME) -> str:
    tz = ZoneInfo(tz_name)
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

def val(d: dict, key: str, default: str = "") -> str:
    return str(d.get(key, default) or default)

def squash(text: str) -> str:
    return re.sub(r"[\W_]+", "", _ascii_like(str(text or "")).lower())

def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", _ascii_like(str(text or "")).lower()).strip()

def _url_name_tokens(url: str) -> List[str]:
    try:
        path = re.sub(r"[?#].*$", "", str(url or ""))
        name = path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
        name = _ascii_like(name)
        return re.findall(r"[a-z0-9]+", name)
    except Exception:
        return []

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

def _open_ws(title: str):
    sh = get_gs_client().open_by_url(SPREADSHEET_URL)
    return sh.worksheet(title)

def _load_sap_dataframe() -> pd.DataFrame:
    ws = _open_ws(SAP_SHEET_NAME)
    records = ws.get_all_records()
    new_df = pd.DataFrame(records)
    new_df.columns = [c.strip().lower() for c in new_df.columns]
    for col in ("код", "oem", "парт номер", "oem парт номер"):
        if col in new_df.columns:
            new_df[col] = new_df[col].astype(str).map(_norm_str)
    if "image" in new_df.columns:
        new_df["image"] = new_df["image"].astype(str).str.strip()
    for col in ("тип", "наименование", "изготовитель"):
        if col in new_df.columns:
            new_df[col] = new_df[col].astype(str).map(_norm_str)
    return new_df

# ---------- Индексы ----------
def build_search_index(df_: pd.DataFrame) -> Dict[str, Set[int]]:
    idx: Dict[str, Set[int]] = {}
    cols = [c for c in SEARCH_COLUMNS if c in df_.columns]
    for i, row in df_.iterrows():
        for c in cols:
            raw = str(row.get(c, ""))
            if c in ("код", "парт номер", "oem парт номер"):
                core = _norm_code(raw)
                if core:
                    idx.setdefault(core, set()).add(i)
            for t in re.findall(r"[a-z0-9]+", _ascii_like(raw.lower())):
                if t:
                    idx.setdefault(t, set()).add(i)
    return idx

def _rebuild_exact_code_indexes(df_: pd.DataFrame) -> None:
    _code_index.clear()
    _oem_index.clear()
    if "код" in df_.columns:
        for i, v in df_["код"].items():
            key = _norm_code(v)
            if key:
                _code_index.setdefault(key, []).append(i)
    if "oem парт номер" in df_.columns:
        for i, v in df_["oem парт номер"].items():
            key = _norm_code(v)
            if key:
                _oem_index.setdefault(key, []).append(i)

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
        for t in tokens:
            if t in skip or len(t) < 3:
                continue
            index.setdefault(_norm_code(t), url)
        index.setdefault("".join(tokens), url)
    return index

def ensure_fresh_data(force: bool = False):
    global df, _search_index, _image_index, _last_load_ts
    need = force or df is None or (time.time() - _last_load_ts > DATA_TTL)
    if not need:
        return
    new_df = _load_sap_dataframe()
    df = new_df
    _search_index = build_search_index(df)
    _rebuild_exact_code_indexes(df)
    _image_index = build_image_index(df)
    _last_load_ts = time.time()
    logger.info(f"✅ Перезагружено {len(df)} строк, построены индексы")

def sap_count() -> int:
    ensure_fresh_data()
    return 0 if df is None else len(df)

# ---------- Картинки ----------
async def find_image_by_code_async(code: str) -> str:
    ensure_fresh_data()
    if not code:
        return ""
    key = _norm_code(code)
    hit = _image_index.get(key)
    if hit:
        return hit
    try:
        if df is not None and "image" in df.columns:
            for url in df["image"]:
                url = str(url or "").strip()
                if not url:
                    continue
                tokens = _url_name_tokens(url)
                name_join = "".join(tokens)
                if key in tokens or key in name_join:
                    return url
    except Exception as e:
        logger.warning(f"find_image_by_code_async fallback error: {e}")
    logger.info(f"[image] нет записи в индексе для кода: {key}")
    return ""

def normalize_drive_url(url: str) -> str:
    m = re.search(r"drive\.google\.com/(?:file/d/([-\w]{20,})|open\?id=([-\w]{20,}))", str(url or ""))
    if m:
        file_id = m.group(1) or m.group(2)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return str(url or "")

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

# ---------- Поиск ----------
def _tokenize_query(q: str) -> List[str]:
    q = _ascii_like(str(q or "").lower())
    tokens = re.findall(r"[a-z0-9]+", q)
    joined = _norm_code(q)  # склеенный вариант для кодов (LR 7000 -> lr7000)
    if joined and joined not in tokens:
        tokens.append(joined)
    return [t for t in tokens if t]

def match_row_by_index(tokens: List[str]) -> Set[int]:
    ensure_fresh_data()
    if not tokens:
        return set()
    tokens_norm = [_norm_code(t) for t in tokens if t]
    if not tokens_norm:
        return set()

    # Пересечение (все слова)
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
        if acc:
            return acc

    # Иначе объединение
    found = set()
    for t in tokens_norm:
        found |= _search_index.get(t, set())
    return found

def _relevance_score(row: dict, tokens: List[str], q_squash: str, q_code: Optional[str] = None) -> float:
    """Ранжирование: приоритет точного кода/ОЕМ, затем подстроки в ключевых полях."""
    tkns = [_norm_str(t) for t in tokens if t]
    if not tkns:
        return 0.0

    code = _norm_str(row.get("код", ""))
    name = _norm_str(row.get("наименование", ""))
    ttype= _norm_str(row.get("тип", ""))
    oem  = _norm_str(row.get("oem", ""))
    manuf= _norm_str(row.get("изготовитель", ""))

    weights = {"код": 5.0, "наименование": 3.0, "тип": 2.0, "oem": 2.0, "изготовитель": 2.0}
    fields  = {"код": code, "наименование": name, "тип": ttype, "oem": oem, "изготовитель": manuf}

    score = 0.0
    for f, text in fields.items():
        for t in tkns:
            if t and (t in text):
                score += weights[f]

    if q_squash:
        joined = squash(code + name + ttype + oem + manuf)
        if q_squash in joined:
            score += 10.0

    # приоритет точного кода/оема, если передали q_code
    if q_code:
        if _norm_code(code) == q_code or _norm_code(oem) == q_code:
            score += 100.0
        elif q_code and (q_code == squash(code) or q_code == squash(oem)):
            score += 60.0
        elif q_code and (q_code in _norm_code(code) or q_code in _norm_code(oem)):
            score += 30.0

    # лёгкие бонусы за начало строки
    q_full = " ".join(tkns)
    q_full_no_ws = squash(q_full)
    if code:
        if code == q_full:
            score += 20.0
        if code.startswith(q_full) or code.startswith(q_full_no_ws):
            score += 10.0
        for t in tkns:
            if t and code.startswith(t):
                score += 3.0
    return score

def find_rows(query: str, limit: int = 20) -> List[int]:
    """Вернёт индексы строк df под запрос (с сортировкой по релевантности)."""
    ensure_fresh_data()
    if df is None or df.empty:
        return []

    tokens = _tokenize_query(query)
    candidates = list(match_row_by_index(tokens))
    if not candidates:
        candidates = list(range(len(df)))

    q_squash = squash(query)
    q_code   = _norm_code(query)

    scored: List[Tuple[float, int]] = []
    for i in candidates:
        row = df.iloc[i].to_dict()
        sc = _relevance_score(row, tokens, q_squash, q_code)
        if sc > 0:
            scored.append((sc, i))

    if not scored:
        return candidates[:limit]

    scored.sort(key=lambda t: t[0], reverse=True)
    return [i for _, i in scored[:limit]]

def get_row(i: int) -> Dict[str, Any]:
    ensure_fresh_data()
    if df is None or df.empty or i < 0 or i >= len(df):
        return {}
    row = df.iloc[int(i)]
    return {k: ("" if pd.isna(v) else v) for k, v in row.to_dict().items()}

def find_items(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    idxs = find_rows(query, limit=limit)
    return [get_row(i) for i in idxs]

# ---------- Экспорт ----------
def _df_to_xlsx(df_: pd.DataFrame, filename: str = "export.xlsx") -> io.BytesIO:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df_.to_excel(writer, index=False)
    buf.seek(0)
    return buf

# ---------- Пользователи / доступ ----------
def _parse_int(x) -> Optional[int]:
    try:
        v = int(str(x).strip())
        return v if v > 0 else None
    except Exception:
        return None

def _normalize_header_name(h: str, idx: int) -> str:
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

def load_users_from_sheet() -> Tuple[Set[int], Set[int], Set[int]]:
    allowed: Set[int] = set()
    admins:  Set[int] = set()
    blocked: Set[int] = set()
    try:
        ws = _open_ws(USERS_SHEET_NAME)
    except Exception:
        logger.info("Лист пользователей отсутствует — пускаем всех по умолчанию")
        return allowed, admins, blocked

    all_vals = ws.get_all_values()
    if not all_vals:
        return allowed, admins, blocked

    headers_raw = all_vals[0]
    headers = _dedupe_headers(headers_raw)
    rows = all_vals[1:]
    recs: List[dict] = []
    for r in rows:
        recs.append({headers[i]: (r[i] if i < len(r) else "") for i in range(len(headers))})
    dfu = pd.DataFrame(recs)
    dfu.columns = [c.strip().lower() for c in dfu.columns]

    has_role     = "role" in dfu.columns
    has_allowed  = "allowed" in dfu.columns
    has_admin    = "admin" in dfu.columns
    has_blocked  = "blocked" in dfu.columns

    def truthy(v) -> bool:
        s = str(v).strip().lower()
        return s in ("1", "true", "да", "y", "yes", "ok", "ок")

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
        if has_blocked and truthy(r.get("blocked")):
            blocked.add(uid); continue
        if has_admin and truthy(r.get("admin")):
            admins.add(uid); allowed.add(uid); continue
        if has_allowed and truthy(r.get("allowed")):
            allowed.add(uid); continue
        allowed.add(uid)
    return allowed, admins, blocked

# ---------- Запись в листы ----------
def _append_row(ws_title: str, values: List[Any]) -> None:
    ws = _open_ws(ws_title)
    for attempt in range(3):
        try:
            ws.append_row(values, value_input_option="USER_ENTERED")
            return
        except gspread.exceptions.APIError as e:
            msg = str(e)
            if any(x in msg for x in ("429", "500", "502", "503", "504")) and attempt < 2:
                sleep_s = 2 * (attempt + 1)
                logger.warning(f"Google API {msg}, ретрай через {sleep_s}s")
                time.sleep(sleep_s)
                continue
            logger.exception("Ошибка Google API при append_row")
            raise
        except Exception:
            logger.exception("Неожиданная ошибка при append_row")
            raise

def record_user(user_id: int, first_name: str, username: str, query: str) -> None:
    try:
        tz = ZoneInfo(TZ_NAME)
        now = datetime.now(tz)
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        username = (username or "").lstrip("@")

        ws = _open_ws(USERS_SHEET_NAME)
        header = ws.row_values(1)
        if not header:
            ws.insert_row(["ID", "Имя", "Юзернейм", "Дата", "Время", "TZ", "Запрос"], 1)

        _append_row(USERS_SHEET_NAME, [
            user_id, first_name or "", username or "",
            date_str, time_str, TZ_NAME, query or "",
        ])
    except Exception:
        logger.exception("Ошибка записи в 'Пользователи'")

def record_history(user_id: int, title_or_code: str, quantity: Any, comment: str) -> None:
    try:
        tz = ZoneInfo(TZ_NAME)
        now = datetime.now(tz)
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        ws = _open_ws(HISTORY_SHEET_NAME)
        header = ws.row_values(1)
        if not header:
            ws.insert_row(
                ["Дата", "Время", "TZ", "Название/Код", "Кол-во", "Комментарий", "UserID"], 1
            )

        _append_row(HISTORY_SHEET_NAME, [
            date_str, time_str, TZ_NAME,
            title_or_code or "",
            quantity if quantity is not None else "",
            comment or "",
            user_id,
        ])
    except Exception:
        logger.exception("Ошибка записи в 'История'")

# ---------- Async helper ----------
import asyncio
async def asyncio_to_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

async def ensure_fresh_data_async(force: bool = False):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: ensure_fresh_data(force))

# ---------- Инициализация ----------
def initial_load():
    try:
        ensure_fresh_data(force=True)
    except Exception as e:
        logger.exception(f"initial_load: ensure_fresh_data error: {e}")
        raise
    try:
        allowed, admins, blocked = load_users_from_sheet()
        SHEET_ALLOWED.clear(); SHEET_ALLOWED.update(allowed)
        SHEET_ADMINS.clear();  SHEET_ADMINS.update(admins)
        SHEET_BLOCKED.clear(); SHEET_BLOCKED.update(blocked)
    except Exception as e:
        logger.warning(f"initial_load: не удалось загрузить пользователей: {e}")

async def initial_load_async():
    try:
        await asyncio_to_thread(ensure_fresh_data, True)
    except Exception as e:
        logger.exception(f"initial_load_async error: {e}")
        raise
    try:
        allowed, admins, blocked = await asyncio_to_thread(load_users_from_sheet)
        SHEET_ALLOWED.clear(); SHEET_ALLOWED.update(allowed)
        SHEET_ADMINS.clear();  SHEET_ADMINS.update(admins)
        SHEET_BLOCKED.clear(); SHEET_BLOCKED.update(blocked)
    except Exception as e:
        logger.warning(f"initial_load_async: не удалось загрузить пользователей: {e}")
