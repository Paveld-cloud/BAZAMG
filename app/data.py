# app/data.py
import os
import io
import re
import time
import json
import math
import logging
from typing import Dict, Set, Tuple, List, Iterable, Optional

import pandas as pd
import aiohttp
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
from zoneinfo import ZoneInfo

# -------------------- Логгер --------------------
logger = logging.getLogger("bot.data")

# -------------------- Конфиг --------------------
# Читаем из app.config то, что может быть определено там.
try:
    from app.config import (
        SPREADSHEET_URL,
        SAP_SHEET_NAME,          # имя листа с деталями (по умолчанию "SAP")
        IMAGES_SHEET_NAME,       # имя листа с картинками (например, "Изображения"), опционально
        USERS_SHEET_NAME,        # имя листа с пользователями (например, "Пользователи"), опционально
        DATA_TTL,                # сек, сколько держать df в памяти (например 300..900)
        SEARCH_COLUMNS,          # список колонок для поиска (тип, наименование, код, oem, изготовитель)
    )
except Exception:
    # Запасные значения по умолчанию
    SPREADSHEET_URL = os.getenv("SPREADSHEET_URL", "")
    SAP_SHEET_NAME = "SAP"
    IMAGES_SHEET_NAME = "Изображения"   # если листа нет — пропустим
    USERS_SHEET_NAME = "Пользователи"   # если листа нет — пропустим
    DATA_TTL = int(os.getenv("DATA_TTL", "600"))
    SEARCH_COLUMNS = ["тип", "наименование", "код", "oem", "изготовитель"]

GOOGLE_APPLICATION_CREDENTIALS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "")
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# -------------------- Глобальные состояния --------------------
df: Optional[pd.DataFrame] = None
_last_load_ts: float = 0.0

# Поиск и изображения
_search_index: Dict[str, Set[int]] = {}
_image_index: Dict[str, str] = {}

# Диалоги и состояние пользователей (используется в handlers.py)
user_state: Dict[int, dict] = {}
issue_state: Dict[int, dict] = {}

# Списки пользователей из таблицы
SHEET_ALLOWED: Set[int] = set()
SHEET_ADMINS: Set[int] = set()
SHEET_BLOCKED: Set[int] = set()

# Шаги диалога (используются в handlers.py → ConversationHandler)
ASK_QUANTITY, ASK_COMMENT, ASK_CONFIRM = range(3)

# -------------------- Утилиты --------------------
def _norm_code(x: str) -> str:
    """Нормализуем код: lower + убираем пробелы и дефисы."""
    return re.sub(r"[\s\-]+", "", str(x or "").strip().lower())

def _norm_str(x: str) -> str:
    return str(x or "").strip().lower()

def now_local_str(tz_name: str = "Asia/Tashkent") -> str:
    tz = ZoneInfo(tz_name)
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

def val(d: dict, key: str, default: str = "") -> str:
    """Безопасно достать значение из словаря row.to_dict()."""
    return str(d.get(key, default) or default)

def _extract_code_from_url(url: str) -> str:
    """
    Берём последний сегмент пути без расширения:
    https://site.com/path/UZ005399.png -> uz005399
    """
    try:
        path = re.sub(r"[?#].*$", "", url)  # обрежем query/fragment
        tail = path.rsplit("/", 1)[-1]
        name = tail.rsplit(".", 1)[0]
        return _norm_code(name)
    except Exception:
        return ""

def _safe_col(df_: pd.DataFrame, col: str) -> Optional[pd.Series]:
    if col not in df_.columns:
        return None
    s = df_[col].astype(str).fillna("").str.strip().str.lower()
    return s

# -------------------- Формат карточки --------------------
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

# -------------------- Google Sheets клиент --------------------
def get_gs_client():
    if not GOOGLE_APPLICATION_CREDENTIALS_JSON:
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS_JSON не задан")
    try:
        info = json.loads(GOOGLE_APPLICATION_CREDENTIALS_JSON)
        creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    except json.JSONDecodeError:
        # на случай если пользователь положил путь вместо JSON
        creds = Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS_JSON, scopes=SCOPES)
    client = gspread.authorize(creds)
    return client

# -------------------- Загрузка данных --------------------
def _load_sap_dataframe() -> pd.DataFrame:
    client = get_gs_client()
    sh = client.open_by_url(SPREADSHEET_URL)
    ws = sh.worksheet(SAP_SHEET_NAME)
    records = ws.get_all_records()
    new_df = pd.DataFrame(records)

    # нормализуем имена колонок
    new_df.columns = [c.strip().lower() for c in new_df.columns]

    # приведение важных полей
    for col in ("код", "oem"):
        if col in new_df.columns:
            new_df[col] = new_df[col].astype(str).str.strip().str.lower()
    if "image" in new_df.columns:
        new_df["image"] = new_df["image"].astype(str).str.strip()

    return new_df

def _load_images_sheet() -> Dict[str, str]:
    """
    Опционально: загрузим отдельный лист с изображениями (если он есть).
    Ожидаемые колонки: 'код', 'image'. Возвращаем {norm_code: image_url}.
    """
    mapping: Dict[str, str] = {}
    try:
        client = get_gs_client()
        sh = client.open_by_url(SPREADSHEET_URL)
        ws = sh.worksheet(IMAGES_SHEET_NAME)
    except Exception:
        # листа нет — не ошибка
        return mapping

    rows = ws.get_all_records()
    if not rows:
        return mapping

    df_img = pd.DataFrame(rows)
    df_img.columns = [c.strip().lower() for c in df_img.columns]
    if "код" not in df_img.columns or "image" not in df_img.columns:
        return mapping

    for _, r in df_img.iterrows():
        code = _norm_code(r.get("код", ""))
        url = str(r.get("image", "")).strip()
        if code and url:
            mapping[code] = url
    logger.info(f"image-index: из листа '{IMAGES_SHEET_NAME}' прочитано {len(mapping)} ссылок")
    return mapping

def build_search_index(df_: pd.DataFrame) -> Dict[str, Set[int]]:
    """
    Простой индекс: токен -> набор индексов строк, где токен встречается в любой из SEARCH_COLUMNS.
    """
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
    Индекс картинок — два прохода + слияние с отдельным листом (если есть):
    A) ключ = имя файла из URL (последний сегмент без расширения)
    B) фолбэк: ключ = значение в колонке 'код', если есть image и A не покрыл.
    + Сливаем {код:image} из листа IMAGES_SHEET_NAME.
    """
    index: Dict[str, str] = {}

    # A) по имени файла из df.image
    if "image" in df_.columns:
        added_a = 0
        for _, row in df_.iterrows():
            raw_url = str(row.get("image", "")).strip()
            if not raw_url:
                continue
            key = _extract_code_from_url(raw_url)
            if not key:
                continue
            index[key] = raw_url
            added_a += 1
    else:
        added_a = 0

    # B) фолбэк по 'код'
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

    # C) слияние с отдельным листом
    ext = _load_images_sheet()
    before_merge = len(index)
    index.update(ext)
    merged_added = len(index) - before_merge

    logger.info(
        f"image-index: A(filename)={added_a}, B(code-col)={added_b}, "
        f"C(extra-sheet)={merged_added}, unique_keys={len(index)}"
    )
    return index

def ensure_fresh_data(force: bool = False):
    """
    Синхронная перезагрузка данных и индексов (если TTL истёк или force=True).
    """
    global df, _search_index, _image_index, _last_load_ts

    need = (
        force
        or df is None
        or (time.time() - _last_load_ts > DATA_TTL)
    )
    if not need:
        return

    new_df = _load_sap_dataframe()
    df = new_df
    _search_index = build_search_index(df)
    _image_index = build_image_index(df)
    _last_load_ts = time.time()
    logger.info(f"✅ Перезагружено {len(df)} строк и построены индексы")

async def ensure_fresh_data_async(force: bool = False):
    # обёртка над синхронной загрузкой
    await asyncio_to_thread(ensure_fresh_data, force)

# -------------------- Картинки --------------------
async def find_image_by_code_async(code: str) -> str:
    """
    Вернёт URL картинки по коду детали.
    Ищем по индексу, ключ нормализуем (_norm_code).
    """
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
    """
    Если пришлют ibb.co/<page>, пытаемся вытащить прямой og:image.
    Для i.ibb.co/<hash>/file.png — возвращаем как есть.
    """
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
    """
    Приводим ссылку к виду, который Telegram скорее всего примет:
    - Google Drive -> прямой uc?export=download
    - ibb.co page -> og:image
    - Прочее — без изменений
    """
    if not url_raw:
        return ""
    url = normalize_drive_url(url_raw)
    url = await resolve_ibb_direct_async(url)
    return url

# -------------------- Поисковый индекс --------------------
def match_row_by_index(tokens: List[str]) -> Set[int]:
    """
    Возвращает индексы строк, в которых встречаются ВСЕ токены.
    """
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
            return set()  # если какого-то токена нет — пересечение пустое
        sets.append(s)

    # пересечение всех множеств
    acc = sets[0].copy()
    for s in sets[1:]:
        acc &= s
        if not acc:
            break
    return acc

def squash(text: str) -> str:
    """Убираем все небуквенно-цифровые символы, приводим к lower (для «склеенного» поиска)."""
    return re.sub(r"[\W_]+", "", str(text or "").lower())

def normalize(text: str) -> str:
    """Нормализатор пользовательского запроса (для токенизации)."""
    return re.sub(r"[^\w\s]", "", str(text or "").lower()).strip()

# -------------------- Экспорт --------------------
def _df_to_xlsx(df_: pd.DataFrame, filename: str = "export.xlsx") -> io.BytesIO:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df_.to_excel(writer, index=False)
    buf.seek(0)
    return buf

# -------------------- Пользователи --------------------
def _parse_int(x) -> Optional[int]:
    try:
        v = int(str(x).strip())
        return v if v > 0 else None
    except Exception:
        return None

def load_users_from_sheet() -> Tuple[Set[int], Set[int], Set[int]]:
    """
    Пытаемся прочитать лист пользователей.
    Поддерживаем разные схемы:
      - колонки: user_id, role (admin|user|blocked)
      - или отдельные булевы: allowed, admin, blocked
      - или просто список 'user_id' => allowed
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

    rows = ws.get_all_records()
    if not rows:
        logger.info("Лист пользователей пуст")
        return allowed, admins, blocked

    dfu = pd.DataFrame(rows)
    dfu.columns = [c.strip().lower() for c in dfu.columns]

    has_uid = "user_id" in dfu.columns
    has_role = "role" in dfu.columns
    has_allowed = "allowed" in dfu.columns
    has_admin = "admin" in dfu.columns
    has_blocked = "blocked" in dfu.columns

    for _, r in dfu.iterrows():
        uid = _parse_int(r.get("user_id") if has_uid else r.get("uid"))
        if not uid:
            continue

        if has_role:
            role = str(r.get("role", "")).strip().lower()
            if role in ("admin", "админ"):
                admins.add(uid)
                allowed.add(uid)
            elif role in ("blocked", "ban", "заблокирован"):
                blocked.add(uid)
            else:
                allowed.add(uid)
            continue

        # булевые флаги
        if has_blocked and str(r.get("blocked")).strip().lower() in ("1", "true", "да", "y", "yes"):
            blocked.add(uid)
            continue
        if has_admin and str(r.get("admin")).strip().lower() in ("1", "true", "да", "y", "yes"):
            admins.add(uid)
            allowed.add(uid)
            continue
        if has_allowed:
            if str(r.get("allowed")).strip().lower() in ("1", "true", "да", "y", "yes"):
                allowed.add(uid)
            continue

        # просто список
        allowed.add(uid)

    logger.info(f"Пользователи прочитаны: allowed={len(allowed)}, admins={len(admins)}, blocked={len(blocked)}")
    return allowed, admins, blocked

# -------------------- Хелперы для async --------------------
import asyncio
async def asyncio_to_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

