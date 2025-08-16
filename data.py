import time
import asyncio
import logging
import pandas as pd
from pandas import DataFrame
from typing import Optional, Dict, Set, List
from config import DATA_TTL, USERS_TTL
from gsheets import load_data_blocking, load_users_from_sheet
from indexing import build_search_index, build_image_index, SEARCH_FIELDS, _norm_code, _fuse, _filename_from_url

logger = logging.getLogger("data")

df: Optional[DataFrame] = None
_last_load_ts = 0.0
_search_index: Optional[Dict[str, Set[int]]] = None
_image_index: Optional[Dict[str, str]] = None

SHEET_ALLOWED: Set[int] = set()
SHEET_ADMINS: Set[int] = set()
SHEET_BLOCKED: Set[int] = set()
_last_users_ts = 0.0

_loading_data = False
_loading_users = False

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

def is_admin(uid: int) -> bool:
    ensure_users()
    return uid in SHEET_ADMINS

def is_allowed(uid: int) -> bool:
    ensure_users()
    if uid in SHEET_BLOCKED:
        return False
    if SHEET_ALLOWED:
        return (uid in SHEET_ALLOWED) or (uid in SHEET_ADMINS)
    return True

def get_admin_ids() -> Set[int]:
    return set(SHEET_ADMINS)

def get_df():
    return df

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

async def find_image_by_code_async(code: str) -> str:
    global _image_index
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

def get_image_index_copy() -> Dict[str, str]:
    return dict(_image_index or {})