import re
import io
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from zoneinfo import ZoneInfo
from config import TZ_NAME
import asyncio

def now_local_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    try:
        return datetime.now(ZoneInfo(TZ_NAME)).strftime(fmt)
    except Exception:
        return datetime.utcnow().strftime(fmt)

async def to_thread(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)

def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", (text or "")).lower().strip()

def squash(text: str) -> str:
    return re.sub(r"[\W_]+", "", (text or "").lower(), flags=re.UNICODE)

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

async def safe_send_html_message(bot, chat_id: int, text: str, **kwargs):
    try:
        return await bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML", **kwargs)
    except Exception:
        import re as _re
        no_tags = _re.sub(r"</?(b|i|code)>", "", text)
        kwargs.pop("parse_mode", None)
        return await bot.send_message(chat_id=chat_id, text=no_tags, **kwargs)

def df_to_xlsx(df: DataFrame, name: str) -> io.BytesIO:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False)
    buf.seek(0)
    buf.name = name
    return buf