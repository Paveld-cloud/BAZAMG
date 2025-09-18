# app/config.py
import os

def _truthy(x: str | None) -> bool:
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "да", "ok", "ок"} or (s.isdigit() and int(s) > 0)

# --- Бот / вебхук
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
WEBHOOK_URL = (os.getenv("WEBHOOK_URL", "") or "").rstrip("/")
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/webhook")
PORT = int(os.getenv("PORT", "8080"))
WEBHOOK_SECRET_TOKEN = os.getenv("WEBHOOK_SECRET_TOKEN", "")

# --- Google Sheets
SPREADSHEET_URL = os.getenv("SPREADSHEET_URL", "")
GOOGLE_APPLICATION_CREDENTIALS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "{}")
SHEET_NAME = os.getenv("SHEET_NAME", "").strip()

# --- Поведение бота
TZ_NAME = os.getenv("TIMEZONE", "Asia/Tashkent")
PAGE_SIZE = int(os.getenv("PAGE_SIZE", "5"))
MAX_QTY = float(os.getenv("MAX_QTY", "1000"))

# --- Кеши
DATA_TTL = int(os.getenv("DATA_TTL", "300"))
USERS_TTL = int(os.getenv("USERS_TTL", "300"))

# --- Поиск
SEARCH_FIELDS = ["тип", "наименование", "код", "oem", "изготовитель"]

# --- Мультимедиа приветствия
WELCOME_ANIMATION_URL = os.getenv("WELCOME_ANIMATION_URL", "").strip()
WELCOME_PHOTO_URL = os.getenv("WELCOME_PHOTO_URL", "").strip()
WELCOME_MEDIA_ID = os.getenv("WELCOME_MEDIA_ID", "").strip()
SUPPORT_CONTACT = os.getenv("SUPPORT_CONTACT", "👨‍💻 Поддержка: @Paveldemen")

# --- Админы (через ENV ADMINS="123,456")
ADMINS = set()
_adm_env = os.getenv("ADMINS", "")
if _adm_env:
    for p in _adm_env.replace(" ", "").split(","):
        if p.isdigit():
            ADMINS.add(int(p))

# --- Режим сопоставления фото
# 1 = строго: фото только если код содержится в URL
# 0 = мягко: код -> image из строки (как раньше)
IMAGE_STRICT = _truthy(os.getenv("IMAGE_STRICT", "1"))
