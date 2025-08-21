# app/config.py
import os

def _truthy(x: str | None) -> bool:
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {"1","true","yes","y","да","ok","ок"} or (s.isdigit() and int(s) > 0)

# --- Бот / вебхук
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
WEBHOOK_URL = (os.getenv("WEBHOOK_URL", "") or "").rstrip("/")  # если пусто — будет polling
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/webhook")
PORT = int(os.getenv("PORT", "8080"))
WEBHOOK_SECRET_TOKEN = os.getenv("WEBHOOK_SECRET_TOKEN", "")

# --- Данные / Google Sheets
SPREADSHEET_URL = os.getenv("SPREADSHEET_URL", "")
GOOGLE_APPLICATION_CREDENTIALS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "{}")
SHEET_NAME = os.getenv("SHEET_NAME", "")  # если пусто — будет первый лист

# --- Кеш/страницы/лимиты
TZ_NAME = os.getenv("TZ_NAME", "Asia/Tashkent")
DATA_TTL = int(os.getenv("DATA_TTL", "120"))       # сек между авто-перезагрузками df
USERS_TTL = int(os.getenv("USERS_TTL", "300"))     # сек между перечитываниями «Пользователи»
PAGE_SIZE = int(os.getenv("PAGE_SIZE", "5"))
MAX_QTY = float(os.getenv("MAX_QTY", "10000"))

# --- Поиск: какие поля использовать (в таблице)
SEARCH_FIELDS = tuple((os.getenv("SEARCH_FIELDS","тип,наименование,код,oem,изготовитель")
                       .lower().replace(" ", "").split(",")))

# --- Приветствие/контакты
WELCOME_ANIMATION_URL = os.getenv("WELCOME_ANIMATION_URL", "")
WELCOME_PHOTO_URL = os.getenv("WELCOME_PHOTO_URL", "")
WELCOME_MEDIA_ID = os.getenv("WELCOME_MEDIA_ID", "")  # file_id
SUPPORT_CONTACT = os.getenv("SUPPORT_CONTACT", "👨‍💻 Поддержка: @your_support")

# --- Админы (ENV ADMINS="123,456")
ADMINS = set()
_adm_env = os.getenv("ADMINS", "")
if _adm_env:
    for p in _adm_env.replace(" ", "").split(","):
        if p.isdigit():
            ADMINS.add(int(p))

# --- Режим сопоставления фото
IMAGE_STRICT = _truthy(os.getenv("IMAGE_STRICT", "1"))
