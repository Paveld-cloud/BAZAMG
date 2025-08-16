# app/config.py
import os

# ---------------------- БАЗОВОЕ ----------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()

# Google Sheets
SPREADSHEET_URL = os.getenv("SPREADSHEET_URL", "").strip()

# Сервис-аккаунт: поддерживаем ОДНОВРЕМЕННО оба имени переменной,
# чтобы код мог импортировать любой вариант.
GOOGLE_APPLICATION_CREDENTIALS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "").strip()
CREDS_JSON = GOOGLE_APPLICATION_CREDENTIALS_JSON or os.getenv("CREDS_JSON", "").strip()

SHEET_NAME = os.getenv("SHEET_NAME", "").strip()

# Webhook
WEBHOOK_URL = (os.getenv("WEBHOOK_URL", "").strip()).rstrip("/")
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/webhook").strip()
PORT = int(os.getenv("PORT", "8080"))
WEBHOOK_SECRET_TOKEN = os.getenv("WEBHOOK_SECRET_TOKEN", "").strip()

# Прочее
TZ_NAME = os.getenv("TIMEZONE", "Asia/Tashkent").strip()
PAGE_SIZE = int(os.getenv("PAGE_SIZE", "5"))
MAX_QTY = float(os.getenv("MAX_QTY", "1000"))

# Приветствие/медиа/контакты
WELCOME_ANIMATION_URL = os.getenv("WELCOME_ANIMATION_URL", "").strip()  # gif/mp4/file_id
WELCOME_PHOTO_URL = os.getenv("WELCOME_PHOTO_URL", "").strip()          # url/file_id
SUPPORT_CONTACT = os.getenv("SUPPORT_CONTACT", "👨‍💻 Поддержка: @your_support").strip()

# Можно оставить пустым или заменить на свой file_id
WELCOME_MEDIA_ID = os.getenv(
    "WELCOME_MEDIA_ID",
    "AgACAgIAAxkBAAIPVGieF335h6r2xO6EvVxMTTatIs7VAAJg-zEbBUHwSAgsrYCCYGWiAQADAgADeQADNgQ"
).strip()

# ---------------------- КОНСТАНТЫ ----------------------
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
DATA_TTL = int(os.getenv("DATA_TTL", "300"))
USERS_TTL = int(os.getenv("USERS_TTL", "300"))

# Поля, по которым строим индекс поиска
SEARCH_FIELDS = ["тип", "наименование", "код", "oem", "изготовитель"]

# ----------
