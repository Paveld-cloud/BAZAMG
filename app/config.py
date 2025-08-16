# app/config.py
import os

# ---------------------- БАЗОВОЕ ----------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()

# Google Sheets
SPREADSHEET_URL = os.getenv("SPREADSHEET_URL", "").strip()
# JSON сервис-аккаунта (содержимое переменной, а не путь к файлу)
CREDS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "").strip()
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

# Хочешь — оставь пустым. Ниже пример file_id (можешь заменить/очистить):
WELCOME_MEDIA_ID = os.getenv(
    "WELCOME_MEDIA_ID",
    "AgACAgIAAxkBAAIPVGieF335h6r2xO6EvVxMTTatIs7VAAJg-zEbBUHwSAgsrYCCYGWiAQADAgADeQADNgQ"
).strip()

# ---------------------- КОНСТАНТЫ ----------------------
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
DATA_TTL = int(os.getenv("DATA_TTL", "300"))
USERS_TTL = int(os.getenv("USERS_TTL", "300"))

# Список полей, по которым строим индекс и ищем
SEARCH_FIELDS = ["тип", "наименование", "код", "oem", "изготовитель"]

# ---------------------- АДМИНЫ ----------------------
# Можно задать через ENV ADMINS="123,456,789"
def _parse_admins(s: str):
    out = set()
    for piece in (s or "").replace(";", ",").split(","):
        p = piece.strip()
        if not p:
            continue
        try:
            out.add(int(p))
        except Exception:
            pass
    return out

# По умолчанию добавлен один локальный админ (можешь убрать/поменять)
ADMINS = _parse_admins(os.getenv("ADMINS", "225177765"))
