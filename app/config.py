import os

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
WEBHOOK_URL = (os.getenv("WEBHOOK_URL", "") or "").rstrip("/")
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/webhook")
PORT = int(os.getenv("PORT", "8080"))
WEBHOOK_SECRET_TOKEN = os.getenv("WEBHOOK_SECRET_TOKEN", "")
TZ_NAME = os.getenv("TIMEZONE", os.getenv("TZ_NAME", "Asia/Tashkent"))

SPREADSHEET_URL = os.getenv("SPREADSHEET_URL", "")
SHEET_NAME = os.getenv("SHEET_NAME", "").strip()
MAX_QTY = float(os.getenv("MAX_QTY", "1000"))
PAGE_SIZE = int(os.getenv("PAGE_SIZE", "5"))
DATA_TTL = int(os.getenv("DATA_TTL", "300"))
USERS_TTL = int(os.getenv("USERS_TTL", "300"))

WELCOME_ANIMATION_URL = os.getenv("WELCOME_ANIMATION_URL", "").strip()
WELCOME_PHOTO_URL = os.getenv("WELCOME_PHOTO_URL", "").strip()
SUPPORT_CONTACT = os.getenv("SUPPORT_CONTACT", "👨‍💻 Поддержка: @your_support")
WELCOME_MEDIA_ID = os.getenv("WELCOME_MEDIA_ID", "AgACAgIAAxkBAAIPVGieF335h6r2xO6EvVxMTTatIs7VAAJg-zEbBUHwSAgsrYCCYGWiAQADAgADeQADNgQ")

GOOGLE_APPLICATION_CREDENTIALS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "")

# Локальные админы (через запятую)
ADMINS = set()
_raw_admins = os.getenv("ADMINS", "225177765").split(",")
for a in _raw_admins:
    a = a.strip()
    if a.isdigit():
        ADMINS.add(int(a))