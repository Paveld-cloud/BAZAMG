import os, json
from google.oauth2.service_account import Credentials

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
SPREADSHEET_URL = os.getenv("SPREADSHEET_URL")
CREDS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
WEBHOOK_URL = (os.getenv("WEBHOOK_URL") or "").rstrip("/")
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/webhook")
PORT = int(os.getenv("PORT", "8080"))
WEBHOOK_SECRET_TOKEN = os.getenv("WEBHOOK_SECRET_TOKEN", "")
SHEET_NAME = os.getenv("SHEET_NAME", "").strip()
MAX_QTY = float(os.getenv("MAX_QTY", "1000"))
TZ_NAME = os.getenv("TIMEZONE", "Asia/Tashkent")
PAGE_SIZE = int(os.getenv("PAGE_SIZE", "5"))

WELCOME_ANIMATION_URL = os.getenv("WELCOME_ANIMATION_URL", "").strip()
WELCOME_PHOTO_URL = os.getenv("WELCOME_PHOTO_URL", "").strip()
SUPPORT_CONTACT = os.getenv("SUPPORT_CONTACT", "👨‍💻 Поддержка: @your_support")
WELCOME_MEDIA_ID = os.getenv("WELCOME_MEDIA_ID", "AgACAgIAAxkBAAIRRGie3G13XwrO7OGTKd1nqS0pXsMIAAJ28DEbiWb4SLglxfO1HainAQADAgADeAADNgQ")

DATA_TTL = int(os.getenv("DATA_TTL", "300"))
USERS_TTL = int(os.getenv("USERS_TTL", "300"))

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

def get_creds():
    creds_info = json.loads(CREDS_JSON)
    return Credentials.from_service_account_info(creds_info, scopes=SCOPES)