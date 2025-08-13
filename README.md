# Telegram Sheets Inventory Bot

Бот ищет детали в Google Sheets и работает через Telegram Webhook.

## Установка

```bash
pip install -r requirements.txt
python bot.py
```

## Обязательные переменные окружения
- TELEGRAM_TOKEN — токен бота
- SPREADSHEET_URL — ссылка на Google Sheet
- GOOGLE_APPLICATION_CREDENTIALS_JSON — JSON сервисного аккаунта в одной строке
- WEBHOOK_URL — публичный HTTPS базовый URL (домен деплоя)

## Опциональные переменные окружения
- WEBHOOK_PATH — путь вебхука (по умолчанию `/webhook`)
- PORT — порт (по умолчанию `8080`)
- WEBHOOK_SECRET_TOKEN — секрет для проверок Telegram (если нужен)
- TIMEZONE — часовой пояс для записей (по умолчанию `Europe/Moscow`)

## Railway/Heroku
1. Убедитесь, что в `Procfile` указан процесс типа web:
```
web: python bot.py
```
2. Задайте все переменные окружения из списков выше.
3. После деплоя укажите в `WEBHOOK_URL` домен приложения. Бот сам выставит webhook на `${WEBHOOK_URL}${WEBHOOK_PATH}`.

## Быстрый запуск локально (пример)
```bash
export TELEGRAM_TOKEN=xxx
export SPREADSHEET_URL='https://docs.google.com/spreadsheets/d/...'
export GOOGLE_APPLICATION_CREDENTIALS_JSON='{"type":"service_account",...}'
export WEBHOOK_URL='https://your-tunnel-or-domain'
python bot.py
```