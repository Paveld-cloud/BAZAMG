# BAZAMG — Refactored Modular Bot

Это модульная версия твоего бота. Логика сохранена: поиск, фото по коду, списание (qty+коммент+подтверждение), запись в "История", права доступа по листу "Пользователи", экспорт, /fileid, /imgdebug, меню и вебхук.

## Запуск

1. Заполни переменные окружения (Railway/Render/Heroku):
- TELEGRAM_TOKEN
- SPREADSHEET_URL
- GOOGLE_APPLICATION_CREDENTIALS_JSON
- WEBHOOK_URL (например, https://your-app.up.railway.app)
- WEBHOOK_PATH (например, /webhook)
- (опционально) SHEET_NAME, TIMEZONE, PAGE_SIZE, MAX_QTY, WELCOME_*

2. Установи зависимости:
```
pip install -r requirements.txt
```

3. Запусти:
```
python bot.py
```

## Структура
- `bot.py` — точка входа и запуск webhook
- `config.py` — ENV и константы
- `gsheets.py` — работа с Google Sheets
- `data.py` — кеш, индексы, доступы, поиск фото по коду
- `indexing.py` — построение индексов
- `images.py` — обработка URL-ов картинок
- `handlers.py` — все хендлеры/команды, диалог списания, экспорт
- `ui.py` — инлайн-клавиатуры
- `utils.py` — вспомогательные функции

## Совместимость
- Требуется python-telegram-bot 20.x
- Совместим с твоим текущим requirements.txt