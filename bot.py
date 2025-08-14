import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.types import InputFile
from aiogram.utils import executor

API_TOKEN = "ТВОЙ_ТОКЕН_БОТА"
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# Асинхронная версия загрузки пользователей
async def ensure_users_async(force=True):
    print("📥 Загрузка пользователей...")
    await asyncio.sleep(1)  # имитация загрузки
    print("✅ Пользователи загружены")

# Приветственное сообщение с фото
@dp.message_handler(commands=["start"])
async def send_welcome(message: types.Message):
    photo_id = "AgACAgIAAxkBAAIPVGieF335h6r2xO6EvVxMTTatIs7VAAJg-zEbBUHwSAgsrYCCYGWiAQADAgADeQADNgQ"
    await message.answer_photo(
        photo=photo_id,
        caption="Добро пожаловать! 🎉\nЗдесь будет твой красивый текст."
    )

# Главная асинхронная функция запуска
async def main():
    # Выполняем загрузку пользователей
    await ensure_users_async(force=True)

    # Запускаем бота
    print("🚀 Бот запущен")
    await dp.start_polling()

if __name__ == "__main__":
    asyncio.run(main())
