import asyncio
import logging
import sys
import os
import io
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F, Router
from aiogram.types import Message, BufferedInputFile
from aiogram.filters import CommandStart
from aiogram.enums import ContentType, ParseMode

from services.ocr import OCREngine

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
TESSERACT_CMD = os.getenv("TESSERACT_CMD")

if not BOT_TOKEN:
    logging.error("BOT_TOKEN is not set. Please check your .env file.")
    sys.exit(1)

router = Router()
ocr_service = OCREngine(tesseract_cmd=TESSERACT_CMD)


@router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "Привет! Я OCR-бот.\n"
        "Отправь мне изображение (фото или документ), и я извлеку из него текст.\n"
        "Я понимаю русский и английский языки."
    )


async def process_image(message: Message, bot: Bot, file_id: str):
    """Common logic for processing images from photos and documents."""
    status_msg = await message.reply("Обрабатываю изображение...")

    try:
        file_info = await bot.get_file(file_id)
        file_bytes = io.BytesIO()
        await bot.download_file(file_info.file_path, file_bytes)

        result_text = await asyncio.to_thread(
            ocr_service.recognize,
            file_bytes.getvalue()
        )

        if len(result_text) > 4000:
            result_file = BufferedInputFile(
                result_text.encode('utf-8'),
                filename="ocr_result.txt"
            )
            await message.reply_document(
                result_file,
                caption="Текст слишком длинный, отправляю файлом."
            )
        else:
            await message.reply(
                f"**Результат:**\n\n{result_text}",
                parse_mode=ParseMode.MARKDOWN
            )

    except Exception as e:
        logging.exception("Error processing image")
        await message.reply(f"Ошибка: {e}")
    finally:
        await status_msg.delete()


@router.message(F.content_type == ContentType.PHOTO)
async def handle_photo(message: Message, bot: Bot):
    photo = message.photo[-1]
    await process_image(message, bot, photo.file_id)


@router.message(F.content_type == ContentType.DOCUMENT)
async def handle_document(message: Message, bot: Bot):
    if not message.document.mime_type or not message.document.mime_type.startswith('image/'):
        await message.reply("Пожалуйста, отправьте файл изображения.")
        return

    await process_image(message, bot, message.document.file_id)


async def main():
    logging.basicConfig(level=logging.INFO)
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)

    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped")