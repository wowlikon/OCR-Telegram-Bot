import threading
import asyncio
import logging
import sys
import os
import io
import html
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F, Router
from aiogram.types import Message, BufferedInputFile
from aiogram.filters import CommandStart
from aiogram.enums import ContentType, ParseMode

from services.ocr import OCREngine

load_dotenv()

TESSERACT_CMD = os.getenv("TESSERACT_CMD")
MAX_THREADS = int(os.getenv("MAX_THREADS", 2))

router = Router()
ocr_semaphore = asyncio.Semaphore(MAX_THREADS)
ocr_service = OCREngine(tesseract_cmd=TESSERACT_CMD)
MESSAGES = {
    "ru": {
        "HELLO": "Привет! Я OCR-бот.\nОтправь мне изображение (фото или документ), и я извлеку из него текст.\nЯ понимаю разные языки.",
        "IN_QUEUE": "Ожидает обработку...",
        "PROCESSING": "Обрабатываю изображение...",
        "TOO_LONG": "Текст слишком длинный, отправляю файлом.",
        "RESULT": "<b>Результат:</b>\n\n",
        "ERROR": "Ошибка: ",
        "SEND_IMAGE": "Пожалуйста, отправьте файл изображения.",
        "TIMEOUT": "Время обработки изображения было слишком долгое. Попробуйте позже.",
    },
    "en": {
        "HELLO": "Hello! I'm OCR-bot.\nSend to me image (photo or document) and i extract text from it.\nI understand multiple languages.",
        "IN_QUEUE": "Waiting processing...",
        "PROCESSING": "Processing image...",
        "TOO_LONG": "Text too long, sending file.",
        "RESULT": "<b>Result:</b>\n\n",
        "ERROR": "Error: ",
        "SEND_IMAGE": "Please, send image file.",
        "TIMEOUT": "Image processing time is too long. Try later.",
    },
}


def get_lang_code(message: Message) -> str:
    user = message.from_user
    lang = "en"
    if user and user.language_code:
        lang = user.language_code
    return lang


@router.message(CommandStart())
async def cmd_start(message: Message):
    lang = get_lang_code(message)
    text = MESSAGES.get(lang, MESSAGES["en"])

    await message.answer(text["HELLO"])


async def process_image(message: Message, bot: Bot, file_id: str):
    """Common logic for processing images from photos and documents."""
    lang = get_lang_code(message)
    text = MESSAGES.get(lang, MESSAGES["en"])

    status_msg = await message.answer(text["IN_QUEUE"])

    try:
        async with ocr_semaphore:
            file_info = await bot.get_file(file_id)
            file_bytes = io.BytesIO()
            await bot.download_file(file_info.file_path, file_bytes)

            await status_msg.edit_text(text["PROCESSING"])

            result_text = asyncio.to_thread(
                ocr_service.recognize, file_bytes.getvalue()
            )

        if len(result_text) >= 4000:
            result_file = BufferedInputFile(
                result_text.encode("utf-8"), filename="ocr_result.txt"
            )
            await message.reply_document(result_file, caption=text["TOO_LONG"])
        else:
            await message.reply(
                text["RESULT"] + html.escape(result_text), parse_mode=ParseMode.HTML
            )
    except RuntimeError:
        await message.reply(text["TIMEOUT"], parse_mode=ParseMode.HTML)

    except Exception as e:
        logging.exception(f"Error processing image {e}")
        await message.reply(text["ERROR"] + str(e))
    finally:
        await status_msg.delete()


@router.message(F.content_type == ContentType.PHOTO)
async def handle_photo(message: Message, bot: Bot):
    photo = message.photo[-1]
    await process_image(message, bot, photo.file_id)


@router.message(F.content_type == ContentType.DOCUMENT)
async def handle_document(message: Message, bot: Bot):
    lang = get_lang_code(message)
    text = MESSAGES.get(lang, MESSAGES["en"])

    if not message.document.mime_type or not message.document.mime_type.startswith(
        "image/"
    ):
        await message.reply(text["SEND_IMAGE"])
        return

    await process_image(message, bot, message.document.file_id)


async def main():
    BOT_TOKEN = os.getenv("BOT_TOKEN")
    logging.basicConfig(level=logging.INFO)

    if not BOT_TOKEN:
        logging.error("BOT_TOKEN is not set. Please check your .env file.")
        sys.exit(1)
        return

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
