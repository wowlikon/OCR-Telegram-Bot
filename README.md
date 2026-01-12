# OCR Telegram Bot

Telegram-бот для распознавания текста на изображениях с использованием Tesseract OCR и продвинутой обработкой изображений.

## Возможности

- Распознавание текста с фотографий и документов
- Поддержка русского и английского языков
- Мульти-пайплайн обработка с автовыбором лучшего результата
- Автоматическое определение и исправление ориентации
- Отправка длинных результатов файлом

## Технологии обработки

### Предобработка изображений

| Метод | Описание |
|-------|----------|
| **CLAHE** | Адаптивное выравнивание гистограммы для улучшения контраста |
| **Denoise** | Удаление шума (fastNlMeansDenoising) |
| **Deskew** | Выравнивание наклона текста |
| **Auto-rotate** | Автоповорот на 90°/180°/270° через OSD |

### Методы бинаризации

| Метод | Лучше всего для |
|-------|-----------------|
| **Otsu** | Документы с равномерным освещением |
| **Adaptive** | Фото с тенями и неравномерным светом |
| **Sauvola** | Исторические документы, низкий контраст |

### Постобработка

- Исправление типичных OCR-ошибок (0→О, 1→l)
- Склейка переносов слов
- Удаление мусорных строк
- Нормализация пробелов

## Установка

### 1. Tesseract OCR

**Windows:**
```
https://github.com/UB-Mannheim/tesseract/wiki
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-rus
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

### 2. Python-зависимости

```bash
git clone <repository-url>
cd tesseract

python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### 3. Конфигурация

```bash
cp .env.example .env
```

```env
BOT_TOKEN=токен_от_@BotFather
TESSERACT_CMD=/usr/bin/tesseract
```

## Запуск

```bash
python main.py
```

## API движка

```python
from services import OCREngine

engine = OCREngine(langs='rus+eng')

text = engine.recognize(image_bytes)

detailed = engine.recognize_detailed(image_bytes)
# {
#   'results': {
#     'standard': {'text': '...', 'confidence': 92.5},
#     'adaptive': {'text': '...', 'confidence': 88.3},
#     ...
#   },
#   'best_method': 'standard',
#   'best_text': '...'
# }
```

## Структура

```
tesseract/
├── main.py
├── services/
│   ├── __init__.py
│   └── ocr.py
├── requirements.txt
├── .env.example
└── README.md
```

## Автор

Канал: [@mcodeg](https://t.me/mcodeg)
