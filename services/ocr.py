import cv2
import numpy as np
import pytesseract
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class OCRResult:
    text: str
    confidence: float


class OCREngine:
    def __init__(self, tesseract_cmd: str | None = None, langs: str = "rus+eng"):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        self.langs = langs
        self.configs = [
            r"--oem 3 --psm 6",
            r"--oem 3 --psm 3",
            r"--oem 3 --psm 4",
            r"--oem 1 --psm 6",
        ]

        self.ocr_corrections = {
            r"\b0\b": "О",
            r"(?<=[А-Яа-яЁё])0(?=[А-Яа-яЁё])": "о",
            r"(?<=[А-Яа-яЁё])1(?=[А-Яа-яЁё])": "л",
            r"(?<=[a-zA-Z])0(?=[a-zA-Z])": "o",
            r"(?<=[a-zA-Z])1(?=[a-zA-Z])": "l",
            r"\|": "I",
            r",,": '"',
            r"''": '"',
            r"«\s*": "«",
            r"\s*»": "»",
        }

    def _decode_image(self, image_bytes: bytes) -> Optional[np.ndarray]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image

    def _resize_if_needed(
        self, image: np.ndarray, min_height: int = 1000
    ) -> np.ndarray:
        h, w = image.shape[:2]
        if h < min_height:
            scale = min_height / h
            image = cv2.resize(
                image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
        elif h > 4000:
            scale = 4000 / h
            image = cv2.resize(
                image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        return image

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(
            image, h=10, templateWindowSize=7, searchWindowSize=21
        )

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        coords = np.column_stack(np.where(image < 128))
        if len(coords) < 100:
            return image

        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90

        if abs(angle) < 0.5:
            return image

        h, w = image.shape
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated

    def _auto_rotate(self, image: np.ndarray) -> np.ndarray:
        try:
            osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
            rotation = osd.get("rotate", 0)
            if rotation == 90:
                return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                return cv2.rotate(image, cv2.ROTATE_180)
            elif rotation == 270:
                return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        except pytesseract.TesseractError:
            pass
        return image

    def _binarize_otsu(self, image: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _binarize_adaptive(self, image: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

    def _binarize_sauvola(
        self, image: np.ndarray, window_size: int = 25, k: float = 0.2
    ) -> np.ndarray:
        mean = cv2.blur(image, (window_size, window_size))
        mean_sq = cv2.blur(image.astype(np.float32) ** 2, (window_size, window_size))
        std = np.sqrt(np.maximum(mean_sq - mean.astype(np.float32) ** 2, 0))
        threshold = mean * (1 + k * (std / 128 - 1))
        binary = (image > threshold).astype(np.uint8) * 255
        return binary

    def _morphology_clean(self, image: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return image

    def _remove_borders(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        border = int(min(h, w) * 0.02)
        if border > 5:
            mask = np.zeros_like(image)
            mask[border:-border, border:-border] = 255
            contours, _ = cv2.findContours(
                cv2.bitwise_not(image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if (
                    x < border
                    or y < border
                    or x + cw > w - border
                    or y + ch > h - border
                ):
                    if cw > w * 0.8 or ch > h * 0.8:
                        cv2.drawContours(image, [cnt], -1, 255, -1)
        return image

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def _preprocess_pipeline(
        self, image: np.ndarray, method: str = "standard"
    ) -> np.ndarray:
        image = self._resize_if_needed(image)
        gray = self._to_grayscale(image)

        if method == "standard":
            gray = self._enhance_contrast(gray)
            gray = self._denoise(gray)
            gray = self._auto_rotate(gray)
            gray = self._deskew(gray)
            binary = self._binarize_otsu(gray)
            binary = self._morphology_clean(binary)

        elif method == "adaptive":
            gray = self._enhance_contrast(gray)
            gray = self._auto_rotate(gray)
            gray = self._deskew(gray)
            binary = self._binarize_adaptive(gray)
            binary = self._morphology_clean(binary)

        elif method == "sauvola":
            gray = self._auto_rotate(gray)
            gray = self._deskew(gray)
            binary = self._binarize_sauvola(gray)

        elif method == "clean":
            gray = self._denoise(gray)
            gray = self._auto_rotate(gray)
            binary = self._binarize_otsu(gray)
            binary = self._remove_borders(binary)

        else:
            gray = self._auto_rotate(gray)
            binary = self._binarize_otsu(gray)

        return binary

    def _get_confidence(self, image: np.ndarray, config: str) -> Tuple[str, float]:
        data = pytesseract.image_to_data(
            image,
            timeout=30,
            lang=self.langs,
            config=config,
            output_type=pytesseract.Output.DICT,
        )

        confidences = [int(c) for c in data["conf"] if int(c) > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        text = pytesseract.image_to_string(image, lang=self.langs, config=config)

        return text, avg_confidence

    def _postprocess(self, text: str) -> str:
        for pattern, replacement in self.ocr_corrections.items():
            text = re.sub(pattern, replacement, text)

        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        lines = []
        for line in text.split("\n"):
            line = line.strip()
            if line and not re.match(r"^[\W\d_]+$", line):
                lines.append(line)

        text = "\n".join(lines)

        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

        return text.strip()

    def _recognize_with_method(self, image: np.ndarray, method: str) -> OCRResult:
        processed = self._preprocess_pipeline(image, method)

        best_text = ""
        best_confidence = 0

        for config in self.configs:
            try:
                text, confidence = self._get_confidence(processed, config)
                if confidence > best_confidence and len(text.strip()) > 0:
                    best_confidence = confidence
                    best_text = text
            except pytesseract.TesseractError:
                continue

        return OCRResult(text=best_text, confidence=best_confidence)

    def recognize(self, image_bytes: bytes) -> str:
        try:
            image = self._decode_image(image_bytes)
            if image is None:
                return "Ошибка: не удалось декодировать изображение."

            methods = ["standard", "adaptive", "sauvola", "clean"]
            results: List[OCRResult] = []

            for method in methods:
                try:
                    result = self._recognize_with_method(image, method)
                    if result.text.strip():
                        results.append(result)
                except Exception:
                    continue

            if not results:
                return "Текст не найден или изображение слишком размытое."

            best_result = max(results, key=lambda r: (r.confidence, len(r.text)))
            clean_text = self._postprocess(best_result.text)

            if not clean_text:
                return "Текст не найден или изображение слишком размытое."

            return clean_text

        except Exception as e:
            return f"Системная ошибка: {str(e)}"

    def recognize_detailed(self, image_bytes: bytes) -> dict:
        try:
            image = self._decode_image(image_bytes)
            if image is None:
                return {"error": "Не удалось декодировать изображение"}

            methods = ["standard", "adaptive", "sauvola", "clean"]
            results = {}

            for method in methods:
                try:
                    result = self._recognize_with_method(image, method)
                    results[method] = {
                        "text": self._postprocess(result.text),
                        "confidence": round(result.confidence, 2),
                    }
                except Exception as e:
                    results[method] = {"error": str(e)}

            best_method = max(
                [(m, r) for m, r in results.items() if "text" in r and r["text"]],
                key=lambda x: (x[1]["confidence"], len(x[1]["text"])),
                default=(None, None),
            )

            return {
                "results": results,
                "best_method": best_method[0] if best_method[0] else None,
                "best_text": (
                    best_method[1]["text"]
                    if best_method[1] and "text" in best_method[1]
                    else None
                ),
            }

        except Exception as e:
            return {"error": str(e)}
