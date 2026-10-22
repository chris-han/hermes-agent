from __future__ import annotations

from pathlib import Path
from typing import Any

PROVIDER_NAME = "local_optional_ocr"
MODEL_VERSION = "pytesseract-or-unavailable.v1"


def extract_image_text(path: Path, *, language_hints: list[str] | None = None) -> dict[str, Any]:
    langs = language_hints or []
    try:
        import pytesseract
        from PIL import Image
    except Exception:
        return {
            "status": "unavailable",
            "provider": PROVIDER_NAME,
            "model_version": MODEL_VERSION,
            "text": "",
            "tokens": [],
            "languages": langs,
        }
    lang = "chi_sim+eng" if any(lang.lower().startswith("zh") for lang in langs) else "eng"
    image = Image.open(path)
    text = pytesseract.image_to_string(image, lang=lang)
    tokens: list[dict[str, Any]] = []
    try:
        data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
        for index, value in enumerate(data.get("text", [])):
            token = str(value or "").strip()
            if not token:
                continue
            tokens.append(
                {
                    "text": token,
                    "confidence": float(data.get("conf", [0])[index]),
                    "bbox": [
                        float(data.get("left", [0])[index]),
                        float(data.get("top", [0])[index]),
                        float(data.get("width", [0])[index]),
                        float(data.get("height", [0])[index]),
                    ],
                }
            )
    except Exception:
        tokens = []
    return {
        "status": "ok",
        "provider": PROVIDER_NAME,
        "model_version": MODEL_VERSION,
        "text": text,
        "tokens": tokens,
        "languages": langs,
    }
