from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import base_result


def parse(path: Path, *, allow_ocr: bool, language_hints: list[str]) -> dict[str, Any]:
    method = "text_markdown" if path.suffix.lower() in {".md", ".markdown"} else "plain_text"
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    return base_result(path, method=method, text=text)
