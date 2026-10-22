from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from .common import base_result


def parse(path: Path, *, allow_ocr: bool, language_hints: list[str]) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return base_result(path, method="json_structured", text=json.dumps(payload, ensure_ascii=False, indent=2))
    root = ElementTree.fromstring(path.read_text(encoding="utf-8"))
    rows = []
    for element in root.iter():
        if element.text and element.text.strip():
            rows.append(f"{element.tag}: {element.text.strip()}")
    return base_result(path, method="xml_structured", text="\n".join(rows))
