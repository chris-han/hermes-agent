from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .common import key_value_candidates, text_blocks


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.path: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.path.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if self.path:
            self.path.pop()

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)


def parse(path: Path, *, allow_ocr: bool, language_hints: list[str]) -> dict[str, Any]:
    parser = _TextParser()
    parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
    blocks = text_blocks("\n".join(parser.parts), evidence_prefix="html.dom", page=None)
    return {
        "pages": [],
        "blocks": blocks,
        "tables": [],
        "key_value_candidates": key_value_candidates(blocks),
        "barcodes": [],
        "qr_codes": [],
        "evidence": [
            {"id": eid, "source": path.name, "locator": {"kind": "html_dom", "path": None}}
            for block in blocks
            for eid in block.get("evidence_ids", [])
        ],
        "parser": {"method": "html_parser", "version": "document_extraction.local.v1"},
        "provider_artifacts": [],
    }
