#!/usr/bin/env python3
"""Document tools for local PDF reading."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from tools.registry import registry

logger = logging.getLogger(__name__)

_MAX_CHARS = 15_000
_MIN_TEXT_PER_PAGE = 50


def _parse_pages(pages_str: str, total: int) -> list[int]:
    """Parse a page-range string into zero-based page indices."""
    result: set[int] = set()
    for part in pages_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            try:
                s = max(int(start.strip()) - 1, 0)
                e = min(int(end.strip()), total)
            except ValueError:
                continue
            result.update(range(s, e))
            continue
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < total:
                result.add(idx)
    return sorted(result)


def _ocr_requested() -> bool:
    raw = os.getenv("HERMES_ENABLE_PDF_OCR", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def read_document(file_path: str, pages: str = "") -> str:
    """Extract text from a PDF document.

    OCR is feature-flagged for later enablement but is not implemented here.
    """
    path = Path(file_path).expanduser()
    if not path.exists():
        return json.dumps({"status": "error", "error": f"File not found: {file_path}"}, ensure_ascii=False)
    if path.suffix.lower() != ".pdf":
        return json.dumps({"status": "error", "error": f"Only PDF supported, got: {path.suffix}"}, ensure_ascii=False)

    try:
        import pypdfium2 as pdfium
    except Exception as exc:
        return json.dumps({"status": "error", "error": f"PDF support unavailable: {exc}"}, ensure_ascii=False)

    try:
        doc = pdfium.PdfDocument(str(path))
        total_pages = len(doc)
        target_pages = _parse_pages(pages, total_pages) if pages.strip() else list(range(total_pages))

        texts: list[str] = []
        low_text_pages = 0
        for i in target_pages:
            if not (0 <= i < total_pages):
                continue
            page = doc[i]
            text = page.get_textpage().get_text_range().strip()
            if len(text) < _MIN_TEXT_PER_PAGE:
                low_text_pages += 1
            if text:
                texts.append(f"--- Page {i + 1} ---\n{text}")

        doc.close()
        full_text = "\n\n".join(texts)
        char_count = len(full_text)
        truncated = False
        if char_count > _MAX_CHARS:
            full_text = full_text[:_MAX_CHARS] + f"\n\n... (truncated, total {char_count} chars)"
            truncated = True

        ocr_enabled = _ocr_requested()
        return json.dumps(
            {
                "status": "ok",
                "file": path.name,
                "file_path": str(path.resolve()),
                "total_pages": total_pages,
                "pages_read": len(target_pages),
                "char_count": char_count,
                "truncated": truncated,
                "low_text_pages": low_text_pages,
                "ocr_requested": ocr_enabled,
                "ocr_available": False,
                "ocr_note": (
                    "OCR is feature-flagged but not installed in this build."
                    if ocr_enabled and low_text_pages
                    else ""
                ),
                "text": full_text,
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        logger.exception("read_document tool error")
        return json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False)


READ_DOCUMENT_SCHEMA = {
    "name": "read_document",
    "description": (
        "Read a local PDF document and extract text from text-based pages. "
        "Use this for uploaded filings, reports, and research PDFs. "
        "Accepts an absolute file path and optional page ranges."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute or workspace-relative path to a local PDF file.",
            },
            "pages": {
                "type": "string",
                "description": "Optional page range like '1-5', '3,7,9-12'. Leave empty for all pages.",
                "default": "",
            },
        },
        "required": ["file_path"],
    },
}


def _read_document(args: dict, **_) -> str:
    return read_document(args["file_path"], args.get("pages", ""))


registry.register(
    name="read_document",
    toolset="documents",
    schema=READ_DOCUMENT_SCHEMA,
    handler=_read_document,
    description=READ_DOCUMENT_SCHEMA["description"],
    emoji="📄",
)
