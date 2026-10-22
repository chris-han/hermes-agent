from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import key_value_candidates, text_blocks
from ..providers import ocr


def parse(path: Path, *, allow_ocr: bool, language_hints: list[str]) -> dict[str, Any]:
    if not allow_ocr:
        text = ""
        ocr_result = {"status": "disabled", "provider": ocr.PROVIDER_NAME, "model_version": ocr.MODEL_VERSION, "tokens": []}
    else:
        ocr_result = ocr.extract_image_text(path, language_hints=language_hints)
        text = str(ocr_result.get("text") or "")
    blocks = text_blocks(text, evidence_prefix="ocr.p1", page=1)
    evidence = []
    for index, block in enumerate(blocks):
        token = (ocr_result.get("tokens") or [None] * len(blocks))[index] if index < len(ocr_result.get("tokens") or []) else None
        locator: dict[str, Any] = {"kind": "image_ocr", "page": 1}
        if isinstance(token, dict) and token.get("bbox"):
            locator["bbox"] = token["bbox"]
            block["confidence"] = token.get("confidence")
        for evidence_id in block.get("evidence_ids", []):
            evidence.append({"id": evidence_id, "source": path.name, "locator": locator})
    return {
        "pages": [{"page_number": 1, "width": None, "height": None, "rotation": 0}],
        "blocks": blocks,
        "tables": [],
        "key_value_candidates": key_value_candidates(blocks),
        "barcodes": [],
        "qr_codes": [],
        "evidence": evidence,
        "parser": {"method": "image_ocr", "version": "document_extraction.local.v1"},
        "provider_artifacts": [ocr_result] if ocr_result.get("status") == "ok" else [],
    }
