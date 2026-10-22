from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def text_blocks(text: str, *, evidence_prefix: str, page: int | None = None) -> list[dict[str, Any]]:
    blocks = []
    for index, line in enumerate(line for line in text.splitlines() if line.strip()):
        evidence_id = f"{evidence_prefix}.block.{index + 1}"
        blocks.append(
            {
                "id": f"block_{len(blocks) + 1}",
                "kind": "text",
                "text": line.strip(),
                "evidence_ids": [evidence_id],
                "page": page,
            }
        )
    return blocks


def key_value_candidates(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for block in blocks:
        text = str(block.get("text") or "")
        if ":" not in text and "：" not in text:
            continue
        key, value = re.split(r"[:：]", text, maxsplit=1)
        key = key.strip()
        value = value.strip()
        if key and value:
            candidates.append(
                {
                    "key": key,
                    "value": value,
                    "confidence": 0.6,
                    "evidence_ids": list(block.get("evidence_ids") or []),
                }
            )
    return candidates


def base_result(path: Path, *, method: str, text: str = "") -> dict[str, Any]:
    blocks = text_blocks(text, evidence_prefix=method, page=1)
    return {
        "pages": [{"page_number": 1, "width": None, "height": None, "rotation": 0}],
        "blocks": blocks,
        "tables": [],
        "key_value_candidates": key_value_candidates(blocks),
        "barcodes": [],
        "qr_codes": [],
        "evidence": [
            {
                "id": evidence_id,
                "source": path.name,
                "locator": {"kind": method, "page": block.get("page")},
            }
            for block in blocks
            for evidence_id in block.get("evidence_ids", [])
        ],
        "parser": {"method": method, "version": "document_extraction.local.v1"},
        "provider_artifacts": [],
    }
