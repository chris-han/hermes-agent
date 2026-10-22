from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return ["".join(t.text or "" for t in si.iter(f"{_NS}t")) for si in root.findall(f"{_NS}si")]


def _sheet_names(archive: zipfile.ZipFile) -> dict[str, str]:
    try:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    except KeyError:
        return {"rId1": "Sheet1"}
    return {
        str(sheet.attrib.get(f"{_REL}id") or f"rId{idx}"): str(sheet.attrib.get("name") or f"Sheet{idx}")
        for idx, sheet in enumerate(workbook.findall(f"{_NS}sheets/{_NS}sheet"), start=1)
    }


def _cell_text(cell: ElementTree.Element, shared: list[str]) -> str:
    if cell.attrib.get("t") == "inlineStr":
        return "".join(t.text or "" for t in cell.iter(f"{_NS}t"))
    value = cell.find(f"{_NS}v")
    if value is None or value.text is None:
        return ""
    if cell.attrib.get("t") == "s":
        try:
            return shared[int(value.text)]
        except Exception:
            return ""
    return value.text


def parse(path: Path, *, allow_ocr: bool, language_hints: list[str]) -> dict[str, Any]:
    tables: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        shared = _shared_strings(archive)
        sheet_names = _sheet_names(archive)
        sheet_files = sorted(name for name in archive.namelist() if name.startswith("xl/worksheets/sheet") and name.endswith(".xml"))
        for sheet_index, sheet_file in enumerate(sheet_files, start=1):
            sheet_name = sheet_names.get(f"rId{sheet_index}", f"Sheet{sheet_index}")
            root = ElementTree.fromstring(archive.read(sheet_file))
            rows = []
            for row in root.findall(f"{_NS}sheetData/{_NS}row"):
                row_number = int(row.attrib.get("r") or len(rows) + 1)
                cells = []
                for cell in row.findall(f"{_NS}c"):
                    cell_ref = str(cell.attrib.get("r") or "")
                    text = _cell_text(cell, shared)
                    if not text:
                        continue
                    evidence_id = f"xlsx.s{sheet_index}.{cell_ref}"
                    cells.append({"cell": cell_ref, "text": text, "evidence_ids": [evidence_id]})
                    blocks.append({"id": f"block_{len(blocks) + 1}", "kind": "cell", "text": text, "evidence_ids": [evidence_id]})
                    evidence.append({"id": evidence_id, "source": path.name, "locator": {"kind": "xlsx_cell", "sheet": sheet_name, "cell": cell_ref}})
                if cells:
                    rows.append({"row": row_number, "cells": cells})
            tables.append({"id": f"table_{sheet_index}", "kind": "xlsx_sheet", "sheet": sheet_name, "rows": rows, "evidence_ids": [f"xlsx.s{sheet_index}"]})
    return {
        "pages": [],
        "blocks": blocks,
        "tables": tables,
        "key_value_candidates": [],
        "barcodes": [],
        "qr_codes": [],
        "evidence": evidence,
        "parser": {"method": "xlsx_ooxml", "version": "document_extraction.local.v1"},
        "provider_artifacts": [],
    }
