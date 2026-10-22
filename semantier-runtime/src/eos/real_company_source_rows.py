from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from eos import company_dataset_import_store, real_company_setup_store


PARSER_PROFILE_VERSION = "real_company_source_parser.v1"
FIELD_MAPPING_PROFILE_VERSION = "real_company_field_mapping.v1"


def reconcile_promoted_imports(
    *,
    organization_id: str,
    requested_by: str,
    dataset_version_id: str | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    active = _active_dataset(organization_id, dataset_version_id)
    if active is None:
        return {
            "ok": True,
            "setup_run": None,
            "dataset_version_id": dataset_version_id,
            "created_source_row_count": 0,
            "existing_source_row_count": 0,
            "blocking_reasons": ["ACTIVE_DATASET_VERSION_REQUIRED"],
        }
    dataset_version_id = str(active["dataset_version_id"])
    run = real_company_setup_store.create_or_get_setup_run(
        organization_id=organization_id,
        dataset_version_id=dataset_version_id,
        requested_by=requested_by,
        idempotency_key=idempotency_key or f"reconcile:{dataset_version_id}",
        stage="reconcile_promoted_imports",
    )
    if run.get("ok") is False:
        return {
            "ok": False,
            "setup_run": run,
            "dataset_version_id": dataset_version_id,
            "created_source_row_count": 0,
            "existing_source_row_count": 0,
            "blocking_reasons": run.get("blocking_reasons", ["SETUP_RUN_LOCKED"]),
        }
    if run.get("status") in real_company_setup_store.TERMINAL_RUN_STATUSES:
        rows = real_company_setup_store.list_source_rows(
            organization_id=organization_id,
            dataset_version_id=dataset_version_id,
        )
        return {
            "ok": True,
            "setup_run": run,
            "dataset_version_id": dataset_version_id,
            "created_source_row_count": 0,
            "existing_source_row_count": len(rows),
            "blocked_files": [],
            "blocking_reasons": [],
        }

    import_record = company_dataset_import_store.get_import(
        organization_id=organization_id,
        import_id=str(active["import_id"]),
    )
    parser_profile = import_record.get("parser_profile") if import_record else {}
    parser_profile_hash = str(
        active.get("artifact_hash_manifest", {}).get("parser_profile_hash")
        or (import_record or {}).get("parser_profile_hash")
        or _hash_json({"version": PARSER_PROFILE_VERSION, "parser_profile": parser_profile})
    )
    field_mapping_profile_hash = _hash_json(
        {"version": FIELD_MAPPING_PROFILE_VERSION, "mode": "ascii_column_index"}
    )

    created = 0
    existing = 0
    blocked_files: list[dict[str, Any]] = []
    for file_manifest in _manifest_files(active):
        result = _normalize_manifest_file(
            organization_id=organization_id,
            dataset_version_id=dataset_version_id,
            import_id=str(active["import_id"]),
            file_manifest=file_manifest,
            parser_profile=parser_profile or {},
            parser_profile_hash=parser_profile_hash,
            field_mapping_profile_hash=field_mapping_profile_hash,
        )
        created += result["created_source_row_count"]
        existing += result["existing_source_row_count"]
        blocked_files.extend(result["blocked_files"])

    real_company_setup_store.append_setup_event(
        setup_run_id=str(run["setup_run_id"]),
        organization_id=organization_id,
        dataset_version_id=dataset_version_id,
        stage="reconcile_promoted_imports",
        event_type="source_rows_reconciled",
        event_status="ready" if not blocked_files else "blocked",
        details={
            "created_source_row_count": created,
            "existing_source_row_count": existing,
            "blocked_files": blocked_files,
            "parser_profile_hash": parser_profile_hash,
            "field_mapping_profile_hash": field_mapping_profile_hash,
        },
    )
    real_company_setup_store.transition_setup_run(
        setup_run_id=str(run["setup_run_id"]),
        status="completed" if not blocked_files else "blocked",
        current_stage="reconcile_promoted_imports",
        blocking_reason="SOURCE_FILE_BLOCKED" if blocked_files else None,
    )
    return {
        "ok": True,
        "setup_run": run,
        "dataset_version_id": dataset_version_id,
        "created_source_row_count": created,
        "existing_source_row_count": existing,
        "blocked_files": blocked_files,
        "blocking_reasons": ["SOURCE_FILE_BLOCKED"] if blocked_files else [],
    }


def preview_source_rows(
    *,
    organization_id: str,
    dataset_version_id: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    active = _active_dataset(organization_id, dataset_version_id)
    if active is None:
        return {
            "ok": False,
            "dataset_version_id": dataset_version_id,
            "rows": [],
            "blocking_reasons": ["ACTIVE_DATASET_VERSION_REQUIRED"],
        }
    rows = real_company_setup_store.list_source_rows(
        organization_id=organization_id,
        dataset_version_id=str(active["dataset_version_id"]),
        limit=limit,
    )
    return {
        "ok": True,
        "dataset_version_id": str(active["dataset_version_id"]),
        "rows": [
            {
                "source_row_hash": row["source_row_hash"],
                "row_type": row.get("row_type"),
                "classification_status": row.get("classification_status"),
                "row": row.get("row", {}),
                "metadata": row.get("metadata", {}),
            }
            for row in rows
        ],
        "blocking_reasons": [],
    }


def _normalize_manifest_file(
    *,
    organization_id: str,
    dataset_version_id: str,
    import_id: str,
    file_manifest: dict[str, Any],
    parser_profile: dict[str, Any],
    parser_profile_hash: str,
    field_mapping_profile_hash: str,
) -> dict[str, Any]:
    file_id = str(file_manifest.get("file_id") or "")
    artifact_path = Path(str(file_manifest.get("artifact_path") or ""))
    source_file_hash = str(file_manifest.get("source_file_hash") or "")
    if not artifact_path.exists():
        row = _blocked_file_row(
            reason_code="SOURCE_ARTIFACT_MISSING",
            file_manifest=file_manifest,
        )
        return _store_rows(
            organization_id=organization_id,
            dataset_version_id=dataset_version_id,
            import_id=import_id,
            file_id=file_id,
            source_file_hash=source_file_hash,
            parser_profile_hash=parser_profile_hash,
            field_mapping_profile_hash=field_mapping_profile_hash,
            rows=[row],
            blocked_files=[{"file_id": file_id, "reason_code": "SOURCE_ARTIFACT_MISSING"}],
        )
    content = artifact_path.read_bytes()
    actual_hash = hashlib.sha256(content).hexdigest()
    if source_file_hash and actual_hash != source_file_hash:
        row = _blocked_file_row(
            reason_code="SOURCE_FILE_HASH_MISMATCH",
            file_manifest=file_manifest,
        )
        return _store_rows(
            organization_id=organization_id,
            dataset_version_id=dataset_version_id,
            import_id=import_id,
            file_id=file_id,
            source_file_hash=source_file_hash,
            parser_profile_hash=parser_profile_hash,
            field_mapping_profile_hash=field_mapping_profile_hash,
            rows=[row],
            blocked_files=[{"file_id": file_id, "reason_code": "SOURCE_FILE_HASH_MISMATCH"}],
        )

    suffix = artifact_path.suffix.lower()
    if suffix == ".csv":
        rows = _normalize_csv(content, parser_profile)
    elif suffix == ".xlsx":
        rows = _normalize_xlsx(content, parser_profile)
    elif suffix == ".xls":
        rows = _normalize_xls(content, parser_profile)
    else:
        rows = [
            _blocked_file_row(
                reason_code="UNSUPPORTED_SOURCE_FILE_TYPE",
                file_manifest=file_manifest,
            )
        ]
    return _store_rows(
        organization_id=organization_id,
        dataset_version_id=dataset_version_id,
        import_id=import_id,
        file_id=file_id,
        source_file_hash=actual_hash,
        parser_profile_hash=parser_profile_hash,
        field_mapping_profile_hash=field_mapping_profile_hash,
        rows=rows,
        blocked_files=[
            {"file_id": file_id, "reason_code": "UNSUPPORTED_SOURCE_FILE_TYPE"}
        ]
        if suffix not in {".csv", ".xls", ".xlsx"}
        else [],
    )


def _store_rows(
    *,
    organization_id: str,
    dataset_version_id: str,
    import_id: str,
    file_id: str,
    source_file_hash: str,
    parser_profile_hash: str,
    field_mapping_profile_hash: str,
    rows: list[dict[str, Any]],
    blocked_files: list[dict[str, Any]],
) -> dict[str, Any]:
    created = 0
    existing = 0
    for row in rows:
        source_row_hash = _hash_json(
            {
                "organization_id": organization_id,
                "dataset_version_id": dataset_version_id,
                "import_id": import_id,
                "file_id": file_id,
                "row_index": row["metadata"].get("row_index"),
                "row": row["row"],
                "source_file_hash": source_file_hash,
                "parser_profile_hash": parser_profile_hash,
            }
        )
        prior = real_company_setup_store.get_source_row(
            organization_id=organization_id,
            dataset_version_id=dataset_version_id,
            source_row_hash=source_row_hash,
        )
        real_company_setup_store.upsert_source_row(
            organization_id=organization_id,
            dataset_version_id=dataset_version_id,
            import_id=import_id,
            file_id=file_id,
            source_row_hash=source_row_hash,
            source_file_hash=source_file_hash,
            parser_profile_hash=parser_profile_hash,
            field_mapping_profile_hash=field_mapping_profile_hash,
            classification_status=row["classification_status"],
            row_type=row["row_type"],
            row=row["row"],
            metadata=row["metadata"],
        )
        if prior is None:
            created += 1
        else:
            existing += 1
    return {
        "created_source_row_count": created,
        "existing_source_row_count": existing,
        "blocked_files": blocked_files,
    }


def _normalize_csv(content: bytes, parser_profile: dict[str, Any]) -> list[dict[str, Any]]:
    encoding = str(parser_profile.get("encoding") or "utf-8")
    text = _decode_text(content, encoding)
    raw_rows = list(csv.reader(io.StringIO(text)))
    return _normalize_tabular_rows(raw_rows, parser_profile, sheet_name="Sheet1")


def _normalize_xlsx(content: bytes, parser_profile: dict[str, Any]) -> list[dict[str, Any]]:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        shared_strings = _xlsx_shared_strings(archive)
        sheet_name, sheet_path = _first_xlsx_sheet(archive)
        root = ElementTree.fromstring(archive.read(sheet_path))
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    raw_rows: list[list[str]] = []
    for row in root.findall(".//x:sheetData/x:row", namespace):
        values: list[str] = []
        for cell in row.findall("x:c", namespace):
            ref = cell.attrib.get("r", "")
            column_index = _excel_column_index(ref)
            while len(values) < column_index:
                values.append("")
            value_node = cell.find("x:v", namespace)
            value = value_node.text if value_node is not None else ""
            if cell.attrib.get("t") == "s" and value:
                try:
                    value = shared_strings[int(value)]
                except (IndexError, ValueError):
                    pass
            values.append(str(value or "").strip())
        raw_rows.append(values)
    return _normalize_tabular_rows(raw_rows, parser_profile, sheet_name=sheet_name)


def _normalize_xls(content: bytes, parser_profile: dict[str, Any]) -> list[dict[str, Any]]:
    import xlrd

    workbook = xlrd.open_workbook(file_contents=content)
    sheet_names = _selected_sheet_names(workbook.sheet_names(), parser_profile)
    rows: list[dict[str, Any]] = []
    for sheet_name in sheet_names:
        sheet = workbook.sheet_by_name(sheet_name)
        raw_rows: list[list[str]] = []
        for row_index in range(sheet.nrows):
            raw_rows.append([_xls_cell_to_text(sheet.cell(row_index, col_index)) for col_index in range(sheet.ncols)])
        rows.extend(_normalize_tabular_rows(raw_rows, parser_profile, sheet_name=sheet_name))
    return rows


def _selected_sheet_names(all_sheet_names: list[str], parser_profile: dict[str, Any]) -> list[str]:
    if bool(parser_profile.get("all_sheets")):
        return all_sheet_names
    requested = [
        str(item).strip()
        for item in (parser_profile.get("selected_sheets") or [])
        if str(item).strip()
    ]
    selected = [name for name in requested if name in all_sheet_names]
    return selected or all_sheet_names[:1]


def _xls_cell_to_text(cell: Any) -> str:
    import xlrd

    if cell.ctype == xlrd.XL_CELL_EMPTY:
        return ""
    if cell.ctype == xlrd.XL_CELL_NUMBER:
        value = float(cell.value)
        return str(int(value)) if value.is_integer() else str(value)
    return str(cell.value or "").strip()


def _normalize_tabular_rows(
    raw_rows: list[list[str]],
    parser_profile: dict[str, Any],
    *,
    sheet_name: str,
) -> list[dict[str, Any]]:
    if not raw_rows:
        return []
    header_rows = parser_profile.get("header_rows") or {}
    header_row = int(header_rows.get(sheet_name) or header_rows.get("Sheet1") or 1) if isinstance(header_rows, dict) else 1
    header_index = max(header_row - 1, 0)
    headers = [
        str(value or "").strip() or f"column_{index + 1}"
        for index, value in enumerate(raw_rows[header_index] if header_index < len(raw_rows) else [])
    ]
    normalized: list[dict[str, Any]] = []
    for row_index, raw in enumerate(raw_rows[header_index + 1 :], start=header_index + 2):
        if not any(str(value or "").strip() for value in raw):
            continue
        row_payload: dict[str, str] = {}
        columns: list[dict[str, str]] = []
        for index, header in enumerate(headers):
            field_key = f"column_{index + 1}"
            value = str(raw[index] if index < len(raw) else "").strip()
            row_payload[field_key] = value
            columns.append(
                {
                    "field_key": field_key,
                    "source_column_label": header,
                }
            )
        classification = _classify_row(row_payload, columns)
        normalized.append(
            {
                "row": row_payload,
                "row_type": classification["row_type"],
                "classification_status": classification["classification_status"],
                "metadata": {
                    "row_index": row_index,
                    "sheet_name": sheet_name,
                    "columns": columns,
                    "classification": classification,
                },
            }
        )
    return normalized


def _classify_row(row: dict[str, str], columns: list[dict[str, str]]) -> dict[str, Any]:
    amount_key = _find_amount_key(row, columns)
    if amount_key is None:
        return {
            "classification_status": "blocked",
            "row_type": "unsupported",
            "blocking_reasons": ["AMOUNT_REQUIRED"],
        }
    return {
        "classification_status": "classified",
        "row_type": "economic_event",
        "amount_field_key": amount_key,
        "blocking_reasons": [],
    }


def _find_amount_key(row: dict[str, str], columns: list[dict[str, str]]) -> str | None:
    labeled_keys = {
        item["field_key"]: str(item.get("source_column_label") or "").casefold()
        for item in columns
    }
    preferred = [
        key
        for key, label in labeled_keys.items()
        if any(token in label for token in ("amount", "金额", "借方", "贷方", "收入", "支出"))
    ]
    for key in [*preferred, *row.keys()]:
        if _parse_amount(row.get(key)) is not None:
            return key
    return None


def _parse_amount(value: Any) -> float | None:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return None
    text = re.sub(r"^[^\d.-]+", "", text)
    text = re.sub(r"[^\d.-]+$", "", text)
    try:
        return float(text)
    except ValueError:
        return None


def _blocked_file_row(*, reason_code: str, file_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "row": {
            "column_1": str(file_manifest.get("original_filename") or file_manifest.get("file_id") or ""),
        },
        "row_type": "unsupported",
        "classification_status": "blocked",
        "metadata": {
            "row_index": 0,
            "columns": [{"field_key": "column_1", "source_column_label": "source_file"}],
            "classification": {
                "classification_status": "blocked",
                "row_type": "unsupported",
                "blocking_reasons": [reason_code],
            },
        },
    }


def _decode_text(content: bytes, encoding: str) -> str:
    for candidate in [encoding, "utf-8-sig", "utf-8", "gb18030", "gbk", "big5"]:
        if not candidate or candidate == "auto":
            continue
        try:
            return content.decode(candidate)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    return [
        "".join(node.text or "" for node in item.findall(".//x:t", namespace))
        for item in root.findall("x:si", namespace)
    ]


def _first_xlsx_sheet(archive: zipfile.ZipFile) -> tuple[str, str]:
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    rels = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    rel_targets = {
        rel.attrib.get("Id"): rel.attrib.get("Target", "")
        for rel in rels.findall(f"{{{rel_ns}}}Relationship")
    }
    sheet = workbook.find(f".//{{{main_ns}}}sheet")
    if sheet is None:
        raise ValueError("xlsx workbook has no sheets")
    name = sheet.attrib.get("name", "Sheet1")
    relationship_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
    target = rel_targets.get(relationship_id, "worksheets/sheet1.xml")
    return name, "xl/" + target.lstrip("/")


def _excel_column_index(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha()).upper()
    if not letters:
        return 0
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return max(index - 1, 0)


def _manifest_files(active_dataset: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = active_dataset.get("artifact_hash_manifest") or {}
    return [item for item in manifest.get("files") or [] if isinstance(item, dict)]


def _active_dataset(
    organization_id: str,
    dataset_version_id: str | None,
) -> dict[str, Any] | None:
    if dataset_version_id:
        return company_dataset_import_store.get_dataset_version(
            organization_id=organization_id,
            dataset_version_id=dataset_version_id,
        )
    return company_dataset_import_store.get_active_dataset_version(
        organization_id=organization_id,
    )


def _hash_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
