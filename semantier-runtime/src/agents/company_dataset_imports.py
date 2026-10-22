from __future__ import annotations

import hashlib
import importlib
import csv
import io
import json
import mimetypes
import os
import stat
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
import uuid

from charset_normalizer import from_bytes

from agents import runtime_inventory
from eos import company_dataset_import_store as store
from eos.db import now_iso
from eos.real_company_setup_state import real_company_setup_status_payload
from runtime_paths import platform_runtime_root


PARSER_PROFILE_SCHEMA_VERSION = "company_dataset_parser_profile.v1"
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_BATCH_BYTES = 250 * 1024 * 1024
MAX_SOURCE_PATH_FILES = 200
MAX_PREVIEW_ROWS = 20
ACCEPTED_EXTENSIONS = {".xlsx", ".xls", ".csv"}
ACCEPTED_MIME_HINTS = {
    "text/csv",
    "application/csv",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}
PROMOTION_ERROR_CATEGORIES = {
    "transient_retry_safe",
    "permanent_operator_action",
    "data_quality_remap_required",
    "authorization_revoked",
    "internal_invariant_violation",
}


class CompanyDatasetImportError(RuntimeError):
    pass


class CompanyDatasetAuthorizationError(CompanyDatasetImportError):
    pass


class CompanyDatasetValidationError(CompanyDatasetImportError):
    pass


@dataclass(frozen=True)
class UploadedDatasetFile:
    filename: str
    content_type: str
    content: bytes


def _normalize_source_path(value: str) -> Path:
    raw = value.strip()
    if not raw:
        raise CompanyDatasetValidationError("source_path required")
    path = Path(raw).expanduser()
    if path.exists():
        return path
    windows_drive = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
    if windows_drive:
        drive, rest = windows_drive.groups()
        wsl_path = Path("/mnt") / drive.lower() / rest.replace("\\", "/")
        if wsl_path.exists():
            return wsl_path
    return path


def uploaded_dataset_files_from_path(source_path: str) -> list[UploadedDatasetFile]:
    path = _normalize_source_path(source_path)
    if not path.exists():
        raise CompanyDatasetValidationError("source_path does not exist")

    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in ACCEPTED_EXTENSIONS
        )
    else:
        raise CompanyDatasetValidationError("source_path must be a file or directory")

    if not candidates:
        raise CompanyDatasetValidationError(
            "source_path contains no accepted spreadsheet files"
        )
    if len(candidates) > MAX_SOURCE_PATH_FILES:
        raise CompanyDatasetValidationError(
            "source_path contains too many spreadsheet files"
        )

    files: list[UploadedDatasetFile] = []
    for candidate in candidates:
        content_type = mimetypes.guess_type(candidate.name)[0] or ""
        files.append(
            UploadedDatasetFile(
                filename=candidate.name,
                content_type=content_type,
                content=candidate.read_bytes(),
            )
        )
    return files


def _artifact_root() -> Path:
    root = platform_runtime_root() / "artifacts" / "company_dataset_imports"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _magic_ok(extension: str, content: bytes) -> bool:
    if extension == ".csv":
        return b"\x00" not in content[:4096]
    if extension == ".xlsx":
        return content.startswith(b"PK\x03\x04")
    if extension == ".xls":
        return content.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")
    return False


def _validate_file(file: UploadedDatasetFile) -> None:
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()
    if extension not in ACCEPTED_EXTENSIONS:
        raise CompanyDatasetValidationError("unsupported file extension")
    if not file.content:
        raise CompanyDatasetValidationError("empty file")
    if len(file.content) > MAX_FILE_BYTES:
        raise CompanyDatasetValidationError("file exceeds 50 MiB limit")
    if not _magic_ok(extension, file.content):
        raise CompanyDatasetValidationError("file magic bytes do not match accepted spreadsheet formats")
    guessed = mimetypes.guess_type(filename)[0]
    content_type = (file.content_type or guessed or "").split(";", 1)[0].strip().lower()
    if content_type and content_type not in ACCEPTED_MIME_HINTS:
        raise CompanyDatasetValidationError("unsupported file MIME type")


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _suggest_header_row(rows: list[list[str]]) -> int:
    best_index = 0
    best_score = -1.0
    for index, row in enumerate(rows[:MAX_PREVIEW_ROWS]):
        values = [_cell_text(value) for value in row]
        non_empty = [value for value in values if value]
        if not non_empty:
            continue
        numeric = 0
        for value in non_empty:
            try:
                float(value.replace(",", ""))
                numeric += 1
            except ValueError:
                pass
        score = len(non_empty) + (len(non_empty) - numeric) * 0.5 - numeric * 0.25
        if score > best_score:
            best_index = index
            best_score = score
    return best_index + 1


def _decode_csv_text(content: bytes) -> tuple[str, str]:
    if content.startswith(b"\xef\xbb\xbf"):
        return content.decode("utf-8-sig"), "utf-8-sig"
    if content.startswith(b"\xff\xfe") or content.startswith(b"\xfe\xff"):
        return content.decode("utf-16"), "utf-16"

    for encoding in ("utf-8", "gb18030", "gbk", "big5"):
        try:
            return content.decode(encoding), encoding
        except UnicodeDecodeError:
            continue

    detected = from_bytes(content).best()
    if detected and detected.encoding:
        encoding = detected.encoding
        try:
            return content.decode(encoding), encoding
        except (LookupError, UnicodeDecodeError):
            pass

    return content.decode("utf-8", errors="replace"), "utf-8-replacement"


def _preview_csv(file: UploadedDatasetFile) -> list[dict[str, Any]]:
    text, encoding = _decode_csv_text(file.content)
    rows = [
        [_cell_text(cell) for cell in row]
        for row in csv.reader(io.StringIO(text))
    ][:MAX_PREVIEW_ROWS]
    return [
        {
            "sheet_name": "Sheet1",
            "suggested_header_row": _suggest_header_row(rows),
            "encoding": encoding,
            "rows": rows,
        }
    ]


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        raw = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ElementTree.fromstring(raw)
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    values: list[str] = []
    for item in root.findall("x:si", namespace):
        parts = [node.text or "" for node in item.findall(".//x:t", namespace)]
        values.append("".join(parts))
    return values


def _xlsx_sheet_targets(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    rels = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    office_rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    rel_targets = {
        rel.attrib.get("Id"): rel.attrib.get("Target", "")
        for rel in rels.findall(f"{{{rel_ns}}}Relationship")
    }
    sheets: list[tuple[str, str]] = []
    for sheet in workbook.findall(f".//{{{main_ns}}}sheet"):
        name = sheet.attrib.get("name", "Sheet")
        rel_id = sheet.attrib.get(f"{{{office_rel_ns}}}id")
        target = rel_targets.get(rel_id, "")
        if not target:
            continue
        if target.startswith("/"):
            path_name = target.lstrip("/")
        else:
            path_name = f"xl/{target}"
        sheets.append((name, path_name))
    return sheets


def _xlsx_cell_value(cell: ElementTree.Element, shared_strings: list[str]) -> str:
    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    cell_type = cell.attrib.get("t")
    value_node = cell.find(f"{{{main_ns}}}v")
    if value_node is None or value_node.text is None:
        inline = cell.find(f".//{{{main_ns}}}t")
        return _cell_text(inline.text if inline is not None else "")
    raw = value_node.text
    if cell_type == "s":
        try:
            return _cell_text(shared_strings[int(raw)])
        except (ValueError, IndexError):
            return ""
    return _cell_text(raw)


def _preview_xlsx(file: UploadedDatasetFile) -> list[dict[str, Any]]:
    with zipfile.ZipFile(io.BytesIO(file.content)) as archive:
        shared_strings = _xlsx_shared_strings(archive)
        previews: list[dict[str, Any]] = []
        main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
        for sheet_name, target in _xlsx_sheet_targets(archive):
            root = ElementTree.fromstring(archive.read(target))
            rows: list[list[str]] = []
            for row in root.findall(f".//{{{main_ns}}}sheetData/{{{main_ns}}}row"):
                values = [
                    _xlsx_cell_value(cell, shared_strings)
                    for cell in row.findall(f"{{{main_ns}}}c")
                ]
                rows.append(values)
                if len(rows) >= MAX_PREVIEW_ROWS:
                    break
            previews.append(
                {
                    "sheet_name": sheet_name,
                    "suggested_header_row": _suggest_header_row(rows),
                    "rows": rows,
                }
            )
        return previews


def _preview_xls(file: UploadedDatasetFile) -> list[dict[str, Any]]:
    import xlrd

    workbook = xlrd.open_workbook(file_contents=file.content)
    previews: list[dict[str, Any]] = []
    for sheet in workbook.sheets():
        rows: list[list[str]] = []
        for index in range(min(sheet.nrows, MAX_PREVIEW_ROWS)):
            rows.append([_cell_text(value) for value in sheet.row_values(index)])
        previews.append(
            {
                "sheet_name": sheet.name,
                "suggested_header_row": _suggest_header_row(rows),
                "rows": rows,
            }
        )
    return previews


def preview_dataset_files(files: list[UploadedDatasetFile]) -> dict[str, Any]:
    if not files:
        raise CompanyDatasetValidationError("at least one file required")
    total_bytes = sum(len(item.content) for item in files)
    if total_bytes > MAX_BATCH_BYTES:
        raise CompanyDatasetValidationError("preview batch exceeds 250 MiB limit")

    file_previews: list[dict[str, Any]] = []
    for file in files:
        _validate_file(file)
        filename = Path(file.filename or "").name
        extension = Path(filename).suffix.lower()
        if extension == ".csv":
            sheets = _preview_csv(file)
        elif extension == ".xlsx":
            sheets = _preview_xlsx(file)
        elif extension == ".xls":
            sheets = _preview_xls(file)
        else:
            sheets = []
        file_previews.append(
            {
                "filename": filename,
                "byte_size": len(file.content),
                "encoding": sheets[0].get("encoding") if sheets else None,
                "sheets": sheets,
            }
        )
    all_sheet_names = sorted(
        {
            sheet["sheet_name"]
            for preview in file_previews
            for sheet in preview.get("sheets", [])
            if sheet.get("sheet_name")
        }
    )
    suggested_header_rows = {
        sheet["sheet_name"]: sheet["suggested_header_row"]
        for preview in file_previews
        for sheet in preview.get("sheets", [])
        if sheet.get("sheet_name")
    }
    return {
        "files": file_previews,
        "sheet_names": all_sheet_names,
        "suggested_header_rows": suggested_header_rows,
    }


def build_parser_profile(
    *,
    selected_sheets: list[str] | None = None,
    header_rows: dict[str, int] | None = None,
    all_sheets: bool = False,
    auto_header: bool = False,
    locale: str = "zh-CN",
    timezone: str = "UTC",
    encoding: str = "utf-8",
    required_column_profile: str = "company_dataset_v1",
) -> dict[str, Any]:
    profile = {
        "parser_profile_schema_version": PARSER_PROFILE_SCHEMA_VERSION,
        "locale": locale.strip() or "zh-CN",
        "timezone": timezone.strip() or "UTC",
        "encoding": encoding.strip() or "utf-8",
        "all_sheets": bool(all_sheets),
        "auto_header": bool(auto_header),
        "selected_sheets": []
        if all_sheets
        else sorted([sheet.strip() for sheet in selected_sheets or [] if sheet.strip()]),
        "header_rows": {}
        if auto_header
        else {
            str(key).strip(): int(value)
            for key, value in sorted((header_rows or {}).items())
            if str(key).strip()
        },
        "required_column_profile": required_column_profile.strip() or "company_dataset_v1",
    }
    profile["parser_profile_hash"] = _sha256_json(profile)
    return profile


def _assert_active_real_context(ctx: Any) -> None:
    if not getattr(ctx, "organization_id", None):
        raise CompanyDatasetAuthorizationError("Active organization required")
    if str(getattr(ctx, "dataset_type", "") or "").upper() != "REAL":
        raise CompanyDatasetAuthorizationError("Active organization must be REAL")


def _source_record(*, import_id: str, file: UploadedDatasetFile, content_hash: str, artifact_path: Path) -> dict[str, Any]:
    return {
        "file_id": f"file_{uuid.uuid4().hex}",
        "import_id": import_id,
        "original_filename": Path(file.filename or "dataset").name,
        "content_type": file.content_type,
        "byte_size": len(file.content),
        "artifact_path": str(artifact_path),
        "source_file_hash": content_hash,
    }


def create_dataset_import(
    *,
    ctx: Any,
    idempotency_key: str,
    files: list[UploadedDatasetFile],
    parser_profile: dict[str, Any] | None = None,
    purpose: str = "company_dataset",
) -> dict[str, Any]:
    if purpose == "receivables_candidate":
        _assert_candidate_context(ctx)
    elif purpose == "company_dataset":
        _assert_active_real_context(ctx)
    else:
        raise CompanyDatasetValidationError("unsupported import purpose")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", idempotency_key):
        raise CompanyDatasetValidationError("safe idempotency_key required")
    if not files:
        raise CompanyDatasetValidationError("at least one file required")
    total_bytes = sum(len(item.content) for item in files)
    if total_bytes > MAX_BATCH_BYTES:
        raise CompanyDatasetValidationError("import batch exceeds 250 MiB limit")
    for file in files:
        _validate_file(file)

    profile = dict(parser_profile or build_parser_profile())
    reserved = {"candidate_purpose", "candidate_workspace_id"}
    if reserved.intersection(profile):
        raise CompanyDatasetValidationError("candidate binding is server-owned")
    if purpose == "receivables_candidate":
        profile.update(candidate_purpose=purpose, candidate_workspace_id=str(ctx.workspace_id))
        profile["parser_profile_hash"] = _sha256_json({key: value for key, value in profile.items() if key != "parser_profile_hash"})
    existing = store.get_import_for_idempotency(organization_id=str(ctx.organization_id), idempotency_key=idempotency_key)
    if existing is not None:
        _assert_import_request_binding(ctx, existing, purpose)
        return with_authority_state(existing, dataset_type=str(getattr(ctx, "dataset_type", "") or ""))
    import_dir = _artifact_root() / str(ctx.organization_id) / idempotency_key
    import_dir.mkdir(parents=True, exist_ok=True)
    source_records: list[dict[str, Any]] = []
    provisional_import_id = f"import_{uuid.uuid4().hex}"
    for file in files:
        content_hash = _sha256_bytes(file.content)
        safe_name = Path(file.filename or f"{content_hash}.bin").name
        artifact_path = import_dir / f"{content_hash}_{safe_name}"
        if not artifact_path.exists():
            artifact_path.write_bytes(file.content)
        source_records.append(
            _source_record(
                import_id=provisional_import_id,
                file=file,
                content_hash=content_hash,
                artifact_path=artifact_path,
            )
        )

    payload = store.create_import(
        organization_id=str(ctx.organization_id),
        actor_user_id=str(ctx.user_id),
        idempotency_key=idempotency_key,
        parser_profile_schema_version=PARSER_PROFILE_SCHEMA_VERSION,
        parser_profile_hash=str(profile["parser_profile_hash"]),
        parser_profile=profile,
        source_files=source_records,
    )
    _assert_import_request_binding(ctx, payload, purpose)
    return with_authority_state(payload, dataset_type=str(getattr(ctx, "dataset_type", "") or ""))



def _assert_import_request_binding(ctx: Any, record: dict, purpose: str) -> None:
    profile = record["parser_profile"]
    recorded_purpose = profile.get("candidate_purpose") or "company_dataset"
    bound_workspace = profile.get("candidate_workspace_id")
    if bound_workspace is not None and bound_workspace != getattr(ctx, "workspace_id", None):
        raise CompanyDatasetAuthorizationError("candidate import idempotency outside workspace")
    if recorded_purpose != purpose:
        raise CompanyDatasetAuthorizationError("import idempotency purpose mismatch")

def _registered_toolset_tools(toolset_id: str) -> set[str]:
    inventory = runtime_inventory.list_toolsets_inventory(workspace_hermes_home=None)
    tools = {
        str(tool)
        for item in inventory.get("tools", [])
        if isinstance(item, dict) and item.get("id") == toolset_id
        for tool in item.get("tools", [])
    }
    return tools


def _plugin_import_roots() -> tuple[Path, ...]:
    repo_root = Path(__file__).resolve().parents[2]
    return (
        repo_root / "src" / "plugins",
        repo_root / "semantier-skills" / "plugins",
    )


def _ensure_plugin_import_roots() -> None:
    for plugin_path in reversed(_plugin_import_roots()):
        if plugin_path.exists() and str(plugin_path) not in sys.path:
            sys.path.insert(0, str(plugin_path))


def _load_automate_excel_tools_module():
    if "validate_excel_file" not in _registered_toolset_tools("automate_excel"):
        raise CompanyDatasetValidationError("automate_excel plugin must be installed or enabled")
    _ensure_plugin_import_roots()
    return importlib.import_module("automate_excel.tools")


def _load_governed_data_engineering_tools_module():
    if "governed_profile_table" not in _registered_toolset_tools("governed_data_engineering"):
        raise CompanyDatasetValidationError("governed_data_engineering plugin must be installed or enabled")
    _ensure_plugin_import_roots()
    return importlib.import_module("governed_data_engineering.tools")


def _validation_tool_payload(source: dict[str, Any], parser_profile: dict[str, Any]) -> dict[str, Any]:
    artifact_path = str(source["artifact_path"])
    extension = Path(artifact_path).suffix.lower()
    selected_sheets = parser_profile.get("selected_sheets") or []
    selected_sheet = str(selected_sheets[0]) if selected_sheets else "Sheet1"
    payload: dict[str, Any] = {
        "input": artifact_path,
        "encoding": str(parser_profile.get("encoding") or "utf-8"),
    }
    if extension in {".xlsx", ".xls"} and selected_sheets:
        payload["sheet"] = selected_sheet
    if not parser_profile.get("auto_header"):
        header_rows = parser_profile.get("header_rows") or {}
        if isinstance(header_rows, dict) and selected_sheet in header_rows:
            payload["header_row"] = int(header_rows[selected_sheet])
    return payload


def _data_engineering_profile_payload(source: dict[str, Any], parser_profile: dict[str, Any]) -> dict[str, Any] | None:
    artifact_path = str(source["artifact_path"])
    if Path(artifact_path).suffix.lower() != ".csv":
        return None
    return {
        "csv_path": artifact_path,
        "encoding": str(parser_profile.get("encoding") or "utf-8"),
        "sample_limit": 5,
    }


def _promotion_manifest(record: dict[str, Any]) -> dict[str, Any]:
    validation_result = record.get("validation_result") or {}
    normalized_files = validation_result.get("normalized_files") or []
    files_by_id = {
        str(file.get("file_id") or ""): file
        for file in record.get("files", [])
        if isinstance(file, dict)
    }
    files: list[dict[str, Any]] = []
    for normalized in normalized_files:
        if not isinstance(normalized, dict):
            continue
        file_id = str(normalized.get("file_id") or "")
        source = files_by_id.get(file_id, {})
        files.append(
            {
                "file_id": file_id,
                "original_filename": source.get("original_filename"),
                "content_type": source.get("content_type"),
                "artifact_path": normalized.get("artifact_path") or source.get("artifact_path"),
                "source_file_hash": source.get("source_file_hash"),
                "normalized_file_hash": normalized.get("normalized_file_hash"),
            }
        )
    return {
        "manifest_schema_version": "company_dataset_version_manifest.v1",
        "parser_profile_hash": record.get("parser_profile_hash"),
        "parser_profile_schema_version": record.get("parser_profile_schema_version"),
        "source_import_id": record.get("import_id"),
        "files": files,
    }


def _set_active_dataset_version(
    *,
    organization_id: str,
    dataset_version_id: str,
    actor_user_id: str,
) -> None:
    from agents import gateway_identity

    gateway_identity.set_organization_active_dataset_version(
        organization_id=organization_id,
        dataset_version_id=dataset_version_id,
        actor_user_id=actor_user_id,
    )


def _complete_dataset_promotion(
    *,
    ctx: Any,
    record: dict[str, Any],
    idempotency_key: str,
) -> dict[str, Any]:
    validation_result = record.get("validation_result") or {}
    if validation_result.get("blocking_errors"):
        raise CompanyDatasetValidationError("cannot promote import with blocking validation errors")
    manifest = _promotion_manifest(record)
    if not manifest["files"]:
        raise CompanyDatasetValidationError("cannot promote import without normalized files")
    dataset_version = store.create_dataset_version(
        organization_id=str(ctx.organization_id),
        import_id=str(record["import_id"]),
        activated_by=str(ctx.user_id),
        artifact_hash_manifest=manifest,
    )
    promotion_result = {
        "promotion_idempotency_key": idempotency_key,
        "started_at": (record.get("promotion_result") or {}).get("started_at") or now_iso(),
        "promoted_at": dataset_version["activated_at"],
        "mode": "sync",
        "dataset_version_id": dataset_version["dataset_version_id"],
        "artifact_hash_manifest": manifest,
    }
    promoted = store.transition_import_status(
        organization_id=str(ctx.organization_id),
        import_id=str(record["import_id"]),
        from_statuses={"promoting"},
        to_status="promoted",
        promotion_result=promotion_result,
    )
    _set_active_dataset_version(
        organization_id=str(ctx.organization_id),
        dataset_version_id=str(dataset_version["dataset_version_id"]),
        actor_user_id=str(ctx.user_id),
    )
    return promoted


def validate_dataset_import(
    *,
    ctx: Any,
    import_id: str,
    idempotency_key: str,
) -> dict[str, Any]:
    _assert_active_real_context(ctx)
    record = store.get_import(organization_id=str(ctx.organization_id), import_id=import_id)
    if record is None:
        raise KeyError(import_id)
    if record["parser_profile"].get("candidate_purpose") == "receivables_candidate":
        raise CompanyDatasetAuthorizationError("Read-only candidate imports cannot validate or promote company datasets")
    if record["status"] == "validating":
        return with_authority_state(
            record,
            dataset_type=str(getattr(ctx, "dataset_type", "") or ""),
        )
    existing_result = store.get_validation_run_result(
        organization_id=str(ctx.organization_id),
        idempotency_key=idempotency_key,
    )
    if existing_result is not None:
        return with_authority_state(record, dataset_type=str(getattr(ctx, "dataset_type", "") or ""))

    validating = store.transition_import_status(
        organization_id=str(ctx.organization_id),
        import_id=import_id,
        from_statuses={"uploaded", "validation_failed"},
        to_status="validating",
    )
    tools = _load_automate_excel_tools_module()
    try:
        data_engineering_tools = _load_governed_data_engineering_tools_module()
    except CompanyDatasetValidationError:
        data_engineering_tools = None
    file_results: list[dict[str, Any]] = []
    data_engineering_results: list[dict[str, Any]] = []
    blocking_errors: list[dict[str, Any]] = []
    normalized_files: list[dict[str, Any]] = []
    for source in validating.get("files", []):
        if source.get("file_role") != "source":
            continue
        tool_payload = _validation_tool_payload(source, validating.get("parser_profile", {}))
        result_raw = tools.validate_excel_file(tool_payload)
        result = json.loads(result_raw)
        file_result = {
            "file_id": source["file_id"],
            "source_file_hash": source["source_file_hash"],
            "tool_name": "validate_excel_file",
            "toolset": "automate_excel",
            "tool_result": result,
        }
        file_results.append(file_result)
        if not result.get("ok"):
            blocking_errors.append(
                {
                    "file_id": source["file_id"],
                    "error": result.get("stderr") or result.get("error") or "validation failed",
                }
            )
        else:
            profile_payload = _data_engineering_profile_payload(
                source,
                validating.get("parser_profile", {}),
            )
            if profile_payload is None:
                data_engineering_results.append(
                    {
                        "file_id": source["file_id"],
                        "toolset": "governed_data_engineering",
                        "tool_name": "governed_profile_table",
                        "status": "skipped",
                        "reason": "profile_table_v1_accepts_csv_only",
                    }
                )
            elif data_engineering_tools is None:
                data_engineering_results.append(
                    {
                        "file_id": source["file_id"],
                        "toolset": "governed_data_engineering",
                        "tool_name": "governed_profile_table",
                        "status": "unavailable",
                        "reason": "governed_data_engineering plugin must be installed or enabled",
                    }
                )
            else:
                profile_raw = data_engineering_tools.governed_profile_table(profile_payload)
                data_engineering_results.append(
                    {
                        "file_id": source["file_id"],
                        "source_file_hash": source["source_file_hash"],
                        "tool_name": "governed_profile_table",
                        "toolset": "governed_data_engineering",
                        "tool_result": json.loads(profile_raw),
                    }
                )
            normalized_files.append(
                {
                    "file_id": source["file_id"],
                    "normalized_file_hash": source["source_file_hash"],
                    "artifact_path": source["artifact_path"],
                }
            )

    validation_result = {
        "status": "validation_failed" if blocking_errors else "staged",
        "validated_at": now_iso(),
        "row_counts": {},
        "sheet_names": validating.get("parser_profile", {}).get("selected_sheets") or [],
        "required_column_failures": blocking_errors,
        "blocking_errors": blocking_errors,
        "normalized_files": normalized_files,
        "file_results": file_results,
        "data_engineering_results": data_engineering_results,
    }
    store.record_validation_run(
        organization_id=str(ctx.organization_id),
        import_id=import_id,
        idempotency_key=idempotency_key,
        parser_profile_schema_version=PARSER_PROFILE_SCHEMA_VERSION,
        parser_profile_hash=str(record["parser_profile_hash"]),
        validation_result=validation_result,
    )
    final_status = "validation_failed" if blocking_errors else "staged"
    updated = store.transition_import_status(
        organization_id=str(ctx.organization_id),
        import_id=import_id,
        from_statuses={"validating"},
        to_status=final_status,
        validation_result=validation_result,
    )
    return with_authority_state(updated, dataset_type=str(getattr(ctx, "dataset_type", "") or ""))


def promote_dataset_import(
    *,
    ctx: Any,
    import_id: str,
    idempotency_key: str,
) -> dict[str, Any]:
    _assert_active_real_context(ctx)
    if not idempotency_key.strip():
        raise CompanyDatasetValidationError("idempotency_key required")
    record = store.get_import(organization_id=str(ctx.organization_id), import_id=import_id)
    if record is None:
        raise KeyError(import_id)
    if record["parser_profile"].get("candidate_purpose") == "receivables_candidate":
        raise CompanyDatasetAuthorizationError("Read-only candidate imports cannot validate or promote company datasets")
    if record["status"] == "promoting":
        completed = _complete_dataset_promotion(
            ctx=ctx,
            record=record,
            idempotency_key=str((record.get("promotion_result") or {}).get("promotion_idempotency_key") or idempotency_key),
        )
        return with_authority_state(completed, dataset_type=str(getattr(ctx, "dataset_type", "") or ""))
    promoting = store.transition_import_status(
        organization_id=str(ctx.organization_id),
        import_id=import_id,
        from_statuses={"staged", "promotion_failed"},
        to_status="promoting",
        promotion_result={
            "promotion_idempotency_key": idempotency_key,
            "started_at": now_iso(),
            "mode": "sync",
        },
    )
    completed = _complete_dataset_promotion(
        ctx=ctx,
        record=promoting,
        idempotency_key=idempotency_key,
    )
    return with_authority_state(completed, dataset_type=str(getattr(ctx, "dataset_type", "") or ""))


def _rows_from_csv_artifact(path: Path, *, encoding: str, header_row: int) -> list[dict[str, str]]:
    raw = path.read_bytes()
    text, _ = _decode_csv_text(raw) if encoding == "auto" else (raw.decode(encoding), encoding)
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return []
    header_index = max(int(header_row) - 1, 0)
    if header_index >= len(rows):
        return []
    headers = [
        _cell_text(value) or f"column_{index + 1}"
        for index, value in enumerate(rows[header_index])
    ]
    payload: list[dict[str, str]] = []
    for row in rows[header_index + 1 :]:
        payload.append(
            {
                headers[index]: _cell_text(row[index]) if index < len(row) else ""
                for index in range(len(headers))
            }
        )
    return payload


def _filter_rows(rows: list[dict[str, str]], filter_text: str) -> list[dict[str, str]]:
    needle = filter_text.strip().casefold()
    if not needle:
        return rows
    return [
        row
        for row in rows
        if any(needle in str(value).casefold() for value in row.values())
    ]


def get_active_dataset_explorer(
    *,
    ctx: Any,
    file_id: str | None = None,
    page: int = 1,
    page_size: int = 25,
    filter_text: str = "",
) -> dict[str, Any]:
    _assert_active_real_context(ctx)
    active = store.get_active_dataset_version(organization_id=str(ctx.organization_id))
    if active is None:
        raise CompanyDatasetValidationError("no promoted real dataset")
    manifest = active.get("artifact_hash_manifest") or {}
    parser_profile_hash = str(manifest.get("parser_profile_hash") or "")
    import_record = store.get_import(
        organization_id=str(ctx.organization_id),
        import_id=str(active["import_id"]),
    )
    parser_profile = import_record.get("parser_profile") if import_record else {}
    encoding = str((parser_profile or {}).get("encoding") or "utf-8")
    header_rows = (parser_profile or {}).get("header_rows") or {}
    header_row = int(header_rows.get("Sheet1") or 1) if isinstance(header_rows, dict) else 1
    files: list[dict[str, Any]] = []
    rows_by_file_id: dict[str, list[dict[str, str]]] = {}
    for file_manifest in manifest.get("files") or []:
        if not isinstance(file_manifest, dict):
            continue
        artifact_path = Path(str(file_manifest.get("artifact_path") or ""))
        rows: list[dict[str, str]] = []
        if artifact_path.suffix.lower() == ".csv" and artifact_path.exists():
            rows = _rows_from_csv_artifact(
                artifact_path,
                encoding=encoding,
                header_row=header_row,
            )
        current_file_id = str(file_manifest.get("file_id") or "")
        rows_by_file_id[current_file_id] = rows
        files.append(
            {
                "file_id": current_file_id,
                "original_filename": file_manifest.get("original_filename"),
                "source_file_hash": file_manifest.get("source_file_hash"),
                "normalized_file_hash": file_manifest.get("normalized_file_hash"),
                "row_count": len(rows),
                "promoted_rows": len(rows),
                "deduped_rows": 0,
                "error_rows": 0,
                "fixed_rows": 0,
            }
        )
    all_selected = str(file_id or "").strip().lower() == "all"
    selected_file_id = (
        "all"
        if all_selected
        else file_id if file_id in rows_by_file_id else (files[0]["file_id"] if files else "")
    )
    if selected_file_id == "all":
        selected_rows = [
            {
                "source_file": str(current_file.get("original_filename") or current_file.get("file_id") or ""),
                **row,
            }
            for current_file in files
            for row in rows_by_file_id.get(str(current_file.get("file_id") or ""), [])
        ]
        selected_file_stats = {
            "files_promoted": len(files),
            "rows_promoted": len(selected_rows),
            "deduped_rows": 0,
            "error_rows": 0,
            "fixed_rows": 0,
        }
    else:
        selected_rows = rows_by_file_id.get(str(selected_file_id), [])
        selected_file = next(
            (item for item in files if item.get("file_id") == selected_file_id),
            None,
        )
        selected_file_stats = {
            "files_promoted": 1 if selected_file else 0,
            "rows_promoted": int((selected_file or {}).get("promoted_rows") or 0),
            "deduped_rows": int((selected_file or {}).get("deduped_rows") or 0),
            "error_rows": int((selected_file or {}).get("error_rows") or 0),
            "fixed_rows": int((selected_file or {}).get("fixed_rows") or 0),
        }
    filtered_rows = _filter_rows(selected_rows, filter_text)
    safe_page_size = max(1, min(int(page_size), 100))
    total_rows = len(filtered_rows)
    total_pages = max(1, (total_rows + safe_page_size - 1) // safe_page_size)
    safe_page = max(1, min(int(page), total_pages))
    start = (safe_page - 1) * safe_page_size
    page_rows = filtered_rows[start : start + safe_page_size]
    columns = list(page_rows[0].keys()) if page_rows else (list(selected_rows[0].keys()) if selected_rows else [])
    promoted_rows = sum(int(file.get("promoted_rows") or 0) for file in files)
    return {
        "dataset_version_id": active["dataset_version_id"],
        "organization_id": active["organization_id"],
        "import_id": active["import_id"],
        "activated_at": active["activated_at"],
        "parser_profile_hash": parser_profile_hash,
        "stats": {
            "files_promoted": len(files),
            "rows_promoted": promoted_rows,
            "deduped_rows": 0,
            "error_rows": 0,
            "fixed_rows": 0,
        },
        "files": files,
        "selected_file_id": selected_file_id,
        "selected_file_stats": selected_file_stats,
        "columns": columns,
        "rows": page_rows,
        "pagination": {
            "page": safe_page,
            "page_size": safe_page_size,
            "total_rows": total_rows,
            "total_pages": total_pages,
            "filter": filter_text,
        },
    }


def get_dataset_import(*, ctx: Any, import_id: str) -> dict[str, Any] | None:
    if not getattr(ctx, "organization_id", None):
        raise CompanyDatasetAuthorizationError("Active organization required")
    record = store.get_import(organization_id=str(ctx.organization_id), import_id=import_id)
    if record is None:
        return None
    candidate_workspace = record["parser_profile"].get("candidate_workspace_id")
    if candidate_workspace is not None and candidate_workspace != getattr(ctx, "workspace_id", None):
        raise CompanyDatasetAuthorizationError("candidate import outside workspace")
    return with_authority_state(record, dataset_type=str(getattr(ctx, "dataset_type", "") or ""))


def list_dataset_imports(*, ctx: Any) -> list[dict[str, Any]]:
    if not getattr(ctx, "organization_id", None):
        raise CompanyDatasetAuthorizationError("Active organization required")
    return [
        with_authority_state(record, dataset_type=str(getattr(ctx, "dataset_type", "") or ""))
        for record in store.list_imports(organization_id=str(ctx.organization_id))
        if record["parser_profile"].get("candidate_workspace_id") in (None, getattr(ctx, "workspace_id", None))
    ]


def authority_state_for(*, dataset_type: str | None, active_dataset_version_id: str | None = None, latest_import_status: str | None = None) -> str:
    """Legacy import-row display state.

    REAL organization readiness must use `setup_status_for_context()` instead.
    """
    if str(dataset_type or "").upper() == "DEMO":
        return "DEMO_ACTIVE"
    if str(dataset_type or "").upper() != "REAL":
        return "REAL_EMPTY"
    if active_dataset_version_id:
        return "REAL_PROMOTED"
    if latest_import_status in {"uploaded", "validating", "validation_failed", "staged"}:
        return "REAL_STAGED_ONLY"
    if latest_import_status == "promoting":
        return "REAL_PROMOTING"
    if latest_import_status == "promotion_failed":
        return "REAL_PROMOTION_FAILED"
    if latest_import_status == "promoted":
        return "REAL_PROMOTED"
    return "REAL_EMPTY"


def with_authority_state(record: dict[str, Any], *, dataset_type: str | None) -> dict[str, Any]:
    payload = dict(record)
    payload["authority_state"] = authority_state_for(
        dataset_type=dataset_type,
        active_dataset_version_id=None,
        latest_import_status=str(record.get("status") or ""),
    )
    return payload


def setup_status_for_context(ctx: Any, org_context: dict[str, Any] | None = None) -> dict[str, Any]:
    org_context = org_context or {}
    dataset_type = str(
        org_context.get("dataset_type")
        or getattr(ctx, "dataset_type", "")
        or ""
    )
    active_dataset_version_id = (
        str(org_context.get("active_dataset_version_id"))
        if org_context.get("active_dataset_version_id")
        else None
    )
    return real_company_setup_status_payload(
        organization_id=str(getattr(ctx, "organization_id", "") or ""),
        dataset_type=dataset_type,
        active_dataset_version_id=active_dataset_version_id,
    )


def _assert_candidate_context(ctx: Any) -> None:
    """Explicit candidate reads/uploads do not confer validation/promotion rights."""
    if (not getattr(ctx, "authenticated", False) or not getattr(ctx, "organization_id", None)
            or not getattr(ctx, "workspace_id", None)):
        raise CompanyDatasetAuthorizationError("Authenticated organization/workspace context required")
    if str(getattr(ctx, "dataset_type", "")).upper() not in {"REAL", "DEVELOPMENT"}:
        raise CompanyDatasetAuthorizationError("REAL or DEVELOPMENT candidate context required")


def read_authorized_import_file(*, ctx: Any, import_id: str, file_id: str | None = None) -> tuple[dict, dict, bytes]:
    """Resolve full immutable bytes from the launcher-owned import boundary."""
    _assert_candidate_context(ctx)
    record = get_dataset_import(ctx=ctx, import_id=import_id)
    if record is None:
        raise CompanyDatasetValidationError("IMPORT_NOT_FOUND")
    expiry = record.get("retention_until")
    # Promoted company datasets have an explicit indefinite-retention policy;
    # candidate/staged imports always require a concrete retention deadline.
    if expiry is None and record.get("status") == "promoted" and not record["parser_profile"].get("candidate_purpose"):
        pass
    else:
        try:
            expires_at = datetime.fromisoformat(str(expiry).replace("Z", "+00:00"))
            # Explicit normalization for legacy receipts precedes comparison.
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError) as exc:
            raise CompanyDatasetValidationError("IMPORT_RETENTION_UNAVAILABLE") from exc
        if expires_at <= datetime.now(timezone.utc):
            raise CompanyDatasetValidationError("IMPORT_SOURCE_EXPIRED")
    files = [item for item in record.get("files", []) if item.get("file_role") == "source"]
    selected = [item for item in files if file_id is None or item["file_id"] == file_id]
    if len(selected) != 1:
        raise CompanyDatasetValidationError("ONE_IMPORT_FILE_REQUIRED")
    source = selected[0]
    path = Path(source["artifact_path"])
    root = _artifact_root().resolve() / str(ctx.organization_id)
    try:
        parts = path.relative_to(root).parts
    except ValueError as exc:
        raise CompanyDatasetAuthorizationError("IMPORT_ARTIFACT_OUTSIDE_OWNER") from exc
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise CompanyDatasetAuthorizationError("IMPORT_ARTIFACT_OUTSIDE_OWNER")
    parent_fd = None
    try:
        parent_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
        with os.fdopen(file_fd, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_FILE_BYTES:
                raise CompanyDatasetValidationError("IMPORT_SOURCE_PIN_MISMATCH")
            data = stream.read(MAX_FILE_BYTES + 1)
    except OSError as exc:
        raise CompanyDatasetAuthorizationError("IMPORT_ARTIFACT_UNAVAILABLE") from exc
    finally:
        if parent_fd is not None:
            os.close(parent_fd)
    if len(data) != source["byte_size"] or _sha256_bytes(data) != source["source_file_hash"]:
        raise CompanyDatasetValidationError("IMPORT_SOURCE_PIN_MISMATCH")
    profile = record["parser_profile"]
    basis = {key: value for key, value in profile.items() if key != "parser_profile_hash"}
    if _sha256_json(basis) != profile["parser_profile_hash"] or record["parser_profile_hash"] != profile["parser_profile_hash"]:
        raise CompanyDatasetValidationError("IMPORT_PARSER_PIN_MISMATCH")
    return record, source, data
