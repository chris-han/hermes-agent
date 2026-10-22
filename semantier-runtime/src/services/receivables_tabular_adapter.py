"""Bounded full-file tabular parsing with original sheet/row/cell identities."""

import csv
import io
import re
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta
from xml.etree import ElementTree as ET
from agents.company_dataset_imports import (
    _decode_csv_text,
    _xlsx_shared_strings,
    _xlsx_sheet_targets,
    _xlsx_cell_value,
    MAX_FILE_BYTES,
)
from contracts.canonical_source_ir import (
    CanonicalSourceDocument,
    SourceLocator,
    SourceBlock,
    SourceMapping,
    SourceMap,
    Transformation,
    ParserTrace,
    source_block_identity,
    content_hash,
)

NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
MAX_ROWS = 100000
MAX_CELLS = 1000000
MAX_COLUMNS = 256


@dataclass(frozen=True)
class TabularRows:
    sheet_name: str
    header_row: int
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    row_numbers: tuple[int, ...]


def _xml(archive, name):
    data = archive.read(name)
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ValueError("UNSAFE_XLSX_XML")
    return ET.fromstring(data)


def _column(ref):
    letters = re.match(r"^[A-Z]+", ref or "")
    if not letters:
        raise ValueError("INVALID_XLSX_CELL_REFERENCE")
    index = 0
    for letter in letters[0]:
        index = index * 26 + ord(letter) - 64
    return index - 1


def _parse_tabular(data, *, original_name, parser_profile):
    if not 0 < len(data) <= MAX_FILE_BYTES:
        raise ValueError("INVALID_BYTE_LENGTH")
    if original_name.lower().endswith(".csv"):
        text, _ = _decode_csv_text(data)
        reader = csv.reader(io.StringIO(text), strict=True)
        all_rows = []
        cell_count = 0
        for number, row in enumerate(reader, 1):
            if len(row) > MAX_COLUMNS:
                raise ValueError("TABULAR_COLUMN_LIMIT")
            cell_count += len(row)
            if cell_count > MAX_CELLS:
                raise ValueError("TABULAR_CELL_LIMIT")
            all_rows.append((number, tuple(value.strip() for value in row)))
            if number > MAX_ROWS:
                raise ValueError("TABULAR_ROW_LIMIT")
        sheet = "Sheet1"
    elif original_name.lower().endswith(".xlsx"):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if sum(
                item.file_size for item in archive.infolist()
            ) > MAX_FILE_BYTES * 4 or any(
                "vbaproject" in n.lower() for n in archive.namelist()
            ):
                raise ValueError("UNSAFE_XLSX_PACKAGE")
            # Validate every parsed XML member before invoking native helpers.
            for name in (
                "xl/workbook.xml",
                "xl/_rels/workbook.xml.rels",
                "xl/sharedStrings.xml",
            ):
                if name in archive.namelist():
                    _xml(archive, name)
            for name in archive.namelist():
                if name.endswith(".rels"):
                    relations = _xml(archive, name)
                    if any(rel.get("TargetMode") == "External" for rel in relations):
                        raise ValueError("XLSX_EXTERNAL_RELATIONSHIP_UNSUPPORTED")
            sheets = _xlsx_sheet_targets(archive)
            selected = parser_profile.get("selected_sheets") or []
            if len(selected) != 1 and len(sheets) != 1:
                raise ValueError("ONE_WORKSHEET_REQUIRED")
            sheet = selected[0] if selected else sheets[0][0]
            targets = [target for name, target in sheets if name == sheet]
            if len(targets) != 1:
                raise ValueError("WORKSHEET_NOT_FOUND")
            shared = _xlsx_shared_strings(archive)
            date_styles = set()
            if "xl/styles.xml" in archive.namelist():
                styles = _xml(archive, "xl/styles.xml")
                custom = {
                    int(e.get("numFmtId")): e.get("formatCode", "")
                    for e in styles.findall(f"{{{NS}}}numFmts/{{{NS}}}numFmt")
                }
                for index, style in enumerate(
                    styles.findall(f"{{{NS}}}cellXfs/{{{NS}}}xf")
                ):
                    fmt = int(style.get("numFmtId", "0"))
                    if 14 <= fmt <= 22 or (
                        fmt in custom and re.search(r"[dy]", custom[fmt], re.I)
                    ):
                        date_styles.add(index)
            workbook = _xml(archive, "xl/workbook.xml")
            props = workbook.find(f"{{{NS}}}workbookPr")
            epoch = (
                date(1904, 1, 1)
                if props is not None and props.get("date1904") in ("1", "true")
                else date(1899, 12, 30)
            )
            root = _xml(archive, targets[0])
            all_rows = []
            cell_count = 0
            for row in root.findall(f".//{{{NS}}}sheetData/{{{NS}}}row"):
                number = int(row.get("r", len(all_rows) + 1))
                values = []
                occupied_columns = set()
                for cell in row.findall(f"{{{NS}}}c"):
                    index = _column(cell.get("r"))
                    if index in occupied_columns:
                        raise ValueError("DUPLICATE_XLSX_CELL")
                    occupied_columns.add(index)
                    if index >= MAX_COLUMNS:
                        raise ValueError("TABULAR_COLUMN_LIMIT")
                    if cell_count + index + 1 > MAX_CELLS:
                        raise ValueError("TABULAR_CELL_LIMIT")
                    while len(values) <= index:
                        values.append("")
                    if cell.find(f"{{{NS}}}f") is not None:
                        raise ValueError("XLSX_FORMULA_UNSUPPORTED")
                    value = _xlsx_cell_value(cell, shared).strip()
                    if (
                        value
                        and int(cell.get("s", "0")) in date_styles
                        and cell.get("t") not in ("s", "inlineStr")
                    ):
                        try:
                            value = (
                                epoch + timedelta(days=int(float(value)))
                            ).isoformat()
                        except (ValueError, OverflowError):
                            raise ValueError("INVALID_XLSX_DATE")
                    values[index] = value
                cell_count += len(values)
                all_rows.append((number, tuple(values)))
                if len(all_rows) > MAX_ROWS:
                    raise ValueError("TABULAR_ROW_LIMIT")
    else:
        raise ValueError("CSV_XLSX_REQUIRED")
    if len({number for number, _ in all_rows}) != len(all_rows):
        raise ValueError("DUPLICATE_XLSX_ROW")
    if sum(len(row) for _, row in all_rows) > MAX_CELLS:
        raise ValueError("TABULAR_CELL_LIMIT")
    header_row = int((parser_profile.get("header_rows") or {}).get(sheet, 1))
    candidates = [row for number, row in all_rows if number == header_row]
    if len(candidates) != 1:
        raise ValueError("HEADER_ROW_REQUIRED")
    headers = candidates[0]
    if not headers or any(not h for h in headers):
        raise ValueError("EMPTY_HEADER")
    if len(set(headers)) != len(headers):
        raise ValueError("DUPLICATE_HEADER")
    rows, numbers = [], []
    for number, row in all_rows:
        if number <= header_row or not any(row):
            continue
        if len(row) > len(headers):
            raise ValueError("ROW_WIDTH_MISMATCH")
        rows.append(row + ("",) * (len(headers) - len(row)))
        numbers.append(number)
    return TabularRows(sheet, header_row, headers, tuple(rows), tuple(numbers))


def parse_tabular(data, *, original_name, parser_profile):
    try:
        return _parse_tabular(
            data, original_name=original_name, parser_profile=parser_profile
        )
    except (
        csv.Error,
        zipfile.BadZipFile,
        ET.ParseError,
        KeyError,
        UnicodeError,
        OverflowError,
    ) as exc:
        raise ValueError("TABULAR_PARSE_FAILED") from exc


def tabular_canonical_ir(parsed, *, source_identity_ref, source_hash, media_type):
    blocks, mappings = [], []
    for row_number, row in (
        (parsed.header_row, parsed.headers),
        *zip(parsed.row_numbers, parsed.rows),
    ):
        for column, value in enumerate(row, 1):
            locator = SourceLocator(
                sourceIdentityRef=source_identity_ref,
                sourceHash=source_hash,
                mediaType=media_type,
                format="tabular",
                tableId="t1",
                rowId=f"t1.r{row_number}",
                columnId=f"t1.c{column}",
                nativeId=f"{parsed.sheet_name}!R{row_number}C{column}",
                nativeMetadata={
                    "sheet_name": parsed.sheet_name,
                    "row_number": row_number,
                    "column_number": column,
                    "header_row": parsed.header_row,
                },
            )
            block_id = source_block_identity(source_hash, locator, "cell")
            blocks.append(
                SourceBlock(
                    blockId=block_id,
                    blockType="cell",
                    content=value,
                    ordinal=len(blocks),
                    structure={"tableOrdinal": 1},
                    contentHash=content_hash(value),
                )
            )
            mappings.append(
                SourceMapping(
                    sourceLocators=[locator],
                    blockIds=[block_id],
                    transformation=Transformation(
                        kind="one_to_one", transformRef="receivables_tabular.v1"
                    ),
                )
            )
    return CanonicalSourceDocument(
        sourceIdentityRef=source_identity_ref,
        blocks=blocks,
        sourceMap=SourceMap(mappings=mappings),
        parserTrace=ParserTrace(
            provider="semantier-native-tabular",
            adapterVersion="receivables_tabular.v1",
            canonicalIrVersion="canonical_source_ir.v1",
        ),
    )


def tabular_evidence_refs(document, *, source_identity, parsed, created_at):
    """Build native cell evidence in one pass, preserving qualified selector identity."""
    from contracts.evidence_location import EvidenceRef, EvidenceSelector, TableSelector
    from semantier.knowledge_feature_packs.evidence_selector_builder import (
        evidence_ref_identity,
    )

    mappings = {
        mapping.block_ids[0]: mapping.source_locators[0]
        for mapping in document.source_map.mappings
    }
    # Preserve native evidence selector semantics for legacy header-row profiles;
    # the original header row is separately pinned in each source locator.
    headers = list(parsed.headers) if parsed.header_row == 1 else []
    evidence = []
    for block in document.blocks:
        locator = mappings[block.block_id]
        value = str(block.content)
        selector = EvidenceSelector(
            schemaVersion="semantier.evidence_selector.v1",
            selectorKind="table_cell",
            exactQuote=value or None,
            prefix=None,
            suffix=None,
            normalizedQuote=re.sub(r"\s+", " ", value).strip().casefold()
            if value
            else None,
            structuralPath=[
                "block:cell",
                "ordinal:" + str(block.ordinal),
                "localContentHash:" + block.content_hash,
            ],
            table=TableSelector(
                tableOrdinal=1,
                headingPath=[],
                headerPath=headers,
                rowKey=locator.row_id,
                columnKey=locator.column_id,
                cellText=value,
            ),
            sourceElementRef=locator.native_id,
        )
        selector_hash = selector.selector_hash()
        reference = evidence_ref_identity(
            organization_id=source_identity.organization_id,
            workspace_id=source_identity.workspace_id,
            source_identity_ref=source_identity.source_identity_ref,
            selector_hash=selector_hash,
            evidence_role="source",
        )
        evidence.append(
            EvidenceRef(
                schemaVersion="semantier.evidence_ref.v1",
                evidenceRef=reference,
                organizationId=source_identity.organization_id,
                workspaceId=source_identity.workspace_id,
                sourceIdentityRef=source_identity.source_identity_ref,
                selectorSchemaVersion="semantier.evidence_selector.v1",
                selectorHash=selector_hash,
                selector=selector,
                evidenceRole="source",
                provenance={
                    "source_block_id": block.block_id,
                    "source_hash": locator.source_hash,
                    "source_locator": locator.model_dump(mode="json", by_alias=True),
                    "artifact_ref": source_identity.artifact_ref,
                    "canonical_ir_version": document.canonical_ir_version,
                },
                createdAt=created_at,
            )
        )
    return tuple(evidence)
