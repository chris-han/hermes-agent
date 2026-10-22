from __future__ import annotations

import zipfile
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
from typing import Any
from xml.etree import ElementTree

from services.governed_artifact_service import StoreBackedGovernedArtifactService

COMMENT_WRITER_VERSION = "native_ooxml_comments.v2"
_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
_XML_NS = "http://www.w3.org/XML/1998/namespace"
_W = f"{{{_W_NS}}}"
_REL = f"{{{_REL_NS}}}"
_CT = f"{{{_CT_NS}}}"
_XML = f"{{{_XML_NS}}}"

ElementTree.register_namespace("w", _W_NS)
ElementTree.register_namespace("r", _R_NS)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _docx_targets_by_body_index(
    document_xml: bytes,
) -> tuple[ElementTree.Element, dict[int, ElementTree.Element], dict[tuple[int, int, int], ElementTree.Element]]:
    root = ElementTree.fromstring(document_xml)
    body = root.find(f"{_W}body")
    if body is None:
        raise ValueError("DOCX_BODY_MISSING")
    body_index = 0
    table_index = 0
    paragraphs: dict[int, ElementTree.Element] = {}
    table_cells: dict[tuple[int, int, int], ElementTree.Element] = {}
    for child in list(body):
        if child.tag not in {f"{_W}p", f"{_W}tbl"}:
            continue
        body_index += 1
        if child.tag == f"{_W}p":
            paragraphs[body_index] = child
            continue
        table_index += 1
        for row_index, row in enumerate(child.findall(f"{_W}tr"), start=1):
            for cell_index, cell in enumerate(row.findall(f"{_W}tc"), start=1):
                paragraph = cell.find(f"{_W}p")
                if paragraph is not None:
                    table_cells[(table_index, row_index, cell_index)] = paragraph
    return root, paragraphs, table_cells


def _ensure_comments_relationship(rels_xml: bytes | None) -> bytes:
    if rels_xml:
        root = ElementTree.fromstring(rels_xml)
    else:
        root = ElementTree.Element(f"{_REL}Relationships")
    for rel in root.findall(f"{_REL}Relationship"):
        if rel.attrib.get("Type") == (
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments"
        ):
            return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)
    max_id = 0
    for rel in root.findall(f"{_REL}Relationship"):
        rel_id = rel.attrib.get("Id", "")
        if rel_id.startswith("rId") and rel_id[3:].isdigit():
            max_id = max(max_id, int(rel_id[3:]))
    ElementTree.SubElement(
        root,
        f"{_REL}Relationship",
        {
            "Id": f"rId{max_id + 1}",
            "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments",
            "Target": "comments.xml",
        },
    )
    return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)


def _ensure_comments_content_type(content_types_xml: bytes) -> bytes:
    root = ElementTree.fromstring(content_types_xml)
    for override in root.findall(f"{_CT}Override"):
        if override.attrib.get("PartName") == "/word/comments.xml":
            return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)
    ElementTree.SubElement(
        root,
        f"{_CT}Override",
        {
            "PartName": "/word/comments.xml",
            "ContentType": "application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml",
        },
    )
    return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)


def _comments_xml(comments: list[dict[str, Any]]) -> bytes:
    root = ElementTree.Element(f"{_W}comments")
    for index, comment in enumerate(comments):
        node = ElementTree.SubElement(
            root,
            f"{_W}comment",
            {
                f"{_W}id": str(index),
                f"{_W}author": str(comment.get("author") or "Semantier"),
                f"{_W}date": str(comment.get("created_at") or _now_iso()),
            },
        )
        paragraph = ElementTree.SubElement(node, f"{_W}p")
        run = ElementTree.SubElement(paragraph, f"{_W}r")
        text = ElementTree.SubElement(run, f"{_W}t")
        text.text = str(comment["text"])
    return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)


def _append_comment_reference(paragraph: ElementTree.Element, comment_id: int) -> None:
    start = ElementTree.Element(f"{_W}commentRangeStart", {f"{_W}id": str(comment_id)})
    end = ElementTree.Element(f"{_W}commentRangeEnd", {f"{_W}id": str(comment_id)})
    ref_run = ElementTree.Element(f"{_W}r")
    ElementTree.SubElement(ref_run, f"{_W}commentReference", {f"{_W}id": str(comment_id)})
    insert_at = len(paragraph)
    paragraph.insert(insert_at, start)
    paragraph.insert(insert_at + 1, end)
    paragraph.insert(insert_at + 2, ref_run)


def _paragraph_text(paragraph: ElementTree.Element) -> str:
    return "".join(text.text or "" for text in paragraph.iter(f"{_W}t"))


def _text_run(text_value: str) -> ElementTree.Element:
    run = ElementTree.Element(f"{_W}r")
    text = ElementTree.SubElement(run, f"{_W}t")
    if text_value[:1].isspace() or text_value[-1:].isspace():
        text.attrib[f"{_XML}space"] = "preserve"
    text.text = text_value
    return run


def _append_exact_comment_range(
    paragraph: ElementTree.Element,
    comment_id: int,
    matched_text: str,
) -> None:
    start = ElementTree.Element(f"{_W}commentRangeStart", {f"{_W}id": str(comment_id)})
    end = ElementTree.Element(f"{_W}commentRangeEnd", {f"{_W}id": str(comment_id)})
    ref_run = ElementTree.Element(f"{_W}r")
    ElementTree.SubElement(ref_run, f"{_W}commentReference", {f"{_W}id": str(comment_id)})
    paragraph.append(start)
    paragraph.append(_text_run(matched_text))
    paragraph.append(end)
    paragraph.append(ref_run)


def _apply_paragraph_comment_ranges(
    paragraph: ElementTree.Element,
    ranges: list[dict[str, Any]],
) -> None:
    paragraph_text = _paragraph_text(paragraph)
    sorted_ranges = sorted(ranges, key=lambda item: int(item["start_offset"]))
    previous_end = 0
    paragraph_properties = [child for child in list(paragraph) if child.tag == f"{_W}pPr"][:1]
    source_runs = [
        child
        for child in list(paragraph)
        if child.tag == f"{_W}r" and _run_text(child)
    ]
    for child in list(paragraph):
        paragraph.remove(child)
    for child in paragraph_properties:
        paragraph.append(child)
    if not source_runs:
        source_runs = []

    def append_text_segment(start: int, end: int) -> None:
        if end <= start:
            return
        cursor = 0
        for run in source_runs:
            run_text = _run_text(run)
            run_start = cursor
            run_end = cursor + len(run_text)
            overlap_start = max(start, run_start)
            overlap_end = min(end, run_end)
            if overlap_end > overlap_start:
                clone = deepcopy(run)
                text_nodes = list(clone.iter(f"{_W}t"))
                if text_nodes:
                    text_nodes[0].text = run_text[overlap_start - run_start : overlap_end - run_start]
                    for text_node in text_nodes[1:]:
                        text_node.text = ""
                    paragraph.append(clone)
            cursor = run_end
        if not source_runs:
            paragraph.append(_text_run(paragraph_text[start:end]))

    for item in sorted_ranges:
        start = int(item["start_offset"])
        end = int(item["end_offset"])
        append_text_segment(previous_end, start)
        paragraph.append(
            ElementTree.Element(f"{_W}commentRangeStart", {f"{_W}id": str(item["comment_id"])})
        )
        append_text_segment(start, end)
        paragraph.append(
            ElementTree.Element(f"{_W}commentRangeEnd", {f"{_W}id": str(item["comment_id"])})
        )
        ref_run = ElementTree.Element(f"{_W}r")
        ElementTree.SubElement(ref_run, f"{_W}commentReference", {f"{_W}id": str(item["comment_id"])})
        paragraph.append(ref_run)
        previous_end = end
    if previous_end < len(paragraph_text):
        append_text_segment(previous_end, len(paragraph_text))


def _run_text(run: ElementTree.Element) -> str:
    return "".join(text.text or "" for text in run.iter(f"{_W}t"))


def _comment_text(finding: dict[str, Any]) -> str:
    verdict = finding.get("verdict") if isinstance(finding.get("verdict"), dict) else {}
    issue = finding.get("issue_type") or verdict.get("severity") or verdict.get("decision") or "finding"
    basis = finding.get("judgment_basis") or verdict.get("remediation_action") or ""
    ontology_rule_id = finding.get("ontology_rule_id") or verdict.get("ontology_rule_id") or ""
    priority = finding.get("priority") or verdict.get("priority") or ""
    evidence = " ".join(
        item
        for item in (
            f"rule={ontology_rule_id}" if ontology_rule_id else "",
            f"priority={priority}" if priority else "",
        )
        if item
    )
    return f"{issue}: {finding.get('matched_text', '')}. {evidence} {basis}".strip()


def write_labeled_docx_bytes(
    *,
    source_docx_bytes: bytes,
    canonical_document: dict[str, Any],
    findings: list[dict[str, Any]],
    created_by: str,
    created_at: str | None = None,
) -> tuple[bytes, dict[str, Any]]:
    if created_at is not None:
        instant = datetime.fromisoformat(created_at)
        if instant.tzinfo is None or instant.utcoffset() != timezone.utc.utcoffset(instant):
            raise ValueError("DOCX_COMMENT_UTC_REQUIRED")
    elements_by_anchor = {
        str(element.get("anchor_ref") or element.get("id")): element
        for element in canonical_document.get("elements") or []
    }
    for element in canonical_document.get("elements") or []:
        if element.get("kind") != "table":
            continue
        for row in element.get("rows") or []:
            for cell in row.get("cells") or []:
                anchor_ref = str(cell.get("anchor_ref") or "")
                if anchor_ref:
                    elements_by_anchor[anchor_ref] = {
                        **cell,
                        "kind": "table_cell",
                        "body_index": element.get("body_index"),
                        "table_index": element.get("table_index"),
                        "row_index": cell.get("row") or row.get("row"),
                        "cell_index": cell.get("column"),
                    }
    paragraph_comments: list[dict[str, Any]] = []
    pending_ranges_by_target: dict[tuple[str, int | tuple[int, int, int]], list[dict[str, Any]]] = {}
    unsupported: list[dict[str, Any]] = []
    output_buffer = BytesIO()
    with zipfile.ZipFile(BytesIO(source_docx_bytes)) as source:
        document_root, paragraphs, table_cells = _docx_targets_by_body_index(source.read("word/document.xml"))
        for finding in findings:
            element = elements_by_anchor.get(str(finding.get("target_anchor_ref") or ""))
            finding_id = str(finding.get("finding_id") or "")
            if not element:
                unsupported.append({"finding_id": finding_id, "reason": "DOCX_ANCHOR_NOT_FOUND"})
                continue
            target_kind = str(element.get("kind") or "")
            if target_kind == "paragraph":
                target_key: tuple[str, int | tuple[int, int, int]] = (
                    "paragraph",
                    int(element.get("body_index") or 0),
                )
                paragraph = paragraphs.get(int(element.get("body_index") or 0))
                missing_reason = "DOCX_PARAGRAPH_NOT_FOUND"
            elif target_kind == "table_cell":
                try:
                    cell_key = (
                        int(element.get("table_index") or 0),
                        int(element.get("row_index") or element.get("row") or 0),
                        int(element.get("cell_index") or element.get("column") or 0),
                    )
                except (TypeError, ValueError):
                    unsupported.append({"finding_id": finding_id, "reason": "DOCX_TABLE_CELL_COORDINATES_REQUIRED"})
                    continue
                target_key = ("table_cell", cell_key)
                paragraph = table_cells.get(cell_key)
                missing_reason = "DOCX_TABLE_CELL_NOT_FOUND"
            else:
                unsupported.append({"finding_id": finding_id, "reason": "DOCX_ANCHOR_KIND_UNSUPPORTED"})
                continue
            if paragraph is None:
                unsupported.append({"finding_id": finding_id, "reason": missing_reason})
                continue
            paragraph_text = _paragraph_text(paragraph)
            canonical_text = str(element.get("text") or "")
            if canonical_text and canonical_text != paragraph_text.strip():
                unsupported.append({"finding_id": finding_id, "reason": "DOCX_ANCHOR_TEXT_MISMATCH"})
                continue
            try:
                start = int(finding.get("start_offset"))
                end = int(finding.get("end_offset"))
            except (TypeError, ValueError):
                unsupported.append({"finding_id": finding_id, "reason": "DOCX_FINDING_OFFSETS_REQUIRED"})
                continue
            if start < 0 or end <= start:
                unsupported.append({"finding_id": finding_id, "reason": "DOCX_FINDING_OFFSETS_INVALID"})
                continue
            leading_trim = len(paragraph_text) - len(paragraph_text.lstrip())
            absolute_start = leading_trim + start
            absolute_end = leading_trim + end
            matched_text = str(finding.get("matched_text") or "")
            if absolute_end > len(paragraph_text):
                unsupported.append({"finding_id": finding_id, "reason": "DOCX_FINDING_RANGE_OUT_OF_BOUNDS"})
                continue
            if matched_text and paragraph_text[absolute_start:absolute_end] != matched_text:
                unsupported.append({"finding_id": finding_id, "reason": "DOCX_FINDING_RANGE_TEXT_MISMATCH"})
                continue
            ranges = pending_ranges_by_target.setdefault(target_key, [])
            if any(
                absolute_start < int(item["end_offset"]) and absolute_end > int(item["start_offset"])
                for item in ranges
            ):
                unsupported.append({"finding_id": finding_id, "reason": "DOCX_FINDING_RANGE_OVERLAP_UNSUPPORTED"})
                continue
            comment_id = len(paragraph_comments)
            ranges.append(
                {
                    "finding_id": finding_id,
                    "comment_id": comment_id,
                    "target_anchor_ref": str(finding.get("target_anchor_ref") or ""),
                    "start_offset": absolute_start,
                    "end_offset": absolute_end,
                }
            )
            paragraph_comments.append(
                {
                    "author": created_by,
                    "created_at": created_at if created_at is not None else _now_iso(),
                    "text": _comment_text(finding),
                }
            )
        for target_key, ranges in pending_ranges_by_target.items():
            if target_key[0] == "paragraph":
                paragraph = paragraphs.get(int(target_key[1]))
            else:
                paragraph = table_cells.get(target_key[1])  # type: ignore[arg-type]
            if paragraph is not None:
                _apply_paragraph_comment_ranges(paragraph, ranges)
        replacements = {
            "word/document.xml": ElementTree.tostring(
                document_root, encoding="utf-8", xml_declaration=True
            ),
            "word/comments.xml": _comments_xml(paragraph_comments),
        }
        try:
            content_types_xml = source.read("[Content_Types].xml")
        except KeyError:
            content_types_xml = (
                b'<?xml version="1.0" encoding="UTF-8"?>'
                b'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                b'<Override PartName="/word/document.xml" '
                b'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                b"</Types>"
            )
        replacements["[Content_Types].xml"] = _ensure_comments_content_type(content_types_xml)
        try:
            rels_xml = source.read("word/_rels/document.xml.rels")
        except KeyError:
            rels_xml = None
        replacements["word/_rels/document.xml.rels"] = _ensure_comments_relationship(rels_xml)

        with zipfile.ZipFile(output_buffer, "w", compression=zipfile.ZIP_DEFLATED) as output:
            def write_replacement(name, data):
                if created_at is None:
                    output.writestr(name, data)
                else:
                    # Fixed ZIP metadata, independent of process clock/timezone.
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    output.writestr(info, data)
            written = set()
            for info in source.infolist():
                if info.filename in replacements:
                    write_replacement(info.filename, replacements[info.filename])
                    written.add(info.filename)
                else:
                    output.writestr(info, source.read(info.filename))
            for name, data in replacements.items():
                if name not in written:
                    write_replacement(name, data)
    projection_manifest = {
        "comment_projection_schema_version": "tender_docx_comment_projection.v1",
        "comment_writer_version": COMMENT_WRITER_VERSION,
        "comments_written": len(paragraph_comments),
        "unsupported_findings": unsupported,
        "comment_ranges": [
            {
                "finding_id": str(item["finding_id"]),
                "comment_id": int(item["comment_id"]),
                "target_anchor_ref": str(item["target_anchor_ref"]),
                "start_offset": int(item["start_offset"]),
                "end_offset": int(item["end_offset"]),
            }
            for ranges in pending_ranges_by_target.values()
            for item in sorted(ranges, key=lambda value: int(value["comment_id"]))
        ],
    }
    return output_buffer.getvalue(), projection_manifest


def generate_labeled_docx(*, ctx: Any, **kwargs: Any) -> dict[str, Any]:
    db_path = kwargs.pop("db_path", None)
    service = kwargs.pop("artifact_service", None) or StoreBackedGovernedArtifactService(db_path=db_path)
    return service.create_labeled_docx_derivative(
        ctx=ctx,
        comment_writer_version=COMMENT_WRITER_VERSION,
        writer=write_labeled_docx_bytes,
        **kwargs,
    )
