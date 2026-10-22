from __future__ import annotations

from typing import Any

from contracts.knowledge_classification import (
    CanonicalClassificationInputV1,
    CanonicalFieldValueV1,
    CanonicalRecordV1,
    CanonicalSegmentV1,
    RuntimeContextDTO,
    canonical_content_hash,
)


def canonicalize_structured_records(
    *,
    records: list[dict[str, Any]],
    context: RuntimeContextDTO,
    source_ref: str,
    source_hash: str,
    parser_profile_version: str,
    normalized_artifact_ref: str | None,
    normalized_artifact_hash: str | None,
) -> CanonicalClassificationInputV1:
    canonical_records: list[CanonicalRecordV1] = []
    for record_index, record in enumerate(records, start=1):
        record_id = str(record.get("record_id") or f"record:{record_index}")
        record_type = str(record.get("record_type") or "structured_record")
        raw_fields = record.get("fields") or record.get("field_values") or {}
        if not isinstance(raw_fields, dict):
            raise ValueError("STRUCTURED_RECORD_FIELDS_REQUIRED")
        fields: list[CanonicalFieldValueV1] = []
        for field_index, (field_name, value) in enumerate(sorted(raw_fields.items()), start=1):
            normalized_value = "" if value is None else str(value).strip()
            field_path = str(field_name)
            anchor_ref = f"{source_ref}#record={record_id};field={field_path}"
            field_payload = {
                "record_id": record_id,
                "field_path": field_path,
                "normalized_value": normalized_value,
                "source_anchor_ref": anchor_ref,
            }
            fields.append(
                CanonicalFieldValueV1(
                    field_id=f"{record_id}:field:{field_index}",
                    field_name=str(field_name),
                    field_path=field_path,
                    declared_type=type(value).__name__,
                    normalized_value=normalized_value,
                    display_value_ref=record.get("display_value_refs", {}).get(field_name)
                    if isinstance(record.get("display_value_refs"), dict)
                    else None,
                    source_anchor_ref=anchor_ref,
                    value_hash=canonical_content_hash(field_payload),
                )
            )
        record_payload = {
            "record_id": record_id,
            "record_type": record_type,
            "fields": [field.value_hash for field in fields],
        }
        canonical_records.append(
            CanonicalRecordV1(
                record_id=record_id,
                record_type=record_type,
                field_values=tuple(fields),
                source_anchor_ref=f"{source_ref}#record={record_id}",
                record_hash=canonical_content_hash(record_payload),
            )
        )
    return CanonicalClassificationInputV1.build(
        input_id="kci_" + canonical_content_hash(
            {
                "source_hash": source_hash,
                "source_ref": source_ref,
                "records": [record.record_hash for record in canonical_records],
            }
        ).removeprefix("sha256:")[:32],
        organization_id=context.organization_id,
        workspace_id=context.workspace_id,
        source_kind="structured_records",
        source_ref=source_ref,
        source_hash=source_hash,
        normalized_artifact_ref=normalized_artifact_ref,
        normalized_artifact_hash=normalized_artifact_hash,
        parser_profile_version=parser_profile_version,
        schema_version="canonical_classification_input.v1",
        records=tuple(canonical_records),
        lineage_manifest={
            "normalization_stage": "structured_record_adapter",
            "authority_role": "evidence_only",
        },
    )


def canonicalize_unstructured_document(
    *,
    document_text: str,
    canonical_document: dict[str, Any] | None,
    context: RuntimeContextDTO,
    source_kind: str,
    source_ref: str,
    source_hash: str,
    parser_profile_version: str,
    normalized_artifact_ref: str | None,
    normalized_artifact_hash: str | None,
) -> CanonicalClassificationInputV1:
    segments: list[CanonicalSegmentV1] = []
    if canonical_document is not None:
        elements = canonical_document.get("elements") or []
        for index, element in enumerate(elements, start=1):
            text = str(element.get("text") or "")
            if not text:
                continue
            anchor_ref = str(element.get("anchor_ref") or element.get("id") or f"{source_ref}#element={index}")
            local_text_hash = str(element.get("local_text_hash") or canonical_content_hash(text))
            segments.append(
                CanonicalSegmentV1(
                    segment_id=f"segment:{index}",
                    segment_kind=str(element.get("kind") or "document_element"),
                    text=text,
                    normalized_text=text.strip().lower(),
                    source_anchor_ref=anchor_ref,
                    parent_segment_id=None,
                    order_index=index,
                    local_text_hash=local_text_hash,
                    layout_metadata={
                        key: value
                        for key, value in element.items()
                        if key not in {"text", "anchor_ref", "local_text_hash"}
                    },
                )
            )
    else:
        for index, line in enumerate([item for item in document_text.splitlines() if item.strip()] or [document_text], start=1):
            anchor_ref = f"{source_ref}#segment={index}"
            segments.append(
                CanonicalSegmentV1(
                    segment_id=f"segment:{index}",
                    segment_kind="text",
                    text=line,
                    normalized_text=line.strip().lower(),
                    source_anchor_ref=anchor_ref,
                    parent_segment_id=None,
                    order_index=index,
                    local_text_hash=canonical_content_hash(line),
                    layout_metadata={},
                )
            )
    return CanonicalClassificationInputV1.build(
        input_id="kci_" + canonical_content_hash(
            {
                "source_hash": source_hash,
                "source_ref": source_ref,
                "segments": [segment.local_text_hash for segment in segments],
            }
        ).removeprefix("sha256:")[:32],
        organization_id=context.organization_id,
        workspace_id=context.workspace_id,
        source_kind=source_kind,
        source_ref=source_ref,
        source_hash=source_hash,
        normalized_artifact_ref=normalized_artifact_ref,
        normalized_artifact_hash=normalized_artifact_hash,
        parser_profile_version=parser_profile_version,
        schema_version="canonical_classification_input.v1",
        segments=tuple(segments),
        lineage_manifest={
            "normalization_stage": "unstructured_document_adapter",
            "authority_role": "evidence_only",
            "graph_projection": "optional_not_requested",
        },
    )
