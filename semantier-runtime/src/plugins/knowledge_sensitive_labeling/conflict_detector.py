from __future__ import annotations

from dataclasses import asdict

from contracts.knowledge_classification import (
    FieldFindingV1,
    OccurrenceFindingV1,
    RuntimeConflictV1,
    canonical_content_hash,
)


def detect_runtime_conflicts(
    *,
    source_hash: str,
    occurrences: list[OccurrenceFindingV1],
    fields: list[FieldFindingV1],
) -> list[RuntimeConflictV1]:
    conflicts: list[RuntimeConflictV1] = []
    by_span: dict[tuple[str, int, int], list[OccurrenceFindingV1]] = {}
    for finding in occurrences:
        by_span.setdefault(
            (finding.source_anchor_ref, finding.start_offset, finding.end_offset),
            [],
        ).append(finding)
    for (anchor_ref, start, end), items in by_span.items():
        classification_ids = tuple(sorted({item.classification_id for item in items}))
        if len(classification_ids) > 1:
            conflicts.append(
                _conflict(
                    conflict_type="same_span_incompatible_classifications",
                    source_hash=source_hash,
                    evidence_ref=f"{anchor_ref}:{start}:{end}",
                    classification_ids=classification_ids,
                    metadata={"finding_ids": sorted(item.finding_id for item in items)},
                )
            )
    anchored = sorted(
        occurrences,
        key=lambda item: (item.source_anchor_ref, item.start_offset, item.end_offset, item.classification_id),
    )
    for index, left in enumerate(anchored):
        for right in anchored[index + 1 :]:
            if left.source_anchor_ref != right.source_anchor_ref:
                continue
            if left.start_offset == right.start_offset and left.end_offset == right.end_offset:
                continue
            if left.start_offset < right.end_offset and right.start_offset < left.end_offset:
                classification_ids = tuple(sorted({left.classification_id, right.classification_id}))
                conflicts.append(
                    _conflict(
                        conflict_type="overlapping_spans_incompatible_actions",
                        source_hash=source_hash,
                        evidence_ref=f"{left.source_anchor_ref}:{min(left.start_offset, right.start_offset)}:{max(left.end_offset, right.end_offset)}",
                        classification_ids=classification_ids,
                        metadata={"finding_ids": sorted({left.finding_id, right.finding_id})},
                    )
                )
    by_field: dict[tuple[str, str, str], list[FieldFindingV1]] = {}
    for finding in fields:
        by_field.setdefault(
            (finding.record_id, finding.field_path, finding.value_hash),
            [],
        ).append(finding)
    for (record_id, field_path, value_hash), items in by_field.items():
        classification_ids = tuple(sorted({item.classification_id for item in items}))
        if len(classification_ids) > 1:
            conflicts.append(
                _conflict(
                    conflict_type="field_value_incompatible_classifications",
                    source_hash=source_hash,
                    evidence_ref=f"{record_id}:{field_path}:{value_hash}",
                    classification_ids=classification_ids,
                    metadata={"finding_ids": sorted(item.finding_id for item in items)},
                )
            )
    return conflicts


def _conflict(
    *,
    conflict_type: str,
    source_hash: str,
    evidence_ref: str,
    classification_ids: tuple[str, ...],
    metadata: dict[str, object],
) -> RuntimeConflictV1:
    payload = {
        "conflict_type": conflict_type,
        "source_hash": source_hash,
        "evidence_ref": evidence_ref,
        "classification_ids": classification_ids,
        "metadata": metadata,
    }
    conflict_hash = canonical_content_hash(payload)
    return RuntimeConflictV1(
        conflict_id="krc_" + conflict_hash.removeprefix("sha256:")[:24],
        conflict_type=conflict_type,
        source_hash=source_hash,
        evidence_ref=evidence_ref,
        classification_ids=classification_ids,
        content_hash=canonical_content_hash({**payload, "content_hash": None}),
        metadata={**metadata, "conflict_payload": asdict(RuntimeConflictV1(
            conflict_id="",
            conflict_type=conflict_type,
            source_hash=source_hash,
            evidence_ref=evidence_ref,
            classification_ids=classification_ids,
            content_hash="",
            metadata=metadata,
        ))},
    )
