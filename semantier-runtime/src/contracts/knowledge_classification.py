from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from eos.artifact_hashing import content_hash

CONTRACT_VERSION = "knowledge_classification.v1"


def canonical_content_hash(value: Any) -> str:
    return "sha256:" + content_hash(value)


@dataclass(frozen=True)
class RuntimeContextDTO:
    organization_id: str
    workspace_id: str | None
    actor_id: str
    actor_role: str | None = None
    channel: str | None = None
    session_id: str | None = None
    capabilities: tuple[str, ...] = ()
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class CanonicalFieldValueV1:
    field_id: str
    field_name: str
    field_path: str
    declared_type: str
    normalized_value: str
    display_value_ref: str | None
    source_anchor_ref: str
    value_hash: str
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class CanonicalRecordV1:
    record_id: str
    record_type: str
    field_values: tuple[CanonicalFieldValueV1, ...]
    source_anchor_ref: str
    record_hash: str
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class CanonicalSegmentV1:
    segment_id: str
    segment_kind: str
    text: str
    normalized_text: str
    source_anchor_ref: str
    parent_segment_id: str | None
    order_index: int
    local_text_hash: str
    layout_metadata: dict[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class CanonicalEntityV1:
    entity_id: str
    entity_type: str
    normalized_name: str
    mention_anchor_refs: tuple[str, ...]
    confidence_evidence: dict[str, Any]
    content_hash: str
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class CanonicalRelationV1:
    relation_id: str
    subject_entity_id: str
    predicate: str
    object_entity_id_or_value: str
    evidence_anchor_refs: tuple[str, ...]
    confidence_evidence: dict[str, Any]
    content_hash: str
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class CanonicalClassificationInputV1:
    input_id: str
    organization_id: str
    workspace_id: str | None
    source_kind: str
    source_ref: str
    source_hash: str
    normalized_artifact_ref: str | None
    normalized_artifact_hash: str | None
    parser_profile_version: str
    schema_version: str
    records: tuple[CanonicalRecordV1, ...] = ()
    segments: tuple[CanonicalSegmentV1, ...] = ()
    entities: tuple[CanonicalEntityV1, ...] = ()
    relations: tuple[CanonicalRelationV1, ...] = ()
    graph_projection_ref: str | None = None
    lineage_manifest: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    contract_version: str = CONTRACT_VERSION

    @staticmethod
    def build(
        *,
        input_id: str,
        organization_id: str,
        workspace_id: str | None,
        source_kind: str,
        source_ref: str,
        source_hash: str,
        normalized_artifact_ref: str | None,
        normalized_artifact_hash: str | None,
        parser_profile_version: str,
        schema_version: str,
        records: tuple[CanonicalRecordV1, ...] = (),
        segments: tuple[CanonicalSegmentV1, ...] = (),
        entities: tuple[CanonicalEntityV1, ...] = (),
        relations: tuple[CanonicalRelationV1, ...] = (),
        graph_projection_ref: str | None = None,
        lineage_manifest: dict[str, Any] | None = None,
    ) -> "CanonicalClassificationInputV1":
        hash_payload = {
            "input_id": input_id,
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "source_kind": source_kind,
            "source_ref": source_ref,
            "source_hash": source_hash,
            "normalized_artifact_ref": normalized_artifact_ref,
            "normalized_artifact_hash": normalized_artifact_hash,
            "parser_profile_version": parser_profile_version,
            "schema_version": schema_version,
            "records": [asdict(item) for item in records],
            "segments": [asdict(item) for item in segments],
            "entities": [asdict(item) for item in entities],
            "relations": [asdict(item) for item in relations],
            "graph_projection_ref": graph_projection_ref,
            "lineage_manifest": lineage_manifest or {},
            "contract_version": CONTRACT_VERSION,
        }
        return CanonicalClassificationInputV1(
            input_id=input_id,
            organization_id=organization_id,
            workspace_id=workspace_id,
            source_kind=source_kind,
            source_ref=source_ref,
            source_hash=source_hash,
            normalized_artifact_ref=normalized_artifact_ref,
            normalized_artifact_hash=normalized_artifact_hash,
            parser_profile_version=parser_profile_version,
            schema_version=schema_version,
            records=records,
            segments=segments,
            entities=entities,
            relations=relations,
            graph_projection_ref=graph_projection_ref,
            lineage_manifest=lineage_manifest or {},
            content_hash=canonical_content_hash(hash_payload),
        )


@dataclass(frozen=True)
class ParseRunCreateDTO:
    source_kind: str
    source_ref: str
    source_hash: str
    parser_profile_version: str
    normalized_artifact_ref: str | None
    normalized_artifact_hash: str | None
    idempotency_key: str
    metadata: dict[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class ParseRunDTO:
    parse_run_id: str
    input_id: str
    status: str
    source_hash: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class ParseEvidenceDTO:
    run: ParseRunDTO
    canonical_input: CanonicalClassificationInputV1
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class ClassificationRunCreateDTO:
    input_id: str
    source_hash: str
    activation_set_snapshot_id: str
    resolved_activation_set_hash: str
    rule_version_set_hash: str
    classifier_profile_version: str
    ontology_version: str
    idempotency_key: str
    metadata: dict[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class ClassificationRunV1:
    run_id: str
    input_id: str
    status: str
    source_hash: str
    activation_set_snapshot_id: str
    resolved_activation_set_hash: str
    rule_version_set_hash: str
    classifier_profile_version: str
    ontology_version: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class OccurrenceFindingV1:
    finding_id: str
    source_hash: str
    source_anchor_ref: str
    start_offset: int
    end_offset: int
    occurrence_index: int
    classification_id: str
    matched_text: str
    local_text_hash: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class FieldFindingV1:
    finding_id: str
    source_hash: str
    record_id: str
    field_path: str
    value_hash: str
    classification_id: str
    matched_text: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class EntityFindingV1:
    finding_id: str
    source_hash: str
    entity_id: str
    evidence_anchor_hash: str
    classification_id: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class RelationFindingV1:
    finding_id: str
    source_hash: str
    relation_id: str
    evidence_anchor_hash: str
    classification_id: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class RuntimeConflictV1:
    conflict_id: str
    conflict_type: str
    source_hash: str
    evidence_ref: str
    classification_ids: tuple[str, ...]
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class ClassificationSummaryV1:
    run_id: str
    occurrence_count: int
    field_count: int
    entity_count: int
    relation_count: int
    conflict_count: int
    evidence_set_hash: str
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class ClassificationEvidenceDTO:
    run: ClassificationRunV1
    occurrences: tuple[OccurrenceFindingV1, ...] = ()
    fields: tuple[FieldFindingV1, ...] = ()
    entities: tuple[EntityFindingV1, ...] = ()
    relations: tuple[RelationFindingV1, ...] = ()
    runtime_conflicts: tuple[RuntimeConflictV1, ...] = ()
    summary: ClassificationSummaryV1 | None = None
    contract_version: str = CONTRACT_VERSION
