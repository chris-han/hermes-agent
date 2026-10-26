"""Shared derived contracts for governed cross-artifact delta comparison.

These contracts are evidence/read projections only. They do not own admission,
activation, semantic authority, or persistence lifecycle.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eos.artifact_hashing import canonical_json, content_hash


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


class _FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DeltaTargetType(StrEnum):
    WORKFLOW = "WORKFLOW"
    KNOWLEDGE_GRAPH = "KNOWLEDGE_GRAPH"
    ONTOLOGY = "ONTOLOGY"
    POLICY = "POLICY"
    DECISION_SPACE = "DECISION_SPACE"
    PROJECTION_BUNDLE = "PROJECTION_BUNDLE"
    OTHER = "OTHER"


class DeltaLens(StrEnum):
    STRUCTURAL = "STRUCTURAL"
    SEMANTIC = "SEMANTIC"
    AUTHORITY = "AUTHORITY"
    BEHAVIOR = "BEHAVIOR"
    EVIDENCE = "EVIDENCE"
    EXECUTION_BINDING = "EXECUTION_BINDING"
    PRESENTATION = "PRESENTATION"


class DeltaLensStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class DeltaOperationKind(StrEnum):
    ADD = "ADD"
    REMOVE = "REMOVE"
    MODIFY = "MODIFY"
    MOVE = "MOVE"
    REBIND = "REBIND"
    SUPERSEDE = "SUPERSEDE"
    SPLIT = "SPLIT"
    MERGE = "MERGE"
    NO_CHANGE = "NO_CHANGE"


class DeltaMateriality(StrEnum):
    MATERIAL = "MATERIAL"
    NON_MATERIAL = "NON_MATERIAL"
    UNKNOWN = "UNKNOWN"


class ImpactRelation(StrEnum):
    DEPENDS_ON = "DEPENDS_ON"
    PROJECTS_FROM = "PROJECTS_FROM"
    EXECUTES = "EXECUTES"
    VERIFIES = "VERIFIES"
    GOVERNS = "GOVERNS"
    BINDS = "BINDS"
    REFERENCES = "REFERENCES"


class ImpactStatus(StrEnum):
    DIRECT = "DIRECT"
    TRANSITIVE = "TRANSITIVE"
    POSSIBLE = "POSSIBLE"
    UNAVAILABLE = "UNAVAILABLE"


class InvariantDisposition(StrEnum):
    PRESERVED = "PRESERVED"
    CHANGED = "CHANGED"
    REQUIRES_VERIFICATION = "REQUIRES_VERIFICATION"
    UNAVAILABLE = "UNAVAILABLE"


class QualificationStatus(StrEnum):
    REQUIRED = "REQUIRED"
    SATISFIED = "SATISFIED"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"


class DeltaLifecycleState(StrEnum):
    CURRENT = "CURRENT"
    SUCCESSOR_DRAFTED = "SUCCESSOR_DRAFTED"
    DELTA_BOUND = "DELTA_BOUND"
    IMPACT_PROJECTED = "IMPACT_PROJECTED"
    QUALIFIED = "QUALIFIED"
    CHANGE_ASSESSED = "CHANGE_ASSESSED"
    ADMITTED = "ADMITTED"
    ACTIVATED = "ACTIVATED"
    BOUND_IN_FUTURE_EXECUTION = "BOUND_IN_FUTURE_EXECUTION"
    SUPERSEDED = "SUPERSEDED"
    REJECTED = "REJECTED"


class ArtifactVersionPinV1(_FrozenContract):
    schema_version: Literal["governed_artifact_version_pin.v1"] = (
        "governed_artifact_version_pin.v1"
    )
    artifact_type: str
    artifact_id: str
    version_id: str
    content_hash: str = Field(pattern=SHA256_PATTERN)
    owner_ref: str
    organization_id: str | None = None
    workspace_id: str | None = None
    effective_from: str | None = None
    effective_to: str | None = None
    parent_version_ref: str | None = None

    @model_validator(mode="after")
    def validate_identity(self) -> "ArtifactVersionPinV1":
        for field_name in ("artifact_type", "artifact_id", "version_id", "owner_ref"):
            value = getattr(self, field_name)
            if not value or not value.isascii():
                raise ValueError(f"{field_name} must be a non-empty ASCII-stable identifier")
        for field_name in ("organization_id", "workspace_id", "parent_version_ref"):
            value = getattr(self, field_name)
            if value is not None and (not value or not value.isascii()):
                raise ValueError(f"{field_name} must be null or a non-empty ASCII-stable identifier")
        return self


class DeltaOperationV1(_FrozenContract):
    schema_version: Literal["governed_delta_operation.v1"] = "governed_delta_operation.v1"
    operation_id: str
    operation_kind: DeltaOperationKind
    lens: DeltaLens
    subject_ref: str
    before_ref: str | None = None
    after_ref: str | None = None
    changed_fields: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    materiality: DeltaMateriality

    @model_validator(mode="after")
    def validate_operation(self) -> "DeltaOperationV1":
        if not self.operation_id or not self.operation_id.isascii():
            raise ValueError("operation_id must be a non-empty ASCII-stable identifier")
        if not self.subject_ref or not self.subject_ref.isascii():
            raise ValueError("subject_ref must be a non-empty ASCII-stable identifier")
        if tuple(sorted(set(self.changed_fields))) != self.changed_fields:
            raise ValueError("changed_fields must be sorted and unique")
        if tuple(sorted(set(self.evidence_refs))) != self.evidence_refs:
            raise ValueError("evidence_refs must be sorted and unique")
        return self


class DeltaLensResultV1(_FrozenContract):
    schema_version: Literal["governed_delta_lens_result.v1"] = (
        "governed_delta_lens_result.v1"
    )
    lens: DeltaLens
    status: DeltaLensStatus
    operations: tuple[DeltaOperationV1, ...] = ()
    basis_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_lens(self) -> "DeltaLensResultV1":
        if self.status is not DeltaLensStatus.AVAILABLE and self.operations:
            raise ValueError("non-AVAILABLE lens cannot carry operations")
        if tuple(sorted(set(self.basis_refs))) != self.basis_refs:
            raise ValueError("basis_refs must be sorted and unique")
        op_ids = tuple(op.operation_id for op in self.operations)
        if tuple(sorted(set(op_ids))) != op_ids:
            raise ValueError("operations must be sorted by unique operation_id")
        if any(op.lens is not self.lens for op in self.operations):
            raise ValueError("operation lens must match containing lens")
        return self


class DeltaAncestryV1(_FrozenContract):
    common_ancestor_ref: str | None = None
    branch_ref: str | None = None


class MaterialChangeSummaryV1(_FrozenContract):
    material_lenses: tuple[DeltaLens, ...] = ()
    unchanged_lenses: tuple[DeltaLens, ...] = ()
    unavailable_lenses: tuple[DeltaLens, ...] = ()

    @model_validator(mode="after")
    def validate_partition(self) -> "MaterialChangeSummaryV1":
        groups = [
            set(self.material_lenses),
            set(self.unchanged_lenses),
            set(self.unavailable_lenses),
        ]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("material-change lens groups must be disjoint")
        return self


class DeltaEnvelopeV1(_FrozenContract):
    schema_version: Literal["governed_delta_envelope.v1"] = "governed_delta_envelope.v1"
    delta_id: str
    target_type: DeltaTargetType
    predecessor: ArtifactVersionPinV1
    successor_candidate: ArtifactVersionPinV1
    ancestry: DeltaAncestryV1 = DeltaAncestryV1()
    lenses: tuple[DeltaLensResultV1, ...]
    material_change_summary: MaterialChangeSummaryV1
    content_hash: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_envelope(self) -> "DeltaEnvelopeV1":
        if self.predecessor.artifact_type != self.successor_candidate.artifact_type:
            raise ValueError("predecessor and successor artifact_type must match")
        if self.predecessor.artifact_id != self.successor_candidate.artifact_id:
            raise ValueError("predecessor and successor artifact_id must match")
        lens_values = tuple(item.lens for item in self.lenses)
        expected = tuple(DeltaLens)
        if lens_values != expected:
            raise ValueError("lenses must contain all seven canonical lenses in canonical order")
        expected_hash = "sha256:" + content_hash(self.hash_payload())
        if self.content_hash != expected_hash:
            raise ValueError("delta envelope content_hash does not match canonical payload")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("content_hash")
        return payload

    def canonical_json(self) -> str:
        return canonical_json(self.model_dump(mode="json"))


class AffectedArtifactV1(_FrozenContract):
    artifact_ref: str
    relation: ImpactRelation
    impact_kind: DeltaLens
    impact_status: ImpactStatus
    basis_refs: tuple[str, ...] = ()


class AffectedInvariantV1(_FrozenContract):
    invariant_ref: str
    disposition: InvariantDisposition


class RequiredQualificationV1(_FrozenContract):
    qualification_ref: str
    owner_ref: str
    status: QualificationStatus


class ImpactPopulationScopeV2(_FrozenContract):
    schema_version: Literal["impact_population_scope.v2"] = "impact_population_scope.v2"
    basis: Literal["SUPPLIED_VIEWS"] = "SUPPLIED_VIEWS"
    organization_id: str | None
    workspace_id: str | None
    query: Literal["EXACT_EVIDENCE_DEPENDENTS"] = "EXACT_EVIDENCE_DEPENDENTS"
    member_pins: tuple[tuple[str, str, str, str], ...]
    supplied_enumeration_complete: Literal[True] = True
    owner_population: Literal["UNESTABLISHED"] = "UNESTABLISHED"
    legal_population: Literal["UNESTABLISHED"] = "UNESTABLISHED"
    traversal: Literal["DECLARED_EVIDENCE_MEANING_STRUCTURE_DECISION"] = "DECLARED_EVIDENCE_MEANING_STRUCTURE_DECISION"

    @model_validator(mode="after")
    def exact_members(self):
        if self.member_pins != tuple(sorted(set(self.member_pins))):
            raise ValueError("impact members must be unique and ordered")
        if len({p[2] for p in self.member_pins}) != len(self.member_pins):
            raise ValueError("impact contains conflicting decision versions")
        return self


class ImpactProjectionV1(_FrozenContract):
    population_scope: ImpactPopulationScopeV2 | None = Field(default=None, exclude_if=lambda v: v is None)
    schema_version: Literal["governed_delta_impact_projection.v1"] = (
        "governed_delta_impact_projection.v1"
    )
    delta_ref: str
    delta_hash: str = Field(pattern=SHA256_PATTERN)
    affected_artifacts: tuple[AffectedArtifactV1, ...] = ()
    affected_invariants: tuple[AffectedInvariantV1, ...] = ()
    required_qualification: tuple[RequiredQualificationV1, ...] = ()
    content_hash: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_projection(self) -> "ImpactProjectionV1":
        expected_hash = "sha256:" + content_hash(self.hash_payload())
        if self.content_hash != expected_hash:
            raise ValueError("impact projection content_hash does not match canonical payload")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("content_hash")
        return payload


class DeltaTransitionReceiptV1(_FrozenContract):
    schema_version: Literal["governed_delta_transition_receipt.v1"] = (
        "governed_delta_transition_receipt.v1"
    )
    delta_ref: str
    predecessor_ref: str
    successor_ref: str
    change_candidate_ref: str | None = None
    assessment_ref: str | None = None
    admission_receipt_ref: str | None = None
    activation_receipt_ref: str | None = None
    lifecycle_state: DeltaLifecycleState
    owner_ref: str
    occurred_at: str
    content_hash: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_receipt(self) -> "DeltaTransitionReceiptV1":
        if self.lifecycle_state in {
            DeltaLifecycleState.ADMITTED,
            DeltaLifecycleState.ACTIVATED,
            DeltaLifecycleState.BOUND_IN_FUTURE_EXECUTION,
            DeltaLifecycleState.SUPERSEDED,
        } and self.admission_receipt_ref is None:
            raise ValueError("admitted-or-later lifecycle state requires admission receipt")
        if self.lifecycle_state in {
            DeltaLifecycleState.ACTIVATED,
            DeltaLifecycleState.BOUND_IN_FUTURE_EXECUTION,
            DeltaLifecycleState.SUPERSEDED,
        } and self.activation_receipt_ref is None:
            raise ValueError("activated-or-later lifecycle state requires activation receipt")
        expected_hash = "sha256:" + content_hash(self.hash_payload())
        if self.content_hash != expected_hash:
            raise ValueError("transition receipt content_hash does not match canonical payload")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("content_hash")
        return payload


__all__ = [
    "AffectedArtifactV1",
    "AffectedInvariantV1",
    "ArtifactVersionPinV1",
    "DeltaAncestryV1",
    "DeltaEnvelopeV1",
    "DeltaLens",
    "DeltaLensResultV1",
    "DeltaLensStatus",
    "DeltaLifecycleState",
    "DeltaMateriality",
    "DeltaOperationKind",
    "DeltaOperationV1",
    "DeltaTargetType",
    "DeltaTransitionReceiptV1",
    "ImpactProjectionV1",
    "ImpactRelation",
    "ImpactStatus",
    "InvariantDisposition",
    "MaterialChangeSummaryV1",
    "QualificationStatus",
    "RequiredQualificationV1",
]
