"""Additive normative-schema and competency representability contracts."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import math
import re
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eos.artifact_hashing import content_hash


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def timestamps_are_utc(self) -> "_Contract":
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime) and (
                value.tzinfo is None or value.utcoffset() != timedelta(0)
            ):
                raise ValueError(f"{name} must be timezone-aware UTC")
        return self


class SourceSearchCoverageManifestV1(_Contract):
    schema_version: Literal["source_search_coverage_manifest.v1"] = (
        "source_search_coverage_manifest.v1"
    )
    coverage_ref: str = Field(min_length=1)
    coverage_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    source_region_refs: tuple[str, ...] = Field(min_length=1)
    excluded_region_refs: tuple[str, ...]
    search_methods: tuple[str, ...] = Field(min_length=1)
    pattern_classes: tuple[str, ...] = Field(min_length=1)
    parser_ref: str = Field(min_length=1)
    provider_ref: str = Field(min_length=1)
    config_hash: str = Field(pattern=SHA256_PATTERN)
    dependency_traversal_refs: tuple[str, ...] = Field(min_length=1)
    result_refs: tuple[str, ...] = Field(min_length=1)
    governed_reviewer_ref: str = Field(min_length=1)
    reviewed_at: datetime


class NormativeSchemaSlotV1(_Contract):
    dimension: Literal[
        "rule",
        "modality",
        "exception",
        "scope",
        "qualification",
        "precedence",
        "temporal_condition",
        "entity_distinction",
        "evidence_binding",
    ]
    epistemic_status: Literal["FOUND", "NONE_EVIDENCED", "UNRESOLVED"]
    schema_object_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    search_coverage_ref: str | None = None
    search_coverage_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def evidence_matches_status(self) -> "NormativeSchemaSlotV1":
        pair = (self.search_coverage_ref, self.search_coverage_hash)
        if (pair[0] is None) != (pair[1] is None):
            raise ValueError("search coverage ref/hash must be supplied together")
        if self.epistemic_status == "NONE_EVIDENCED" and pair[0] is None:
            raise ValueError("NONE_EVIDENCED requires audited search coverage")
        if self.epistemic_status == "FOUND" and not self.schema_object_refs:
            raise ValueError("FOUND requires representing schema objects")
        return self


class NormativeFactDefinitionV1(_Contract):
    """Source-derived input vocabulary, without observations or expected judgments."""
    fact_ref: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$")
    value_type: Literal["boolean", "integer", "number", "string", "date", "boolean_list"]
    definition: str = Field(min_length=1)
    schema_object_refs: tuple[str, ...] = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)

    allowed_values: tuple[str, ...] | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def categorical_domain_is_valid(self):
        if self.allowed_values is not None and (
            self.value_type != "string"
            or len(set(self.allowed_values)) != len(self.allowed_values)
            or any(not value or value != value.strip() for value in self.allowed_values)
        ):
            raise ValueError("allowed values require a string fact and unique nonblank exact values")
        return self

    def accepts_value(self, value) -> bool:
        if self.value_type == "boolean":
            return type(value) is bool
        if self.value_type == "integer":
            return type(value) is int
        if self.value_type == "number":
            return type(value) is int or (type(value) is float and math.isfinite(value))
        if self.value_type == "boolean_list":
            return isinstance(value, (list, tuple)) and all(type(item) is bool for item in value)
        if self.value_type == "string":
            return isinstance(value, str) and (self.allowed_values is None or value in self.allowed_values)
        if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
            return False
        try:
            date.fromisoformat(value)
        except ValueError:
            return False
        return True

    def accepts_predicate(self, operator: str, expected_value) -> bool:
        if operator == "EXISTS":
            return expected_value is None
        if operator == "NOT":
            return self.value_type == "boolean" and expected_value is None
        if operator in {"AND", "OR"}:
            return self.value_type == "boolean_list" and expected_value is None
        if operator == "IN":
            return self.value_type != "boolean_list" and isinstance(expected_value, (tuple, list)) and all(
                self.accepts_value(value) for value in expected_value)
        if operator in {"GE", "GT", "LE", "LT"}:
            return self.value_type in {"integer", "number"} and self.accepts_value(expected_value)
        if operator in {"DATE_ON_OR_AFTER", "DATE_ON_OR_BEFORE"}:
            return self.value_type == "date" and self.accepts_value(expected_value)
        return operator == "EQ" and self.value_type != "boolean_list" and self.accepts_value(expected_value)


class NormativeSchemaContractV1(_Contract):
    teaching_candidate: Literal[True] | None = Field(default=None, exclude_if=lambda value: value is None)
    REQUIRED_DIMENSIONS: ClassVar[tuple[str, ...]] = (
        "rule",
        "modality",
        "exception",
        "scope",
        "qualification",
        "precedence",
        "temporal_condition",
        "entity_distinction",
        "evidence_binding",
    )

    schema_version: Literal["normative_schema_contract.v1"] = (
        "normative_schema_contract.v1"
    )
    schema_ref: str = Field(min_length=1)
    schema_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    boundary_ref: str = Field(min_length=1)
    boundary_hash: str = Field(pattern=SHA256_PATTERN)
    competency_contract_ref: str = Field(min_length=1)
    competency_contract_hash: str = Field(pattern=SHA256_PATTERN)
    graph_intent_brief_ref: str = Field(min_length=1)
    graph_intent_brief_hash: str = Field(pattern=SHA256_PATTERN)
    corpus_schema_candidate_ref: str = Field(min_length=1)
    corpus_schema_candidate_hash: str = Field(pattern=SHA256_PATTERN)
    schema_decision_ref: str | None = Field(default=None, min_length=1, exclude_if=lambda value: value is None)
    schema_decision_hash: str | None = Field(default=None, pattern=SHA256_PATTERN, exclude_if=lambda value: value is None)
    profile_ref: str = Field(min_length=1)
    profile_hash: str = Field(pattern=SHA256_PATTERN)
    compiler_ref: str = Field(min_length=1)
    compiler_hash: str = Field(pattern=SHA256_PATTERN)
    proposal_evidence_ref: str | None = Field(default=None, exclude_if=lambda value: value is None)
    proposal_evidence_hash: str | None = Field(
        default=None, pattern=SHA256_PATTERN, exclude_if=lambda value: value is None,
    )
    source_review_ref: str | None = Field(default=None, exclude_if=lambda value: value is None)
    source_review_hash: str | None = Field(default=None, pattern=SHA256_PATTERN, exclude_if=lambda value: value is None)
    mapping_review_ref: str | None = Field(default=None, exclude_if=lambda value: value is None)
    mapping_review_hash: str | None = Field(default=None, pattern=SHA256_PATTERN, exclude_if=lambda value: value is None)
    slots: tuple[NormativeSchemaSlotV1, ...]
    fact_definitions: tuple[NormativeFactDefinitionV1, ...] | None = Field(
        default=None, exclude_if=lambda value: value is None)
    created_at: datetime

    @model_validator(mode="after")
    def every_dimension_appears_once(self) -> "NormativeSchemaContractV1":
        if self.teaching_candidate is not None and self.teaching_candidate is not True:
            raise ValueError("teaching candidate marker must be true or absent")
        if self.teaching_candidate:
            if self.schema_decision_ref is not None or self.schema_decision_hash is not None:
                raise ValueError("teaching candidate cannot claim a schema admission decision")
        elif self.schema_decision_ref is None or self.schema_decision_hash is None:
            raise ValueError("admitted normative schema requires an exact schema decision")
        if (self.proposal_evidence_ref is None) != (self.proposal_evidence_hash is None):
            raise ValueError("proposal evidence ref/hash must be supplied together")
        if (self.source_review_ref is None) != (self.source_review_hash is None):
            raise ValueError("source review ref/hash must be supplied together")
        if (self.mapping_review_ref is None) != (self.mapping_review_hash is None):
            raise ValueError("mapping review ref/hash must be supplied together")
        if self.mapping_review_ref is not None and self.source_review_ref is not None:
            raise ValueError("mapping and absence reviews cannot share a successor")
        if self.fact_definitions is not None:
            refs = [item.fact_ref for item in self.fact_definitions]
            if len(refs) != len(set(refs)) or refs != sorted(refs):
                raise ValueError("fact definitions must have unique canonically ordered identifiers")
        dimensions = [slot.dimension for slot in self.slots]
        if set(dimensions) != set(self.REQUIRED_DIMENSIONS) or len(dimensions) != len(
            set(dimensions)
        ):
            raise ValueError("normative schema must declare every semantic dimension exactly once")
        return self


class NormativeSchemaMappingReviewV1(_Contract):
    """Explicit human review of one retained conditional precedence record."""
    schema_version: Literal["normative_schema_mapping_review.v1"] = "normative_schema_mapping_review.v1"
    review_ref: str
    review_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str
    workspace_id: str
    actor_ref: str = Field(pattern=r"^user:.+")
    reviewer_kind: Literal["human"] = "human"
    predecessor_schema_ref: str
    predecessor_schema_hash: str = Field(pattern=SHA256_PATTERN)
    proposal_ref: str
    proposal_hash: str = Field(pattern=SHA256_PATTERN)
    schema_object_ref: str = Field(min_length=1)
    binding_record_hash: str = Field(pattern=SHA256_PATTERN)
    source_evidence_ref: str = Field(min_length=1)
    package_hash: str = Field(pattern=SHA256_PATTERN)
    rationale: str = Field(min_length=1)
    compiler_ref: Literal["compiler:source-normative-schema-mapping-review:v1"] = "compiler:source-normative-schema-mapping-review:v1"
    compiler_hash: str = Field(pattern=SHA256_PATTERN)
    reviewed_at: datetime

    @model_validator(mode="after")
    def exact_content_identity(self):
        digest = content_hash(self.model_dump(mode="json", exclude={"review_ref", "review_hash"}))
        if (self.review_ref, self.review_hash) != ("normative-schema-mapping-review:"+digest, "sha256:"+digest):
            raise ValueError("normative schema mapping review hash mismatch")
        return self


class NormativeSchemaProposalEvidenceV1(_Contract):
    """Retained advisory generation; neither review nor certification authority."""

    schema_version: Literal["normative_schema_proposal_evidence.v1"] = (
        "normative_schema_proposal_evidence.v1"
    )
    evidence_ref: str
    evidence_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str
    workspace_id: str
    instructions: str = Field(min_length=1)
    input_text: str = Field(min_length=1)
    output_schema: dict
    output: dict
    provider_input_text: str | None = Field(default=None, exclude_if=lambda value: value is None)
    rejected_response_text: str | None = Field(default=None, exclude_if=lambda value: value is None)
    usage: dict | None = Field(default=None, exclude_if=lambda value: value is None)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    compiler_ref: str = Field(min_length=1)
    compiler_hash: str = Field(pattern=SHA256_PATTERN)
    created_at: datetime

    @model_validator(mode="after")
    def exact_content_identity(self) -> "NormativeSchemaProposalEvidenceV1":
        digest = content_hash(self.model_dump(mode="json", exclude={"evidence_ref", "evidence_hash"}))
        if (self.evidence_ref, self.evidence_hash) != (
            "normative-schema-proposal:" + digest, "sha256:" + digest,
        ):
            raise ValueError("normative schema proposal evidence hash mismatch")
        return self


class SchemaAdequacyRowV1(_Contract):
    competency_id: str = Field(min_length=1)
    critical: bool
    required_semantic_dimensions: tuple[str, ...] = Field(min_length=1)
    representing_schema_objects: tuple[str, ...]
    representability: Literal["REPRESENTABLE", "UNREPRESENTABLE", "UNRESOLVED"]
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    decision_refs: tuple[str, ...]
    blocking_reason: str | None = None
    allowed_resolution_path: Literal[
        "schema_successor", "competency_scope_successor", "source_evidence_successor"
    ] | None = None


class SchemaAdequacyReportV1(_Contract):
    schema_version: Literal["schema_adequacy_report.v1"] = "schema_adequacy_report.v1"
    report_ref: str = Field(min_length=1)
    report_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    normative_schema_ref: str = Field(min_length=1)
    normative_schema_hash: str = Field(pattern=SHA256_PATTERN)
    rows: tuple[SchemaAdequacyRowV1, ...]
    overall_decision: Literal["PASS", "FAIL"]
    certifiable_s3_allowed: bool
    exploration_allowed: Literal[True] = True
    created_at: datetime

    @model_validator(mode="after")
    def critical_rows_cannot_be_hidden(self) -> "SchemaAdequacyReportV1":
        blocking = any(
            row.critical and row.representability != "REPRESENTABLE"
            for row in self.rows
        )
        if blocking and (self.overall_decision != "FAIL" or self.certifiable_s3_allowed):
            raise ValueError("failed critical competency cannot be hidden by aggregate status")
        if not blocking and (
            self.overall_decision != "PASS" or not self.certifiable_s3_allowed
        ):
            raise ValueError("passing critical competency rows require a passing gate")
        return self


__all__ = [
    "NormativeFactDefinitionV1",
    "NormativeSchemaContractV1",
    "NormativeSchemaSlotV1",
    "NormativeSchemaProposalEvidenceV1",
    "SchemaAdequacyReportV1",
    "SchemaAdequacyRowV1",
    "SourceSearchCoverageManifestV1",
]
