"""Version-bound, evidence-bearing contracts for ontology seed supply."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import to_jsonable_python

from eos.artifact_hashing import content_hash

SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
CapabilityStatus = Literal["supported", "partial", "unsupported"]
CapabilityName = Literal[
    "conditional_constraint", "scoped_requirement", "exception_structure",
    "qualification_structure", "precedence_structure", "temporal_constraint",
    "entity_distinction",
]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def persisted_timestamps_are_utc(self) -> "_Contract":
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime) and (
                value.tzinfo is None or value.utcoffset() != timedelta(0)
            ):
                raise ValueError(f"{name} must be timezone-aware UTC")
        return self


class OntologySnapshotIdentityV1(_Contract):
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    ontology_ref: str = Field(min_length=1)
    version: str = Field(min_length=1)
    source_content_hash: str = Field(pattern=SHA256_PATTERN)
    graph_content_hash: str = Field(pattern=SHA256_PATTERN)
    name: str = Field(min_length=1)
    domain_primary: str = Field(min_length=1)
    domain_secondary: tuple[str, ...]
    provenance: dict[str, str]
    license: str | None = None
    seed_class: Literal["REFERENCE", "ORGANIZATION", "DOMAIN_TEMPLATE"]
    authority_role: str = Field(min_length=1)
    governance_state: str = Field(min_length=1)


class OntologyIdentityV1(_Contract):
    name: str = Field(min_length=1)
    domain_primary: str = Field(min_length=1)
    domain_secondary: tuple[str, ...]
    provenance: dict[str, str]
    license: str | None = None


class StructuralTripleV1(_Contract):
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)


class StructuralEvidenceV1(_Contract):
    evidence_kind: Literal[
        "class_declaration", "property_signature", "shacl_property_constraint",
        "owl_disjointness", "graph_inspection",
    ]
    triples: tuple[StructuralTripleV1, ...]
    note: str | None = None


class EvidenceBoundValueV1(_Contract):
    value: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    review_state: Literal["proposed", "reviewed"]
    claim_method: Literal["deterministic", "ia_skill", "hybrid", "human"]


class OntologyCompetencyProfileV1(_Contract):
    questions: tuple[EvidenceBoundValueV1, ...]
    intended_tasks: tuple[EvidenceBoundValueV1, ...]


class OntologySemanticCoverageV1(_Contract):
    definition_families: tuple[EvidenceBoundValueV1, ...]
    concepts: tuple[EvidenceBoundValueV1, ...]
    relation_families: tuple[EvidenceBoundValueV1, ...]


class RepresentationCapabilityClaimV1(_Contract):
    status: CapabilityStatus
    review_state: Literal["proposed", "reviewed"]
    structural_evidence: tuple[StructuralEvidenceV1, ...]
    assessment_evidence_refs: tuple[str, ...]
    semantic_rationale: str | None = None
    claim_method: Literal["deterministic", "ia_skill", "hybrid", "human"]

    @model_validator(mode="after")
    def supported_claim_has_evidence_or_review(self) -> "RepresentationCapabilityClaimV1":
        if self.status == "supported" and not self.structural_evidence and self.review_state != "reviewed":
            raise ValueError("unreviewed supported claims require structural evidence")
        return self


class RepresentationCapabilitiesV1(_Contract):
    conditional_constraint: RepresentationCapabilityClaimV1
    scoped_requirement: RepresentationCapabilityClaimV1
    exception_structure: RepresentationCapabilityClaimV1
    qualification_structure: RepresentationCapabilityClaimV1
    precedence_structure: RepresentationCapabilityClaimV1
    temporal_constraint: RepresentationCapabilityClaimV1
    entity_distinction: RepresentationCapabilityClaimV1


class OntologyAuthorityV1(_Contract):
    seed_class: Literal["REFERENCE", "ORGANIZATION", "DOMAIN_TEMPLATE"]
    authority_role: str = Field(min_length=1)
    governance_state: str = Field(min_length=1)


class ProfileProvenanceV1(_Contract):
    method: Literal["deterministic", "ia_skill", "hybrid", "human"]
    state: Literal["proposed", "reviewed"]
    generated_at: datetime
    prompt_asset: str | None = None
    prompt_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    analyst_ref: str | None = None
    analysis_input_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    analysis_output_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    based_on_analysis_output_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    analysis_input_envelope: dict[str, object] | None = None
    analysis_output_envelope: dict[str, object] | None = None
    reviewer_identity_ref: str | None = None
    reviewed_at: datetime | None = None
    changed_claims: tuple[CapabilityName, ...]

    @model_validator(mode="after")
    def review_identity_is_complete(self) -> "ProfileProvenanceV1":
        review_values = (self.reviewer_identity_ref, self.reviewed_at)
        if self.state == "reviewed" and any(value is None for value in review_values):
            raise ValueError("reviewed profiles require reviewer identity and reviewed_at")
        if self.state == "proposed" and any(value is not None for value in review_values):
            raise ValueError("proposed profiles cannot carry review identity")
        if (self.analysis_input_hash is None) != (self.analysis_output_hash is None):
            raise ValueError("analysis input/output hashes must both be present or absent")
        if (self.analysis_input_envelope is None) != (self.analysis_output_envelope is None):
            raise ValueError("analysis input/output envelopes must both be present or absent")
        if (self.analysis_input_hash is None) != (self.analysis_input_envelope is None):
            raise ValueError("analysis hashes and envelopes must all be absent or all be present")
        if self.analysis_input_envelope is not None:
            analysis_input = IAAnalysisInputV1.model_validate(self.analysis_input_envelope)
            analysis_output = IAAnalysisOutputV1.model_validate(self.analysis_output_envelope)
            if (
                analysis_input.analysis_input_hash != self.analysis_input_hash
                or analysis_output.analysis_hash != self.analysis_output_hash
                or analysis_output.analysis_input_hash != analysis_input.analysis_input_hash
            ):
                raise ValueError("embedded analysis envelopes do not match provenance pins")
        return self

    def source_analysis(self) -> tuple["IAAnalysisInputV1", "IAAnalysisOutputV1"]:
        validated = type(self).model_validate(self.model_dump(mode="json"))
        if validated.analysis_input_envelope is None or validated.analysis_output_envelope is None:
            raise ValueError("profile provenance has no embedded IA analysis")
        return (
            IAAnalysisInputV1.model_validate(validated.analysis_input_envelope),
            IAAnalysisOutputV1.model_validate(validated.analysis_output_envelope),
        )


class OntologyCapabilityProfileV1(_Contract):
    schema_version: Literal["ontology_capability_profile.v1"] = "ontology_capability_profile.v1"
    profile_ref: str = Field(min_length=1)
    profile_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    ontology_ref: str = Field(min_length=1)
    version: str = Field(min_length=1)
    source_content_hash: str = Field(pattern=SHA256_PATTERN)
    graph_content_hash: str = Field(pattern=SHA256_PATTERN)
    revision: int = Field(ge=1)
    predecessor_profile_ref: str | None = None
    predecessor_profile_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    identity: OntologyIdentityV1
    structural_inventory: tuple[StructuralTripleV1, ...]
    competency_profile: OntologyCompetencyProfileV1
    semantic_coverage: OntologySemanticCoverageV1
    representation_capabilities: RepresentationCapabilitiesV1
    authority: OntologyAuthorityV1
    profile_provenance: ProfileProvenanceV1

    def hash_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"profile_ref", "profile_hash"})

    @model_validator(mode="after")
    def identity_and_hash_are_bound(self) -> "OntologyCapabilityProfileV1":
        if (self.predecessor_profile_ref is None) != (self.predecessor_profile_hash is None):
            raise ValueError("predecessor profile ref/hash must both be present or absent")
        if self.revision == 1 and self.predecessor_profile_ref is not None:
            raise ValueError("initial profile cannot have a predecessor")
        if self.revision > 1 and self.predecessor_profile_ref is None:
            raise ValueError("successor profile requires predecessor ref/hash")
        expected = "sha256:" + content_hash(self.hash_payload())
        if self.profile_hash != expected:
            raise ValueError("profile_hash does not bind exact profile content")
        if self.profile_ref != f"ontology-capability-profile:{expected[7:]}":
            raise ValueError("profile_ref does not bind exact profile content")
        return self

    @classmethod
    def create(cls, **values: object) -> "OntologyCapabilityProfileV1":
        material = to_jsonable_python({"schema_version": "ontology_capability_profile.v1", **values})
        digest = content_hash(material)
        return cls(profile_ref=f"ontology-capability-profile:{digest}", profile_hash=f"sha256:{digest}", **values)


class CapabilityAssessmentV1(_Contract):
    status: CapabilityStatus
    rationale: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)


class MetadataValueAssessmentV1(_Contract):
    value: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)


class CompetencyMetadataAssessmentV1(_Contract):
    questions: tuple[MetadataValueAssessmentV1, ...] | None = None
    intended_tasks: tuple[MetadataValueAssessmentV1, ...] | None = None


class SemanticCoverageAssessmentV1(_Contract):
    definition_families: tuple[MetadataValueAssessmentV1, ...] | None = None
    concepts: tuple[MetadataValueAssessmentV1, ...] | None = None
    relation_families: tuple[MetadataValueAssessmentV1, ...] | None = None


class IAAnalysisInputV1(_Contract):
    schema_version: Literal["ontology_capability_analysis_input.v1"] = "ontology_capability_analysis_input.v1"
    analysis_input_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    ontology_ref: str = Field(min_length=1)
    version: str = Field(min_length=1)
    source_content_hash: str = Field(pattern=SHA256_PATTERN)
    graph_content_hash: str = Field(pattern=SHA256_PATTERN)
    base_profile_ref: str = Field(min_length=1)
    base_profile_hash: str = Field(pattern=SHA256_PATTERN)
    structural_inventory: tuple[StructuralTripleV1, ...]
    prompt_asset: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def hash_is_bound(self) -> "IAAnalysisInputV1":
        payload = self.model_dump(mode="json", exclude={"analysis_input_hash"})
        if self.analysis_input_hash != "sha256:" + content_hash(payload):
            raise ValueError("analysis_input_hash does not bind exact analysis input")
        return self

    @classmethod
    def create(cls, **values: object) -> "IAAnalysisInputV1":
        material = to_jsonable_python({"schema_version": "ontology_capability_analysis_input.v1", **values})
        return cls(analysis_input_hash="sha256:" + content_hash(material), **values)


class IAAnalysisOutputV1(_Contract):
    schema_version: Literal["ontology_capability_analysis_output.v1"] = "ontology_capability_analysis_output.v1"
    analysis_hash: str = Field(pattern=SHA256_PATTERN)
    analysis_input_hash: str = Field(pattern=SHA256_PATTERN)
    analyst_ref: str = Field(min_length=1)
    assessments: dict[CapabilityName, CapabilityAssessmentV1]
    competency: CompetencyMetadataAssessmentV1 | None = None
    semantic_coverage: SemanticCoverageAssessmentV1 | None = None
    completed_at: datetime

    @model_validator(mode="after")
    def hash_is_bound(self) -> "IAAnalysisOutputV1":
        payload = self.model_dump(mode="json", exclude={"analysis_hash"})
        if self.analysis_hash != "sha256:" + content_hash(payload):
            raise ValueError("analysis_hash does not bind exact analysis output")
        return self

    @classmethod
    def create(cls, **values: object) -> "IAAnalysisOutputV1":
        defaults = {"competency": None, "semantic_coverage": None}
        material_values = {**defaults, **values}
        material = to_jsonable_python({"schema_version": "ontology_capability_analysis_output.v1", **material_values})
        return cls(analysis_hash="sha256:" + content_hash(material), **material_values)


__all__ = [
    "CapabilityAssessmentV1", "CapabilityName", "CompetencyMetadataAssessmentV1",
    "EvidenceBoundValueV1", "IAAnalysisInputV1", "IAAnalysisOutputV1",
    "MetadataValueAssessmentV1", "OntologyCapabilityProfileV1",
    "OntologySnapshotIdentityV1", "SemanticCoverageAssessmentV1",
]
