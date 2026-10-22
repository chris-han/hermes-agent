"""Read-only Tender Decision Basis transport contract."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.ontology_building import ArtifactPinV1
from contracts.evidence_location import DocxStructuralRange, PdfPageRect, TextFallbackLocation


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TenderDecisionBasisSourceOccurrenceV1(_Contract):
    schema_version: Literal["tender_decision_basis_source_occurrence.v1"] = (
        "tender_decision_basis_source_occurrence.v1"
    )
    evidence_ref: str = Field(min_length=1)
    occurrence_ref: str | None = Field(default=None, min_length=1)
    status: Literal["EXACT", "UNRESOLVED"]
    source_identity_ref: str = Field(min_length=1)
    document_name: str | None = Field(default=None, min_length=1)
    source_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    canonical_source_ir_pin: ArtifactPinV1
    source_region_ref: str = Field(min_length=1)
    source_block_ref: str | None = Field(default=None, min_length=1)
    media_type: str | None = Field(default=None, min_length=1)
    evidence_role: str = Field(min_length=1)
    selector_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    exact_text: str = Field(min_length=1)
    presentation_location: DocxStructuralRange | PdfPageRect | TextFallbackLocation | None = None


class TenderDecisionBasisNodeV1(_Contract):
    node_ref: str = Field(min_length=1)
    layer: Literal["EVIDENCE", "MEANING", "STRUCTURE", "DECISION"]
    label: str = Field(min_length=1)
    artifact_pin: ArtifactPinV1 | None = None
    recorded_content: dict | None = None

    @model_validator(mode="after")
    def bind_recorded_content(self):
        if self.recorded_content is not None:
            identities = {
                "observed_constraint_hypothesis": ("hypothesisId", "hypothesisHash"),
                "frame_match_result": ("resultId", "resultHash"),
            }
            if self.artifact_pin is None or self.artifact_pin.artifact_kind not in identities:
                raise ValueError("unsupported recorded content")
            ref, digest = identities[self.artifact_pin.artifact_kind]
            if (self.recorded_content.get(ref), self.recorded_content.get(digest)) != (
                self.artifact_pin.artifact_ref, self.artifact_pin.artifact_hash,
            ):
                raise ValueError("recorded content pin mismatch")
        return self



class TenderDecisionBasisEdgeV1(_Contract):
    edge_ref: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    target_ref: str = Field(min_length=1)
    relation: Literal["RECORDED_INPUT"] = "RECORDED_INPUT"
    basis_finding_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class TenderDecisionBasisAffectedFindingV1(_Contract):
    finding_ref: str = Field(min_length=1)
    finding_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    decision_status: Literal["candidate", "violation", "acceptable", "uncertain", "escalate"]
    rationale: str
    shared_evidence_refs: tuple[str, ...]
    dependency_edge_refs: tuple[str, ...]


class TenderFactObservationV1(_Contract):
    value: str = Field(min_length=1, max_length=2000)
    provenance: Literal["OBSERVED", "INFERRED"]
    evidence_ref: str = Field(min_length=1)
    occurrence_ref: str = Field(min_length=1)
    source_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    selector_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    exact_text: str = Field(min_length=1)
    inference_basis: str | None = None

    @model_validator(mode="after")
    def inferred_basis(self):
        if self.provenance == "INFERRED" and not self.inference_basis:
            raise ValueError("inference basis required")
        return self


class TenderConditionalOutcomeV1(_Contract):
    if_established: str = Field(min_length=1, max_length=2000)
    proposed_effect: str = Field(min_length=1, max_length=2000)


class TenderFactRequirementV1(_Contract):
    requirement_ref: str = Field(
        min_length=1, max_length=200, pattern=r"^[\x21-\x7e]+$"
    )
    requirement_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    question: str = Field(min_length=1, max_length=1000)
    evidence_state: Literal["MISSING", "NEEDS_REVIEW", "CONFLICT"]
    observations: tuple[TenderFactObservationV1, ...] = Field(max_length=20)
    effect: Literal[
        "ROUTE_SELECTION",
        "ROUTE_EXCLUSION",
        "PREMISE_INVALIDATION",
        "VERSION_SELECTION",
        "NECESSARY_CONDITION",
        "SUPPORTING",
        "UNCLASSIFIED",
    ]
    consequence: str = Field(min_length=1, max_length=2000)
    source_start: int = Field(ge=0)
    source_end: int = Field(gt=0)
    source_excerpt: str = Field(min_length=1)
    outcomes: tuple[TenderConditionalOutcomeV1, ...] = Field(max_length=20)
    remaining_unknowns: tuple[str, ...] = Field(max_length=50)
    joint_requirement_refs: tuple[str, ...] = Field(max_length=50)

    @model_validator(mode="after")
    def bind_requirement(self):
        from eos.artifact_hashing import content_hash

        if self.requirement_hash != "sha256:" + content_hash(
            self.model_dump(
                mode="json", exclude={"requirement_hash"}, exclude_none=True
            )
        ):
            raise ValueError("requirement hash mismatch")
        if self.evidence_state == "MISSING" and self.observations:
            raise ValueError("missing requirement cannot contain observations")
        if self.evidence_state == "CONFLICT" and len(self.observations) < 2:
            raise ValueError("conflict requires separate assertions")
        if self.effect != "UNCLASSIFIED" and not self.outcomes:
            raise ValueError("classified effect requires proposed alternatives")
        return self


class TenderApplicabilityReviewV1(_Contract):
    schema_version: Literal["tender_applicability_review.v1"]
    version_ref: str = Field(min_length=1)
    section_ref: str = Field(min_length=1)
    quote_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    coverage: Literal["PARTIAL"] = "PARTIAL"
    requirements: tuple[TenderFactRequirementV1, ...] = Field(max_length=100)

    @model_validator(mode="after")
    def unique_requirements(self):
        refs = [r.requirement_ref for r in self.requirements]
        if len(set(refs)) != len(refs):
            raise ValueError("duplicate requirement identity")
        for requirement in self.requirements:
            if any(
                ref not in refs or ref == requirement.requirement_ref
                for ref in requirement.joint_requirement_refs
            ):
                raise ValueError("unbound joint requirement")
        return self


class TenderProvisionCandidateV1(_Contract):
    candidate_ref: str = Field(min_length=1)
    candidate_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    finding_ref: str = Field(min_length=1)
    finding_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    pattern_ref: str = Field(min_length=1)
    pattern_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    source_ref: str = Field(min_length=1)
    source_title: str = Field(min_length=1)
    version_ref: str = Field(min_length=1)
    version_identity: str = Field(min_length=1)
    section_ref: str = Field(min_length=1)
    section_locator: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    quote_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    proposed_roles: tuple[Literal["rule_authority", "scope_authority", "effectivity_authority",
        "precedence_authority", "exception_authority"], ...] = Field(min_length=1)
    review_state: Literal["UNRESOLVED"] = "UNRESOLVED"
    applicability: Literal["UNKNOWN"] = "UNKNOWN"
    test_only: bool = False
    applicability_review: TenderApplicabilityReviewV1 | None = None
    review_context_status: Literal["AVAILABLE", "UNAVAILABLE", "INVALID"] = (
        "UNAVAILABLE"
    )

    @model_validator(mode="after")
    def bind_review(self):
        review = self.applicability_review
        if review is None:
            if self.review_context_status == "AVAILABLE":
                raise ValueError("available review requires payload")
            return self
        if self.review_context_status != "AVAILABLE" or (
            review.version_ref,
            review.section_ref,
            review.quote_hash,
        ) != (self.version_ref, self.section_ref, self.quote_hash):
            raise ValueError("review source pin mismatch")
        for requirement in review.requirements:
            if self.quote[
                requirement.source_start : requirement.source_end
            ] != requirement.source_excerpt or requirement.source_end > len(self.quote):
                raise ValueError("review source span mismatch")
        return self


class TenderDecisionBasisV1(_Contract):
    schema_version: Literal["tender_decision_basis.v1"] = "tender_decision_basis.v1"
    journey_id: str = Field(min_length=1)
    transition_id: str = Field(min_length=1)
    inspection_result_ref: str = Field(min_length=1)
    inspection_result_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    finding_ref: str = Field(min_length=1)
    finding_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    decision_status: Literal["candidate", "violation", "acceptable", "uncertain", "escalate"]
    rationale: str
    structural_match: bool
    semantic_relation: Literal[
        "FRAME_EQUIVALENT", "FRAME_ENTAILS", "NO_MATCH", "UNRESOLVED"
    ]
    frame_match_state: Literal["MATCH", "NO_MATCH", "UNKNOWN"]

    hypothesis_pin: ArtifactPinV1
    frame_match_pin: ArtifactPinV1
    canonical_pattern_pin: ArtifactPinV1
    accepted_graph_release_pin: ArtifactPinV1
    governed_rule_pin: ArtifactPinV1

    source_occurrences: tuple[TenderDecisionBasisSourceOccurrenceV1, ...]
    nodes: tuple[TenderDecisionBasisNodeV1, ...] = ()
    edges: tuple[TenderDecisionBasisEdgeV1, ...] = ()
    affected_findings: tuple[TenderDecisionBasisAffectedFindingV1, ...] = ()
    impact_scope: Literal["THIS_INSPECTION_EPISODE"] = "THIS_INSPECTION_EPISODE"
    impact_coverage: Literal["COMPLETE", "PARTIAL"] = "PARTIAL"
    completeness: Literal["COMPLETE", "PARTIAL"]
    gaps: tuple[str, ...] = ()
    provision_candidates: tuple[TenderProvisionCandidateV1, ...] = ()
    candidate_context_status: Literal["AVAILABLE", "UNAVAILABLE", "ERROR"] = "UNAVAILABLE"

    @model_validator(mode="after")
    def bind_candidate_context(self):
        identities: dict[str, str] = {}
        for candidate in self.provision_candidates:
            if (candidate.finding_ref, candidate.finding_hash) != (
                self.finding_ref,
                self.finding_hash,
            ) or (candidate.pattern_ref, candidate.pattern_hash) != (
                self.canonical_pattern_pin.artifact_ref,
                self.canonical_pattern_pin.artifact_hash,
            ):
                raise ValueError("candidate trace pin mismatch")
            if candidate.applicability_review:
                for requirement in candidate.applicability_review.requirements:
                    previous = identities.setdefault(
                        requirement.requirement_ref, requirement.requirement_hash
                    )
                    if previous != requirement.requirement_hash:
                        raise ValueError("conflicting requirement identity")
                    for observation in requirement.observations:
                        if not any(
                            occurrence.status == "EXACT"
                            and (
                                observation.evidence_ref,
                                observation.occurrence_ref,
                                observation.source_hash,
                                observation.selector_hash,
                                observation.exact_text,
                            )
                            == (
                                occurrence.evidence_ref,
                                occurrence.occurrence_ref,
                                occurrence.source_hash,
                                occurrence.selector_hash,
                                occurrence.exact_text,
                            )
                            for occurrence in self.source_occurrences
                        ):
                            raise ValueError("review observation pin mismatch")
        return self
