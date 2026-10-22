"""Independent external-fidelity validation contracts."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts.artifact_pin import ArtifactPinV1
from eos.artifact_hashing import content_hash


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
_ExpectationId = Annotated[str, Field(strict=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$")]
RuleOutcomeExpectations = Annotated[dict[_ExpectationId, Literal["MATCH", "NO_MATCH", "UNKNOWN"]], Field(min_length=1)]
CaseResultRuleRefs = Annotated[dict[_ExpectationId, _ExpectationId], Field(min_length=1)]
CaseRuleExpectations = Annotated[dict[_ExpectationId, RuleOutcomeExpectations], Field(min_length=1)]



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


class SourceRequirementPredicateV1(_Contract):
    """Source meaning operands; deliberately has no candidate/predicate ID."""
    operator: str = Field(min_length=1)
    fact_ref: _ExpectationId
    expected_value: Any = None
    comparison_fact_ref: str | None = None

    @model_validator(mode="after")
    def executable_source_expression(self):
        from contracts.normative_semantics import SemanticPredicateV1
        SemanticPredicateV1(predicate_id="source.expression", evidence_refs=("source.expression",),
            **self.model_dump())
        return self


class SourceRequirementInstanceV1(_Contract):
    instance_id: _ExpectationId
    occurrence_ref: _ExpectationId
    meaning_selector: _ExpectationId
    semantic_dimension: Literal["rule", "exception"]
    owning_rule_id: _ExpectationId | None = None
    modality: Literal["obligation", "prohibition", "permission"] | None = None
    # OR between adequate alternatives; AND inside each bundle. No candidate IDs.
    support_alternatives: tuple[tuple[SourceRequirementPredicateV1, ...], ...] = Field(min_length=1)
    evidence_refs: tuple[_ExpectationId, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def explicit_meaning(self):
        if any(not bundle for bundle in self.support_alternatives):
            raise ValueError("source support bundles must be nonempty")
        if self.semantic_dimension == "rule":
            if self.owning_rule_id is not None or self.modality is None:
                raise ValueError("source rule requires modality and no parent")
        elif self.owning_rule_id is None or self.modality is not None:
            raise ValueError("source exception requires owning source rule")
        return self


class SourceRequirementInventoryV1(_Contract):
    schema_version: Literal["source_requirement_inventory.v1"] = "source_requirement_inventory.v1"
    source_identity_ref: str = Field(min_length=1)
    source_hash: str = Field(pattern=SHA256_PATTERN)
    boundary_ref: str = Field(min_length=1)
    boundary_hash: str = Field(pattern=SHA256_PATTERN)
    purpose: str = Field(min_length=1)
    system_boundary: str = Field(min_length=1)
    requirements: str = Field(min_length=1)
    competency_questions: dict[_ExpectationId, str] = Field(min_length=1)
    exposure: Literal["DEVELOPMENT"]
    authorship: Literal["ASSISTANT_AUTHORED"]
    completeness: Literal["NOT_QUALIFIED"]
    eligibility: Literal["ELIGIBLE", "UNRESOLVED"]
    instances: tuple[SourceRequirementInstanceV1, ...]

    @model_validator(mode="after")
    def independent_identity(self):
        ids = {i.instance_id: i for i in self.instances}
        meanings = {(i.occurrence_ref, i.meaning_selector) for i in self.instances}
        if len(ids) != len(self.instances) or len(meanings) != len(self.instances):
            raise ValueError("duplicate source requirement identity or meaning selector")
        for item in self.instances:
            if item.owning_rule_id is not None and (item.owning_rule_id not in ids
                or ids[item.owning_rule_id].semantic_dimension != "rule"):
                raise ValueError("unknown owning source rule")
        return self


class SourceRequirementAssessmentV1(_Contract):
    instance_id: _ExpectationId
    owning_rule_id: _ExpectationId | None
    mapping_status: Literal["MAPPED", "MISSING", "AMBIGUOUS"]
    reason: str = Field(min_length=1)
    supporting_frame_refs: tuple[str, ...] = ()
    witness_status: Literal["PASS", "FAIL", "UNKNOWN", "NOT_RUN"] = "NOT_RUN"
    witness_case_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def attributable_mapping(self):
        if (self.mapping_status == "MAPPED") != bool(self.supporting_frame_refs):
            raise ValueError("mapped source requirements require adequate support refs")
        for values in (self.supporting_frame_refs, self.witness_case_ids):
            if values != tuple(sorted(set(values))):
                raise ValueError("source requirement assessment refs must be unique and ordered")
        if self.mapping_status != "MAPPED" and self.witness_status != "NOT_RUN":
            raise ValueError("unmapped requirements cannot claim executed support witnesses")
        return self


class SourceValidationCaseV1(_Contract):
    """Advisory hypothetical scenario; expected results still require review."""
    required_source_instance_ids: tuple[_ExpectationId, ...] = Field(default=(), exclude_if=lambda value: not value)
    expected_instance_results: dict[_ExpectationId, Literal["MATCH", "NO_MATCH", "UNKNOWN"]] | None = Field(default=None, exclude_if=lambda value: value is None)
    case_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$")
    competency_id: str = Field(min_length=1)
    case_kind: Literal["positive", "hard_negative", "boundary", "ambiguity"]
    question: str = Field(min_length=1)
    observed_facts: dict[str, Any]
    expected_result: Literal["MATCH", "NO_MATCH", "UNKNOWN"]
    expected_compatibility: Literal["COMPATIBLE", "CONFLICT", "UNKNOWN"] | None = Field(default=None, exclude_if=lambda value: value is None)
    expected_conflict_rule_refs: tuple[_ExpectationId, ...] = Field(default=(), exclude_if=lambda value: not value)
    expected_resolution: Literal["RESOLVED_BY_PRECEDENCE", "UNRESOLVED"] | None = Field(default=None, exclude_if=lambda value: value is None)
    expected_controlling_rule_refs: tuple[_ExpectationId, ...] = Field(default=(), exclude_if=lambda value: not value)
    required_semantic_dimensions: tuple[str, ...] = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1)


class SourceValidationCasesV1(_Contract):
    cases: tuple[SourceValidationCaseV1, ...]
    unresolved_competency_ids: tuple[str, ...]


class SourceValidationProposalEvidenceV1(_Contract):
    """Exact independent generation receipt, without review or oracle authority."""
    schema_version: Literal["source_validation_proposal_evidence.v1"] = "source_validation_proposal_evidence.v1"
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
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    compiler_ref: str = Field(min_length=1)
    compiler_hash: str = Field(pattern=SHA256_PATTERN)
    created_at: datetime
    predecessor_proposal_pin: ArtifactPinV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    review_actor_ref: str | None = Field(default=None, pattern=r'^user:.+$', exclude_if=lambda value: value is None)
    review_rationale: str | None = Field(default=None, min_length=1, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def exact_content_identity(self) -> "SourceValidationProposalEvidenceV1":
        fields=(self.predecessor_proposal_pin,self.review_actor_ref,self.review_rationale)
        if any(value is not None for value in fields) and (any(value is None for value in fields)
            or self.provider!='governed-source-case-review:delegated_ai' or self.model!='not_applicable'):
            raise ValueError('source-case review requires exact lineage and explicit attribution')
        digest = content_hash(self.model_dump(mode="json", exclude={"evidence_ref", "evidence_hash"}))
        if (self.evidence_ref, self.evidence_hash) != ("source-validation-proposal:" + digest, "sha256:" + digest):
            raise ValueError("source validation proposal evidence hash mismatch")
        return self


class SourceValidationCaseReviewV1(_Contract):
    """Human review of one outcome; grants no case-set acceptance or certification."""
    schema_version: Literal["source_validation_case_review.v1"] = "source_validation_case_review.v1"
    review_ref: str
    review_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str
    workspace_id: str
    actor_ref: str = Field(min_length=1)
    proposal_ref: str = Field(min_length=1)
    proposal_hash: str = Field(pattern=SHA256_PATTERN)
    case_id: str = Field(min_length=1)
    original_case_hash: str = Field(pattern=SHA256_PATTERN)
    expected_result: Literal["MATCH", "NO_MATCH", "UNKNOWN"]
    rationale: str = Field(min_length=1)
    decided_at: datetime

    @model_validator(mode="after")
    def exact_content_identity(self) -> "SourceValidationCaseReviewV1":
        digest = content_hash(self.model_dump(mode="json", exclude={"review_ref", "review_hash"}))
        if (self.review_ref, self.review_hash) != ("source-validation-case-review:" + digest, "sha256:" + digest):
            raise ValueError("source validation case review hash mismatch")
        if not self.rationale.strip():
            raise ValueError("case review rationale required")
        return self


class SourceValidationReviewDecisionV1(_Contract):
    source_requirement_inventory: SourceRequirementInventoryV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    schema_version: Literal["source_validation_review_decision.v1"] = "source_validation_review_decision.v1"
    decision_ref: str
    decision_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str
    workspace_id: str
    actor_ref: str = Field(min_length=1)
    proposal_ref: str = Field(min_length=1)
    proposal_hash: str = Field(pattern=SHA256_PATTERN)
    decision: Literal["accepted", "rejected"]
    rationale: str = Field(min_length=1)
    compiler_ref: str = Field(min_length=1)
    compiler_hash: str = Field(pattern=SHA256_PATTERN)
    case_review_pins: tuple[ArtifactPinV1, ...] = Field(default=(), exclude_if=lambda value: not value)
    case_rule_expectations: CaseRuleExpectations | None = Field(default=None, exclude_if=lambda value: value is None)
    case_result_rule_refs: CaseResultRuleRefs | None = Field(default=None, exclude_if=lambda value: value is None)
    decided_at: datetime

    @model_validator(mode="after")
    def exact_content_identity(self) -> "SourceValidationReviewDecisionV1":
        digest = content_hash(self.model_dump(mode="json", exclude={"decision_ref", "decision_hash"}))
        if (self.decision_ref, self.decision_hash) != ("source-validation-review:" + digest, "sha256:" + digest):
            raise ValueError("source validation review decision hash mismatch")
        return self


class ValidationObligationV1(_Contract):
    obligation_id: str = Field(min_length=1)
    obligation_hash: str = Field(pattern=SHA256_PATTERN)
    semantic_dimension: Literal[
        "rule",
        "modality",
        "exception",
        "scope",
        "qualification",
        "temporal_condition",
        "precedence",
        "entity_distinction",
        "evidence_binding",
        "expected_inference",
        "prohibited_inference",
    ]
    competency_ids: tuple[str, ...] = Field(min_length=1)
    acceptance_criterion: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    provenance_kind: Literal[
        "source_derived", "human_confirmed", "authority_reviewed", "synthetic_mutation"
    ]


class ExecutableCompetencyTestV1(_Contract):
    expected_instance_results: dict[_ExpectationId, Literal["MATCH", "NO_MATCH", "UNKNOWN"]] | None = Field(default=None, exclude_if=lambda value: value is None)
    test_id: str = Field(min_length=1)
    test_hash: str = Field(pattern=SHA256_PATTERN)
    competency_id: str = Field(min_length=1)
    case_kind: Literal["positive", "hard_negative", "boundary", "ambiguity"]
    question: str = Field(min_length=1)
    expected_result: str = Field(min_length=1)
    expected_rule_results: RuleOutcomeExpectations | None = Field(default=None, exclude_if=lambda value: value is None)
    result_rule_ref: _ExpectationId | None = Field(default=None, exclude_if=lambda value: value is None)
    expected_compatibility: Literal["COMPATIBLE", "CONFLICT", "UNKNOWN"] | None = Field(default=None, exclude_if=lambda value: value is None)
    expected_conflict_rule_refs: tuple[_ExpectationId, ...] = Field(default=(), exclude_if=lambda value: not value)
    expected_resolution: Literal["RESOLVED_BY_PRECEDENCE", "UNRESOLVED"] | None = Field(default=None, exclude_if=lambda value: value is None)
    expected_controlling_rule_refs: tuple[_ExpectationId, ...] = Field(default=(), exclude_if=lambda value: not value)
    source_case_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]*$",
        exclude_if=lambda value: value is None)
    observed_facts: dict[str, Any] | None = Field(default=None, exclude_if=lambda value: value is None)
    required_obligation_ids: tuple[str, ...] = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    provenance_kind: Literal[
        "source_derived", "human_confirmed", "authority_reviewed", "synthetic_mutation"
    ]

    @model_validator(mode="after")
    def normative_stage_expectations(self):
        if self.result_rule_ref is not None and (self.expected_rule_results is None
            or self.result_rule_ref not in self.expected_rule_results):
            raise ValueError("result rule must belong to the reviewed rule expectations")
        for refs in (self.expected_conflict_rule_refs, self.expected_controlling_rule_refs):
            if refs != tuple(sorted(set(refs))):
                raise ValueError("normative expectation refs must be unique and ordered")
        if self.expected_compatibility is not None and self.expected_rule_results is None:
            raise ValueError("normative oracle requires intrinsic rule expectations")
        if self.expected_compatibility == "CONFLICT":
            if len(self.expected_conflict_rule_refs) < 2 or self.expected_resolution is None:
                raise ValueError("conflict oracle requires membership and resolution expectations")
        elif self.expected_conflict_rule_refs or self.expected_resolution is not None:
            raise ValueError("non-conflict oracle cannot declare conflict or resolution")
        if self.expected_resolution == "RESOLVED_BY_PRECEDENCE":
            if not self.expected_controlling_rule_refs or not set(self.expected_controlling_rule_refs).issubset(self.expected_conflict_rule_refs):
                raise ValueError("controller must belong to the expected conflict")
        elif self.expected_controlling_rule_refs:
            raise ValueError("unresolved oracle cannot declare a controller")
        return self

    @model_validator(mode="after")
    def source_case_and_facts_are_paired(self) -> "ExecutableCompetencyTestV1":
        if (self.source_case_id is None) != (self.observed_facts is None):
            raise ValueError("source case identity and observed facts must be paired")
        if self.source_case_id is not None:
            digest = content_hash(self.model_dump(mode="json", exclude={"test_id", "test_hash", "provenance_kind"}))
            if self.test_hash != "sha256:" + digest:
                raise ValueError("reviewed source test hash mismatch")
        return self


class ValidationIndependenceV1(_Contract):
    compiler_ref: str = Field(min_length=1)
    compiler_hash: str = Field(pattern=SHA256_PATTERN)
    run_ref: str = Field(min_length=1)
    prompt_hash: str = Field(pattern=SHA256_PATTERN)
    config_hash: str = Field(pattern=SHA256_PATTERN)
    provider_ref: str = Field(min_length=1)
    model_ref: str = Field(min_length=1)
    source_region_coverage_ref: str = Field(min_length=1)
    semantic_candidate_input_ref: None = None


class ValidationContractV1(_Contract):
    source_requirement_inventory: SourceRequirementInventoryV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    schema_version: Literal["validation_contract.v1"] = "validation_contract.v1"
    validation_contract_ref: str = Field(min_length=1)
    validation_contract_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    source_identity_ref: str = Field(min_length=1)
    source_hash: str = Field(pattern=SHA256_PATTERN)
    boundary_ref: str = Field(min_length=1)
    boundary_hash: str = Field(pattern=SHA256_PATTERN)
    competency_contract_ref: str = Field(min_length=1)
    competency_contract_hash: str = Field(pattern=SHA256_PATTERN)
    normative_schema_ref: str = Field(min_length=1)
    normative_schema_hash: str = Field(pattern=SHA256_PATTERN)
    human_oracle_ref: str = Field(min_length=1)
    human_oracle_hash: str = Field(pattern=SHA256_PATTERN)
    obligations: tuple[ValidationObligationV1, ...] = Field(min_length=1)
    executable_tests: tuple[ExecutableCompetencyTestV1, ...] = Field(min_length=1)
    independence: ValidationIndependenceV1
    review_state: Literal["proposed", "accepted", "rejected"]
    predecessor_validation_contract_ref: str | None = None
    predecessor_validation_contract_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    review_decision_ref: str | None = None
    compiled_at: datetime

    @model_validator(mode="after")
    def exact_mapping(self) -> "ValidationContractV1":
        inventory = self.source_requirement_inventory
        if inventory is not None:
            if (inventory.source_identity_ref, inventory.source_hash, inventory.boundary_ref, inventory.boundary_hash) != (
                self.source_identity_ref, self.source_hash, self.boundary_ref, self.boundary_hash):
                raise ValueError("source requirement inventory pin mismatch")
            ids = {i.instance_id for i in inventory.instances}
            if ids != {o.obligation_id for o in self.obligations}:
                raise ValueError("source requirement obligation coverage mismatch")
            for test in self.executable_tests:
                if test.expected_instance_results is None or set(test.expected_instance_results) != set(test.required_obligation_ids):
                    raise ValueError("source requirement case witness coverage mismatch")
                if test.competency_id not in inventory.competency_questions:
                    raise ValueError("source requirement competency mismatch")
        elif any(t.expected_instance_results is not None for t in self.executable_tests):
            raise ValueError("instance witnesses require source inventory")
        obligation_ids = {item.obligation_id for item in self.obligations}
        if len(obligation_ids) != len(self.obligations):
            raise ValueError("validation obligation ids must be unique")
        test_ids = [test.test_id for test in self.executable_tests]
        if len(test_ids) != len(set(test_ids)):
            raise ValueError("validation test ids must be unique")
        source_ids = [test.source_case_id for test in self.executable_tests if test.source_case_id is not None]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("validation source case ids must be unique")
        by_id = {item.obligation_id: item for item in self.obligations}
        for test in self.executable_tests:
            if not set(test.required_obligation_ids).issubset(obligation_ids):
                raise ValueError("competency test references an unknown obligation")
            if any(test.competency_id not in by_id[ref].competency_ids for ref in test.required_obligation_ids):
                raise ValueError("competency test references an obligation for another competency")
        if (self.predecessor_validation_contract_ref is None) != (
            self.predecessor_validation_contract_hash is None
        ):
            raise ValueError("validation predecessor ref/hash must be supplied together")
        if self.review_state != "proposed" and (
            self.predecessor_validation_contract_ref is None or self.review_decision_ref is None
        ):
            raise ValueError("reviewed validation contracts require predecessor and decision lineage")
        return self


__all__ = [
    "SourceRequirementPredicateV1",
    "SourceRequirementInstanceV1",
    "SourceRequirementInventoryV1",
    "SourceRequirementAssessmentV1",
    "SourceValidationReviewDecisionV1",
    "SourceValidationCaseV1",
    "SourceValidationCasesV1",
    "SourceValidationProposalEvidenceV1",
    "ExecutableCompetencyTestV1",
    "ValidationContractV1",
    "ValidationIndependenceV1",
    "ValidationObligationV1",
]
