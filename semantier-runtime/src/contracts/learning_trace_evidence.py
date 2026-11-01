"""Companion observations in the existing artifact owner, never semantic authority."""
from datetime import datetime
from typing import Literal
from pydantic import Field, model_validator
from contracts.semantic_kernel import _ContentAddressed, _Contract
from eos.artifact_hashing import content_hash


class LearningTraceObservationV1(_ContentAddressed):
    ARTIFACT_KIND = 'learning_trace_observation'
    schema_version: Literal['learning_trace_observation.v1'] = 'learning_trace_observation.v1'
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    owner_ref: str = Field(min_length=1)
    backend: Literal['sqlite', 'postgresql']
    source_store_id: str = Field(min_length=1)
    source_epoch: str | None
    profile_id: str = Field(min_length=1)
    adapter_revision: Literal['hermes-learning-trace.v1'] = 'hermes-learning-trace.v1'
    session_id: str = Field(min_length=1)
    collection_method: Literal['hook', 'poll', 'native_receipt']
    source_locator: str = Field(min_length=1, max_length=500)
    event_kind: Literal['tool', 'request', 'response', 'message', 'approval']
    phase: str = Field(min_length=1, max_length=100)
    native_occurrence_id: str | None
    occurrence_id: str | None
    payload: dict
    source_occurred_at: datetime | None
    captured_at: datetime

    @model_validator(mode='after')
    def bound_occurrence(self):
        key = None
        if self.source_epoch and self.native_occurrence_id:
            key = 'sha256:' + content_hash([self.organization_id, self.workspace_id,
                self.owner_ref, self.backend, self.source_store_id, self.source_epoch,
                self.profile_id, self.session_id, self.event_kind, self.phase,
                self.native_occurrence_id])
        if self.occurrence_id != key:
            raise ValueError('TRACE_OCCURRENCE_ID_MISMATCH')
        return self


class ActionableLearningLessonV1(_Contract):
    """Structured draft metadata for existing Memory/Skill writers; not execution."""
    schema_version: Literal['actionable_learning_lesson.v1'] = 'actionable_learning_lesson.v1'
    target_ref: str = Field(min_length=1)
    target_hash: str = Field(pattern=r'^sha256:[0-9a-f]{64}$')
    content_role: Literal['memory', 'knowledge']
    capabilities: tuple[Literal['extraction', 'organization', 'update_invalidation',
                                'retrieval_assembly', 'utility_verification'], ...]
    applicability: str = Field(min_length=1)
    exclusions: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    verification_method: str = Field(min_length=1)
    rollback: str = Field(min_length=1)
    evidence_refs: tuple[dict, ...] = Field(min_length=1)
    counter_evidence_refs: tuple[dict, ...] = ()
    change_kind: Literal['knowledge_content', 'maintenance_policy', 'retrieval_assembly_policy', 'unspecified'] = 'unspecified'
    utility: Literal['unmeasured'] = 'unmeasured'


class RoutingFeedbackObservationV1(_ContentAddressed):
    """Producer-attached feedback, never an independent quality certification."""
    ARTIFACT_KIND = 'routing_feedback_observation'
    schema_version: Literal['routing_feedback_observation.v1'] = 'routing_feedback_observation.v1'
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    owner_ref: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    account_id: str = Field(min_length=1)
    root_task_id: str = Field(min_length=1)
    invocation_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    evaluator_version: str = Field(min_length=1)
    acceptance_rule_ref: str = Field(min_length=1)
    dimension: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    severity: Literal['minor', 'major', 'critical', 'unspecified']
    outcome: Literal['PASS', 'FAIL', 'UNRESOLVED', 'UNKNOWN', 'UNLABELLED', 'DISPUTED']
    source_kind: Literal['mechanical', 'expert', 'judge', 'implicit']
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    reference_refs: tuple[str, ...]
    observed_at: datetime
    synthetic: bool
    review_method: str = Field(min_length=1)
    review_inclusion_probability: str | None
    review_probability_source: str | None
    action_propensity: str | None
    propensity_status: Literal['KNOWN', 'UNAVAILABLE']
    propensity_source: str | None
    supersedes_ref: str | None = None
    quality_qualification: Literal['UNMEASURED'] = 'UNMEASURED'

    @model_validator(mode='after')
    def probabilities(self):
        from decimal import Decimal, InvalidOperation
        for value, source in ((self.review_inclusion_probability, self.review_probability_source),
                              (self.action_propensity, self.propensity_source)):
            if value is not None:
                try:
                    number = Decimal(value)
                    if not number.is_finite() or not 0 < number <= 1 or not source:
                        raise ValueError('PROBABILITY_BASIS_REQUIRED')
                except InvalidOperation as exc:
                    raise ValueError('INVALID_PROBABILITY') from exc
        if (self.propensity_status == 'KNOWN') != (self.action_propensity is not None):
            raise ValueError('PROPENSITY_STATUS_MISMATCH')
        return self


LEARNING_TRACE_ARTIFACT_MODELS = {
    'learning_trace_observation.v1': LearningTraceObservationV1,
    'routing_feedback_observation.v1': RoutingFeedbackObservationV1,
}

class LearningTraceObservationV2(_ContentAddressed):
    """Public-source observation; no fabricated native database or producer identity."""
    ARTIFACT_KIND = 'learning_trace_observation'
    schema_version: Literal['learning_trace_observation.v2'] = 'learning_trace_observation.v2'
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    owner_ref: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    profile_id: str = Field(min_length=1)
    source_kind: Literal['public_hook', 'public_export', 'process_result', 'core_receipt']
    source_version: str = Field(min_length=1)
    source_locator: str = Field(min_length=1, max_length=500)
    source_event_id: str | None = None
    source_incarnation: str | None = None
    sequence: int = Field(ge=0)
    event_kind: Literal['tool', 'response', 'request', 'lifecycle', 'coverage', 'package', 'assessment']
    stage: Literal['REQUESTED', 'TOOL_REPORTED', 'EFFECT_CONFIRMED', 'OBSERVED'] = 'OBSERVED'
    raw_source_base64: str
    raw_source_hash: str = Field(pattern=r'^sha256:[0-9a-f]{64}$')
    payload: dict
    gaps: tuple[str, ...] = ()
    source_occurred_at: datetime | None = None
    captured_at: datetime
    producer_trust: Literal['public_unverified'] = 'public_unverified'

    @model_validator(mode='after')
    def exact_source(self):
        import base64
        import hashlib
        if self.stage == "EFFECT_CONFIRMED" and self.source_kind != "core_receipt":
            raise ValueError("EFFECT_CONFIRMATION_REQUIRES_DOMAIN_RECEIPT")
        raw = base64.b64decode(self.raw_source_base64, validate=True)
        if base64.b64encode(raw).decode('ascii') != self.raw_source_base64:
            raise ValueError('NONCANONICAL_PUBLIC_SOURCE')
        if self.raw_source_hash != 'sha256:' + hashlib.sha256(raw).hexdigest():
            raise ValueError('PUBLIC_SOURCE_HASH_MISMATCH')
        return self


class DelegationBasisV1(_Contract):
    """Frozen explicit evaluation constraints, never a runtime grant."""
    schema_version: Literal['authorization_monitor_basis.v1'] = 'authorization_monitor_basis.v1'
    task_id: str = Field(min_length=1)
    instruction_hash: str = Field(pattern=r'^sha256:[0-9a-f]{64}$')
    allowed_operations: tuple[str, ...]
    allowed_objects: tuple[str, ...]
    allowed_recipients: tuple[str, ...] | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    stop_after_sequence: int | None = Field(default=None, ge=0)


class AuthorizationMonitorAssessmentV1(_Contract):
    schema_version: Literal['authorization_monitor_assessment.v1'] = 'authorization_monitor_assessment.v1'
    evaluator_version: Literal['explicit-delegation.v1'] = 'explicit-delegation.v1'
    basis_hash: str
    snapshot_hash: str
    surface: str = Field(min_length=1)
    completion: Literal['PASS', 'FAIL', 'UNKNOWN']
    authorization_status: Literal['WITHIN_SCOPE', 'OUT_OF_SCOPE', 'UNKNOWN']
    coverage: Literal['COMPLETE_FOR_DECLARED_SURFACE', 'PARTIAL', 'MISSING']
    actions: tuple[dict, ...]
    violations: tuple[dict, ...]
    gaps: tuple[str, ...]
    recorded_at: datetime
    authoritative: Literal[False] = False
    learning_gain: Literal['NOT_MEASURED'] = 'NOT_MEASURED'


LEARNING_TRACE_ARTIFACT_MODELS['learning_trace_observation.v2'] = LearningTraceObservationV2
