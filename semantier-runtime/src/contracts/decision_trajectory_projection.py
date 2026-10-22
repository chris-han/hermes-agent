"""Derived decision transition/trajectory projections and experiment-readiness receipt.

These contracts bind existing DecisionStep / DecisionRecord / Evidence / DecisionLocalProjection
owners. They are deterministic projections and receipts, not semantic-kernel objects, a second
trajectory database, or learning/admission authority.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from contracts.ontology_building import ArtifactPinV1
from eos.artifact_hashing import content_hash

SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
_TRAJECTORY_CHECKS = tuple(f"TIG-{index:02d}" for index in range(1, 13))


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def utc_datetimes(self) -> "_Contract":
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime):
                if value.tzinfo is None or value.utcoffset() != timedelta(0):
                    raise ValueError(f"{name} must be timezone-aware UTC")
        return self


class InvocationUsageV1(_Contract):
    invocation_ref: str = Field(min_length=1)
    measurement_evidence: ArtifactPinV1 | None
    beneficiary_logical_step_ids: tuple[str, ...] = Field(min_length=1)
    producer_config: ArtifactPinV1
    model_visible_input_tokens: int | None = Field(default=None, ge=0)
    generated_output_tokens: int | None = Field(default=None, ge=0)
    reported_reasoning_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    physical_prefill_tokens: int | None = Field(default=None, ge=0)
    wall_latency_ms: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def unique_beneficiaries(self) -> "InvocationUsageV1":
        if len(self.beneficiary_logical_step_ids) != len(set(self.beneficiary_logical_step_ids)):
            raise ValueError("beneficiary_logical_step_ids must be unique")
        metrics = (
            self.model_visible_input_tokens,
            self.generated_output_tokens,
            self.reported_reasoning_tokens,
            self.cached_input_tokens,
            self.physical_prefill_tokens,
            self.wall_latency_ms,
        )
        if any(value is not None for value in metrics) and self.measurement_evidence is None:
            raise ValueError("numeric usage requires observed measurement_evidence")
        return self


class DecisionTransitionSourceV1(_Contract):
    trajectory_ref: str = Field(min_length=1)
    trajectory_snapshot: ArtifactPinV1
    session_id: str | None
    decision_step_ref: str = Field(min_length=1)
    logical_step_id: str = Field(min_length=1)
    execution_attempt: ArtifactPinV1
    attempt_index: int = Field(ge=0)
    lane: Literal["TASK", "RECOVERY", "DIAGNOSTIC"]


class DecisionTransitionBasisV1(_Contract):
    quality_evaluation: ArtifactPinV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    context_graph_before: ArtifactPinV1 | None
    working_context: ArtifactPinV1 | None
    decision_local_projection: ArtifactPinV1
    decision_contract: ArtifactPinV1 | None
    input_evidence: tuple[ArtifactPinV1, ...]
    world_time: str | None
    information_cutoff: str | None
    producer_config: ArtifactPinV1
    rendered_request: ArtifactPinV1

    @model_validator(mode="after")
    def canonical_quality_pin(self):
        if self.quality_evaluation is not None and self.quality_evaluation.artifact_kind != "quality_evaluation_run":
            raise ValueError("quality_evaluation must pin a canonical evaluation run")
        return self


class DecisionTransitionObservedV1(_Contract):
    semantic_proposal: ArtifactPinV1 | None
    deterministic_evaluation: ArtifactPinV1 | None
    decision_record: ArtifactPinV1 | None
    verification: ArtifactPinV1 | None
    outcome: ArtifactPinV1 | None
    execution_status: Literal["COMPLETE", "FAILED", "REJECTED", "PARTIAL", "INCONCLUSIVE"]
    outcome_provenance: Literal["OBSERVED", "REPLAYED", "SIMULATED", "INFERRED", "ASSUMED"] | None
    semantic_result: Literal["SEMANTIC_TRUE", "SEMANTIC_FALSE", "SEMANTIC_UNKNOWN"] | None = None
    producer_state: Literal["PRODUCER_OBSERVATION", "PRODUCER_ABSTENTION", "INVALID_OR_CENSORED_SCORE", "PROVIDER_FAILURE"] | None = None
    semantic_error: bool = False
    action_receipts: tuple[ArtifactPinV1, ...] = ()
    tool_receipts: tuple[ArtifactPinV1, ...] = ()


class DecisionStateEffectV1(_Contract):
    status: Literal["UNCHANGED", "CHANGED", "NOT_EXECUTED", "UNKNOWN"]
    context_graph_after: ArtifactPinV1 | None = None
    committed_delta: ArtifactPinV1 | None = None
    commit_receipt: ArtifactPinV1 | None = None

    @model_validator(mode="after")
    def changed_requires_commit_binding(self) -> "DecisionStateEffectV1":
        bound = (self.context_graph_after, self.committed_delta, self.commit_receipt)
        if self.status == "CHANGED" and any(item is None for item in bound):
            raise ValueError("CHANGED state effect requires after graph, committed delta, and commit receipt")
        if self.status != "CHANGED" and any(item is not None for item in bound):
            raise ValueError("unchanged/not-executed/unknown state effect cannot declare committed graph change")
        return self


class DecisionTransitionProjectionV1(_Contract):
    schema_version: Literal["decision_transition_projection.v1"] = "decision_transition_projection.v1"
    projection_ref: str = Field(min_length=1)
    projection_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    projection_policy: ArtifactPinV1
    source: DecisionTransitionSourceV1
    basis: DecisionTransitionBasisV1
    observed: DecisionTransitionObservedV1
    state_effect: DecisionStateEffectV1
    invocation_usage: tuple[InvocationUsageV1, ...]
    started_at: datetime
    observed_through: datetime
    closure_status: Literal["COMPLETE", "PARTIAL", "INVALID"]
    missing_dependencies: tuple[str, ...] = ()
    resolution_of_transition: ArtifactPinV1 | None = None
    successor_working_context: ArtifactPinV1 | None = None
    successor_decision_local_projection: ArtifactPinV1 | None = None
    language_evidence: ArtifactPinV1 | None = None
    test_control_evidence: ArtifactPinV1 | None = None
    counterfactual_of: ArtifactPinV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    counterfactual_kind: Literal["CF-1", "CF-2", "CF-3", "CF-4", "CF-5", "CF-6", "CF-7", "D3"] | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def exact_identity_and_closure(self) -> "DecisionTransitionProjectionV1":
        if (self.counterfactual_of is None) != (self.counterfactual_kind is None):
            raise ValueError("counterfactual requires both baseline transition and declared characteristic")
        if self.counterfactual_of is not None:
            if self.counterfactual_of.artifact_kind != "decision_transition_projection":
                raise ValueError("counterfactual must pin a baseline transition")
            if self.state_effect.status == "CHANGED":
                raise ValueError("counterfactual cannot masquerade as an admitted graph successor")
        if self.observed_through < self.started_at:
            raise ValueError("observed_through cannot precede started_at")
        if self.closure_status == "COMPLETE" and self.missing_dependencies:
            raise ValueError("complete transition cannot declare missing dependencies")
        resolution = self.resolution_of_transition
        successor = (self.successor_working_context, self.successor_decision_local_projection)
        if resolution is None and any(item is not None for item in successor):
            raise ValueError("successor pins require resolution_of_transition")
        if resolution is not None and any(item is None for item in successor):
            raise ValueError("resolution requires successor WorkingContext and DLP pins")
        if resolution is not None and resolution.artifact_kind != "decision_transition_projection":
            raise ValueError("resolution_of_transition must pin a decision transition")
        for name, pin in (("test_control_evidence", self.test_control_evidence),
                          ("language_evidence", self.language_evidence)):
            if pin is not None and pin.artifact_kind != "evidence":
                raise ValueError(f"{name} must pin persisted Evidence")
        invocation_refs = tuple(item.invocation_ref for item in self.invocation_usage)
        if len(invocation_refs) != len(set(invocation_refs)):
            raise ValueError("invocation_ref must be unique within one transition")
        expected = "sha256:" + content_hash(self.hash_payload())
        if self.projection_hash != expected or self.projection_ref != "decision-transition-projection:" + expected[7:]:
            raise ValueError("decision transition projection ref/hash does not bind exact content")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("projection_ref", None)
        payload.pop("projection_hash", None)
        return payload

    def pin(self) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind="decision_transition_projection",
            artifact_ref=self.projection_ref,
            artifact_hash=self.projection_hash,
        )


class DecisionTrajectoryProjectionV1(_Contract):
    continuation_basis: ArtifactPinV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    dependencies: tuple["DependencyEdgeV1", ...] = Field(default=(), exclude_if=lambda value: not value)
    schema_version: Literal["decision_trajectory_projection.v1"] = "decision_trajectory_projection.v1"
    projection_ref: str = Field(min_length=1)
    projection_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    trajectory_ref: str = Field(min_length=1)
    branch_id: str = Field(min_length=1)
    source_snapshot: ArtifactPinV1
    projection_policy: ArtifactPinV1
    coverage_scope: Literal["WHOLE_RUN", "DEPENDENCY_SLICE"] = "WHOLE_RUN"
    boundary_dependencies: tuple[ArtifactPinV1, ...] = ()
    task_definition: ArtifactPinV1 | None = None
    experiment_integrity_snapshot: ArtifactPinV1 | None = None
    transition_sequence: tuple[ArtifactPinV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def exact_identity(self) -> "DecisionTrajectoryProjectionV1":
        refs = tuple(item.artifact_ref for item in self.transition_sequence)
        if len(refs) != len(set(refs)):
            raise ValueError("trajectory transition refs must be unique")
        if any(item.artifact_kind != "decision_transition_projection" for item in self.transition_sequence):
            raise ValueError("trajectory sequence requires decision_transition_projection pins")
        if self.experiment_integrity_snapshot is not None and self.experiment_integrity_snapshot.artifact_kind != "experiment_integrity_snapshot":
            raise ValueError("trajectory experiment pin must identify an experiment integrity snapshot")
        if self.continuation_basis is not None and self.continuation_basis.artifact_kind != "semantic_continuation_basis":
            raise ValueError("trajectory continuation must pin semantic_continuation_basis")
        if any((edge.organization_id,edge.workspace_id) != (self.organization_id,self.workspace_id) for edge in self.dependencies):
            raise ValueError("trajectory dependency scope mismatch")
        expected = "sha256:" + content_hash(self.hash_payload())
        if self.projection_hash != expected or self.projection_ref != "decision-trajectory-projection:" + expected[7:]:
            raise ValueError("decision trajectory projection ref/hash does not bind exact content")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("projection_ref", None)
        payload.pop("projection_hash", None)
        return payload

    def pin(self) -> ArtifactPinV1:
        return ArtifactPinV1(
            artifact_kind="decision_trajectory_projection",
            artifact_ref=self.projection_ref,
            artifact_hash=self.projection_hash,
        )


class TrajectoryInstrumentationGateReceiptV1(_Contract):
    schema_version: Literal["trajectory_instrumentation_gate_receipt.v1"] = (
        "trajectory_instrumentation_gate_receipt.v1"
    )
    gate_ref: str = Field(min_length=1)
    gate_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    projection_policy: ArtifactPinV1
    runtime_contract: ArtifactPinV1
    scorer_contract: ArtifactPinV1
    collector_contract: ArtifactPinV1
    trajectory: ArtifactPinV1
    related_trajectories: tuple[ArtifactPinV1, ...] = ()
    verification_evidence: tuple[ArtifactPinV1, ...] = Field(min_length=1)
    languages: tuple[Literal["en", "zh"], ...]
    has_failed_recovery_attempt: bool
    fresh_session_reconstruction: bool
    passed_check_ids: tuple[str, ...]
    check_results: tuple["TrajectoryInstrumentationCheckResultV1", ...] = Field(min_length=12)
    attested_properties: tuple[Literal["admissibility", "exact_binding", "replayability"], ...]
    status: Literal["GREEN"] = "GREEN"
    validated_at: datetime

    @model_validator(mode="after")
    def green_requires_complete_evidence(self) -> "TrajectoryInstrumentationGateReceiptV1":
        if any(pin.artifact_kind != "decision_trajectory_projection" for pin in self.related_trajectories):
            raise ValueError("related trajectories must be exact decision trajectory projections")
        if self.trajectory in self.related_trajectories or len(set(self.related_trajectories)) != len(self.related_trajectories):
            raise ValueError("receipt trajectory pins must be unique")
        if set(self.languages) != {"en", "zh"} or len(self.languages) != 2:
            raise ValueError("trajectory instrumentation languages must contain exact en and zh")
        if not self.has_failed_recovery_attempt:
            raise ValueError("trajectory instrumentation requires a failed/recovery attempt")
        if not self.fresh_session_reconstruction:
            raise ValueError("trajectory instrumentation requires fresh-session reconstruction")
        present = set(self.passed_check_ids)
        required = set(_TRAJECTORY_CHECKS)
        missing = tuple(item for item in _TRAJECTORY_CHECKS if item not in present)
        extra = tuple(sorted(present - required))
        if missing or extra or len(self.passed_check_ids) != len(_TRAJECTORY_CHECKS):
            detail = ",".join((*missing, *extra)) or "duplicate check ids"
            raise ValueError(f"trajectory instrumentation checks incomplete: {detail}")
        if {item.check_id for item in self.check_results} != required or len(self.check_results) != 12:
            raise ValueError("trajectory instrumentation check results incomplete")
        if any(item.status != "PASS" for item in self.check_results):
            raise ValueError("GREEN receipt requires every mechanically-derived TIG result to PASS")
        if set(self.attested_properties) != {"admissibility", "exact_binding", "replayability"}:
            raise ValueError("gate receipt must attest admissibility, exact binding, and replayability")
        expected = "sha256:" + content_hash(self.hash_payload())
        if self.gate_hash != expected or self.gate_ref != "trajectory-instrumentation-gate:" + expected[7:]:
            raise ValueError("trajectory instrumentation gate ref/hash does not bind exact content")
        return self

    def hash_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("gate_ref", None)
        payload.pop("gate_hash", None)
        return payload


class TrajectoryInstrumentationCheckResultV1(_Contract):
    check_id: str = Field(pattern=r"^TIG-(0[1-9]|1[0-2])$")
    status: Literal["PASS", "FAIL"]
    evidence_refs: tuple[ArtifactPinV1, ...] = ()
    failure_reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def result_has_explanation(self) -> "TrajectoryInstrumentationCheckResultV1":
        if self.status == "PASS" and self.failure_reasons:
            raise ValueError("passing TIG result cannot contain failure reasons")
        if self.status == "FAIL" and not self.failure_reasons:
            raise ValueError("failing TIG result must explain failure")
        return self


TRAJECTORY_INSTRUMENTATION_CHECK_IDS = _TRAJECTORY_CHECKS

__all__ = [
    "DecisionStateEffectV1",
    "DecisionTrajectoryProjectionV1",
    "DecisionTransitionBasisV1",
    "DecisionTransitionObservedV1",
    "DecisionTransitionProjectionV1",
    "DecisionTransitionSourceV1",
    "InvocationUsageV1",
    "TRAJECTORY_INSTRUMENTATION_CHECK_IDS",
    "TrajectoryInstrumentationGateReceiptV1",
]


class DependencyEdgeV1(_Contract):
    """A scoped derived relation; its pins require resolution by their real owners."""

    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    relation_class: Literal["EXECUTION", "SEMANTIC", "EPISTEMIC", "AUTHORITY", "TEMPORAL"]
    relation_type: Literal["WORKING_CONTEXT_OF", "SUPPORTED_BY", "AUTHORIZED_BY", "CONTINUES", "PRECEDES"]
    source: ArtifactPinV1
    target: ArtifactPinV1
    evidence: tuple[ArtifactPinV1, ...] = ()

    @model_validator(mode="after")
    def typed_relation(self) -> "DependencyEdgeV1":
        expected = {
            "WORKING_CONTEXT_OF": ("SEMANTIC", "working_context", "context_graph"),
            "SUPPORTED_BY": ("EPISTEMIC", "working_context", "evidence"),
            "AUTHORIZED_BY": ("AUTHORITY", "working_context", "decision_record"),
            "CONTINUES": ("EXECUTION", "effective_context_checkpoint", "working_context"),
            "PRECEDES": ("TEMPORAL", "effective_context_checkpoint", "effective_context_checkpoint"),
        }[self.relation_type]
        if (self.relation_class, self.source.artifact_kind, self.target.artifact_kind) != expected:
            raise ValueError("DEPENDENCY_RELATION_TYPE_MISMATCH")
        if any(pin.artifact_kind != "evidence" for pin in self.evidence):
            raise ValueError("DEPENDENCY_EVIDENCE_KIND_MISMATCH")
        return self


class SemanticContinuationBasisV1(_Contract):
    """Reconstruction manifest in the existing derived-projection store."""

    schema_version: Literal['semantic_continuation_basis.v1'] = 'semantic_continuation_basis.v1'
    projection_ref: str = Field(min_length=1)
    projection_hash: str = Field(pattern=SHA256_PATTERN)
    organization_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    trajectory_ref: str = Field(min_length=1)
    branch_id: str = Field(min_length=1)
    source_session_id: str = Field(min_length=1)
    checkpoint: ArtifactPinV1
    working_context: ArtifactPinV1
    context_graph: ArtifactPinV1
    evidence: tuple[ArtifactPinV1, ...]
    admission_decision: ArtifactPinV1
    dependencies: tuple[DependencyEdgeV1, ...]
    # Opaque native references remain pending; this manifest grants no execution truth.
    pending_owner_refs: tuple[str, ...] = ()
    unresolved_questions: tuple[str, ...] = ()

    def hash_payload(self):
        return self.model_dump(mode='json', exclude={'projection_ref', 'projection_hash'})

    @model_validator(mode='after')
    def exact_basis(self, info: ValidationInfo):
        kinds = (self.checkpoint.artifact_kind, self.working_context.artifact_kind,
                 self.context_graph.artifact_kind, self.admission_decision.artifact_kind)
        if kinds != ('effective_context_checkpoint','working_context','context_graph','decision_record'):
            raise ValueError('CONTINUATION_PIN_KIND_MISMATCH')
        if any(p.artifact_kind != 'evidence' for p in self.evidence):
            raise ValueError('CONTINUATION_EVIDENCE_KIND_MISMATCH')
        required = {('EXECUTION','CONTINUES',self.checkpoint,self.working_context),
                    ('SEMANTIC','WORKING_CONTEXT_OF',self.working_context,self.context_graph),
                    ('AUTHORITY','AUTHORIZED_BY',self.working_context,self.admission_decision)}
        required.update(('EPISTEMIC','SUPPORTED_BY',self.working_context,p) for p in self.evidence)
        actual = set()
        for edge in self.dependencies:
            if (edge.organization_id,edge.workspace_id) != (self.organization_id,self.workspace_id):
                raise ValueError('CONTINUATION_DEPENDENCY_SCOPE_MISMATCH')
            actual.add((edge.relation_class,edge.relation_type,edge.source,edge.target))
        if actual != required or len(actual) != len(self.dependencies):
            raise ValueError('CONTINUATION_DEPENDENCY_CLOSURE_MISMATCH')
        if info.context and info.context.get("continuation_create_draft"):
            return self
        digest = content_hash(self.hash_payload())
        if (self.projection_ref,self.projection_hash) != ('semantic-continuation-basis:'+digest,'sha256:'+digest):
            raise ValueError('CONTINUATION_IDENTITY_MISMATCH')
        return self

    @classmethod
    def create(cls, **values):
        draft = cls.model_validate(dict(projection_ref='pending',projection_hash='sha256:'+'0'*64,**values),context={'continuation_create_draft':True})
        digest = content_hash(draft.hash_payload())
        return cls.model_validate({**values,'projection_ref':'semantic-continuation-basis:'+digest,
                                   'projection_hash':'sha256:'+digest})

    def pin(self):
        return ArtifactPinV1(artifact_kind='semantic_continuation_basis',
                             artifact_ref=self.projection_ref,artifact_hash=self.projection_hash)
