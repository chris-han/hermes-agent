"""Purpose-bounded operational producer records; no semantic authority."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, model_validator

from contracts.ontology_building import ArtifactPinV1
from contracts.runtime_context_reconstruction import CapturedJsonV1
from contracts.semantic_kernel import WorkingContext, ContextWorld, IntentSnapshot
from contracts.runtime_semantic_evaluation import (
    _AddressedRuntimeIdentity,
    _RuntimeIdentityContract,
    _hash_basis,
    RuntimeEvaluationStateV1,
    RuntimeExecutionAdmissionV1,
    RuntimeEvaluationRequestIdentityV1,
    RuntimeEvaluationRequestOutcomeV1,
    RuntimeTemporalCaptureV1,
    GovernedTargetPinV1,
)

_Count = Annotated[int, Field(strict=True, ge=0)]
_Positive = Annotated[int, Field(strict=True, ge=1)]
_Budget = Annotated[int, Field(strict=True, ge=1, le=4)]


class RuntimeContextReasoningRequestV1(_AddressedRuntimeIdentity):
    ARTIFACT_KIND = "runtime_provider_request"
    REF_FIELD = "request_ref"
    HASH_FIELD = "request_hash"
    REF_PREFIX = "runtime-context-reasoning-request"

    schema_version: Literal["runtime_context_reasoning_request.v1"]
    request_ref: str
    request_hash: str
    organization_id: str
    workspace_id: str
    context: ArtifactPinV1
    world: ArtifactPinV1
    case: ArtifactPinV1
    world_basis: ContextWorld
    intent: IntentSnapshot
    source_inputs: tuple[ArtifactPinV1, ...]
    target: GovernedTargetPinV1
    working_context: WorkingContext
    facts: tuple[str, ...] = Field(min_length=1)
    rules: tuple[str, ...] = Field(min_length=1)
    goal: str = Field(min_length=1)
    executable: ArtifactPinV1
    adapter: ArtifactPinV1
    projector: ArtifactPinV1
    projection_recipe: Literal["all_declared_assertions.v1"]
    max_depth: Literal[10]
    max_dispatches: _Budget

    @classmethod
    def _basis(cls, value):
        return _hash_basis(value, cls.REF_FIELD, cls.HASH_FIELD)

    @model_validator(mode="after")
    def validate_context_input(self):
        wc = self.working_context
        scope = (self.organization_id, self.workspace_id)
        if self.context.artifact_kind != "context_graph" or self.world != self.context:
            raise ValueError("CONTEXT_GRAPH_REQUIRED")
        if (
            self.case.artifact_kind != "context_graph"
            or self.target.artifact != self.case
        ):
            raise ValueError("CASE_TARGET_MISMATCH")
        if (
            (wc.organization_id, wc.workspace_id) != scope
            or (self.target.organization_id, self.target.workspace_id) != scope
            or wc.context_graph != self.context
            or wc.task_goal != self.goal
        ):
            raise ValueError("REQUEST_CONTEXT_MISMATCH")
        if (
            (
                self.executable.artifact_kind,
                self.adapter.artifact_kind,
                self.projector.artifact_kind,
            )
            != ("reasoner_executable", "reasoning_adapter", "context_projector")
            or self.target.target_type != "graph"
            or self.target.scope_bridge is not None
        ):
            raise ValueError("INVALID_PRODUCER_BASIS")
        if len(self.facts) != len(wc.selected_assertion_ids):
            raise ValueError("FACT_SELECTION_MISMATCH")
        return self


class RuntimeContextReasoningCaptureV1(_AddressedRuntimeIdentity):
    ARTIFACT_KIND = "runtime_reasoning_capture"
    REF_FIELD = "capture_ref"
    HASH_FIELD = "capture_hash"
    REF_PREFIX = "runtime-context-reasoning-capture"

    schema_version: Literal["runtime_context_reasoning_capture.v1"]
    capture_ref: str
    capture_hash: str
    organization_id: str
    workspace_id: str
    run: ArtifactPinV1
    provider_request: ArtifactPinV1
    attempt_id: str = Field(min_length=1)
    raw_output: CapturedJsonV1
    captured_at: datetime

    @classmethod
    def _basis(cls, value):
        return _hash_basis(value, cls.REF_FIELD, cls.HASH_FIELD)

    @property
    def native_output(self):
        return json.loads(self.raw_output.canonical_json)

    @model_validator(mode="after")
    def validate_native_shape(self):
        if (
            self.run.artifact_kind != "runtime_evaluation_run"
            or self.provider_request.artifact_kind != "runtime_provider_request"
        ):
            raise ValueError("CAPTURE_BINDING_MISMATCH")
        value = self.native_output
        if value is not None and (
            not isinstance(value, dict)
            or set(value)
            != {"conclusion", "premises", "confidence", "metadata", "rule_used"}
        ):
            raise ValueError("INVALID_NATIVE_CAPTURE")
        return self


class RuntimeTemporalCaptureV2(RuntimeTemporalCaptureV1):
    schema_version: Literal["runtime_temporal_capture.v2"]
    attempt_id: str = Field(min_length=1)


class RuntimeContextExecutionAdmissionV2(RuntimeExecutionAdmissionV1):
    schema_version: Literal["runtime_execution_admission.v2"]
    purpose: Literal["CONTEXT_REASONING"]


class RuntimeContextRequestIdentityV2(RuntimeEvaluationRequestIdentityV1):
    schema_version: Literal["runtime_evaluation_request_identity.v2"]
    operation: Literal["EVALUATE_RUNTIME_CONTEXT_REASONING"]


class RuntimeContextRequestOutcomeV2(RuntimeEvaluationRequestOutcomeV1):
    schema_version: Literal["runtime_evaluation_request_outcome.v2"]
    operation: Literal["EVALUATE_RUNTIME_CONTEXT_REASONING"]


class RuntimeDispatchEventV1(_RuntimeIdentityContract):
    sequence: _Count
    attempt_id: str = Field(min_length=1)
    ordinal: _Positive
    run: ArtifactPinV1
    request: ArtifactPinV1
    fence: _Positive
    role: Literal["ACTOR"]
    kind: Literal[
        "DISPATCH_REGISTERED", "RETURN_CAPTURED", "CALL_EXCEPTION", "OUTCOME_UNCERTAIN"
    ]
    lease_until: datetime | None
    capture: ArtifactPinV1 | None
    diagnostic_code: str | None
    recorded_at: datetime

    @model_validator(mode="after")
    def validate_event_shape(self):
        if (
            self.run.artifact_kind != "runtime_evaluation_run"
            or self.request.artifact_kind != "runtime_provider_request"
        ):
            raise ValueError("INVALID_DISPATCH_BINDING")
        if self.kind == "DISPATCH_REGISTERED":
            if (
                self.lease_until is None
                or self.lease_until <= self.recorded_at
                or self.capture is not None
                or self.diagnostic_code is not None
            ):
                raise ValueError("INVALID_REGISTRATION")
        elif self.lease_until is not None:
            raise ValueError("LEASE_BELONGS_TO_REGISTRATION")
        if (self.kind == "RETURN_CAPTURED") != (self.capture is not None):
            raise ValueError("CAPTURE_EVENT_REQUIRED")
        if (
            self.kind in {"CALL_EXCEPTION", "OUTCOME_UNCERTAIN"}
            and not self.diagnostic_code
        ):
            raise ValueError("DIAGNOSTIC_REQUIRED")
        return self


class RuntimeEvaluationStateV2(RuntimeEvaluationStateV1):
    schema_version: Literal["runtime_evaluation_state.v2"]
    producer_mode: Literal["TEMPORAL", "CONTEXT_REASONING"]
    max_dispatches: _Budget
    dispatch_lease_seconds: Literal[30] = 30
    attempt_events: tuple[RuntimeDispatchEventV1, ...] = ()

    @model_validator(mode="after")
    def validate_state_shape(self):
        if self.producer_mode == "TEMPORAL":
            super().validate_state_shape()
        else:
            if self.run.artifact_kind != "runtime_evaluation_run" or self.results:
                raise ValueError("INVALID_CONTEXT_PRODUCER_STATE")
            if (self.revision == 0) != (self.predecessor is None):
                raise ValueError("INVALID_STATE_LINEAGE")
            if (
                self.predecessor is not None
                and self.predecessor.artifact_kind != "runtime_evaluation_state"
            ):
                raise ValueError("INVALID_STATE_LINEAGE")
            if (
                self.capture is not None
                and self.capture.artifact_kind != "runtime_reasoning_capture"
            ):
                raise ValueError("INVALID_CONTEXT_CAPTURE")
            if len(self.decisions) > 1 or any(
                p.artifact_kind != "decision_record" for p in self.decisions
            ):
                raise ValueError("INVALID_CONTEXT_DECISION")
            if any(
                p.artifact_kind != "runtime_evaluation_diagnostic"
                for p in self.diagnostics
            ):
                raise ValueError("INVALID_CONTEXT_DIAGNOSTIC")
        registered = {}
        terminal = {}
        for sequence, event in enumerate(self.attempt_events):
            expected_capture_kind = (
                "runtime_temporal_capture"
                if self.producer_mode == "TEMPORAL"
                else "runtime_reasoning_capture"
            )
            if (
                event.capture is not None
                and event.capture.artifact_kind != expected_capture_kind
            ):
                raise ValueError("INVALID_ATTEMPT_CAPTURE_KIND")
            if event.sequence != sequence or event.run != self.run:
                raise ValueError("INVALID_ATTEMPT_HISTORY")
            if event.kind == "DISPATCH_REGISTERED":
                if (
                    event.attempt_id in registered
                    or event.ordinal != len(registered) + 1
                ):
                    raise ValueError("DUPLICATE_DISPATCH")
                registered[event.attempt_id] = event
            else:
                original = registered.get(event.attempt_id)
                if (
                    original is None
                    or (event.request, event.ordinal, event.fence, event.role)
                    != (
                        original.request,
                        original.ordinal,
                        original.fence,
                        original.role,
                    )
                    or event.recorded_at < original.recorded_at
                ):
                    raise ValueError("INVALID_ATTEMPT_HISTORY")
                previous = terminal.get(event.attempt_id)
                if previous in {"RETURN_CAPTURED", "CALL_EXCEPTION"} or (
                    previous == "OUTCOME_UNCERTAIN"
                    and event.kind not in {"RETURN_CAPTURED", "CALL_EXCEPTION"}
                ):
                    raise ValueError("DUPLICATE_ATTEMPT_OUTCOME")
                terminal[event.attempt_id] = event.kind
        if len(registered) > self.max_dispatches:
            raise ValueError("DISPATCH_BUDGET_EXCEEDED")
        captured = {
            event.capture
            for event in self.attempt_events
            if event.kind == "RETURN_CAPTURED"
        }
        if self.capture is not None and self.capture not in captured:
            raise ValueError("SELECTED_CAPTURE_NOT_RETAINED")
        return self


class RuntimeContextReferenceCaseV1(_RuntimeIdentityContract):
    """One independently authored DEVELOPMENT case, frozen before execution."""

    case: ArtifactPinV1
    baseline_request: ArtifactPinV1
    successor_request: ArtifactPinV1
    reference: ArtifactPinV1
    expected_native: Literal["PROOF", "UNKNOWN"]
    critical: Literal[True] = True


class RuntimeContextReferencePanelV1(_AddressedRuntimeIdentity):
    """Bounded two-case operational basis inside the existing runtime store."""

    ARTIFACT_KIND = "runtime_context_reference_panel"
    REF_FIELD = "panel_ref"
    HASH_FIELD = "panel_hash"
    REF_PREFIX = "runtime-context-reference-panel"

    schema_version: Literal["runtime_context_reference_panel.v1"]
    panel_ref: str
    panel_hash: str
    organization_id: str
    workspace_id: str
    matched: RuntimeContextReferenceCaseV1
    holdout: RuntimeContextReferenceCaseV1
    scorer: ArtifactPinV1
    scorer_source: str
    scorer_environment: tuple[tuple[str, str], ...]
    comparison_policy: ArtifactPinV1
    comparison_source: str
    policy: Literal["context-proof-independent-reference.v1"]
    exposure: Literal["DEVELOPMENT"]
    frozen_at: datetime

    @classmethod
    def _basis(cls, value):
        return _hash_basis(value, cls.REF_FIELD, cls.HASH_FIELD)

    @model_validator(mode="after")
    def validate_panel(self):
        from hashlib import sha256

        if self.matched.case == self.holdout.case or self.matched.reference == self.holdout.reference:
            raise ValueError("INDEPENDENT_HOLDOUT_REQUIRED")
        if (self.scorer.artifact_kind != "context_reference_scorer"
                or self.scorer.artifact_hash != "sha256:" + sha256(self.scorer_source.encode()).hexdigest()):
            raise ValueError("SCORER_SOURCE_MISMATCH")
        if (self.comparison_policy.artifact_kind != "context_comparison_policy"
                or self.comparison_policy.artifact_hash != "sha256:" + sha256(self.comparison_source.encode()).hexdigest()
                or tuple(name for name, _ in self.scorer_environment) != ("python", "pydantic")):
            raise ValueError("SCORER_CONFIGURATION_MISMATCH")
        for case in (self.matched, self.holdout):
            if (case.case.artifact_kind != "context_graph" or case.reference.artifact_kind != "evidence"
                    or any(pin.artifact_kind != "runtime_provider_request"
                           for pin in (case.baseline_request, case.successor_request))):
                raise ValueError("INVALID_CONTEXT_REFERENCE_BASIS")
        return self


PRODUCER_RECORD_MODELS = (
    RuntimeContextReferencePanelV1,
    RuntimeContextReasoningRequestV1,
    RuntimeContextReasoningCaptureV1,
    RuntimeTemporalCaptureV2,
    RuntimeContextExecutionAdmissionV2,
    RuntimeContextRequestOutcomeV2,
    RuntimeEvaluationStateV2,
)
