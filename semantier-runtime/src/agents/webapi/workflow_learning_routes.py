"""Authenticated workflow-learning adapter into the canonical ChangeCandidate lifecycle."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from agents.meeting_workflow_installation import assess as assess_meeting_installation, verify as verify_meeting_installation
from agents.webapi.deps import require_authenticated_context
from contracts.evidence_location import EvidenceSelector
from contracts.ontology_building import ArtifactPinV1
from contracts.semantic_kernel import (
    ChangeAssessment,
    ChangeDelta,
    ChangeObservation,
    ComparisonExpectation,
    EvaluationBasis,
    Evidence,
    ProtectedInput,
    WorkflowEditOrigin,
)
from eos.artifact_hashing import content_hash
from services.change_candidate_service import ChangeCandidateError, ChangeCandidateService
from storage.ontology_build_store import OntologyBuildStore, ontology_build_store_path


router = APIRouter(tags=["workflow-learning"])
ROUTE_WORKFLOW_LEARNING_CANDIDATE = "/api/workflow-learning/candidates"
ROUTE_WORKFLOW_LEARNING_CANDIDATE_VERIFY = "/api/workflow-learning/candidates/verify"
_SHA256 = r"^sha256:[0-9a-f]{64}$"


class WorkflowLearningCandidateRequest(BaseModel):
    workflow_id: str = Field(min_length=1)
    owner_ref: str = Field(min_length=1)
    source_workflow_ref: str = Field(min_length=1)
    source_workflow_version: str = Field(min_length=1)
    source_workflow_hash: str = Field(pattern=_SHA256)
    successor_workflow_ref: str = Field(min_length=1)
    successor_workflow_version: str = Field(min_length=1)
    successor_workflow_hash: str = Field(pattern=_SHA256)
    feedback_ref: str = Field(min_length=1)
    feedback_hash: str = Field(pattern=_SHA256)
    simulation_ref: str = Field(min_length=1)
    simulation_hash: str = Field(pattern=_SHA256)
    comparison_ref: str = Field(min_length=1)
    comparison_hash: str = Field(pattern=_SHA256)
    matched_case_ref: str = Field(min_length=1)
    holdout_case_ref: str = Field(min_length=1)
    recorded_at: datetime
    attribution: Literal["local_procedure"]
    candidate_owner: Literal["skill_workflow"]
    provider_calls: int = Field(ge=0)
    external_effects: int = Field(ge=0)
    matched_improved: bool
    independent_holdout_stable: bool
    unknown_preserved: bool

    @model_validator(mode="after")
    def validate_bounded_workflow_trial(self):
        if self.source_workflow_hash == self.successor_workflow_hash:
            raise ValueError("WORKFLOW_SUCCESSOR_MUST_DIFFER")
        if self.matched_case_ref == self.holdout_case_ref:
            raise ValueError("MATCHED_HOLDOUT_OVERLAP")
        if self.provider_calls != 0 or self.external_effects != 0:
            raise ValueError("SIMULATION_SIDE_EFFECT_BOUNDARY_VIOLATION")
        if not (
            self.matched_improved
            and self.independent_holdout_stable
            and self.unknown_preserved
        ):
            raise ValueError("WORKFLOW_COMPARISON_NOT_ADMISSIBLE")
        if self.recorded_at.tzinfo is None or self.recorded_at.utcoffset() is None:
            raise ValueError("RECORDED_AT_MUST_BE_TIMEZONE_AWARE")
        return self


class WorkflowLearningCandidateResponse(BaseModel):
    attribution: Literal["local_procedure"]
    candidate_owner: Literal["skill_workflow"]
    proposal: ArtifactPinV1
    observation: ArtifactPinV1
    assessment: ArtifactPinV1
    successor_input: ArtifactPinV1
    disposition: Literal["ACCEPTED_FOR_SUCCESSOR_INPUT"]


class WorkflowLearningCandidateVerifyRequest(BaseModel):
    assessment: ArtifactPinV1
    successor_input: ArtifactPinV1


class WorkflowLearningCandidateVerifyResponse(BaseModel):
    verified: Literal[True]
    assessment: ArtifactPinV1
    successor_input: ArtifactPinV1
    disposition: Literal["ACCEPTED_FOR_SUCCESSOR_INPUT"]


def _derived_pin(kind: str, ref_prefix: str, value: object) -> ArtifactPinV1:
    digest = content_hash(value)
    return ArtifactPinV1(
        artifact_kind=kind,
        artifact_ref=f"{ref_prefix}:{digest}",
        artifact_hash=f"sha256:{digest}",
    )


def _evidence(
    *,
    organization_id: str,
    workspace_id: str,
    source_ref: str,
    source_hash: str,
    role: str,
    recorded_at: datetime,
) -> Evidence:
    selector = EvidenceSelector(
        schema_version="semantier.evidence_selector.v1",
        selector_kind="structure",
        exact_quote=None,
        prefix=None,
        suffix=None,
        normalized_quote=None,
        structural_path=["workflow-learning", role],
        table=None,
        source_element_ref=source_ref,
    )
    return Evidence.create(
        schema_version="semantier.evidence.v1",
        organization_id=organization_id,
        workspace_id=workspace_id,
        source_ref=source_ref,
        source_hash=source_hash,
        selector=selector,
        selector_hash=selector.selector_hash(),
        evidence_role=role,
        provenance=(),
        observed_at=recorded_at,
        source_time=None,
        recorded_at=recorded_at,
    )


@router.post(
    ROUTE_WORKFLOW_LEARNING_CANDIDATE,
    response_model=WorkflowLearningCandidateResponse,
)
def post_workflow_learning_candidate(
    payload: WorkflowLearningCandidateRequest,
    request: Request,
) -> WorkflowLearningCandidateResponse:
    ctx = require_authenticated_context(request)
    if not ctx.organization_id or not ctx.workspace_id:
        raise HTTPException(status_code=409, detail="Workspace organization context is required")

    store = OntologyBuildStore(ontology_build_store_path(ctx.workspace_root))
    try:
        feedback = _evidence(
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            source_ref=payload.feedback_ref,
            source_hash=payload.feedback_hash,
            role="workflow_edit_feedback",
            recorded_at=payload.recorded_at,
        )
        simulation = _evidence(
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            source_ref=payload.simulation_ref,
            source_hash=payload.simulation_hash,
            role="workflow_simulation_receipt",
            recorded_at=payload.recorded_at,
        )
        comparison = _evidence(
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            source_ref=payload.comparison_ref,
            source_hash=payload.comparison_hash,
            role="workflow_behavior_comparison",
            recorded_at=payload.recorded_at,
        )
        for item in (feedback, simulation, comparison):
            store.put(item)

        source = ArtifactPinV1(
            artifact_kind="workflow_definition",
            artifact_ref=payload.source_workflow_ref,
            artifact_hash=payload.source_workflow_hash,
        )
        successor = ArtifactPinV1(
            artifact_kind="workflow_definition",
            artifact_ref=payload.successor_workflow_ref,
            artifact_hash=payload.successor_workflow_hash,
        )
        matched_case = _derived_pin(
            "case",
            "workflow-case",
            {"workflow_id": payload.workflow_id, "case": payload.matched_case_ref},
        )
        holdout_case = _derived_pin(
            "case",
            "workflow-case",
            {"workflow_id": payload.workflow_id, "case": payload.holdout_case_ref},
        )
        matched_execution = _derived_pin(
            "execution_attempt",
            "workflow-simulation-attempt",
            {"simulation": payload.simulation_hash, "case": payload.matched_case_ref},
        )
        holdout_execution = _derived_pin(
            "execution_attempt",
            "workflow-simulation-attempt",
            {"simulation": payload.simulation_hash, "case": payload.holdout_case_ref},
        )
        evaluator = _derived_pin(
            "evaluator",
            "workflow-evaluator",
            {"protocol": "paired-workflow-simulation.v1"},
        )
        suite = _derived_pin(
            "obligation_suite",
            "workflow-obligation-suite",
            {"matched": payload.matched_case_ref, "holdout": payload.holdout_case_ref},
        )
        policy = _derived_pin(
            "comparison_policy",
            "workflow-comparison-policy",
            {
                "policy": "matched-improves-holdout-stable-unknown-preserved.v1",
                "provider_calls": 0,
                "external_effects": 0,
            },
        )
        basis = EvaluationBasis(
            evaluator=evaluator,
            obligation_suite=suite,
            comparison_policy=policy,
            protected_inputs=(
                ProtectedInput(slot_id="source_workflow", artifact=source),
                ProtectedInput(slot_id="workflow_feedback", artifact=feedback.pin()),
                ProtectedInput(slot_id="paired_simulation", artifact=simulation.pin()),
            ),
            runtime_pins=(),
            provider_pins=(),
            config_pins=(),
            candidate_delta_bounds="workflow topology only; semantic factors, Tender context, authority/policy basis, and Skill selection frozen",
            comparison_expectation=ComparisonExpectation(
                matched_baseline_result="MISSING_AUTHORITY_RESOLUTION",
                matched_successor_result="AUTHORITY_RESOLUTION_PRESENT",
                holdout_policy="UNCHANGED",
            ),
        )
        service = ChangeCandidateService(store=store)
        proposal = service.propose(
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            origin=WorkflowEditOrigin(
                workflow=source,
                evidence=(feedback.pin(), simulation.pin(), comparison.pin()),
            ),
            failure_class="STRUCTURE",
            target="WORKFLOW",
            hypothesis="The bounded workflow topology correction improves the matched Tender case while preserving the independent holdout and structural UNKNOWN behavior.",
            evaluation_basis=basis,
            delta=ChangeDelta(
                description="Adopt the simulated Tender workflow topology correction only.",
                claims_semantic_graph_change=False,
                basis_refs=(feedback.pin(), simulation.pin(), comparison.pin(), source),
                structural_diff=_derived_pin(
                    "workflow_diff",
                    "workflow-diff",
                    {
                        "source": payload.source_workflow_hash,
                        "successor": payload.successor_workflow_hash,
                    },
                ),
            ),
            matched_cases=(matched_case,),
            independent_holdout_cases=(holdout_case,),
            recorded_at=payload.recorded_at,
        )
        observation = service.record_observations(
            proposal=proposal.pin(),
            observations=(
                ChangeObservation(
                    case=matched_case,
                    execution_attempt=matched_execution,
                    status="COMPLETE",
                    metric_refs=(comparison.pin(),),
                    diagnostics=(),
                ),
                ChangeObservation(
                    case=holdout_case,
                    execution_attempt=holdout_execution,
                    status="COMPLETE",
                    metric_refs=(comparison.pin(),),
                    diagnostics=(),
                ),
            ),
            recorded_at=payload.recorded_at,
        )
        assessment = service.assess(
            observation_revision=observation.pin(),
            assessment=ChangeAssessment(
                trial_validity="VALID",
                verdict="BETTER",
                metric_refs=(comparison.pin(),),
                disposition="ACCEPTED_FOR_SUCCESSOR_INPUT",
                reason_code="MATCHED_IMPROVED_INDEPENDENT_HOLDOUT_STABLE_UNKNOWN_PRESERVED",
            ),
            successor_inputs=(successor,),
            recorded_at=payload.recorded_at,
        )
        return WorkflowLearningCandidateResponse(
            attribution="local_procedure",
            candidate_owner="skill_workflow",
            proposal=proposal.pin(),
            observation=observation.pin(),
            assessment=assessment.pin(),
            successor_input=successor,
            disposition="ACCEPTED_FOR_SUCCESSOR_INPUT",
        )
    except ChangeCandidateError as exc:
        raise HTTPException(status_code=409, detail=exc.code) from exc
    finally:
        store.close()


@router.post(
    ROUTE_WORKFLOW_LEARNING_CANDIDATE_VERIFY,
    response_model=WorkflowLearningCandidateVerifyResponse,
)
def post_workflow_learning_candidate_verify(
    payload: WorkflowLearningCandidateVerifyRequest,
    request: Request,
) -> WorkflowLearningCandidateVerifyResponse:
    ctx = require_authenticated_context(request)
    if not ctx.organization_id or not ctx.workspace_id:
        raise HTTPException(status_code=409, detail="Workspace organization context is required")
    if payload.assessment.artifact_kind != "change_candidate":
        raise HTTPException(status_code=409, detail="CHANGE_CANDIDATE_ASSESSMENT_REQUIRED")
    if payload.successor_input.artifact_kind != "workflow_definition":
        raise HTTPException(status_code=409, detail="WORKFLOW_SUCCESSOR_INPUT_REQUIRED")

    store = OntologyBuildStore(ontology_build_store_path(ctx.workspace_root))
    try:
        service = ChangeCandidateService(store=store)
        candidate = service.get(pin=payload.assessment)
        if (candidate.organization_id, candidate.workspace_id) != (
            ctx.organization_id,
            ctx.workspace_id,
        ):
            raise HTTPException(status_code=404, detail="Workflow candidate not found")
        if (
            candidate.stage != "ASSESSMENT"
            or candidate.target != "WORKFLOW"
            or candidate.assessment is None
            or candidate.assessment.disposition != "ACCEPTED_FOR_SUCCESSOR_INPUT"
            or candidate.successor_inputs != (payload.successor_input,)
        ):
            raise HTTPException(status_code=409, detail="WORKFLOW_CANDIDATE_NOT_ADMISSIBLE")
        return WorkflowLearningCandidateVerifyResponse(
            verified=True,
            assessment=candidate.pin(),
            successor_input=payload.successor_input,
            disposition="ACCEPTED_FOR_SUCCESSOR_INPUT",
        )
    except ChangeCandidateError as exc:
        raise HTTPException(status_code=404, detail=exc.code) from exc
    finally:
        store.close()


__all__ = [
    "ROUTE_WORKFLOW_LEARNING_CANDIDATE",
    "ROUTE_WORKFLOW_LEARNING_CANDIDATE_VERIFY",
    "WorkflowLearningCandidateRequest",
    "WorkflowLearningCandidateResponse",
    "WorkflowLearningCandidateVerifyRequest",
    "WorkflowLearningCandidateVerifyResponse",
    "post_workflow_learning_candidate",
    "post_workflow_learning_candidate_verify",
    "router",
]


class MeetingWorkflowInstallationRequest(BaseModel):
    definition: dict
    assessment: ArtifactPinV1 | None = None


@router.post('/api/workflow-learning/meeting-installation')
def post_meeting_workflow_installation(payload: MeetingWorkflowInstallationRequest, request: Request):
    ctx = require_authenticated_context(request)
    try:
        if payload.assessment:
            return verify_meeting_installation(ctx, payload.definition, payload.assessment.model_dump(mode='json'))
        return assess_meeting_installation(ctx, payload.definition)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
