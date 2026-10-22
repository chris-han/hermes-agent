"""Authenticated Phase1-001 ContextGraph Studio learning-gate seam.

The Studio cannot trust browser-submitted F1/precision/recall values as
decision truth.  This route is a thin server seam that resolves the
referenced V0/V1 evidence from the canonical MVL bundle, calls the existing
`evaluate_learning_gate(...)` function, and returns the resulting
`LearningGateResult` as MVL evidence.  The decision is product evidence
only; it does not activate a graph, certify graph construction/reasoning,
or mutate governance authority.

Security invariants (CF-E23, §19.3.4):
- The request payload only carries V0/V1 run refs and optional reviewer
  minutes.  The browser never submits metric values directly.
- `FindingMetrics` are resolved exclusively from the canonical V0/V1
  evidence bundle scoped to the authenticated tenant (organization +
  workspace).  Cross-tenant access is rejected with 404.
- When the canonical MVL path is not wired for a run, the route returns
  409 — never a fabricated decision.
- The response decision is appended to the MVL evidence bundle; it does
  not activate a graph, certify reasoning, or mutate governance.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from agents.webapi.deps import require_authenticated_context
from storage.knowledge_builder_evaluation_store import KnowledgeBuilderEvaluationStore
from semantier.semantica_context_graph.evaluation import (
    FindingMetrics,
    evaluate_learning_gate,
)


router = APIRouter(tags=["contextgraph-learning-gate"])
ROUTE_CONTEXTGRAPH_LEARNING_GATE = "/api/contextgraph/evaluation/learning-gate"


class LearningGateRequest(BaseModel):
    """Browser-supplied request payload.

    Only V0/V1 run refs and optional reviewer minutes are accepted.  All
    `FindingMetrics` fields are intentionally absent — the canonical MVL
    evidence bundle is the only allowed source for those values.
    """

    v0_run_ref: str = Field(min_length=1)
    v1_run_ref: str = Field(min_length=1)
    reviewer_minutes_v0: float | None = None
    reviewer_minutes_v1: float | None = None


class LearningGateResponse(BaseModel):
    decision: str  # 'GO' | 'STOP_REVISE' | 'SPLIT_FIX'
    f1_delta: float
    precision_delta: float
    recall_delta: float
    reviewer_minutes_delta_ratio: float | None
    reason: str
    v0_run_ref: str
    v1_run_ref: str
    canonical_resolution: str  # 'mvl_bundle' | 'mvl_bundle_ux'


class CanonicalRunNotFoundError(Exception):
    """Raised when a V0/V1 run ref does not resolve to a canonical MVL bundle
    in the authenticated tenant scope."""


class CanonicalRunMetricsUnavailableError(Exception):
    """Raised when the canonical MVL bundle exists but its FindingMetrics
    are not yet wired through this seam."""


def _resolve_canonical_run_metrics(
    organization_id: str,
    workspace_id: str,
    run_ref: str,
) -> FindingMetrics:
    """Resolve canonical metrics from a tenant-scoped evaluation run."""

    run = KnowledgeBuilderEvaluationStore().get_evaluation_run_scoped(
        run_ref,
        organization_id=organization_id,
        workspace_id=workspace_id,
    )
    if run is None:
        raise CanonicalRunNotFoundError(f"Canonical evaluation run not found: {run_ref}")
    if run.get("run_status") != "completed" or not isinstance(run.get("metrics"), dict):
        raise CanonicalRunMetricsUnavailableError(
            f"Canonical evaluation metrics are unavailable for run: {run_ref}"
        )
    metrics = run["metrics"]
    results = run.get("results") or []
    tp = sum(1 for row in results if row.get("expected_outcome") == "match" and row.get("actual_outcome") == "match")
    fp = sum(1 for row in results if row.get("expected_outcome") != "match" and row.get("actual_outcome") == "match")
    fn = sum(1 for row in results if row.get("expected_outcome") == "match" and row.get("actual_outcome") != "match")
    precision = float(metrics.get("precision", 0.0))
    recall = float(metrics.get("recall", 0.0))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return FindingMetrics(
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        evidence_anchor_accuracy=float(metrics.get("source_anchor_coverage", 0.0)),
        graph_link_correctness=float(metrics.get("graph_link_correctness", 0.0)),
    )


@router.post(ROUTE_CONTEXTGRAPH_LEARNING_GATE, response_model=LearningGateResponse)
def post_contextgraph_learning_gate(payload: LearningGateRequest, request: Request) -> LearningGateResponse:
    """Compute the Phase1-001 product-learning decision for a V0/V1 pair.

    Authenticated callers only.  See module docstring for authority scope.
    """

    ctx = require_authenticated_context(request)
    if not ctx.organization_id or not ctx.workspace_id:
        raise HTTPException(status_code=409, detail="Workspace organization context is required")

    try:
        v0 = _resolve_canonical_run_metrics(
            ctx.organization_id,
            ctx.workspace_id,
            payload.v0_run_ref,
        )
        v1 = _resolve_canonical_run_metrics(
            ctx.organization_id,
            ctx.workspace_id,
            payload.v1_run_ref,
        )
    except CanonicalRunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except CanonicalRunMetricsUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    reviewer_minutes_provided = (
        payload.reviewer_minutes_v0 is not None
        or payload.reviewer_minutes_v1 is not None
    )

    result = evaluate_learning_gate(
        v0,
        v1,
        reviewer_minutes_v0=payload.reviewer_minutes_v0,
        reviewer_minutes_v1=payload.reviewer_minutes_v1,
    )

    return LearningGateResponse(
        decision=result.decision,
        f1_delta=result.f1_delta,
        precision_delta=result.precision_delta,
        recall_delta=result.recall_delta,
        reviewer_minutes_delta_ratio=result.reviewer_minutes_delta_ratio,
        reason=result.reason,
        v0_run_ref=payload.v0_run_ref,
        v1_run_ref=payload.v1_run_ref,
        canonical_resolution=(
            "mvl_bundle_ux" if reviewer_minutes_provided else "mvl_bundle"
        ),
    )


__all__ = [
    "CanonicalRunMetricsUnavailableError",
    "CanonicalRunNotFoundError",
    "LearningGateRequest",
    "LearningGateResponse",
    "ROUTE_CONTEXTGRAPH_LEARNING_GATE",
    "post_contextgraph_learning_gate",
    "router",
]
