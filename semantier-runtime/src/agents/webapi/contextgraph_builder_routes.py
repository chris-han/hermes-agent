"""Candidate-only Builder routes for the tender ContextGraph MVL."""

from __future__ import annotations

from pathlib import Path
import asyncio
import json

from fastapi import APIRouter, HTTPException, Request

from agents.corpus_graph_schema_capability_factory import (
    build_corpus_graph_schema_capability,
)
from agents.webapi.deps import require_route_authorization
from agents.route_policy import RouteAuthorizationClass
from agents.workspace_uploads import save_workspace_upload
from semantier.knowledge_feature_packs.source_elements import source_elements
from semantier.semantica_context_graph.artifact_repository import ContextGraphArtifactRepository
from semantier.semantica_context_graph.native_builder import build_v0_candidate
from services.corpus_graph_schema_capability_service import (
    CorpusGraphSchemaCapabilityService,
)


router = APIRouter(tags=["contextgraph-builder"])


def _ctx(request: Request):
    return require_route_authorization(request, authorization_class=RouteAuthorizationClass.TENANT_MEMBER,
                                       route_hint="contextgraph builder")


def _schema_capability(ctx) -> CorpusGraphSchemaCapabilityService:
    authority = {
        "workspace_id": getattr(ctx, "workspace_id", None),
        "user_id": getattr(ctx, "user_id", None),
        "organization_id": getattr(ctx, "organization_id", None),
    }
    if any(value is None or not str(value).strip() for value in authority.values()):
        raise HTTPException(
            status_code=403, detail="authenticated tenant context is incomplete"
        )
    return build_corpus_graph_schema_capability(
        workspace_root=ctx.workspace_root,
        workspace_id=str(authority["workspace_id"]),
        user_id=str(authority["user_id"]),
        organization_id=str(authority["organization_id"]),
    )


def _reject_payload_authority(body: dict) -> None:
    reserved = {
        "workspaceId",
        "workspace_id",
        "organizationId",
        "organization_id",
        "userId",
        "user_id",
        "actorId",
        "actor_id",
    }
    if reserved.intersection(body):
        raise HTTPException(
            status_code=400,
            detail="authority fields are derived from authenticated context",
        )


@router.post("/api/contextgraph/builder/upload")
async def upload_candidate_source(request: Request):
    ctx = _ctx(request)
    form = await request.form()
    upload = form.get("file")
    session_id = str(form.get("session_id") or "").strip()
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(status_code=400, detail="file required")
    if Path(str(getattr(upload, "filename", ""))).suffix.lower() != ".docx":
        raise HTTPException(status_code=400, detail="only DOCX is supported")
    result = save_workspace_upload(workspace_root=Path(ctx.workspace_root), session_id=session_id,
                                   filename=str(upload.filename), content=await upload.read(),
                                   content_type=str(getattr(upload, "content_type", "")))
    return {"status": "ok", "relativePath": result.relative_path, "sourceHash": source_elements(result.file_path)["document_hash"]}


@router.post("/api/contextgraph/builder/preview")
async def preview_candidate(request: Request):
    ctx = _ctx(request)
    body = await request.json()
    relative = str(body.get("relativePath") or "")
    root = Path(ctx.workspace_root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root / "sessions") or not path.is_file() or path.suffix.lower() != ".docx":
        raise HTTPException(status_code=400, detail="source must be a governed session upload")
    parsed = source_elements(path)
    return {"status": "extraction-complete", "sourceHash": parsed["document_hash"],
            "items": [{"text": e["text"], "family": e.get("family"), "anchor": e["anchor"]} for e in parsed["elements"]]}


@router.get("/api/contextgraph/schema-candidates")
async def list_schema_candidates(request: Request):
    ctx = _ctx(request)
    return _schema_capability(ctx).list(limit=20)


@router.post("/api/contextgraph/graph-intents/propose")
async def propose_graph_intent(request: Request):
    ctx = _ctx(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="GraphIntent proposal body must be an object")
    _reject_payload_authority(body)
    required = {
        "lineageId",
        "sourceIdentityRef",
        "canonicalSourceIrRef",
        "profileId",
        "operatingMode",
    }
    if set(body) != required:
        raise HTTPException(status_code=400, detail="GraphIntent proposal requires exact governed source pins")
    try:
        capability = _schema_capability(ctx)
        return await asyncio.to_thread(
            capability.propose_graph_intent,
            lineage_id=body.get("lineageId"),
            source_identity_ref=body.get("sourceIdentityRef"),
            canonical_source_ir_ref=body.get("canonicalSourceIrRef"),
            profile_id=body.get("profileId"),
            operating_mode=body.get("operatingMode"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/contextgraph/graph-intents/{graph_intent_ref}")
async def get_graph_intent(
    request: Request, graph_intent_ref: str, graphIntentHash: str | None = None
):
    try:
        return _schema_capability(_ctx(request)).get_graph_intent(
            graph_intent_ref, graph_intent_hash=graphIntentHash
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/contextgraph/graph-intents/{graph_intent_ref}/review")
async def review_graph_intent(
    request: Request, graph_intent_ref: str
):
    ctx = _ctx(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="GraphIntent review body must be an object")
    _reject_payload_authority(body)
    base = {
        "graphIntentHash",
        "channel",
        "state",
        "operatingMode",
        "selectedQuestionIds",
    }
    allowed = base | {"relationEdit"}
    if not base.issubset(body) or set(body) - allowed:
        raise HTTPException(status_code=400, detail="GraphIntent review requires exact refs and typed operation fields")
    state = body.get("state")
    relation_edit = body.get("relationEdit")
    if (state == "edited") != isinstance(relation_edit, dict):
        raise HTTPException(status_code=400, detail="edited GraphIntent review requires typed relationEdit")
    try:
        return _schema_capability(ctx).review_graph_intent(
            graph_intent_ref,
            graph_intent_hash=body.get("graphIntentHash"),
            channel=body.get("channel"),
            state=state,
            operating_mode=body.get("operatingMode"),
            relation_edit=relation_edit,
            selected_question_ids=body.get("selectedQuestionIds"),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/contextgraph/schema-profiles")
async def list_schema_profiles(request: Request):
    return _schema_capability(_ctx(request)).profiles()


@router.get("/api/contextgraph/schema-candidates/{decision_id}")
async def get_schema_candidate(request: Request, decision_id: str):
    try:
        return _schema_capability(_ctx(request)).get(decision_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/contextgraph/schema-candidates/induce")
async def induce_schema_candidate(request: Request):
    ctx = _ctx(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="induction body must be an object")
    _reject_payload_authority(body)
    try:
        graph_intent_fields = {"graphIntentRef", "graphIntentHash"}
        legacy_fields = {"intendedUse", "competencyQuestions", "relativePath"}
        if graph_intent_fields.intersection(body):
            if legacy_fields.intersection(body):
                raise HTTPException(
                    status_code=400,
                    detail="mixed GraphIntent and legacy induction request shapes are forbidden",
                )
            required = {
                "graphIntentRef",
                "graphIntentHash",
                "canonicalSourceIrRef",
                "profileId",
            }
            if set(body) != required:
                raise HTTPException(
                    status_code=400,
                    detail="GraphIntent induction requires the exact governed request shape",
                )
            return _schema_capability(ctx).induce(
                graph_intent_ref=body.get("graphIntentRef"),
                graph_intent_hash=body.get("graphIntentHash"),
                canonical_source_ir_ref=body.get("canonicalSourceIrRef"),
                profile_id=body.get("profileId"),
            )
        kwargs = {
            "relative_path": body.get("relativePath"),
            "profile_id": body.get("profileId"),
            "intended_use": body.get("intendedUse"),
            "competency_questions": body.get("competencyQuestions", []),
        }
        if body.get("lineageId") is not None:
            kwargs["lineage_id"] = body.get("lineageId")
        if body.get("canonicalSourceIrRef") is not None:
            kwargs["canonical_source_ir_ref"] = body.get("canonicalSourceIrRef")
        if body.get("sourceRef") is not None:
            kwargs["source_ref"] = body.get("sourceRef")
        if body.get("llmProposalEvidenceId") is not None:
            kwargs["llm_proposal_evidence_id"] = body.get("llmProposalEvidenceId")
        return _schema_capability(ctx).induce(**kwargs)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=getattr(exc, "guard_result", str(exc)),
        ) from exc


@router.post("/api/contextgraph/schema-candidates/{decision_id}/review")
async def review_schema_candidate(request: Request, decision_id: str):
    ctx = _ctx(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="review body must be an object")
    _reject_payload_authority(body)
    state = str(body.get("state") or "").strip()
    if state not in {"accepted", "edited", "rejected"}:
        raise HTTPException(status_code=400, detail="invalid review state")
    edited_candidate = body.get("editedCandidate")
    if state == "edited" and not isinstance(edited_candidate, dict):
        raise HTTPException(
            status_code=400,
            detail="editedCandidate must be a complete candidate object for edited review",
        )
    if state != "edited" and "editedCandidate" in body:
        raise HTTPException(
            status_code=400,
            detail="editedCandidate is only valid for edited review",
        )
    note = str(body.get("note") or "").strip()
    try:
        return _schema_capability(ctx).review(
            decision_id,
            state=state,
            note=note,
            edited_candidate=edited_candidate,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/contextgraph/schema-candidates/{decision_id}/sample")
async def execute_schema_candidate_sample(request: Request, decision_id: str):
    try:
        return _schema_capability(_ctx(request)).execute_sample(decision_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/contextgraph/schema-candidates/{decision_id}/decide")
async def decide_schema_candidate(request: Request, decision_id: str):
    ctx = _ctx(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="decision body must be an object")
    _reject_payload_authority(body)
    if set(body) != {"operatingMode"}:
        raise HTTPException(status_code=400, detail="decision body has unexpected fields")
    try:
        return _schema_capability(ctx).decide(
            decision_id,
            operating_mode=body.get("operatingMode"),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/contextgraph/schema-decisions/{schema_decision_id}/extract")
async def execute_governed_extract(request: Request, schema_decision_id: str):
    try:
        return _schema_capability(_ctx(request)).extract(schema_decision_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/contextgraph/schema-extract-runs/{extract_run_id}")
async def inspect_schema_extract_run(
    request: Request, extract_run_id: str, expected_result_hash: str | None = None
):
    try:
        result = _schema_capability(_ctx(request)).read_extract_result(extract_run_id)
        if expected_result_hash is not None and result["extract_result_hash"] != expected_result_hash:
            raise HTTPException(status_code=409, detail="EXTRACT_RESULT_HASH_MISMATCH")
        return result
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404 if str(exc) == "EXTRACT_RESULT_NOT_FOUND" else 409,
                            detail=str(exc)) from exc


@router.post("/api/contextgraph/schema-extract-runs/{extract_run_id}/grounding-gate")
async def evaluate_schema_grounding_gate(request: Request, extract_run_id: str):
    try:
        return _schema_capability(_ctx(request)).evaluate_grounding_gate(
            extract_run_id
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/contextgraph/schema-decisions/{schema_decision_id}/feedback")
async def record_schema_learning_feedback(
    request: Request, schema_decision_id: str
):
    ctx = _ctx(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="feedback body must be an object")
    _reject_payload_authority(body)
    if set(body) == {"correction"}:
        correction = body.get("correction")
        required = {
            "schemaVersion",
            "schemaDecisionHash",
            "graphIntentRef",
            "graphIntentHash",
            "channel",
            "changes",
            "selectedQuestionIds",
        }
        if not isinstance(correction, dict) or set(correction) != required:
            raise HTTPException(
                status_code=400,
                detail="GraphIntent correction must contain exact typed fields",
            )
        _reject_payload_authority(correction)
        if (
            correction.get("schemaVersion")
            != "studio_graph_intent_correction.v1"
            or correction.get("channel") not in {"chat", "direct"}
            or not isinstance(correction.get("changes"), dict)
            or not isinstance(correction.get("selectedQuestionIds"), list)
        ):
            raise HTTPException(
                status_code=400, detail="GraphIntent correction is invalid"
            )
        try:
            return _schema_capability(ctx).record_graph_intent_correction(
                schema_decision_id,
                schema_decision_hash=correction.get("schemaDecisionHash"),
                graph_intent_ref=correction.get("graphIntentRef"),
                graph_intent_hash=correction.get("graphIntentHash"),
                channel=correction.get("channel"),
                graph_intent_changes=correction.get("changes"),
                selected_question_ids=correction.get("selectedQuestionIds"),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    allowed_fields = {"feedback", "candidateDelta", "graphIntentChanges"}
    if "feedback" not in body or not set(body).issubset(allowed_fields):
        raise HTTPException(status_code=400, detail="feedback body has unexpected fields")
    if not isinstance(body["feedback"], dict):
        raise HTTPException(status_code=400, detail="feedback must be an object")
    candidate_delta = body.get("candidateDelta")
    graph_intent_changes = body.get("graphIntentChanges")
    if candidate_delta is not None and not isinstance(candidate_delta, dict):
        raise HTTPException(status_code=400, detail="candidateDelta must be an object")
    if graph_intent_changes is not None and not isinstance(graph_intent_changes, dict):
        raise HTTPException(status_code=400, detail="graphIntentChanges must be an object")
    try:
        return _schema_capability(ctx).record_schema_feedback(
            schema_decision_id,
            feedback=body["feedback"],
            candidate_delta=candidate_delta,
            graph_intent_changes=graph_intent_changes,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/contextgraph/schema-decisions/{schema_decision_id}/feedback-retest")
async def retest_schema_learning_feedback(
    request: Request, schema_decision_id: str
):
    ctx = _ctx(request)
    body = await request.json()
    if not isinstance(body, dict) or set(body) != {"baselineSchemaDecisionId"}:
        raise HTTPException(status_code=400, detail="baselineSchemaDecisionId is required")
    _reject_payload_authority(body)
    try:
        return _schema_capability(ctx).retest_schema_feedback(
            schema_decision_id,
            baseline_schema_decision_id=body.get("baselineSchemaDecisionId"),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/contextgraph/schema-candidates/{decision_id}/feedback-retest")
async def feedback_retest_schema_candidate(request: Request, decision_id: str):
    ctx = _ctx(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="feedback retest body must be an object")
    _reject_payload_authority(body)
    allowed = {"baselineDecisionId", "requestedRelationType"}
    if set(body) - allowed:
        raise HTTPException(status_code=400, detail="feedback retest body has unexpected fields")
    try:
        return _schema_capability(ctx).feedback_retest(
            decision_id,
            baseline_decision_id=body.get("baselineDecisionId"),
            requested_relation_type=body.get("requestedRelationType"),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/contextgraph/builder/build")
async def build_candidate(request: Request):
    ctx = _ctx(request)
    body = await request.json()
    if str(body.get("source") or "fixed-docx") != "fixed-docx":
        raise HTTPException(status_code=400, detail="only the fixed DOCX build is enabled in this MVL")
    repository = ContextGraphArtifactRepository(ctx.workspace_root, organization_id=str(ctx.organization_id),
                                                 workspace_id=str(ctx.workspace_id))
    try:
        metadata = build_v0_candidate(repository, created_by_user_id=str(ctx.user_id))
    except Exception as exc:
        raise HTTPException(status_code=503, detail="candidate graph build failed; retry is safe") from exc
    return {"status": "connected-graph", "artifactId": metadata.artifact_id, "graphVersion": metadata.graph_version,
            "graphHash": metadata.content_hash, "authorityState": metadata.authority_state,
            "counts": {"nodes": len(repository.load_graph(metadata.artifact_id).nodes),
                       "edges": len(repository.load_graph(metadata.artifact_id).edges)}}


@router.post("/api/contextgraph/telemetry")
async def record_telemetry(request: Request):
    ctx = _ctx(request)
    body = await request.json()
    required = {"event", "tenantId", "workspaceId", "graphRef", "graphVersion", "graphHash", "timestamp", "locale"}
    if not isinstance(body, dict) or not required.issubset(body):
        raise HTTPException(status_code=400, detail="MVL telemetry envelope is incomplete")
    if str(body["workspaceId"]) != str(ctx.workspace_id) or (ctx.organization_id and str(body["tenantId"]) != str(ctx.organization_id)):
        raise HTTPException(status_code=403, detail="telemetry tenant binding mismatch")
    target = Path(ctx.workspace_root) / "runs" / "phase1-complete-mvl" / "telemetry.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(body, ensure_ascii=False, sort_keys=True) + "\n")
    return {"status": "recorded"}
