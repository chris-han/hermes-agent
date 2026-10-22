from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from agents.route_policy import classify_authorization_class
from agents.webapi.deps import require_route_authorization as _require_route_authorization
from storage.governed_rule_store import GovernedRuleStore
from storage.policy_rule_candidate_store import PolicyRuleCandidateStore

router = APIRouter(tags=["webapi-gateway"])


def _require_configured_authz(method: str, path: str):
    authz_class = classify_authorization_class(method, path)
    if authz_class is None:
        raise HTTPException(status_code=500, detail="Route auth class not configured")
    return authz_class


async def _json_body(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="JSON body required") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object body required")
    return body


def _text(body: dict, *keys: str, default: str = "") -> str:
    for key in keys:
        value = body.get(key)
        if value is not None:
            return str(value).strip()
    return default


def _list(body: dict, *keys: str) -> list:
    for key in keys:
        value = body.get(key)
        if isinstance(value, list):
            return value
    return []


def _dict(body: dict, *keys: str) -> dict:
    for key in keys:
        value = body.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _candidate_payload(candidate: dict[str, Any], *, audit: bool = False) -> dict[str, Any]:
    payload = {
        "ruleCandidateId": candidate["rule_candidate_id"],
        "organizationId": candidate.get("organization_id"),
        "proposedWorkspaceScope": candidate.get("proposed_workspace_scope"),
        "proposedWorkspaceId": candidate.get("proposed_workspace_id"),
        "authorityDomain": candidate.get("authority_domain"),
        "semanticTier": candidate.get("semantic_tier"),
        "jurisdiction": candidate.get("jurisdiction"),
        "candidateIdempotencyKey": candidate.get("candidate_idempotency_key"),
        "ruleFamily": candidate["rule_family"],
        "candidateState": candidate["candidate_state"],
        "sourceAnchorRefs": candidate["source_anchor_refs"],
        "applicabilityScope": candidate["applicability_scope"],
        "extractedRationale": candidate["extracted_rationale"],
        "draftRuleText": candidate["draft_rule_text"],
        "severity": candidate.get("severity"),
        "confidence": candidate.get("confidence"),
        "uncertaintyNotes": candidate.get("uncertainty_notes"),
        "humanEdits": candidate["human_edits"],
        "approvalEvidence": candidate["approval_evidence"],
        "testEvidence": candidate["test_evidence"],
        "supersedesCandidateId": candidate.get("supersedes_candidate_id"),
        "activationRefs": candidate["activation_refs"],
        "createdBy": candidate["created_by"],
        "createdByActorType": candidate["created_by_actor_type"],
        "createdAt": candidate["created_at"],
        "updatedAt": candidate["updated_at"],
        "isRuntimeAuthority": candidate["candidate_state"] == "ACTIVATED",
        "nonAuthorityReason": (
            None
            if candidate["candidate_state"] == "ACTIVATED"
            else "Candidate is not runtime authority until governed activation and resolver inclusion"
        ),
    }
    if audit:
        payload["contentHash"] = candidate["content_hash"]
    return payload


def _store_error(exc: Exception) -> HTTPException:
    if isinstance(exc, sqlite3.IntegrityError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


@router.post("/api/knowledge/policy-rules")
async def create_policy_rule_candidate(request: Request):
    ctx = _require_route_authorization(
        request,
        authorization_class=_require_configured_authz("POST", "/api/knowledge/policy-rules"),
        route_hint="POST /api/knowledge/policy-rules",
    )
    body = await _json_body(request)
    try:
        candidate = PolicyRuleCandidateStore().create_candidate(
            rule_candidate_id=_text(body, "ruleCandidateId", "rule_candidate_id", default="") or None,
            organization_id=_text(body, "organizationId", "organization_id", default=str(ctx.organization_id or "")) or None,
            proposed_workspace_scope=_text(
                body,
                "proposedWorkspaceScope",
                "proposed_workspace_scope",
                default="workspace" if ctx.workspace_id else "organization",
            ),
            proposed_workspace_id=_text(
                body,
                "proposedWorkspaceId",
                "proposed_workspace_id",
                default=str(ctx.workspace_id or ""),
            )
            or None,
            authority_domain=_text(body, "authorityDomain", "authority_domain", default="tender_compliance") or None,
            semantic_tier=_text(body, "semanticTier", "semantic_tier", default="T4") or None,
            jurisdiction=_text(body, "jurisdiction", default="CN") or None,
            candidate_idempotency_key=_text(
                body,
                "candidateIdempotencyKey",
                "candidate_idempotency_key",
                default="",
            )
            or None,
            rule_family=_text(body, "ruleFamily", "rule_family"),
            candidate_state=_text(body, "candidateState", "candidate_state", default="DRAFT"),
            source_anchor_refs=_list(body, "sourceAnchorRefs", "source_anchor_refs"),
            applicability_scope=_dict(body, "applicabilityScope", "applicability_scope"),
            extracted_rationale=_text(body, "extractedRationale", "extracted_rationale"),
            draft_rule_text=_text(body, "draftRuleText", "draft_rule_text"),
            severity=_text(body, "severity", default="") or None,
            confidence=body.get("confidence") if isinstance(body.get("confidence"), (int, float)) else None,
            uncertainty_notes=_text(body, "uncertaintyNotes", "uncertainty_notes", default="") or None,
            human_edits=_list(body, "humanEdits", "human_edits"),
            approval_evidence=_dict(body, "approvalEvidence", "approval_evidence"),
            test_evidence=_dict(body, "testEvidence", "test_evidence"),
            supersedes_candidate_id=_text(body, "supersedesCandidateId", "supersedes_candidate_id", default="") or None,
            activation_refs=_list(body, "activationRefs", "activation_refs"),
            created_by=str(ctx.user_id),
            created_by_actor_type=_text(body, "createdByActorType", "created_by_actor_type", default="human"),
        )
    except Exception as exc:
        raise _store_error(exc) from exc
    return {"ok": True, "candidate": _candidate_payload(candidate)}


@router.get("/api/knowledge/policy-rules")
async def list_policy_rule_candidates(request: Request, ruleFamily: str | None = None, candidateState: str | None = None):
    _require_route_authorization(
        request,
        authorization_class=_require_configured_authz("GET", "/api/knowledge/policy-rules"),
        route_hint="GET /api/knowledge/policy-rules",
    )
    candidates = PolicyRuleCandidateStore().list_candidates(
        rule_family=ruleFamily,
        candidate_state=candidateState,  # type: ignore[arg-type]
    )
    return {"ok": True, "candidates": [_candidate_payload(candidate) for candidate in candidates]}


@router.get("/api/knowledge/policy-rules/{candidate_id}")
async def get_policy_rule_candidate(candidate_id: str, request: Request):
    _require_route_authorization(
        request,
        authorization_class=_require_configured_authz("GET", "/api/knowledge/policy-rules/{candidate_id}"),
        route_hint="GET /api/knowledge/policy-rules/{candidate_id}",
    )
    candidate = PolicyRuleCandidateStore().get_candidate(candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="rule candidate not found")
    return {"ok": True, "candidate": _candidate_payload(candidate)}


@router.get("/api/knowledge/policy-rules/{candidate_id}/audit")
async def get_policy_rule_candidate_audit(candidate_id: str, request: Request):
    _require_route_authorization(
        request,
        authorization_class=_require_configured_authz("GET", "/api/knowledge/policy-rules/{candidate_id}/audit"),
        route_hint="GET /api/knowledge/policy-rules/{candidate_id}/audit",
    )
    store = PolicyRuleCandidateStore()
    candidate = store.get_candidate(candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="rule candidate not found")
    return {
        "ok": True,
        "candidate": _candidate_payload(candidate, audit=True),
        "events": store.list_events(candidate_id),
    }


@router.post("/api/knowledge/policy-rules/{candidate_id}/reviewer-edit")
async def edit_policy_rule_candidate(candidate_id: str, request: Request):
    ctx = _require_route_authorization(
        request,
        authorization_class=_require_configured_authz("POST", "/api/knowledge/policy-rules/{candidate_id}/reviewer-edit"),
        route_hint="POST /api/knowledge/policy-rules/{candidate_id}/reviewer-edit",
    )
    body = await _json_body(request)
    try:
        candidate = PolicyRuleCandidateStore().record_reviewer_edit(
            candidate_id,
            actor=str(ctx.user_id),
            actor_authority=_text(body, "actorAuthority", "actor_authority", default=str(ctx.member_role or "")),
            human_edits=_list(body, "humanEdits", "human_edits"),
            review_comments=_text(body, "reviewComments", "review_comments", default="") or None,
        )
    except Exception as exc:
        raise _store_error(exc) from exc
    return {"ok": True, "candidate": _candidate_payload(candidate)}


@router.post("/api/knowledge/policy-rules/{candidate_id}/transition-request")
async def request_policy_rule_review(candidate_id: str, request: Request):
    ctx = _require_route_authorization(
        request,
        authorization_class=_require_configured_authz("POST", "/api/knowledge/policy-rules/{candidate_id}/transition-request"),
        route_hint="POST /api/knowledge/policy-rules/{candidate_id}/transition-request",
    )
    try:
        candidate = PolicyRuleCandidateStore().request_review(candidate_id, actor=str(ctx.user_id))
    except Exception as exc:
        raise _store_error(exc) from exc
    return {"ok": True, "candidate": _candidate_payload(candidate)}


@router.post("/api/knowledge/policy-rules/{candidate_id}/approval")
async def approve_policy_rule_candidate(candidate_id: str, request: Request):
    ctx = _require_route_authorization(
        request,
        authorization_class=_require_configured_authz("POST", "/api/knowledge/policy-rules/{candidate_id}/approval"),
        route_hint="POST /api/knowledge/policy-rules/{candidate_id}/approval",
    )
    body = await _json_body(request)
    try:
        candidate = PolicyRuleCandidateStore().approve(
            candidate_id,
            actor=str(ctx.user_id),
            actor_authority=_text(body, "actorAuthority", "actor_authority", default=str(ctx.member_role or "")),
            approval_evidence=_dict(body, "approvalEvidence", "approval_evidence"),
            test_evidence=_dict(body, "testEvidence", "test_evidence"),
        )
        version = GovernedRuleStore().create_approved_version(
            rule_version_id=_text(body, "ruleVersionId", "rule_version_id", default="") or None,
            rule_candidate_id=candidate_id,
            actor=str(ctx.user_id),
            actor_authority=_text(body, "actorAuthority", "actor_authority", default=str(ctx.member_role or "")),
        )
    except Exception as exc:
        raise _store_error(exc) from exc
    return {"ok": True, "candidate": _candidate_payload(candidate), "ruleVersion": version}


@router.post("/api/knowledge/policy-rules/{candidate_id}/rejection")
async def reject_policy_rule_candidate(candidate_id: str, request: Request):
    ctx = _require_route_authorization(
        request,
        authorization_class=_require_configured_authz("POST", "/api/knowledge/policy-rules/{candidate_id}/rejection"),
        route_hint="POST /api/knowledge/policy-rules/{candidate_id}/rejection",
    )
    body = await _json_body(request)
    try:
        candidate = PolicyRuleCandidateStore().reject(
            candidate_id,
            actor=str(ctx.user_id),
            actor_authority=_text(body, "actorAuthority", "actor_authority", default=str(ctx.member_role or "")),
            reason=_text(body, "reason", default="rejected by reviewer"),
        )
    except Exception as exc:
        raise _store_error(exc) from exc
    return {"ok": True, "candidate": _candidate_payload(candidate)}


@router.post("/api/knowledge/policy-rules/{candidate_id}/supersession")
async def supersede_policy_rule_candidate(candidate_id: str, request: Request):
    ctx = _require_route_authorization(
        request,
        authorization_class=_require_configured_authz("POST", "/api/knowledge/policy-rules/{candidate_id}/supersession"),
        route_hint="POST /api/knowledge/policy-rules/{candidate_id}/supersession",
    )
    body = await _json_body(request)
    try:
        candidate = PolicyRuleCandidateStore().supersede(
            candidate_id,
            actor=str(ctx.user_id),
            actor_authority=_text(body, "actorAuthority", "actor_authority", default=str(ctx.member_role or "")),
            superseded_by=_text(body, "supersededBy", "superseded_by"),
        )
    except Exception as exc:
        raise _store_error(exc) from exc
    return {"ok": True, "candidate": _candidate_payload(candidate)}


@router.post("/api/knowledge/policy-rules/{candidate_id}/activation-request")
async def activate_policy_rule_candidate(candidate_id: str, request: Request):
    ctx = _require_route_authorization(
        request,
        authorization_class=_require_configured_authz("POST", "/api/knowledge/policy-rules/{candidate_id}/activation-request"),
        route_hint="POST /api/knowledge/policy-rules/{candidate_id}/activation-request",
    )
    body = await _json_body(request)
    try:
        version = GovernedRuleStore().activate_version(
            _text(body, "ruleVersionId", "rule_version_id"),
            activation_ref=_text(body, "activationRef", "activation_ref"),
            actor=str(ctx.user_id),
            actor_authority=_text(body, "actorAuthority", "actor_authority", default=str(ctx.member_role or "")),
            organization_id=_text(body, "organizationId", "organization_id", default=str(ctx.organization_id or "")) or None,
            activation_scope=_text(body, "activationScope", "activation_scope", default="workspace" if ctx.workspace_id else "organization"),  # type: ignore[arg-type]
            workspace_id=_text(body, "workspaceId", "workspace_id", default=str(ctx.workspace_id or "")) or None,
            effective_from=_text(body, "effectiveFrom", "effective_from", default="") or None,
            effective_to=_text(body, "effectiveTo", "effective_to", default="") or None,
            approval_event_ref=_text(body, "approvalEventRef", "approval_event_ref", default="") or None,
        )
        candidate = PolicyRuleCandidateStore().get_candidate(candidate_id)
    except Exception as exc:
        raise _store_error(exc) from exc
    return {"ok": True, "candidate": _candidate_payload(candidate) if candidate else None, "ruleVersion": version}
