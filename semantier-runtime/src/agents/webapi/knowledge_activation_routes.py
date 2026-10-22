from __future__ import annotations

import sqlite3

from fastapi import APIRouter, HTTPException, Request

from agents.governed_context import resolve_governed_activation_context
from agents.route_policy import classify_authorization_class
from agents.webapi.deps import require_route_authorization as _require_route_authorization
from storage.knowledge_source_activation_store import KnowledgeSourceActivationStore

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


def _bool_value(body: dict, key: str, *, default: bool) -> bool:
    value = body.get(key, default)
    if isinstance(value, bool):
        return value
    raise HTTPException(status_code=400, detail=f"{key} must be boolean")


def _default_dimension_values(control_level: str) -> dict[str, bool]:
    if control_level == "none":
        return {
            "retrievalEnabled": True,
            "promptContextEnabled": True,
            "queryContextEnabled": True,
        }
    if control_level == "prompt_context":
        return {
            "retrievalEnabled": True,
            "promptContextEnabled": True,
            "queryContextEnabled": False,
        }
    if control_level == "query_context":
        return {
            "retrievalEnabled": False,
            "promptContextEnabled": False,
            "queryContextEnabled": True,
        }
    if control_level == "full_optional_context":
        return {
            "retrievalEnabled": True,
            "promptContextEnabled": True,
            "queryContextEnabled": True,
        }
    raise HTTPException(status_code=409, detail="Invalid activation user_context_control_level")


def _preference_decision(body: dict) -> str:
    explicit = str(body.get("decision") or "").strip()
    if explicit:
        return explicit
    if "queryContextEnabled" in body or "query_context_enabled" in body:
        enabled = bool(body.get("queryContextEnabled", body.get("query_context_enabled")))
        return "ENABLE_QUERY_CONTEXT" if enabled else "DISABLE_QUERY_CONTEXT"
    if "promptContextEnabled" in body or "prompt_context_enabled" in body:
        enabled = bool(body.get("promptContextEnabled", body.get("prompt_context_enabled")))
        return "ENABLE_PROMPT_CONTEXT" if enabled else "DISABLE_PROMPT_CONTEXT"
    if "retrievalEnabled" in body or "retrieval_enabled" in body:
        enabled = bool(body.get("retrievalEnabled", body.get("retrieval_enabled")))
        return "ENABLE_RETRIEVAL" if enabled else "DISABLE_RETRIEVAL"
    return "RESET_TO_DEFAULT"


def _preference_id(
    *,
    activation_id: str,
    preference_scope: str,
    subject_id: str | None,
) -> str:
    subject = str(subject_id or "self").strip() or "self"
    return f"pref_{preference_scope}_{subject}_{activation_id}"


def _preference_response(row: dict) -> dict:
    return {
        "preferenceId": row["preference_id"],
        "activationId": row["activation_id"],
        "preferenceScope": row["preference_scope"],
        "principalId": row.get("principal_id"),
        "roleId": row.get("role_id"),
        "organizationId": row["organization_id"],
        "workspaceId": row.get("workspace_id"),
        "retrievalEnabled": bool(row["retrieval_enabled"]),
        "promptContextEnabled": bool(row["prompt_context_enabled"]),
        "queryContextEnabled": bool(row["query_context_enabled"]),
        "preferenceEventHeadHash": row["preference_event_head_hash"],
        "updatedAt": row["updated_at"],
    }


def _text(body: dict, *keys: str, default: str = "") -> str:
    for key in keys:
        value = body.get(key)
        if value is not None:
            return str(value).strip()
    return default


async def _write_preference(
    request: Request,
    *,
    route_path: str,
    preference_scope: str,
) -> dict:
    ctx = _require_route_authorization(
        request,
        authorization_class=_require_configured_authz("POST", route_path),
        route_hint=f"POST {route_path}",
    )
    if not ctx.organization_id:
        raise HTTPException(status_code=409, detail="No organization selected")
    body = await _json_body(request)
    activation_id = str(body.get("activationId") or body.get("activation_id") or "").strip()
    if not activation_id:
        raise HTTPException(status_code=400, detail="activationId required")

    requested_scope = str(body.get("preferenceScope") or body.get("preference_scope") or preference_scope).strip()
    if requested_scope != preference_scope:
        raise HTTPException(status_code=400, detail="preferenceScope route mismatch")

    subject_id = str(body.get("subjectId") or body.get("subject_id") or "").strip() or None
    principal_id = ctx.user_id if preference_scope == "principal" else None
    role_id = None
    if preference_scope == "role":
        role_id = subject_id or str(ctx.member_role or "").strip() or None

    store = KnowledgeSourceActivationStore()
    activation = store.get_activation(activation_id)
    if activation is None:
        raise HTTPException(status_code=404, detail="activation not found")
    defaults = _default_dimension_values(str(activation["user_context_control_level"]))

    try:
        preference = store.write_preference(
            preference_id=_preference_id(
                activation_id=activation_id,
                preference_scope=preference_scope,
                subject_id=principal_id or role_id or subject_id,
            ),
            activation_id=activation_id,
            organization_id=str(ctx.organization_id),
            workspace_id=ctx.workspace_id,
            preference_scope=preference_scope,
            principal_id=principal_id,
            role_id=role_id,
            retrieval_enabled=_bool_value(
                body,
                "retrievalEnabled" if "retrievalEnabled" in body else "retrieval_enabled",
                default=defaults["retrievalEnabled"],
            ),
            prompt_context_enabled=_bool_value(
                body,
                "promptContextEnabled" if "promptContextEnabled" in body else "prompt_context_enabled",
                default=defaults["promptContextEnabled"],
            ),
            query_context_enabled=_bool_value(
                body,
                "queryContextEnabled" if "queryContextEnabled" in body else "query_context_enabled",
                default=defaults["queryContextEnabled"],
            ),
            decision=_preference_decision(body),
            actor=str(ctx.user_id),
            reason=str(body.get("reason") or "Knowledge Base optional context preference"),
        )
    except (ValueError, sqlite3.IntegrityError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "preference": _preference_response(preference)}


@router.post("/api/knowledge/activations")
async def knowledge_activation_create(request: Request):
    ctx = _require_route_authorization(
        request,
        authorization_class=_require_configured_authz("POST", "/api/knowledge/activations"),
        route_hint="POST /api/knowledge/activations",
    )
    if not ctx.organization_id:
        raise HTTPException(status_code=409, detail="No organization selected")
    body = await _json_body(request)
    activation_scope = _text(body, "activationScope", "activation_scope", default="organization")
    workspace_id = (
        _text(body, "workspaceId", "workspace_id", default="")
        or (ctx.workspace_id if activation_scope == "workspace" else None)
    )
    try:
        activation = KnowledgeSourceActivationStore().activate_source(
            activation_id=_text(body, "activationId", "activation_id"),
            knowledge_source_id=_text(body, "knowledgeSourceId", "knowledge_source_id"),
            dataset_asset_version_id=_text(
                body,
                "datasetAssetVersionId",
                "dataset_asset_version_id",
                default="",
            )
            or None,
            organization_id=str(ctx.organization_id),
            workspace_id=workspace_id,
            activation_scope=activation_scope,
            activation_status=_text(
                body,
                "activationStatus",
                "activation_status",
                default="active",
            ),
            workspace_override_policy=_text(
                body,
                "workspaceOverridePolicy",
                "workspace_override_policy",
                default="none",
            ),
            user_context_control_level=_text(
                body,
                "userContextControlLevel",
                "user_context_control_level",
                default="none",
            ),
            decision=_text(body, "decision", default="ACTIVATE_AUTHORITY"),
            actor=str(ctx.user_id),
            actor_authority=str(ctx.member_role or "tenant_admin"),
            reason=_text(body, "reason", default="Knowledge Base activation"),
            artifact_version_hash=_text(
                body,
                "artifactVersionHash",
                "artifact_version_hash",
                default="activation_route_unversioned_request",
            ),
            authority_source_manifest=body.get("authoritySourceManifest")
            if isinstance(body.get("authoritySourceManifest"), dict)
            else body.get("authority_source_manifest")
            if isinstance(body.get("authority_source_manifest"), dict)
            else {},
        )
    except (ValueError, sqlite3.IntegrityError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": True,
        "activation": activation,
        "route_family": "governed_activation_administration",
    }


@router.get("/api/knowledge/activations")
async def knowledge_activation_effective(request: Request):
    ctx = _require_route_authorization(
        request,
        authorization_class=_require_configured_authz("GET", "/api/knowledge/activations"),
        route_hint="GET /api/knowledge/activations",
    )
    context = resolve_governed_activation_context(
        user_id=ctx.user_id,
        workspace_id=ctx.workspace_id,
    )
    if context is None:
        return {
            "ok": True,
            "activation_set_snapshot_id": None,
            "activation_resolver_policy_version": None,
            "resolved_activation_set_hash": None,
            "sources": [],
            "excluded_activation_evidence": [],
        }
    return {
        "ok": True,
        **context,
    }


@router.post("/api/knowledge/activations/{activation_id}/lifecycle")
async def knowledge_activation_lifecycle(activation_id: str, request: Request):
    ctx = _require_route_authorization(
        request,
        authorization_class=_require_configured_authz(
            "POST",
            "/api/knowledge/activations/{activation_id}/lifecycle",
        ),
        route_hint="POST /api/knowledge/activations/{activation_id}/lifecycle",
    )
    body = await _json_body(request)
    try:
        activation = KnowledgeSourceActivationStore().transition_activation(
            activation_id=activation_id,
            activation_status=_text(
                body,
                "activationStatus",
                "activation_status",
                default="active",
            ),
            decision=_text(body, "decision", default="ACTIVATE_AUTHORITY"),
            actor=str(ctx.user_id),
            actor_authority=str(ctx.member_role or "tenant_admin"),
            reason=_text(body, "reason", default="Knowledge Base activation lifecycle transition"),
            artifact_version_hash=_text(
                body,
                "artifactVersionHash",
                "artifact_version_hash",
                default="activation_lifecycle_route_unversioned_request",
            ),
        )
    except (ValueError, sqlite3.IntegrityError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": True,
        "activation": activation,
        "route_family": "governed_activation_administration",
    }


@router.post("/api/knowledge/preferences/principal")
async def knowledge_preference_principal(request: Request):
    return await _write_preference(
        request,
        route_path="/api/knowledge/preferences/principal",
        preference_scope="principal",
    )


@router.post("/api/knowledge/preferences/role")
async def knowledge_preference_role(request: Request):
    return await _write_preference(
        request,
        route_path="/api/knowledge/preferences/role",
        preference_scope="role",
    )


@router.post("/api/knowledge/preferences/workspace-default")
async def knowledge_preference_workspace_default(request: Request):
    return await _write_preference(
        request,
        route_path="/api/knowledge/preferences/workspace-default",
        preference_scope="workspace_default",
    )
