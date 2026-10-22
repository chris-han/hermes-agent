"""Governed session binding shared by the current five-object runtime path."""

from __future__ import annotations

from typing import Literal

from agents.auth_session import RequestContext
from agents.gateway_identity import resolve_workspace_paths
from agents.workspace_session_logs import (
    _load_session_index,
    get_workspace_session_detail,
    resolve_workspace_session_id,
)
from contracts.knowledge_classification import RuntimeContextDTO
from services.runtime_errors import RuntimeContextError


class GovernedRuntimeSessionResolver:
    def __init__(self, *, authority_resolver: object) -> None:
        self._authority_resolver = authority_resolver

    def resolve(
        self, *, actor_context: RuntimeContextDTO, session_ref: str,
        authorization_class: Literal["TENANT_MEMBER", "WORKSPACE_SIMULATION", "WORKSPACE_LOCAL"] = "TENANT_MEMBER",
    ) -> str:
        if authorization_class == "TENANT_MEMBER":
            self._authority_resolver.require_current_actor(actor_context=actor_context)
        else:
            self._authority_resolver.require_current_actor(
                actor_context=actor_context, authorization_class=authorization_class,
            )
        workspace_id = actor_context.workspace_id
        if not workspace_id:
            raise RuntimeContextError("MISSING_WORKSPACE")
        candidate = str(session_ref or "").strip()
        if not candidate or candidate.startswith("public:"):
            raise RuntimeContextError("FORBIDDEN_SESSION")
        if ":session" in candidate and not candidate.startswith(f"{workspace_id}:"):
            raise RuntimeContextError("FORBIDDEN_SESSION")

        _workspace_root, hermes_home = resolve_workspace_paths(workspace_id)
        canonical = resolve_workspace_session_id(hermes_home, candidate)
        if canonical is None:
            raise RuntimeContextError("SESSION_NOT_FOUND")
        if canonical.startswith("public:") or not canonical.startswith(f"{workspace_id}:"):
            raise RuntimeContextError("FORBIDDEN_SESSION")
        detail = get_workspace_session_detail(hermes_home, canonical)
        if not isinstance(detail, dict) or detail.get("session_id") != canonical:
            raise RuntimeContextError("SESSION_NOT_FOUND")
        index = _load_session_index(hermes_home)
        indexed = tuple(
            entry
            for key, entry in index.items()
            if key == canonical or str(entry.get("session_id") or "") == canonical
        )
        if not indexed:
            raise RuntimeContextError("SESSION_NOT_FOUND")
        recorded_workspace = str(detail.get("workspace_id") or "").strip()
        recorded_owner = str(detail.get("workspace_owner_id") or "").strip()
        if recorded_workspace and recorded_workspace != workspace_id:
            raise RuntimeContextError("FORBIDDEN_SESSION")
        if recorded_owner and recorded_owner != workspace_id:
            raise RuntimeContextError("FORBIDDEN_SESSION")
        for entry in indexed:
            indexed_workspace = str(entry.get("workspace_id") or "").strip()
            indexed_owner = str(entry.get("workspace_owner_id") or "").strip()
            if indexed_workspace and indexed_workspace != workspace_id:
                raise RuntimeContextError("FORBIDDEN_SESSION")
            if indexed_owner and indexed_owner != workspace_id:
                raise RuntimeContextError("FORBIDDEN_SESSION")
        return canonical


def runtime_context_dto_from_request_context(
    *, ctx: RequestContext, session_id: str, channel: str | None
) -> RuntimeContextDTO:
    if not ctx.authenticated or not ctx.organization_id or not ctx.workspace_id:
        raise RuntimeContextError("MISSING_WORKSPACE")
    canonical = str(session_id or "").strip()
    if (
        not canonical
        or canonical.startswith("public:")
        or not canonical.startswith(f"{ctx.workspace_id}:")
    ):
        raise RuntimeContextError("FORBIDDEN_SESSION")
    return RuntimeContextDTO(
        organization_id=str(ctx.organization_id),
        workspace_id=str(ctx.workspace_id),
        actor_id=str(ctx.user_id),
        channel=channel,
        session_id=canonical,
    )


__all__ = ["GovernedRuntimeSessionResolver", "runtime_context_dto_from_request_context"]
