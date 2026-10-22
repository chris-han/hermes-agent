"""Server-owned access projection for generic runtime report reads/exports."""

from __future__ import annotations

from typing import Literal

from agents.auth_session import RequestContext


def resolve_runtime_report_access_projection(
    *,
    ctx: RequestContext,
    organization_id: str,
    workspace_id: str,
    requested: Literal["MEMBER", "AUDIT"],
) -> Literal["MEMBER", "AUDIT"]:
    if not isinstance(ctx, RequestContext) or not ctx.authenticated:
        raise PermissionError("AUDIT_FORBIDDEN")
    if (ctx.organization_id, ctx.workspace_id) != (organization_id, workspace_id):
        raise PermissionError("FORBIDDEN_SCOPE")
    if requested == "MEMBER":
        return "MEMBER"
    if requested == "AUDIT":
        return "AUDIT"
    raise PermissionError("AUDIT_FORBIDDEN")


__all__ = ["resolve_runtime_report_access_projection"]
