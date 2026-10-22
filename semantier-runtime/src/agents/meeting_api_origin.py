"""Core admission of prospective native API origins; metadata is never authority."""

from contextlib import contextmanager
from datetime import datetime, timezone


@contextmanager
def admitted_api_actor(adapter, boundary, session_id, session_key, profile):
    from contracts.workflow_execution import meeting_context_from_execution_boundary
    from agents.gateway_identity import get_user_organization_context, get_user_record
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionStore, SessionSource

    from gateway.execution_boundary import current_execution_boundary

    current = current_execution_boundary()
    if (current is None or current is not boundary
            or current.audit_metadata.get("authority_source") != "semantier_authenticated_context"
            or not current.user_id or not current.workspace_id):
        raise ValueError("WORKFLOW_AUTHENTICATED_CONTEXT_REQUIRED")
    user = get_user_record(current.user_id)
    if not user or user.get("workspace_id") != current.workspace_id:
        raise ValueError("WORKFLOW_SCOPE_REQUIRED")
    org = get_user_organization_context(current.user_id) or {}
    if not org.get("organization_id"):
        # This hook may attach a meeting origin; it does not authorize chat.
        # Ordinary authenticated turns continue without meeting authority.
        yield
        return
    ctx = meeting_context_from_execution_boundary()
    if not session_id or not session_key:
        # Unaddressable/stateless API turns are not eligible origins.
        yield
        return
    db = adapter._ensure_session_db()
    if db is None:
        raise RuntimeError("API_ORIGIN_DURABLE_STORE_REQUIRED")
    if db.get_session(session_id) is None:
        db.create_session(session_id, source="api_server", user_id=ctx.user_id)
    adapter._bind_declared_conversation(session_id, session_key, strict=True)
    source = SessionSource(
        platform=Platform.API_SERVER,
        chat_id=session_id,
        user_id=ctx.user_id,
        workspace_owner_id=ctx.workspace_id,
        adapter_key="semantier_embedded_api",
        delivery_adapter_key="semantier_embedded_api",
        profile=profile or None,
    )
    store = SessionStore(
        adapter._session_db_home() / "sessions", GatewayConfig(), session_db=db
    )
    store._db = db
    entry = store.admit_native_api_session(
        session_key=session_key, session_id=session_id, source=source
    )
    origin = {
        "schema_version": "meeting_api_actor_origin.v1",
        "organization_id": ctx.organization_id,
        "workspace_id": ctx.workspace_id,
        "actor_user_id": ctx.user_id,
        "session_id": entry.session_id,
        "session_key": entry.session_key,
        "source": entry.origin.to_dict(),
        "profile": profile or None,
        "adapter_owner": "semantier_embedded_api",
        "authorization_basis": "current_core_identity_scope",
        "authorization_purpose": "meeting_ambiguous_availability_clarification",
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    from gateway.execution_boundary import bind_execution_boundary
    from dataclasses import replace

    with bind_execution_boundary(
        replace(
            boundary,
            audit_metadata={**boundary.audit_metadata, "api_actor_origin": origin},
        )
    ):
        yield


def prospective_meeting_origin(ctx, workflow_binding):
    """Only the worker's Core-owned admission can supply the stored value."""
    from contracts.workflow_execution import hash_value
    from gateway.execution_boundary import current_execution_boundary

    boundary = current_execution_boundary()
    origin = (
        boundary.audit_metadata.get("api_actor_origin")
        if boundary is not None
        else None
    )
    if origin is None:
        return None
    if (origin["actor_user_id"], origin["organization_id"], origin["workspace_id"]) != (
        ctx.user_id,
        ctx.organization_id,
        ctx.workspace_id,
    ):
        raise PermissionError("API_ORIGIN_SCOPE_MISMATCH")
    return {
        **origin,
        "workflow_binding_hash": hash_value(workflow_binding),
        "definition_hash": workflow_binding["definitionArtifactHash"],
        "execution_hash": workflow_binding["executionHash"],
    }


def verify_prospective_origin(origin, ctx, workflow_binding):
    expected = prospective_meeting_origin(ctx, workflow_binding)
    if expected is None or origin != expected:
        raise PermissionError("API_ORIGIN_ATTESTATION_REQUIRED")
    return origin
