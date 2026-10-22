from __future__ import annotations

import hashlib
import re


def creation_identity() -> dict[str, str]:
    """Scope trusted request correlation to the authenticated actor and workspace."""
    from gateway.execution_boundary import current_execution_boundary
    boundary = current_execution_boundary()
    if boundary is None or boundary.audit_metadata.get("authority_source") != "semantier_authenticated_context":
        raise RuntimeError("Durable Chat creation requires authenticated execution context")
    request_id = boundary.audit_metadata.get("durable_request_id")
    digest = boundary.audit_metadata.get("durable_request_hash")
    if not boundary.user_id or not boundary.workspace_id:
        raise RuntimeError("Durable Chat creation requires authenticated actor and workspace")
    if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id):
        raise RuntimeError("Durable Chat creation requires persisted request identity")
    if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise RuntimeError("Durable Chat creation requires original intent hash")
    scope = "\0".join((boundary.workspace_id, boundary.user_id, request_id))
    return {"idempotency_key": hashlib.sha256(scope.encode("utf-8")).hexdigest(),
            "idempotency_payload_hash": digest}
