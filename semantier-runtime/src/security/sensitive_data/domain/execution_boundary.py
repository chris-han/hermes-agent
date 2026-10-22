from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from semantier.sensitive_governance_contracts.hashing import canonical_hash


def bind_governed_execution_boundary(
    *,
    prepared_payload: dict[str, Any],
) -> dict[str, Any]:
    transformed_input = str(prepared_payload.get("server_owned_transformed_input") or "")
    execution_intent = prepared_payload.get("execution_intent")
    input_governance = prepared_payload.get("input_governance")
    if not isinstance(execution_intent, dict) or not isinstance(input_governance, dict):
        raise ValueError("prepared assessment execution boundary is incomplete")
    transformed_input_hash = str(execution_intent.get("transformed_input_hash") or "")
    computed_hash = canonical_hash({"transformed_input": transformed_input})
    if computed_hash != transformed_input_hash:
        raise ValueError("prepared assessment transformed_input_hash mismatch")
    if input_governance.get("status") != "PREPARED":
        raise ValueError("prepared assessment is not admitted for execution boundary")
    expires_at = str(execution_intent.get("expires_at") or "").strip()
    if not expires_at:
        raise ValueError("assessment authorization expired")
    try:
        expires_at_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("assessment authorization expired") from exc
    if expires_at_dt.tzinfo is None or expires_at_dt.utcoffset() is None:
        raise ValueError("assessment authorization expired")
    if expires_at_dt <= datetime.now(timezone.utc):
        raise ValueError("assessment authorization expired")
    boundary_body = {
        "boundary_status": "bound",
        "agent_boundary": execution_intent.get("agent_boundary"),
        "transformed_input_hash": transformed_input_hash,
        "policy_decision_ref": input_governance.get("policy_decision_ref"),
        "validation_result_ref": input_governance.get("validation_result_ref"),
        "admissibility_ref": input_governance.get("admissibility_ref"),
        "destination": input_governance.get("destination"),
        "restricted_request_ref": "restricted://sensitive-governance/transformed-input/"
        + transformed_input_hash[:24],
        "expires_at": expires_at,
    }
    if prepared_payload.get('display_preferences'):
        boundary_body['display_preferences_hash'] = canonical_hash(prepared_payload['display_preferences'])
    return {
        "execution_boundary_ref": "sg_execution_boundary_"
        + canonical_hash(boundary_body)[:24],
        **boundary_body,
    }
