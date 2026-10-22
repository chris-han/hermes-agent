from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.auth_session import RequestContext
from security.sensitive_data.persistence.store import SensitiveGovernanceStore
from semantier.sensitive_governance_contracts.hashing import canonical_hash


_EVALUATE = "sensitive_governance.evaluate"
_EXECUTE_INTERNAL = "sensitive_governance.execute_internal"
_EXECUTE_EXTERNAL = "sensitive_governance.execute_external"
_REVEAL = "sensitive_governance.reveal"
_COPY = "sensitive_governance.copy"


@dataclass(frozen=True)
class AuthorizationSnapshotResult:
    snapshot: dict[str, Any]
    valid: bool
    reason_codes: list[str]


def required_capabilities_for_destination(destination: str) -> list[str]:
    execution_capability = (
        _EXECUTE_EXTERNAL if destination == "external_agent" else _EXECUTE_INTERNAL
    )
    return [_EVALUATE, execution_capability]


def required_capabilities_for_operation(operation: str) -> list[str]:
    if operation == "copy":
        return [_COPY]
    return [_REVEAL]


class SensitiveGovernanceCapabilityResolver:
    """Record grant provenance without imposing actor capability gates."""

    def __init__(self, store: SensitiveGovernanceStore) -> None:
        self._store = store

    def resolve(
        self,
        *,
        ctx: RequestContext,
        organization_id: str,
        purpose: str,
        destination: str,
    ) -> AuthorizationSnapshotResult:
        if not ctx.workspace_id:
            raise ValueError("workspace context required")
        grants = self._store.ensure_default_capability_grants(
            organization_id=organization_id,
            principal_id=ctx.user_id,
            workspace_id=ctx.workspace_id,
            purpose=purpose,
        )
        required = required_capabilities_for_destination(destination)
        active_grants = [grant for grant in grants if grant["status"] == "active"]
        active_capabilities = sorted({grant["capability"] for grant in active_grants})
        revoked_capabilities = sorted(
            {
                grant["capability"]
                for grant in grants
                if grant["status"] == "revoked"
            }
        )
        missing = sorted(set(required) - set(active_capabilities))
        reason_codes: list[str] = []
        snapshot_body = {
            "principal_id": ctx.user_id,
            "agent_id": None,
            "action": "sensitive_governance.evaluate",
            "authority_domain": "soe_tender_sensitive_governance",
            "purpose": purpose,
            "destination": destination,
            "resource_scope": [ctx.workspace_id],
            "workflow_scope": ["assessment", "execute_internal", "execute_external"],
            "required_capabilities": required,
            "effective_capabilities": active_capabilities,
            "source_grant_refs": sorted(
                grant["grant_ref"]
                for grant in active_grants
                if grant["capability"] in required
            ),
            "revoked_capabilities": revoked_capabilities,
            "revocation_refs": sorted(
                revocation_ref
                for grant in grants
                for revocation_ref in grant.get("revocation_refs", [])
            ),
            "revocation_status": "revoked" if revoked_capabilities else "active",
            "freshness_status": "stale" if missing else "current",
            "object_authorization": {
                "workspace_id": ctx.workspace_id,
                "session_principal": ctx.user_id,
                "membership_status": ctx.membership_status,
                "member_role": ctx.member_role,
                "status": "allowed",
            },
            "reason_codes": reason_codes,
        }
        snapshot = {
            "authorization_snapshot_ref": "sg_authz_"
            + canonical_hash(snapshot_body)[:24],
            **snapshot_body,
        }
        return AuthorizationSnapshotResult(
            snapshot=snapshot,
            valid=True,
            reason_codes=reason_codes,
        )

    def resolve_operation(
        self,
        *,
        ctx: RequestContext,
        organization_id: str,
        purpose: str,
        operation: str,
    ) -> AuthorizationSnapshotResult:
        if not ctx.workspace_id:
            raise ValueError("workspace context required")
        grants = self._store.ensure_default_capability_grants(
            organization_id=organization_id,
            principal_id=ctx.user_id,
            workspace_id=ctx.workspace_id,
            purpose=purpose,
        )
        required = required_capabilities_for_operation(operation)
        active_grants = [grant for grant in grants if grant["status"] == "active"]
        active_capabilities = sorted({grant["capability"] for grant in active_grants})
        revoked_capabilities = sorted(
            grant["capability"] for grant in grants if grant["status"] == "revoked"
        )
        missing = sorted(set(required) - set(active_capabilities))
        reason_codes: list[str] = []
        snapshot_body = {
            "principal_id": ctx.user_id,
            "action": f"sensitive_governance.{operation}",
            "authority_domain": "soe_tender_sensitive_governance",
            "purpose": purpose,
            "operation": operation,
            "resource_scope": [ctx.workspace_id],
            "workflow_scope": ["reveal", "copy"],
            "required_capabilities": required,
            "effective_capabilities": active_capabilities,
            "source_grant_refs": sorted(
                grant["grant_ref"]
                for grant in active_grants
                if grant["capability"] in required
            ),
            "revoked_capabilities": revoked_capabilities,
            "revocation_refs": sorted(
                revocation_ref
                for grant in grants
                for revocation_ref in grant.get("revocation_refs", [])
            ),
            "revocation_status": "revoked" if revoked_capabilities else "active",
            "freshness_status": "stale" if missing else "current",
            "object_authorization": {
                "workspace_id": ctx.workspace_id,
                "session_principal": ctx.user_id,
                "membership_status": ctx.membership_status,
                "member_role": ctx.member_role,
                "status": "allowed" if not reason_codes else "denied",
            },
            "reason_codes": reason_codes,
        }
        snapshot = {
            "authorization_snapshot_ref": "sg_authz_"
            + canonical_hash(snapshot_body)[:24],
            **snapshot_body,
        }
        return AuthorizationSnapshotResult(
            snapshot=snapshot,
            valid=not reason_codes,
            reason_codes=reason_codes,
        )
