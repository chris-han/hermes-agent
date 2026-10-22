from __future__ import annotations

from storage import hosted_postgres as hosted

import json
from typing import Any, Protocol

from contracts.tender_sensitive_term_detection_modeling_core import GovernedRuleVersionDTO, RuntimeContextDTO
from eos.artifact_hashing import content_hash


class ActivationSnapshotResolverService(Protocol):
    def materialize_snapshot(
        self,
        *,
        requested_rule_families: list[str],
        context: RuntimeContextDTO,
        effective_at: str | None = None,
        principal_entitlement_ref: str | None = None,
    ) -> dict[str, Any]: ...

    def list_rules_for_activation_snapshot(
        self,
        *,
        activation_set_snapshot_id: str,
        resolved_activation_set_hash: str,
        context: RuntimeContextDTO,
        effective_at: str | None = None,
    ) -> tuple[list[GovernedRuleVersionDTO], str]: ...

    def verify_snapshot_hashes(self, snapshot_id: str, expected_hashes: dict[str, str]) -> bool: ...


class StoreBackedActivationSnapshotResolverService:
    def __init__(self, rule_store: Any | None = None) -> None:
        if rule_store is None:
            from storage.governed_rule_store import GovernedRuleStore

            rule_store = GovernedRuleStore()
        self._rule_store = rule_store

    def resolve_governed_context(self, *, user_id: str, workspace_id: str | None) -> dict[str, Any] | None:
        from agents.governed_context import resolve_governed_activation_context

        return resolve_governed_activation_context(user_id=user_id, workspace_id=workspace_id)

    def materialize_snapshot(
        self,
        *,
        requested_rule_families: list[str],
        context: RuntimeContextDTO,
        effective_at: str | None = None,
        principal_entitlement_ref: str | None = None,
    ) -> dict[str, Any]:
        for snapshot in self._rule_store.list_activation_snapshots(
            organization_id=context.organization_id,
            workspace_id=context.workspace_id,
        ):
            if principal_entitlement_ref is not None and snapshot.get("principal_entitlement_ref") != principal_entitlement_ref:
                continue
            if effective_at and str(snapshot.get("effective_at") or "") > effective_at:
                continue
            try:
                rules, _ = self.list_rules_for_activation_snapshot(
                    activation_set_snapshot_id=snapshot["activation_set_snapshot_id"],
                    resolved_activation_set_hash=snapshot["resolved_activation_set_hash"],
                    context=context,
                    requested_rule_families=requested_rule_families,
                )
            except ValueError:
                continue
            if not rules:
                continue
            return {
                **snapshot,
                "requested_rule_families": list(requested_rule_families),
            }
        raise ValueError("ACTIVATION_SNAPSHOT_NOT_MATERIALIZED")

    def list_rules_for_activation_snapshot(
        self,
        *,
        activation_set_snapshot_id: str,
        resolved_activation_set_hash: str,
        context: RuntimeContextDTO,
        requested_rule_families: list[str] | None = None,
        effective_at: str | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        return self._rule_store.list_rules_for_activation_snapshot(
            activation_set_snapshot_id=activation_set_snapshot_id,
            resolved_activation_set_hash=resolved_activation_set_hash,
            organization_id=context.organization_id,
            workspace_id=context.workspace_id,
            requested_rule_families=requested_rule_families or [],
            effective_at=effective_at,
        )

    def verify_snapshot_hashes(self, snapshot_id: str, expected_hashes: dict[str, str]) -> bool:
        from eos.db import get_conn

        with get_conn() as conn:
            row = conn.execute(
                hosted.sql_query("SELECT * FROM knowledge_activation_set_snapshot WHERE activation_set_snapshot_id = ?", 'SELECT * FROM knowledge_activation_set_snapshot WHERE activation_set_snapshot_id = %s'),
                (snapshot_id,),
            ).fetchone()
        if row is None:
            return False
        snapshot_payload = {
            "organization_id": row["organization_id"],
            "workspace_id": row["workspace_id"],
            "principal_entitlement_ref": row["principal_entitlement_ref"],
            "activation_resolver_policy_version": row[
                "activation_resolver_policy_version"
            ],
            "effective_at": row["effective_at"],
            "included_activation_ids": json.loads(row["included_activation_ids_json"]),
            "excluded_activation_evidence": json.loads(
                row["excluded_activation_evidence_json"]
            ),
            "prior_snapshot_hash": row["prior_snapshot_hash"],
        }
        row_payload = {
            "activation_set_snapshot_id": row["activation_set_snapshot_id"],
            **snapshot_payload,
            "resolved_activation_set_hash": row["resolved_activation_set_hash"],
            "created_at": row["created_at"],
        }
        expected_resolved = expected_hashes.get("resolved_activation_set_hash")
        expected_content = expected_hashes.get("content_hash")
        return (
            (expected_resolved is None or expected_resolved == row["resolved_activation_set_hash"])
            and content_hash(snapshot_payload) == row["resolved_activation_set_hash"]
            and (expected_content is None or expected_content == row["content_hash"])
            and content_hash(row_payload) == row["content_hash"]
        )
