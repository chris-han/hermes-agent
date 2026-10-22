from __future__ import annotations

from storage import hosted_postgres as hosted

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from eos.artifact_hashing import canonical_json
from eos.db import get_conn, now_iso
from semantier.sensitive_governance_contracts.enums import (
    AssessmentState,
    ReplayPhase,
    RunState,
)
from semantier.sensitive_governance_contracts.events import event_now
from semantier.sensitive_governance_contracts.hashing import canonical_hash


class SensitiveGovernanceStore:
    """Authoritative eos.db persistence boundary for the bounded demo slice."""

    def ensure_schema(self, conn: sqlite3.Connection | None = None) -> None:
        close_conn = False
        if conn is None:
            conn = get_conn()
            close_conn = True
        try:
            if hosted.is_postgres(conn):
                hosted.require_tables(conn, 'sensitive_governance_assessments', 'sensitive_governance_authority_objects', 'sensitive_governance_bundle_activations', 'sensitive_governance_capability_grants', 'sensitive_governance_capability_revocations', 'sensitive_governance_dispatch_outbox', 'sensitive_governance_escalation_cases', 'sensitive_governance_escalation_reviews', 'sensitive_governance_events', 'sensitive_governance_response_authority_objects', 'sensitive_governance_restricted_payloads', 'sensitive_governance_reveal_decisions', 'sensitive_governance_reveal_leases', 'sensitive_governance_runs')
                return
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sensitive_governance_assessments (
                  assessment_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  user_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  session_id TEXT,
                  contract_version TEXT NOT NULL,
                  state TEXT NOT NULL,
                  source_kind TEXT NOT NULL,
                  purpose TEXT NOT NULL,
                  destination TEXT NOT NULL,
                  request_hash TEXT NOT NULL,
                  assessment_hash TEXT NOT NULL,
                  idempotency_key TEXT NOT NULL,
                  prepared_payload_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  consumed_at TEXT,
                  consumed_by_run_id TEXT,
                  UNIQUE(organization_id, idempotency_key)
                );

                CREATE TABLE IF NOT EXISTS sensitive_governance_runs (
                  run_id TEXT PRIMARY KEY,
                  assessment_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  user_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  contract_version TEXT NOT NULL,
                  state TEXT NOT NULL,
                  execution_idempotency_key TEXT NOT NULL,
                  governed_response_json TEXT NOT NULL,
                  governed_response_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  completed_at TEXT,
                  UNIQUE(organization_id, execution_idempotency_key),
                  FOREIGN KEY(assessment_id) REFERENCES sensitive_governance_assessments(assessment_id)
                );

                CREATE TABLE IF NOT EXISTS sensitive_governance_events (
                  event_id TEXT PRIMARY KEY,
                  aggregate_id TEXT NOT NULL,
                  event_type TEXT NOT NULL,
                  replay_phase TEXT NOT NULL,
                  occurred_at TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  event_hash TEXT NOT NULL,
                  causation_id TEXT
                );

                CREATE TABLE IF NOT EXISTS sensitive_governance_dispatch_outbox (
                  dispatch_id TEXT PRIMARY KEY,
                  aggregate_id TEXT NOT NULL,
                  adapter TEXT NOT NULL,
                  idempotency_key TEXT NOT NULL,
                  status TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  payload_hash TEXT NOT NULL,
                  claimed_by TEXT,
                  claimed_at TEXT,
                  created_at TEXT NOT NULL,
                  UNIQUE(adapter, idempotency_key)
                );

                CREATE TABLE IF NOT EXISTS sensitive_governance_authority_objects (
                  object_ref TEXT PRIMARY KEY,
                  assessment_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  object_type TEXT NOT NULL,
                  object_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(assessment_id) REFERENCES sensitive_governance_assessments(assessment_id)
                );

                CREATE TABLE IF NOT EXISTS sensitive_governance_bundle_activations (
                  activation_id TEXT PRIMARY KEY,
                  bundle_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  manifest_hash TEXT NOT NULL,
                  artifact_hashes_json TEXT NOT NULL,
                  activated_at TEXT NOT NULL,
                  UNIQUE(bundle_id, organization_id, manifest_hash)
                );

                CREATE TABLE IF NOT EXISTS sensitive_governance_capability_grants (
                  grant_ref TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  principal_id TEXT NOT NULL,
                  capability TEXT NOT NULL,
                  resource_scope_json TEXT NOT NULL,
                  workflow_scope_json TEXT NOT NULL,
                  purpose_scope TEXT NOT NULL,
                  status TEXT NOT NULL,
                  granted_at TEXT NOT NULL,
                  revoked_at TEXT,
                  UNIQUE(organization_id, principal_id, capability, resource_scope_json, purpose_scope)
                );

                CREATE TABLE IF NOT EXISTS sensitive_governance_capability_revocations (
                  revocation_ref TEXT PRIMARY KEY,
                  grant_ref TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  principal_id TEXT NOT NULL,
                  capability TEXT NOT NULL,
                  revoked_at TEXT NOT NULL,
                  reason_code TEXT NOT NULL,
                  FOREIGN KEY(grant_ref) REFERENCES sensitive_governance_capability_grants(grant_ref)
                );

                CREATE TABLE IF NOT EXISTS sensitive_governance_escalation_cases (
                  escalation_id TEXT PRIMARY KEY,
                  assessment_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  assigned_role TEXT NOT NULL,
                  assigned_principal TEXT,
                  status TEXT NOT NULL,
                  reason_codes_json TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(assessment_id) REFERENCES sensitive_governance_assessments(assessment_id)
                );

                CREATE TABLE IF NOT EXISTS sensitive_governance_escalation_reviews (
                  review_ref TEXT PRIMARY KEY,
                  escalation_id TEXT NOT NULL,
                  assessment_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  reviewer_principal TEXT NOT NULL,
                  decision TEXT NOT NULL,
                  reason TEXT NOT NULL,
                  idempotency_key TEXT NOT NULL,
                  review_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  UNIQUE(organization_id, idempotency_key),
                  FOREIGN KEY(escalation_id) REFERENCES sensitive_governance_escalation_cases(escalation_id),
                  FOREIGN KEY(assessment_id) REFERENCES sensitive_governance_assessments(assessment_id)
                );

                CREATE TABLE IF NOT EXISTS sensitive_governance_response_authority_objects (
                  object_ref TEXT PRIMARY KEY,
                  run_id TEXT NOT NULL,
                  assessment_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  object_type TEXT NOT NULL,
                  object_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sensitive_governance_restricted_payloads (
                  restricted_payload_ref TEXT PRIMARY KEY,
                  assessment_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  classification TEXT NOT NULL,
                  clear_value_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(assessment_id) REFERENCES sensitive_governance_assessments(assessment_id)
                );

                CREATE TABLE IF NOT EXISTS sensitive_governance_reveal_leases (
                  lease_id TEXT PRIMARY KEY,
                  response_id TEXT NOT NULL,
                  segment_id TEXT NOT NULL,
                  run_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  principal_id TEXT NOT NULL,
                  purpose TEXT NOT NULL,
                  state TEXT NOT NULL,
                  max_uses INTEGER NOT NULL,
                  use_count INTEGER NOT NULL,
                  issued_at TEXT NOT NULL,
                  expires_at TEXT NOT NULL,
                  policy_hash TEXT NOT NULL,
                  profile_hash TEXT NOT NULL,
                  clear_value_json TEXT NOT NULL,
                  decision_hash TEXT NOT NULL,
                  UNIQUE(organization_id, response_id, segment_id, principal_id, purpose)
                );

                CREATE TABLE IF NOT EXISTS sensitive_governance_reveal_decisions (
                  decision_id TEXT PRIMARY KEY,
                  operation TEXT NOT NULL,
                  response_id TEXT NOT NULL,
                  segment_id TEXT NOT NULL,
                  run_id TEXT,
                  organization_id TEXT NOT NULL,
                  principal_id TEXT NOT NULL,
                  allowed INTEGER NOT NULL,
                  reason_codes_json TEXT NOT NULL,
                  lease_id TEXT,
                  payload_json TEXT NOT NULL,
                  decision_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  UNIQUE(organization_id, operation, response_id, segment_id, principal_id)
                );
                """
            )
        finally:
            if close_conn:
                conn.close()

    def ensure_default_capability_grants(
        self,
        *,
        organization_id: str,
        principal_id: str,
        workspace_id: str,
        purpose: str,
    ) -> list[dict[str, Any]]:
        resource_scope = [workspace_id]
        workflow_scope = ["assessment", "execute_internal", "execute_external"]
        capabilities = [
            "sensitive_governance.evaluate",
            "sensitive_governance.execute_internal",
            "sensitive_governance.execute_external",
            "sensitive_governance.reveal",
        ]
        now = now_iso()
        with get_conn() as conn:
            self.ensure_schema(conn)
            for capability in capabilities:
                grant_ref = self._capability_grant_ref(
                    organization_id=organization_id,
                    principal_id=principal_id,
                    workspace_id=workspace_id,
                    purpose=purpose,
                    capability=capability,
                )
                conn.execute(
                    hosted.sql_query("""
                    INSERT OR IGNORE INTO sensitive_governance_capability_grants (
                      grant_ref, organization_id, principal_id, capability,
                      resource_scope_json, workflow_scope_json, purpose_scope,
                      status, granted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, """
                    INSERT INTO sensitive_governance_capability_grants (
                      grant_ref, organization_id, principal_id, capability,
                      resource_scope_json, workflow_scope_json, purpose_scope,
                      status, granted_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING"""),
                    (
                        grant_ref,
                        organization_id,
                        principal_id,
                        capability,
                        canonical_json(resource_scope),
                        canonical_json(workflow_scope),
                        purpose,
                        "active",
                        now,
                    ),
                )
        return self.list_capability_grants(
            organization_id=organization_id,
            principal_id=principal_id,
            workspace_id=workspace_id,
            purpose=purpose,
        )

    def revoke_capability_grant(
        self,
        *,
        organization_id: str,
        principal_id: str,
        workspace_id: str,
        purpose: str,
        capability: str,
        reason_code: str,
    ) -> dict[str, Any]:
        grant_ref = self._capability_grant_ref(
            organization_id=organization_id,
            principal_id=principal_id,
            workspace_id=workspace_id,
            purpose=purpose,
            capability=capability,
        )
        revoked_at = now_iso()
        revocation_ref = "sg_capability_revocation_" + canonical_hash(
            {
                "grant_ref": grant_ref,
                "organization_id": organization_id,
                "principal_id": principal_id,
                "capability": capability,
                "reason_code": reason_code,
            }
        )[:24]
        self.ensure_default_capability_grants(
            organization_id=organization_id,
            principal_id=principal_id,
            workspace_id=workspace_id,
            purpose=purpose,
        )
        with get_conn() as conn:
            self.ensure_schema(conn)
            conn.execute(
                hosted.sql_query("""
                UPDATE sensitive_governance_capability_grants
                SET status = ?, revoked_at = ?
                WHERE grant_ref = ?
                """, """
                UPDATE sensitive_governance_capability_grants
                SET status = %s, revoked_at = %s
                WHERE grant_ref = %s
                """),
                ("revoked", revoked_at, grant_ref),
            )
            conn.execute(
                hosted.sql_query("""
                INSERT OR IGNORE INTO sensitive_governance_capability_revocations (
                  revocation_ref, grant_ref, organization_id, principal_id,
                  capability, revoked_at, reason_code
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO sensitive_governance_capability_revocations (
                  revocation_ref, grant_ref, organization_id, principal_id,
                  capability, revoked_at, reason_code
                ) VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING"""),
                (
                    revocation_ref,
                    grant_ref,
                    organization_id,
                    principal_id,
                    capability,
                    revoked_at,
                    reason_code,
                ),
            )
        return {
            "revocation_ref": revocation_ref,
            "grant_ref": grant_ref,
            "organization_id": organization_id,
            "principal_id": principal_id,
            "capability": capability,
            "revoked_at": revoked_at,
            "reason_code": reason_code,
        }

    def list_capability_grants(
        self,
        *,
        organization_id: str,
        principal_id: str,
        workspace_id: str,
        purpose: str,
    ) -> list[dict[str, Any]]:
        resource_scope_json = canonical_json([workspace_id])
        with get_conn() as conn:
            self.ensure_schema(conn)
            rows = conn.execute(
                hosted.sql_query("""
                SELECT * FROM sensitive_governance_capability_grants
                WHERE organization_id = ?
                  AND principal_id = ?
                  AND resource_scope_json = ?
                  AND purpose_scope = ?
                ORDER BY capability
                """, """
                SELECT * FROM sensitive_governance_capability_grants
                WHERE organization_id = %s
                  AND principal_id = %s
                  AND resource_scope_json = %s
                  AND purpose_scope = %s
                ORDER BY capability
                """),
                (organization_id, principal_id, resource_scope_json, purpose),
            ).fetchall()
            revocation_rows = conn.execute(
                hosted.sql_query("""
                SELECT grant_ref, revocation_ref
                FROM sensitive_governance_capability_revocations
                WHERE organization_id = ? AND principal_id = ?
                ORDER BY revocation_ref
                """, """
                SELECT grant_ref, revocation_ref
                FROM sensitive_governance_capability_revocations
                WHERE organization_id = %s AND principal_id = %s
                ORDER BY revocation_ref
                """),
                (organization_id, principal_id),
            ).fetchall()
        revocations_by_grant: dict[str, list[str]] = {}
        for row in revocation_rows:
            revocations_by_grant.setdefault(row["grant_ref"], []).append(row["revocation_ref"])
        return [
            {
                "grant_ref": row["grant_ref"],
                "organization_id": row["organization_id"],
                "principal_id": row["principal_id"],
                "capability": row["capability"],
                "resource_scope": json.loads(row["resource_scope_json"]),
                "workflow_scope": json.loads(row["workflow_scope_json"]),
                "purpose_scope": row["purpose_scope"],
                "status": row["status"],
                "granted_at": row["granted_at"],
                "revoked_at": row["revoked_at"],
                "revocation_refs": revocations_by_grant.get(row["grant_ref"], []),
            }
            for row in rows
        ]

    @staticmethod
    def _capability_grant_ref(
        *,
        organization_id: str,
        principal_id: str,
        workspace_id: str,
        purpose: str,
        capability: str,
    ) -> str:
        return "sg_capability_grant_" + canonical_hash(
            {
                "organization_id": organization_id,
                "principal_id": principal_id,
                "workspace_id": workspace_id,
                "purpose": purpose,
                "capability": capability,
            }
        )[:24]

    def create_assessment(
        self,
        *,
        organization_id: str,
        user_id: str,
        workspace_id: str,
        session_id: str | None,
        contract_version: str,
        source_kind: str,
        purpose: str,
        destination: str,
        request_hash: str,
        idempotency_key: str,
        prepared_payload: dict[str, Any],
        state: str = AssessmentState.PREPARED.value,
        authority_objects: list[dict[str, Any]] | None = None,
        bundle_activation: dict[str, Any] | None = None,
        escalation_case: dict[str, Any] | None = None,
        restricted_payloads: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        assessment_id = f"sg_assessment_{uuid4().hex}"
        assessment_payload = {
            "assessment_id": assessment_id,
            "organization_id": organization_id,
            "user_id": user_id,
            "workspace_id": workspace_id,
            "session_id": session_id,
            "contract_version": contract_version,
            "state": state,
            "source_kind": source_kind,
            "purpose": purpose,
            "destination": destination,
            "request_hash": request_hash,
            "prepared_payload": prepared_payload,
        }
        assessment_hash = canonical_hash(assessment_payload)
        created_at = now_iso()
        with get_conn() as conn:
            self.ensure_schema(conn)
            existing = conn.execute(
                hosted.sql_query("""
                SELECT * FROM sensitive_governance_assessments
                WHERE organization_id = ? AND idempotency_key = ?
                """, """
                SELECT * FROM sensitive_governance_assessments
                WHERE organization_id = %s AND idempotency_key = %s
                """),
                (organization_id, idempotency_key),
            ).fetchone()
            if existing:
                if any(existing[field] != value for field, value in (
                    ('user_id', user_id), ('workspace_id', workspace_id),
                    ('session_id', session_id), ('contract_version', contract_version),
                    ('request_hash', request_hash))):
                    raise ValueError('assessment idempotency conflict')
                return self._assessment_from_row(existing)
            conn.execute(
                hosted.sql_query("""
                INSERT INTO sensitive_governance_assessments (
                  assessment_id, organization_id, user_id, workspace_id, session_id,
                  contract_version, state, source_kind, purpose, destination,
                  request_hash, assessment_hash, idempotency_key,
                  prepared_payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO sensitive_governance_assessments (
                  assessment_id, organization_id, user_id, workspace_id, session_id,
                  contract_version, state, source_kind, purpose, destination,
                  request_hash, assessment_hash, idempotency_key,
                  prepared_payload_json, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """),
                (
                    assessment_id,
                    organization_id,
                    user_id,
                    workspace_id,
                    session_id,
                    contract_version,
                    state,
                    source_kind,
                    purpose,
                    destination,
                    request_hash,
                    assessment_hash,
                    idempotency_key,
                    canonical_json(prepared_payload),
                    created_at,
                ),
            )
            for restricted in restricted_payloads or []:
                clear_payload = {"value": restricted["value"]}
                conn.execute(
                    hosted.sql_query("""
                    INSERT INTO sensitive_governance_restricted_payloads (
                      restricted_payload_ref, assessment_id, organization_id,
                      classification, clear_value_json, content_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, """
                    INSERT INTO sensitive_governance_restricted_payloads (
                      restricted_payload_ref, assessment_id, organization_id,
                      classification, clear_value_json, content_hash, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """),
                    (
                        restricted["restricted_payload_ref"],
                        assessment_id,
                        organization_id,
                        restricted["classification"],
                        canonical_json(clear_payload),
                        canonical_hash(clear_payload),
                        created_at,
                    ),
                )
            if bundle_activation:
                activation_id = "sg_bundle_activation_" + canonical_hash(
                    {
                        "bundle_id": bundle_activation["bundle_id"],
                        "organization_id": organization_id,
                        "manifest_hash": bundle_activation["manifest_hash"],
                    }
                )[:24]
                activation_cursor = conn.execute(
                    hosted.sql_query("""
                    INSERT OR IGNORE INTO sensitive_governance_bundle_activations (
                      activation_id, bundle_id, organization_id, manifest_hash,
                      artifact_hashes_json, activated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """, """
                    INSERT INTO sensitive_governance_bundle_activations (
                      activation_id, bundle_id, organization_id, manifest_hash,
                      artifact_hashes_json, activated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING"""),
                    (
                        activation_id,
                        bundle_activation["bundle_id"],
                        organization_id,
                        bundle_activation["manifest_hash"],
                        canonical_json(bundle_activation["artifact_hashes"]),
                        created_at,
                    ),
                )
                if activation_cursor.rowcount == 1:
                    activation_event = event_now(
                        event_id=f"sg_event_{uuid4().hex}",
                        event_type="sensitive_governance.bundle_activated",
                        phase=ReplayPhase.DECISION_COMMITTED,
                        aggregate_id=activation_id,
                        payload={
                            "activation_id": activation_id,
                            "bundle_id": bundle_activation["bundle_id"],
                            "organization_id": organization_id,
                            "manifest_hash": bundle_activation["manifest_hash"],
                            "artifact_hashes": bundle_activation["artifact_hashes"],
                        },
                    )
                    self._append_event(conn, activation_event.to_dict())
            for authority_object in authority_objects or []:
                payload = authority_object["payload"]
                conn.execute(
                    hosted.sql_query("""
                    INSERT OR IGNORE INTO sensitive_governance_authority_objects (
                      object_ref, assessment_id, organization_id, object_type,
                      object_json, content_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, """
                    INSERT INTO sensitive_governance_authority_objects (
                      object_ref, assessment_id, organization_id, object_type,
                      object_json, content_hash, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING"""),
                    (
                        authority_object["object_ref"],
                        assessment_id,
                        organization_id,
                        authority_object["object_type"],
                        canonical_json(payload),
                        canonical_hash(payload),
                        created_at,
                    ),
                )
            if escalation_case:
                conn.execute(
                    hosted.sql_query("""
                    INSERT INTO sensitive_governance_escalation_cases (
                      escalation_id, assessment_id, organization_id, assigned_role,
                      assigned_principal, status, reason_codes_json, payload_json,
                      content_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, """
                    INSERT INTO sensitive_governance_escalation_cases (
                      escalation_id, assessment_id, organization_id, assigned_role,
                      assigned_principal, status, reason_codes_json, payload_json,
                      content_hash, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """),
                    (
                        escalation_case["escalation_id"],
                        assessment_id,
                        organization_id,
                        escalation_case["assigned_role"],
                        escalation_case.get("assigned_principal"),
                        escalation_case["initial_status"],
                        canonical_json(escalation_case["reason_codes"]),
                        canonical_json(escalation_case),
                        canonical_hash(escalation_case),
                        created_at,
                    ),
                )
            event = event_now(
                event_id=f"sg_event_{uuid4().hex}",
                event_type=f"sensitive_governance.assessment_{state.lower()}",
                phase=ReplayPhase.DECISION_COMMITTED,
                aggregate_id=assessment_id,
                payload={
                    "assessment_id": assessment_id,
                    "assessment_hash": assessment_hash,
                    "request_hash": request_hash,
                    "contract_version": contract_version,
                    "state": state,
                },
            )
            self._append_event(conn, event.to_dict())
        return self.get_assessment(assessment_id, organization_id=organization_id) or {}

    def consume_assessment_and_create_run(
        self,
        *,
        assessment_id: str,
        organization_id: str,
        execution_idempotency_key: str,
        assessment_hash: str,
        governed_response: dict[str, Any],
        response_authority_objects: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        run_id = str(governed_response["run_id"])
        response_hash = str(governed_response["response_hash"])
        now = now_iso()
        with get_conn() as conn:
            self.ensure_schema(conn)
            existing_run = conn.execute(
                hosted.sql_query("""
                SELECT * FROM sensitive_governance_runs
                WHERE organization_id = ? AND execution_idempotency_key = ?
                """, """
                SELECT * FROM sensitive_governance_runs
                WHERE organization_id = %s AND execution_idempotency_key = %s
                """),
                (organization_id, execution_idempotency_key),
            ).fetchone()
            if existing_run:
                if existing_run["assessment_id"] != assessment_id:
                    raise ValueError("execution idempotency conflict")
                return self._run_from_row(existing_run)

            assessment = conn.execute(
                hosted.sql_query("""
                SELECT * FROM sensitive_governance_assessments
                WHERE assessment_id = ? AND organization_id = ?
                """, """
                SELECT * FROM sensitive_governance_assessments
                WHERE assessment_id = %s AND organization_id = %s
                """),
                (assessment_id, organization_id),
            ).fetchone()
            if assessment is None:
                raise KeyError("assessment not found")
            if assessment["assessment_hash"] != assessment_hash:
                raise ValueError("assessment_hash mismatch")
            if assessment["state"] != AssessmentState.PREPARED.value:
                raise ValueError("assessment is not executable")
            prepared_payload = json.loads(assessment["prepared_payload_json"])
            execution_intent = prepared_payload.get("execution_intent") or {}
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

            prepared_event_id = f"sg_event_{uuid4().hex}"
            terminal_event_id = f"sg_event_{uuid4().hex}"
            response_with_event_pins = dict(governed_response)
            pins = dict(response_with_event_pins.get("pins") or {})
            pins["execution_prepared_event_id"] = prepared_event_id
            pins["execution_terminal_event_id"] = terminal_event_id
            response_with_event_pins["pins"] = pins
            response_with_event_pins["response_hash"] = canonical_hash(
                {
                    key: value
                    for key, value in response_with_event_pins.items()
                    if key != "response_hash"
                }
            )
            response_hash = str(response_with_event_pins["response_hash"])
            execution_boundary = (
                response_with_event_pins.get("governance_summary", {}).get(
                    "execution_boundary", {}
                )
                or {}
            )

            prepared_event = event_now(
                event_id=prepared_event_id,
                event_type="sensitive_governance.execution_prepared",
                phase=ReplayPhase.EXECUTION_PREPARED,
                aggregate_id=run_id,
                payload={
                    "assessment_id": assessment_id,
                    "run_id": run_id,
                    "assessment_hash": assessment_hash,
                    "adapter_ref": execution_boundary.get("adapter_ref"),
                    "transformed_input_hash": execution_boundary.get("transformed_input_hash"),
                    "restricted_request_ref": execution_boundary.get("restricted_request_ref"),
                },
            )
            terminal_event = event_now(
                event_id=terminal_event_id,
                event_type="sensitive_governance.execution_succeeded",
                phase=ReplayPhase.EXECUTION_TERMINAL,
                aggregate_id=run_id,
                payload={
                    "assessment_id": assessment_id,
                    "run_id": run_id,
                    "governed_response_hash": response_hash,
                },
                causation_id=prepared_event.event_id,
            )
            self._append_event(conn, prepared_event.to_dict())
            self._append_event(conn, terminal_event.to_dict())

            conn.execute(
                hosted.sql_query("""
                INSERT INTO sensitive_governance_runs (
                  run_id, assessment_id, organization_id, user_id, workspace_id,
                  contract_version, state, execution_idempotency_key,
                  governed_response_json, governed_response_hash, created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO sensitive_governance_runs (
                  run_id, assessment_id, organization_id, user_id, workspace_id,
                  contract_version, state, execution_idempotency_key,
                  governed_response_json, governed_response_hash, created_at, completed_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """),
                (
                    run_id,
                    assessment_id,
                    organization_id,
                    assessment["user_id"],
                    assessment["workspace_id"],
                    assessment["contract_version"],
                    RunState.SUCCEEDED.value,
                    execution_idempotency_key,
                    canonical_json(response_with_event_pins),
                    response_hash,
                    now,
                    now,
                ),
            )
            for authority_object in response_authority_objects or []:
                payload = authority_object["payload"]
                conn.execute(
                    hosted.sql_query("""
                    INSERT OR IGNORE INTO sensitive_governance_response_authority_objects (
                      object_ref, run_id, assessment_id, organization_id,
                      object_type, object_json, content_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, """
                    INSERT INTO sensitive_governance_response_authority_objects (
                      object_ref, run_id, assessment_id, organization_id,
                      object_type, object_json, content_hash, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING"""),
                    (
                        authority_object["object_ref"],
                        run_id,
                        assessment_id,
                        organization_id,
                        authority_object["object_type"],
                        canonical_json(payload),
                        canonical_hash(payload),
                        now,
                    ),
                )
            consume_cursor = conn.execute(
                hosted.sql_query("""
                UPDATE sensitive_governance_assessments
                SET state = ?, consumed_at = ?, consumed_by_run_id = ?
                WHERE assessment_id = ? AND organization_id = ? AND state = ?
                """, """
                UPDATE sensitive_governance_assessments
                SET state = %s, consumed_at = %s, consumed_by_run_id = %s
                WHERE assessment_id = %s AND organization_id = %s AND state = %s
                """),
                (
                    AssessmentState.CONSUMED.value,
                    now,
                    run_id,
                    assessment_id,
                    organization_id,
                    AssessmentState.PREPARED.value,
                ),
            )
            if consume_cursor.rowcount != 1:
                raise ValueError("assessment consumption failed")
        return self.get_run(run_id, organization_id=organization_id) or {}

    def get_assessment(self, assessment_id: str, *, organization_id: str) -> dict[str, Any] | None:
        with get_conn() as conn:
            self.ensure_schema(conn)
            row = conn.execute(
                hosted.sql_query("""
                SELECT * FROM sensitive_governance_assessments
                WHERE assessment_id = ? AND organization_id = ?
                """, """
                SELECT * FROM sensitive_governance_assessments
                WHERE assessment_id = %s AND organization_id = %s
                """),
                (assessment_id, organization_id),
            ).fetchone()
        return self._assessment_from_row(row) if row else None

    def review_escalation(
        self,
        *,
        organization_id: str,
        escalation_id: str,
        reviewer_principal: str,
        reviewer_role: str,
        decision: str,
        reason: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        now = now_iso()
        with get_conn() as conn:
            self.ensure_schema(conn)
            prior = conn.execute(
                hosted.sql_query("""SELECT review_json FROM sensitive_governance_escalation_reviews
                   WHERE organization_id = ? AND idempotency_key = ?""", """SELECT review_json FROM sensitive_governance_escalation_reviews
                   WHERE organization_id = %s AND idempotency_key = %s"""),
                (organization_id, idempotency_key),
            ).fetchone()
            if prior:
                review = json.loads(prior["review_json"])
                assessment = conn.execute(
                    hosted.sql_query("""SELECT assessment_hash, state FROM sensitive_governance_assessments
                       WHERE assessment_id = ? AND organization_id = ?""", """SELECT assessment_hash, state FROM sensitive_governance_assessments
                       WHERE assessment_id = %s AND organization_id = %s"""),
                    (review["assessment_id"], organization_id),
                ).fetchone()
                if assessment is None:
                    raise KeyError("review assessment not found")
                return {
                    **review,
                    "assessment_hash": assessment["assessment_hash"],
                    "assessment_state": assessment["state"],
                }
            case = conn.execute(
                hosted.sql_query("""SELECT * FROM sensitive_governance_escalation_cases
                   WHERE escalation_id = ? AND organization_id = ?""", """SELECT * FROM sensitive_governance_escalation_cases
                   WHERE escalation_id = %s AND organization_id = %s"""),
                (escalation_id, organization_id),
            ).fetchone()
            if case is None:
                raise KeyError("escalation not found")
            if case["status"] != "open":
                raise ValueError("escalation is not open")
            assessment = conn.execute(
                hosted.sql_query("""SELECT * FROM sensitive_governance_assessments
                   WHERE assessment_id = ? AND organization_id = ?""", """SELECT * FROM sensitive_governance_assessments
                   WHERE assessment_id = %s AND organization_id = %s"""),
                (case["assessment_id"], organization_id),
            ).fetchone()
            if assessment is None or assessment["state"] != AssessmentState.ESCALATED.value:
                raise ValueError("assessment is not awaiting review")
            decision = decision.upper()
            if decision not in {"APPROVE", "REJECT"}:
                raise ValueError("invalid review decision")
            review_body = {
                "escalation_id": escalation_id,
                "assessment_id": case["assessment_id"],
                "organization_id": organization_id,
                "reviewer_principal": reviewer_principal,
                "reviewer_role": reviewer_role,
                "decision": decision,
                "reason": reason,
                "reviewed_at": now,
            }
            review_hash = canonical_hash(review_body)
            review_ref = "sg_review_" + review_hash[:24]
            review = {**review_body, "review_ref": review_ref, "review_hash": review_hash}
            conn.execute(
                hosted.sql_query("""INSERT INTO sensitive_governance_escalation_reviews (
                     review_ref, escalation_id, assessment_id, organization_id,
                     reviewer_principal, decision, reason, idempotency_key,
                     review_json, content_hash, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", """INSERT INTO sensitive_governance_escalation_reviews (
                     review_ref, escalation_id, assessment_id, organization_id,
                     reviewer_principal, decision, reason, idempotency_key,
                     review_json, content_hash, created_at
                   ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""),
                (
                    review_ref, escalation_id, case["assessment_id"], organization_id,
                    reviewer_principal, decision, reason, idempotency_key,
                    canonical_json(review), canonical_hash(review), now,
                ),
            )
            next_state = (
                AssessmentState.PREPARED.value if decision == "APPROVE"
                else AssessmentState.BLOCKED.value
            )
            prepared_payload = json.loads(assessment["prepared_payload_json"])
            governance = dict(prepared_payload.get("input_governance") or {})
            escalation = dict(governance.get("escalation") or {})
            escalation.update({
                "review_ref": review_ref,
                "review_hash": review_hash,
                "review_decision": decision,
                "reviewed_at": now,
                "execution_blocked": decision != "APPROVE",
            })
            governance["escalation"] = escalation
            governance["status"] = next_state
            prepared_payload["input_governance"] = governance
            assessment_payload = {
                "assessment_id": assessment["assessment_id"],
                "organization_id": assessment["organization_id"],
                "user_id": assessment["user_id"],
                "workspace_id": assessment["workspace_id"],
                "session_id": assessment["session_id"],
                "contract_version": assessment["contract_version"],
                "state": next_state,
                "source_kind": assessment["source_kind"],
                "purpose": assessment["purpose"],
                "destination": assessment["destination"],
                "request_hash": assessment["request_hash"],
                "prepared_payload": prepared_payload,
            }
            next_hash = canonical_hash(assessment_payload)
            conn.execute(
                hosted.sql_query("""UPDATE sensitive_governance_escalation_cases
                   SET status = ?, payload_json = ?, content_hash = ?
                   WHERE escalation_id = ? AND organization_id = ?""", """UPDATE sensitive_governance_escalation_cases
                   SET status = %s, payload_json = %s, content_hash = %s
                   WHERE escalation_id = %s AND organization_id = %s"""),
                (
                    "approved" if decision == "APPROVE" else "rejected",
                    canonical_json({**json.loads(case["payload_json"]), **escalation}),
                    canonical_hash({**json.loads(case["payload_json"]), **escalation}),
                    escalation_id, organization_id,
                ),
            )
            conn.execute(
                hosted.sql_query("""UPDATE sensitive_governance_assessments
                   SET state = ?, assessment_hash = ?, prepared_payload_json = ?
                   WHERE assessment_id = ? AND organization_id = ? AND state = ?""", """UPDATE sensitive_governance_assessments
                   SET state = %s, assessment_hash = %s, prepared_payload_json = %s
                   WHERE assessment_id = %s AND organization_id = %s AND state = %s"""),
                (
                    next_state, next_hash, canonical_json(prepared_payload),
                    assessment["assessment_id"], organization_id, AssessmentState.ESCALATED.value,
                ),
            )
            event = event_now(
                event_id=f"sg_event_{uuid4().hex}",
                event_type="sensitive_governance.escalation_reviewed",
                phase=ReplayPhase.DECISION_COMMITTED,
                aggregate_id=escalation_id,
                payload={
                    "review_ref": review_ref,
                    "review_hash": review_hash,
                    "assessment_id": assessment["assessment_id"],
                    "assessment_hash": next_hash,
                    "decision": decision,
                    "reviewer_principal": reviewer_principal,
                },
            )
            self._append_event(conn, event.to_dict())
        return {**review, "assessment_hash": next_hash, "assessment_state": next_state}

    def get_run(self, run_id: str, *, organization_id: str) -> dict[str, Any] | None:
        with get_conn() as conn:
            self.ensure_schema(conn)
            row = conn.execute(
                hosted.sql_query("""
                SELECT * FROM sensitive_governance_runs
                WHERE run_id = ? AND organization_id = ?
                """, """
                SELECT * FROM sensitive_governance_runs
                WHERE run_id = %s AND organization_id = %s
                """),
                (run_id, organization_id),
            ).fetchone()
        return self._run_from_row(row) if row else None

    def get_run_by_execution_idempotency_key(
        self,
        *,
        organization_id: str,
        execution_idempotency_key: str,
    ) -> dict[str, Any] | None:
        with get_conn() as conn:
            self.ensure_schema(conn)
            row = conn.execute(
                hosted.sql_query("""
                SELECT * FROM sensitive_governance_runs
                WHERE organization_id = ? AND execution_idempotency_key = ?
                """, """
                SELECT * FROM sensitive_governance_runs
                WHERE organization_id = %s AND execution_idempotency_key = %s
                """),
                (organization_id, execution_idempotency_key),
            ).fetchone()
        return self._run_from_row(row) if row else None

    def reveal_segment(
        self,
        *,
        organization_id: str,
        principal_id: str,
        response_id: str,
        segment_id: str,
        purpose: str,
        workspace_id: str | None = None,
        operation: str = "reveal",
        force_stale: bool = False,
        authorization_valid: bool = True,
        authorization_reason_codes: list[str] | None = None,
        authorization_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with get_conn() as conn:
            self.ensure_schema(conn)
            row = conn.execute(
                hosted.sql_query("""
                SELECT * FROM sensitive_governance_runs
                WHERE organization_id = ?
                """, """
                SELECT * FROM sensitive_governance_runs
                WHERE organization_id = %s
                """),
                (organization_id,),
            ).fetchall()
            run_row = None
            response = None
            for candidate in row:
                candidate_response = json.loads(candidate["governed_response_json"])
                if candidate_response.get("response_id") == response_id:
                    run_row = candidate
                    response = candidate_response
                    break
            if (
                run_row is None or response is None
                or (workspace_id is not None and (not workspace_id or run_row["workspace_id"] != workspace_id))
            ):
                raise KeyError("response not found")
            segment = next(
                (
                    item
                    for item in response.get("segments", [])
                    if item.get("segment_id") == segment_id
                ),
                None,
            )
            if not isinstance(segment, dict):
                raise KeyError("segment not found")
            now = datetime.now(timezone.utc)
            now_text = now.isoformat()
            response_stale = force_stale or self._run_is_stale(run_row["completed_at"], now)
            decision = self._build_reveal_decision_payload(
                operation=operation,
                response=response,
                segment=segment,
                run_id=run_row["run_id"],
                organization_id=organization_id,
                principal_id=principal_id,
                purpose=purpose,
                force_stale=response_stale,
                authorization_valid=authorization_valid,
                authorization_reason_codes=authorization_reason_codes or [],
                authorization_snapshot=authorization_snapshot or {},
                now_text=now_text,
            )
            existing = conn.execute(
                hosted.sql_query("""
                SELECT * FROM sensitive_governance_reveal_decisions
                WHERE organization_id = ?
                  AND operation = ?
                  AND response_id = ?
                  AND segment_id = ?
                  AND principal_id = ?
                """, """
                SELECT * FROM sensitive_governance_reveal_decisions
                WHERE organization_id = %s
                  AND operation = %s
                  AND response_id = %s
                  AND segment_id = %s
                  AND principal_id = %s
                """),
                (organization_id, operation, response_id, segment_id, principal_id),
            ).fetchone()
            if existing:
                existing_payload = json.loads(existing["payload_json"])
                if not existing_payload.get("allowed") or not existing_payload.get("lease_id"):
                    return existing_payload
                lease_row = conn.execute(
                    hosted.sql_query("""
                    SELECT * FROM sensitive_governance_reveal_leases
                    WHERE lease_id = ? AND organization_id = ? AND principal_id = ?
                    """, """
                    SELECT * FROM sensitive_governance_reveal_leases
                    WHERE lease_id = %s AND organization_id = %s AND principal_id = %s
                    """),
                    (existing_payload["lease_id"], organization_id, principal_id),
                ).fetchone()
                if lease_row is None:
                    expired_payload = self._expired_reveal_payload(
                        existing_payload=existing_payload,
                        reason_code="REVEAL_LEASE_MISSING",
                        now_text=now_text,
                    )
                    self._append_reveal_terminal_event(
                        conn=conn,
                        operation=operation,
                        response_id=response_id,
                        segment_id=segment_id,
                        principal_id=principal_id,
                        decision=expired_payload,
                        event_suffix="failed",
                    )
                    return expired_payload
                expires_at = datetime.fromisoformat(
                    str(lease_row["expires_at"]).replace("Z", "+00:00")
                )
                if expires_at.tzinfo is None or expires_at.utcoffset() is None:
                    expires_at = expires_at.replace(tzinfo=timezone.utc)
                if expires_at <= now or lease_row["state"] != "ISSUED":
                    expired_payload = self._expired_reveal_payload(
                        existing_payload=existing_payload,
                        reason_code="REVEAL_LEASE_EXPIRED",
                        now_text=now_text,
                    )
                    self._append_reveal_terminal_event(
                        conn=conn,
                        operation=operation,
                        response_id=response_id,
                        segment_id=segment_id,
                        principal_id=principal_id,
                        decision=expired_payload,
                        event_suffix="expired",
                    )
                    return expired_payload
                return existing_payload
            lease_payload = None
            if decision["allowed"]:
                lease_payload = self._build_reveal_lease_payload(
                    response=response,
                    segment=segment,
                    run_id=run_row["run_id"],
                    organization_id=organization_id,
                    principal_id=principal_id,
                    purpose=purpose,
                    now=now,
                )
                shared_lease = conn.execute(
                    hosted.sql_query("""SELECT * FROM sensitive_governance_reveal_leases
                       WHERE organization_id = ? AND response_id = ? AND segment_id = ?
                         AND principal_id = ? AND purpose = ?""", """SELECT * FROM sensitive_governance_reveal_leases
                       WHERE organization_id = %s AND response_id = %s AND segment_id = %s
                         AND principal_id = %s AND purpose = %s"""),
                    (organization_id, response_id, segment_id, principal_id, purpose),
                ).fetchone()
                if shared_lease is not None:
                    expires_at = datetime.fromisoformat(shared_lease["expires_at"].replace("Z", "+00:00"))
                    if expires_at.tzinfo is None or expires_at.utcoffset() is None:
                        expires_at = expires_at.replace(tzinfo=timezone.utc)
                    if expires_at <= now or shared_lease["state"] != "ISSUED":
                        expired_payload = self._expired_reveal_payload(
                            existing_payload=decision,
                            reason_code="REVEAL_LEASE_EXPIRED", now_text=now_text,
                        )
                        self._append_reveal_terminal_event(
                            conn=conn, operation=operation, response_id=response_id,
                            segment_id=segment_id, principal_id=principal_id,
                            decision=expired_payload, event_suffix="expired",
                        )
                        return expired_payload
                    lease_payload = dict(shared_lease)
                    lease_payload["clear_value"] = json.loads(shared_lease["clear_value_json"])
                else:
                    conn.execute(
                        hosted.sql_query("""
                        INSERT INTO sensitive_governance_reveal_leases (
                          lease_id, response_id, segment_id, run_id, organization_id,
                          principal_id, purpose, state, max_uses, use_count,
                          issued_at, expires_at, policy_hash, profile_hash,
                          clear_value_json, decision_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, """
                        INSERT INTO sensitive_governance_reveal_leases (
                          lease_id, response_id, segment_id, run_id, organization_id,
                          principal_id, purpose, state, max_uses, use_count,
                          issued_at, expires_at, policy_hash, profile_hash,
                          clear_value_json, decision_hash
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """),
                        (
                            lease_payload["lease_id"],
                            response_id,
                            segment_id,
                            run_row["run_id"],
                            organization_id,
                            principal_id,
                            purpose,
                            lease_payload["state"],
                            lease_payload["max_uses"],
                            lease_payload["use_count"],
                            lease_payload["issued_at"],
                            lease_payload["expires_at"],
                            lease_payload["policy_hash"],
                            lease_payload["profile_hash"],
                            canonical_json(lease_payload["clear_value"]),
                            lease_payload["decision_hash"],
                        ),
                    )
                decision["lease_id"] = lease_payload["lease_id"]
                decision["expires_at"] = lease_payload["expires_at"]
                decision["clear_value"] = lease_payload["clear_value"]["value"]
            event_refs = self._reveal_event_refs(operation=operation, allowed=decision["allowed"])
            decision["event_refs"] = event_refs
            decision_hash = canonical_hash(decision)
            decision_id = "sg_reveal_decision_" + decision_hash[:24]
            conn.execute(
                hosted.sql_query("""
                INSERT INTO sensitive_governance_reveal_decisions (
                  decision_id, operation, response_id, segment_id, run_id,
                  organization_id, principal_id, allowed, reason_codes_json,
                  lease_id, payload_json, decision_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO sensitive_governance_reveal_decisions (
                  decision_id, operation, response_id, segment_id, run_id,
                  organization_id, principal_id, allowed, reason_codes_json,
                  lease_id, payload_json, decision_hash, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """),
                (
                    decision_id,
                    operation,
                    response_id,
                    segment_id,
                    run_row["run_id"],
                    organization_id,
                    principal_id,
                    1 if decision["allowed"] else 0,
                    canonical_json(decision["reason_codes"]),
                    decision.get("lease_id"),
                    canonical_json(decision),
                    decision_hash,
                    now_text,
                ),
            )
            self._append_reveal_execution_events(
                conn=conn,
                operation=operation,
                response_id=response_id,
                segment_id=segment_id,
                principal_id=principal_id,
                decision=decision,
                decision_hash=decision_hash,
            )
        return decision

    @staticmethod
    def _reveal_event_refs(*, operation: str, allowed: bool) -> list[dict[str, str]]:
        states = ["REQUESTED"]
        states.extend(["AUTHORIZED", "DELIVERED"] if allowed else ["DENIED"])
        return [
            {
                "state": state,
                "event_id": f"sg_event_{uuid4().hex}",
                "event_type": f"sensitive_governance.{operation}_{state.lower()}",
            }
            for state in states
        ]

    def _append_reveal_execution_events(
        self,
        *,
        conn: sqlite3.Connection,
        operation: str,
        response_id: str,
        segment_id: str,
        principal_id: str,
        decision: dict[str, Any],
        decision_hash: str,
    ) -> None:
        causation_id = None
        for ref in decision["event_refs"]:
            event = event_now(
                event_id=ref["event_id"],
                event_type=ref["event_type"],
                phase=ReplayPhase.EXECUTION_TERMINAL,
                aggregate_id=response_id,
                payload={
                    "operation": operation,
                    "response_id": response_id,
                    "segment_id": segment_id,
                    "principal_id": principal_id,
                    "decision_hash": decision_hash,
                    "lease_id": decision.get("lease_id"),
                    "reason_codes": decision["reason_codes"],
                    "authorization_snapshot_hash": decision.get(
                        "authorization_snapshot_hash"
                    ),
                },
                causation_id=causation_id,
            )
            self._append_event(conn, event.to_dict())
            causation_id = event.event_id

    @staticmethod
    def _run_is_stale(completed_at: str | None, now: datetime) -> bool:
        if not completed_at:
            return True
        try:
            completed_at_dt = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        except ValueError:
            return True
        if completed_at_dt.tzinfo is None or completed_at_dt.utcoffset() is None:
            completed_at_dt = completed_at_dt.replace(tzinfo=timezone.utc)
        return completed_at_dt <= now - timedelta(minutes=30)

    @staticmethod
    def _expired_reveal_payload(
        *,
        existing_payload: dict[str, Any],
        reason_code: str,
        now_text: str,
    ) -> dict[str, Any]:
        return {
            key: value
            for key, value in existing_payload.items()
            if key not in {"clear_value", "expires_at"}
        } | {
            "allowed": False,
            "reason_codes": [reason_code],
            "created_at": now_text,
        }

    def _append_reveal_terminal_event(
        self,
        *,
        conn: sqlite3.Connection,
        operation: str,
        response_id: str,
        segment_id: str,
        principal_id: str,
        decision: dict[str, Any],
        event_suffix: str,
    ) -> None:
        event = event_now(
            event_id=f"sg_event_{uuid4().hex}",
            event_type=f"sensitive_governance.{operation}_{event_suffix}",
            phase=ReplayPhase.EXECUTION_TERMINAL,
            aggregate_id=response_id,
            payload={
                "operation": operation,
                "response_id": response_id,
                "segment_id": segment_id,
                "principal_id": principal_id,
                "reason_codes": decision["reason_codes"],
                "lease_id": decision.get("lease_id"),
            },
        )
        self._append_event(conn, event.to_dict())

    def replay_run(self, *, organization_id: str, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id, organization_id=organization_id)
        if run is None:
            raise KeyError("run not found")
        response = run["governed_response"]
        response_body = {key: value for key, value in response.items() if key != "response_hash"}
        recomputed_response_hash = canonical_hash(response_body)
        stored_response_hash = str(response.get("response_hash") or "")
        replay = {
            "run_id": run_id,
            "response": response,
            "replayed_response_hash": recomputed_response_hash,
            "governed_response_hash": run["governed_response_hash"],
            "matches": (
                recomputed_response_hash
                == stored_response_hash
                == run["governed_response_hash"]
            ),
            "pins": response.get("pins", {}),
            "policy_outcomes_hash": canonical_hash(response.get("segments", [])),
        }
        return replay

    @staticmethod
    def _build_reveal_decision_payload(
        *,
        operation: str,
        response: dict[str, Any],
        segment: dict[str, Any],
        run_id: str,
        organization_id: str,
        principal_id: str,
        purpose: str,
        force_stale: bool,
        authorization_valid: bool,
        authorization_reason_codes: list[str],
        authorization_snapshot: dict[str, Any],
        now_text: str,
    ) -> dict[str, Any]:
        action = segment.get("display_action")
        classification = segment.get("classification")
        allowed = (
            operation in {"reveal", "copy"}
            and segment.get("eligible_for_reveal") is True
            and action in {"MASK", "TOKENIZE"}
            and classification != "EXISTING_STATE_SECRET_MARKING"
            and not force_stale
            and authorization_valid
        )
        reason_codes: list[str] = []
        if not authorization_valid:
            allowed = False
            reason_codes.extend(authorization_reason_codes or ["REVEAL_AUTHORIZATION_DENIED"])
        if action == "NEVER_RENDER" or classification == "EXISTING_STATE_SECRET_MARKING":
            allowed = False
            reason_codes.append("NEVER_RENDER_SEGMENT_REVEAL_DENIED")
        if force_stale:
            allowed = False
            reason_codes.append("RESPONSE_STALE")
        if not segment.get("eligible_for_reveal") and operation == "reveal":
            reason_codes.append("SEGMENT_NOT_REVEAL_ELIGIBLE")
        if allowed:
            reason_codes.append("REVEAL_AUTHORIZED")
        return {
            "operation": operation,
            "response_id": response["response_id"],
            "segment_id": segment["segment_id"],
            "run_id": run_id,
            "organization_id": organization_id,
            "principal_id": principal_id,
            "purpose": purpose,
            "allowed": allowed,
            "reason_codes": reason_codes,
            "authorization_snapshot_ref": authorization_snapshot.get(
                "authorization_snapshot_ref"
            ),
            "authorization_snapshot_hash": (
                canonical_hash(authorization_snapshot)
                if authorization_snapshot
                else None
            ),
            "created_at": now_text,
        }

    def _restricted_payload_value(
        self,
        *,
        organization_id: str,
        restricted_payload_ref: str | None,
    ) -> Any | None:
        if not restricted_payload_ref:
            return None
        with get_conn() as conn:
            self.ensure_schema(conn)
            row = conn.execute(
                hosted.sql_query("""
                SELECT clear_value_json, content_hash
                FROM sensitive_governance_restricted_payloads
                WHERE restricted_payload_ref = ? AND organization_id = ?
                """, """
                SELECT clear_value_json, content_hash
                FROM sensitive_governance_restricted_payloads
                WHERE restricted_payload_ref = %s AND organization_id = %s
                """),
                (restricted_payload_ref, organization_id),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["clear_value_json"])
        if canonical_hash(payload) != row["content_hash"]:
            raise ValueError("restricted payload integrity mismatch")
        return payload.get("value")

    def _build_reveal_lease_payload(
        self,
        *,
        response: dict[str, Any],
        segment: dict[str, Any],
        run_id: str,
        organization_id: str,
        principal_id: str,
        purpose: str,
        now: datetime,
    ) -> dict[str, Any]:
        expires_at = now + timedelta(seconds=20)
        restricted_ref = segment.get("restricted_payload_ref")
        resolved_value = self._restricted_payload_value(
            organization_id=organization_id,
            restricted_payload_ref=restricted_ref,
        )
        clear_value = {
            "value": (
                resolved_value
                if resolved_value is not None
                else f"authorized reveal for {segment['segment_id']}"
            ),
            "restricted_payload_ref": restricted_ref,
        }
        lease_body = {
            "response_id": response["response_id"],
            "segment_id": segment["segment_id"],
            "run_id": run_id,
            "organization_id": organization_id,
            "principal_id": principal_id,
            "purpose": purpose,
            "issued_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
            "policy_hash": canonical_hash(response.get("governance_summary", {})),
            "profile_hash": canonical_hash(response.get("pins", {})),
        }
        return {
            "lease_id": "sg_reveal_lease_" + canonical_hash(lease_body)[:24],
            **lease_body,
            "state": "ISSUED",
            "max_uses": 1,
            "use_count": 0,
            "clear_value": clear_value,
            "decision_hash": canonical_hash(lease_body),
        }

    def list_events(self, aggregate_id: str) -> list[dict[str, Any]]:
        with get_conn() as conn:
            self.ensure_schema(conn)
            rows = conn.execute(
                hosted.sql_query("""
                SELECT * FROM sensitive_governance_events
                WHERE aggregate_id = ?
                ORDER BY occurred_at, event_id
                """, """
                SELECT * FROM sensitive_governance_events
                WHERE aggregate_id = %s
                ORDER BY occurred_at, event_id
                """),
                (aggregate_id,),
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def _append_event(self, conn: sqlite3.Connection, event: dict[str, Any]) -> None:
        conn.execute(
            hosted.sql_query("""
            INSERT INTO sensitive_governance_events (
              event_id, aggregate_id, event_type, replay_phase,
              occurred_at, payload_json, event_hash, causation_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, """
            INSERT INTO sensitive_governance_events (
              event_id, aggregate_id, event_type, replay_phase,
              occurred_at, payload_json, event_hash, causation_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """),
            (
                event["event_id"],
                event["aggregate_id"],
                event["event_type"],
                event["phase"],
                event["occurred_at"],
                canonical_json(event["payload"]),
                event["event_hash"],
                event.get("causation_id"),
            ),
        )

    @staticmethod
    def _assessment_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "assessment_id": row["assessment_id"],
            "organization_id": row["organization_id"],
            "user_id": row["user_id"],
            "workspace_id": row["workspace_id"],
            "session_id": row["session_id"],
            "contract_version": row["contract_version"],
            "state": row["state"],
            "source_kind": row["source_kind"],
            "purpose": row["purpose"],
            "destination": row["destination"],
            "request_hash": row["request_hash"],
            "assessment_hash": row["assessment_hash"],
            "prepared_payload": json.loads(row["prepared_payload_json"]),
            "created_at": row["created_at"],
            "consumed_at": row["consumed_at"],
            "consumed_by_run_id": row["consumed_by_run_id"],
        }

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "run_id": row["run_id"],
            "assessment_id": row["assessment_id"],
            "organization_id": row["organization_id"],
            "user_id": row["user_id"],
            "workspace_id": row["workspace_id"],
            "contract_version": row["contract_version"],
            "state": row["state"],
            "governed_response": json.loads(row["governed_response_json"]),
            "governed_response_hash": row["governed_response_hash"],
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
        }

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "event_id": row["event_id"],
            "aggregate_id": row["aggregate_id"],
            "event_type": row["event_type"],
            "phase": row["replay_phase"],
            "occurred_at": row["occurred_at"],
            "payload": json.loads(row["payload_json"]),
            "event_hash": row["event_hash"],
            "causation_id": row["causation_id"],
        }
