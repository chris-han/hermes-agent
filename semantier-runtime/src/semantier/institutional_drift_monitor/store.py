from __future__ import annotations

from storage import hosted_postgres as hosted

import json
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash
from eos.db import get_conn, now_iso

from .contracts import (
    DriftMonitorPolicy,
    DriftObservation,
    DriftValidationError,
    PolicyState,
    canonical_window,
    parse_utc,
)


POLICY_ACTIVE_REGISTRATION_FORBIDDEN = "POLICY_ACTIVE_REGISTRATION_FORBIDDEN"
POLICY_CERTIFICATION_REQUIRED = "POLICY_CERTIFICATION_REQUIRED"
POLICY_APPROVER_ROLE_REQUIRED = "POLICY_APPROVER_ROLE_REQUIRED"
BASELINE_AMBIGUOUS = "BASELINE_AMBIGUOUS"
BASELINE_UNAVAILABLE = "BASELINE_UNAVAILABLE"
POLICY_SCOPE_AMBIGUITY = "POLICY_SCOPE_AMBIGUITY"
POLICY_NOT_FOUND = "POLICY_NOT_FOUND"
LEASE_ALREADY_ACTIVE = "LEASE_ALREADY_ACTIVE"


class InstitutionalDriftStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path is not None else None
        self.ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        if hosted.enabled():
            return get_conn(self._db_path)
        if self._db_path is None:
            return get_conn()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            if hosted.is_postgres(conn):
                hosted.require_tables(conn, 'institutional_drift_baseline_snapshots', 'institutional_drift_findings', 'institutional_drift_learning_signals', 'institutional_drift_observations', 'institutional_drift_recovery_verifications', 'institutional_drift_routing_events', 'institutional_drift_runs', 'metadata_drift_monitor_policies', 'metadata_drift_monitor_schedules', 'metadata_drift_policy_events')
                return
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata_drift_monitor_policies (
                  policy_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT,
                  asset_selector_json TEXT NOT NULL,
                  monitored_dimensions_json TEXT NOT NULL,
                  cadence TEXT NOT NULL,
                  event_subscription_mode TEXT NOT NULL,
                  baseline_resolution_mode TEXT NOT NULL,
                  severity_thresholds_json TEXT NOT NULL,
                  allowed_self_healing_level TEXT NOT NULL,
                  owner_role TEXT NOT NULL,
                  reviewer_role TEXT NOT NULL,
                  governance_state TEXT NOT NULL,
                  policy_version TEXT NOT NULL,
                  effective_from TEXT NOT NULL,
                  effective_to TEXT,
                  policy_hash TEXT NOT NULL,
                  created_by TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS metadata_drift_policy_events (
                  event_id TEXT PRIMARY KEY,
                  policy_id TEXT NOT NULL,
                  event_type TEXT NOT NULL,
                  actor_ref TEXT NOT NULL,
                  actor_role TEXT NOT NULL,
                  from_state TEXT,
                  to_state TEXT NOT NULL,
                  evidence_hash TEXT NOT NULL,
                  event_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(policy_id) REFERENCES metadata_drift_monitor_policies(policy_id)
                );

                CREATE TABLE IF NOT EXISTS institutional_drift_baseline_snapshots (
                  baseline_snapshot_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT,
                  asset_ref TEXT NOT NULL,
                  metadata_asset_version_ref TEXT NOT NULL,
                  activation_set_snapshot_id TEXT NOT NULL,
                  resolved_activation_set_hash TEXT NOT NULL,
                  ontology_version_ref TEXT NOT NULL,
                  semantic_contract_version_ref TEXT NOT NULL,
                  data_contract_version_ref TEXT NOT NULL,
                  policy_version_refs_json TEXT NOT NULL,
                  source_hashes_json TEXT NOT NULL,
                  effective_at TEXT NOT NULL,
                  baseline_snapshot_hash TEXT NOT NULL,
                  lifecycle_state TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS institutional_drift_observations (
                  observation_id TEXT PRIMARY KEY,
                  idempotency_key TEXT NOT NULL UNIQUE,
                  policy_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT,
                  asset_ref TEXT NOT NULL,
                  observation_window TEXT NOT NULL,
                  normalized_payload_json TEXT NOT NULL,
                  source_anchors_json TEXT NOT NULL,
                  collector_version TEXT NOT NULL,
                  observation_hash TEXT NOT NULL,
                  prior_observation_id TEXT,
                  complete INTEGER NOT NULL CHECK(complete IN (0, 1)),
                  created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS institutional_drift_runs (
                  run_id TEXT PRIMARY KEY,
                  idempotency_key TEXT NOT NULL UNIQUE,
                  policy_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT,
                  baseline_snapshot_id TEXT NOT NULL,
                  observation_id TEXT NOT NULL,
                  status TEXT NOT NULL,
                  lease_owner TEXT,
                  lease_expires_at TEXT,
                  created_at TEXT NOT NULL,
                  completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS institutional_drift_findings (
                  finding_id TEXT PRIMARY KEY,
                  run_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT,
                  drift_family TEXT NOT NULL,
                  drift_subtype TEXT NOT NULL,
                  baseline_snapshot_id TEXT NOT NULL,
                  observation_id TEXT NOT NULL,
                  deterministic_differences_json TEXT NOT NULL,
                  semantic_hypotheses_json TEXT NOT NULL,
                  evidence_completeness TEXT NOT NULL,
                  authority_impact_json TEXT NOT NULL,
                  selected_learning_route TEXT NOT NULL,
                  reason_codes_json TEXT NOT NULL,
                  severity TEXT NOT NULL,
                  urgency TEXT NOT NULL,
                  lifecycle_state TEXT NOT NULL,
                  finding_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS institutional_drift_routing_events (
                  event_id TEXT PRIMARY KEY,
                  finding_id TEXT NOT NULL,
                  from_state TEXT,
                  to_state TEXT NOT NULL,
                  route TEXT NOT NULL,
                  reason_codes_json TEXT NOT NULL,
                  actor_ref TEXT NOT NULL,
                  event_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS institutional_drift_learning_signals (
                  signal_id TEXT PRIMARY KEY,
                  finding_id TEXT NOT NULL,
                  signal_type TEXT NOT NULL,
                  lifecycle_state TEXT NOT NULL,
                  candidate_ref TEXT,
                  approval_ref TEXT,
                  activation_ref TEXT,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS institutional_drift_recovery_verifications (
                  verification_id TEXT PRIMARY KEY,
                  finding_id TEXT NOT NULL,
                  remediation_ref TEXT NOT NULL,
                  deployed_artifact_refs_json TEXT NOT NULL,
                  pre_observation_hash TEXT NOT NULL,
                  post_observation_hash TEXT NOT NULL,
                  replay_comparison_result TEXT NOT NULL,
                  non_interference_result TEXT NOT NULL,
                  expected_outcome TEXT NOT NULL,
                  observed_outcome TEXT NOT NULL,
                  regression_result TEXT NOT NULL,
                  verification_status TEXT NOT NULL,
                  evidence_json TEXT NOT NULL,
                  verification_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS metadata_drift_monitor_schedules (
                  schedule_id TEXT PRIMARY KEY,
                  policy_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT,
                  cadence TEXT NOT NULL,
                  cron_ref TEXT NOT NULL,
                  status TEXT NOT NULL,
                  schedule_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  UNIQUE(policy_id, status)
                );
                """
            )

    def register_policy(
        self,
        *,
        organization_id: str,
        workspace_id: str | None,
        asset_selector: dict[str, Any],
        monitored_dimensions: list[str],
        cadence: str,
        baseline_resolution_mode: str = "governed_active",
        severity_thresholds: dict[str, Any] | None = None,
        allowed_self_healing_level: str = "H1",
        owner_role: str = "data_owner",
        reviewer_role: str = "governance_reviewer",
        governance_state: str = "DRAFT",
        policy_version: str = "v1",
        effective_from: str,
        effective_to: str | None = None,
        created_by: str,
    ) -> DriftMonitorPolicy:
        if governance_state == PolicyState.ACTIVE.value:
            raise ValueError(POLICY_ACTIVE_REGISTRATION_FORBIDDEN)
        if governance_state not in {PolicyState.DRAFT.value, PolicyState.PROPOSED.value}:
            raise ValueError("unsupported policy registration state")
        parse_utc(effective_from)
        if effective_to:
            parse_utc(effective_to)
        payload = {
            "schema_version": "metadata_drift_monitor_policy.v1",
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "asset_selector": asset_selector,
            "monitored_dimensions": sorted(monitored_dimensions),
            "cadence": cadence,
            "event_subscription_mode": "cron_or_event",
            "baseline_resolution_mode": baseline_resolution_mode,
            "severity_thresholds": severity_thresholds or {},
            "allowed_self_healing_level": allowed_self_healing_level,
            "owner_role": owner_role,
            "reviewer_role": reviewer_role,
            "governance_state": governance_state,
            "policy_version": policy_version,
            "effective_from": effective_from,
            "effective_to": effective_to,
        }
        policy_hash = content_hash(payload)
        policy_id = f"metadata_drift_policy_{uuid4().hex}"
        with self._connect() as conn:
            conn.execute(
                hosted.sql_query("""
                INSERT INTO metadata_drift_monitor_policies (
                  policy_id, organization_id, workspace_id, asset_selector_json,
                  monitored_dimensions_json, cadence, event_subscription_mode,
                  baseline_resolution_mode, severity_thresholds_json,
                  allowed_self_healing_level, owner_role, reviewer_role,
                  governance_state, policy_version, effective_from, effective_to,
                  policy_hash, created_by, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO metadata_drift_monitor_policies (
                  policy_id, organization_id, workspace_id, asset_selector_json,
                  monitored_dimensions_json, cadence, event_subscription_mode,
                  baseline_resolution_mode, severity_thresholds_json,
                  allowed_self_healing_level, owner_role, reviewer_role,
                  governance_state, policy_version, effective_from, effective_to,
                  policy_hash, created_by, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """),
                (
                    policy_id,
                    organization_id,
                    workspace_id,
                    canonical_json(asset_selector),
                    canonical_json(sorted(monitored_dimensions)),
                    cadence,
                    "cron_or_event",
                    baseline_resolution_mode,
                    canonical_json(severity_thresholds or {}),
                    allowed_self_healing_level,
                    owner_role,
                    reviewer_role,
                    governance_state,
                    policy_version,
                    effective_from,
                    effective_to,
                    policy_hash,
                    created_by,
                    now_iso(),
                ),
            )
            self._record_policy_event(
                conn,
                policy_id=policy_id,
                event_type="PROPOSE",
                actor_ref=created_by,
                actor_role=owner_role,
                from_state=None,
                to_state=governance_state,
                evidence_hash=policy_hash,
                event_payload={"policy_hash": policy_hash},
            )
        return DriftMonitorPolicy(
            policy_id=policy_id,
            organization_id=organization_id,
            workspace_id=workspace_id,
            asset_selector=asset_selector,
            monitored_dimensions=sorted(monitored_dimensions),
            cadence=cadence,
            baseline_resolution_mode=baseline_resolution_mode,
            allowed_self_healing_level=allowed_self_healing_level,
            owner_role=owner_role,
            reviewer_role=reviewer_role,
            governance_state=governance_state,
            policy_version=policy_version,
            effective_from=effective_from,
            effective_to=effective_to,
            policy_hash=policy_hash,
        )

    def certify_policy(self, *, policy_id: str, reviewer_ref: str, reviewer_role: str, evidence_hash: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = self._policy_row(conn, policy_id)
            if row["governance_state"] not in {PolicyState.DRAFT.value, PolicyState.PROPOSED.value}:
                raise ValueError("policy is not certifiable")
            conn.execute(
                hosted.sql_query("UPDATE metadata_drift_monitor_policies SET governance_state = 'CERTIFIED' WHERE policy_id = ?", "UPDATE metadata_drift_monitor_policies SET governance_state = 'CERTIFIED' WHERE policy_id = %s"),
                (policy_id,),
            )
            event = self._record_policy_event(
                conn,
                policy_id=policy_id,
                event_type="CERTIFY",
                actor_ref=reviewer_ref,
                actor_role=reviewer_role,
                from_state=row["governance_state"],
                to_state=PolicyState.CERTIFIED.value,
                evidence_hash=evidence_hash,
                event_payload={"evidence_hash": evidence_hash},
            )
        return event

    def activate_policy(self, *, policy_id: str, approver_ref: str, approver_role: str, certification_event_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = self._policy_row(conn, policy_id)
            cert = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM metadata_drift_policy_events
                WHERE event_id = ? AND policy_id = ? AND event_type = 'CERTIFY'
                """, """
                SELECT *
                FROM metadata_drift_policy_events
                WHERE event_id = %s AND policy_id = %s AND event_type = 'CERTIFY'
                """),
                (certification_event_id, policy_id),
            ).fetchone()
            if row["governance_state"] == PolicyState.ACTIVE.value and cert is not None:
                self._ensure_single_schedule(conn, row)
                return {
                    "event_id": certification_event_id,
                    "policy_id": policy_id,
                    "event_type": "ACTIVATE",
                    "to_state": PolicyState.ACTIVE.value,
                    "idempotent_replay": True,
                }
            if row["governance_state"] != PolicyState.CERTIFIED.value or cert is None:
                raise ValueError(POLICY_CERTIFICATION_REQUIRED)
            conn.execute(
                hosted.sql_query("""
                UPDATE metadata_drift_monitor_policies
                SET governance_state = 'ACTIVE'
                WHERE policy_id = ?
                """, """
                UPDATE metadata_drift_monitor_policies
                SET governance_state = 'ACTIVE'
                WHERE policy_id = %s
                """),
                (policy_id,),
            )
            event = self._record_policy_event(
                conn,
                policy_id=policy_id,
                event_type="ACTIVATE",
                actor_ref=approver_ref,
                actor_role=approver_role,
                from_state=PolicyState.CERTIFIED.value,
                to_state=PolicyState.ACTIVE.value,
                evidence_hash=row["policy_hash"],
                event_payload={"certification_event_id": certification_event_id},
            )
            self._ensure_single_schedule(conn, row)
        return event

    def resolve_policy(self, *, organization_id: str, workspace_id: str | None, as_of: str, asset_ref: str) -> dict[str, Any]:
        parse_utc(as_of)
        with self._connect() as conn:
            rows = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM metadata_drift_monitor_policies
                WHERE organization_id = ?
                  AND governance_state = 'ACTIVE'
                """, """
                SELECT *
                FROM metadata_drift_monitor_policies
                WHERE organization_id = %s
                  AND governance_state = 'ACTIVE'
                """),
                (organization_id,),
            ).fetchall()
        scoped = [
            row for row in rows
            if row["workspace_id"] == workspace_id
            and _selector_matches(json.loads(row["asset_selector_json"]), asset_ref)
            and _effective(row, as_of)
        ]
        org_rows = [
            row for row in rows
            if row["workspace_id"] is None
            and _selector_matches(json.loads(row["asset_selector_json"]), asset_ref)
            and _effective(row, as_of)
        ]
        candidates = scoped or org_rows
        if not candidates:
            return {"ok": False, "error_code": POLICY_NOT_FOUND}
        if len(candidates) > 1:
            return {"ok": False, "error_code": POLICY_SCOPE_AMBIGUITY}
        return {"ok": True, "policy": self._policy_dict(candidates[0])}

    def create_baseline_snapshot(
        self,
        *,
        organization_id: str,
        workspace_id: str | None,
        asset_ref: str,
        metadata_asset_version_ref: str,
        activation_set_snapshot_id: str,
        resolved_activation_set_hash: str,
        ontology_version_ref: str,
        semantic_contract_version_ref: str,
        data_contract_version_ref: str,
        policy_version_refs: list[str],
        source_hashes: dict[str, str],
        effective_at: str,
        lifecycle_state: str = "ACTIVE",
    ) -> dict[str, Any]:
        parse_utc(effective_at)
        payload = {
            "schema_version": "institutional_drift_baseline_snapshot.v1",
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "asset_ref": asset_ref,
            "metadata_asset_version_ref": metadata_asset_version_ref,
            "activation_set_snapshot_id": activation_set_snapshot_id,
            "resolved_activation_set_hash": resolved_activation_set_hash,
            "ontology_version_ref": ontology_version_ref,
            "semantic_contract_version_ref": semantic_contract_version_ref,
            "data_contract_version_ref": data_contract_version_ref,
            "policy_version_refs": sorted(policy_version_refs),
            "source_hashes": source_hashes,
            "effective_at": effective_at,
            "lifecycle_state": lifecycle_state,
        }
        snapshot_hash = content_hash(payload)
        snapshot_id = f"institutional_drift_baseline_{uuid4().hex}"
        with self._connect() as conn:
            conn.execute(
                hosted.sql_query("""
                INSERT INTO institutional_drift_baseline_snapshots (
                  baseline_snapshot_id, organization_id, workspace_id, asset_ref,
                  metadata_asset_version_ref, activation_set_snapshot_id,
                  resolved_activation_set_hash, ontology_version_ref,
                  semantic_contract_version_ref, data_contract_version_ref,
                  policy_version_refs_json, source_hashes_json, effective_at,
                  baseline_snapshot_hash, lifecycle_state, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO institutional_drift_baseline_snapshots (
                  baseline_snapshot_id, organization_id, workspace_id, asset_ref,
                  metadata_asset_version_ref, activation_set_snapshot_id,
                  resolved_activation_set_hash, ontology_version_ref,
                  semantic_contract_version_ref, data_contract_version_ref,
                  policy_version_refs_json, source_hashes_json, effective_at,
                  baseline_snapshot_hash, lifecycle_state, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """),
                (
                    snapshot_id, organization_id, workspace_id, asset_ref,
                    metadata_asset_version_ref, activation_set_snapshot_id,
                    resolved_activation_set_hash, ontology_version_ref,
                    semantic_contract_version_ref, data_contract_version_ref,
                    canonical_json(sorted(policy_version_refs)),
                    canonical_json(source_hashes), effective_at, snapshot_hash,
                    lifecycle_state, now_iso(),
                ),
            )
        return {**payload, "baseline_snapshot_id": snapshot_id, "baseline_snapshot_hash": snapshot_hash}

    def resolve_baseline(self, *, organization_id: str, workspace_id: str | None, asset_ref: str, effective_at: str) -> dict[str, Any]:
        parse_utc(effective_at)
        with self._connect() as conn:
            rows = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM institutional_drift_baseline_snapshots
                WHERE organization_id = ?
                  AND asset_ref = ?
                  AND lifecycle_state = 'ACTIVE'
                  AND effective_at <= ?
                """, """
                SELECT *
                FROM institutional_drift_baseline_snapshots
                WHERE organization_id = %s
                  AND asset_ref = %s
                  AND lifecycle_state = 'ACTIVE'
                  AND effective_at <= %s
                """),
                (organization_id, asset_ref, effective_at),
            ).fetchall()
        scoped = [row for row in rows if row["workspace_id"] == workspace_id]
        org_rows = [row for row in rows if row["workspace_id"] is None]
        candidates = scoped or org_rows
        if not candidates:
            return {"ok": False, "error_code": BASELINE_UNAVAILABLE}
        latest_time = max(row["effective_at"] for row in candidates)
        latest = [row for row in candidates if row["effective_at"] == latest_time]
        if len(latest) != 1:
            return {"ok": False, "error_code": BASELINE_AMBIGUOUS}
        return {"ok": True, "baseline": self._baseline_dict(latest[0])}

    def record_observation(
        self,
        *,
        policy_id: str,
        organization_id: str,
        workspace_id: str | None,
        asset_ref: str,
        observed_from: str,
        observed_to: str,
        payload: dict[str, Any],
        source_anchors: list[str],
        collector_version: str,
        complete: bool = True,
    ) -> DriftObservation:
        window = canonical_window(observed_from, observed_to)
        observation_hash = content_hash(
            {"asset_ref": asset_ref, "window": window, "payload": payload, "source_anchors": source_anchors}
        )
        key = content_hash({"policy_id": policy_id, "organization_id": organization_id, "workspace_id": workspace_id, "asset_ref": asset_ref, "window": window})
        with self._connect() as conn:
            existing = conn.execute(
                hosted.sql_query("SELECT * FROM institutional_drift_observations WHERE idempotency_key = ?", 'SELECT * FROM institutional_drift_observations WHERE idempotency_key = %s'),
                (key,),
            ).fetchone()
            if existing is not None:
                return self._observation_obj(existing)
            observation_id = f"institutional_drift_observation_{uuid4().hex}"
            conn.execute(
                hosted.sql_query("""
                INSERT INTO institutional_drift_observations (
                  observation_id, idempotency_key, policy_id, organization_id, workspace_id,
                  asset_ref, observation_window, normalized_payload_json,
                  source_anchors_json, collector_version, observation_hash,
                  prior_observation_id, complete, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                """, """
                INSERT INTO institutional_drift_observations (
                  observation_id, idempotency_key, policy_id, organization_id, workspace_id,
                  asset_ref, observation_window, normalized_payload_json,
                  source_anchors_json, collector_version, observation_hash,
                  prior_observation_id, complete, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL, %s, %s)
                """),
                (
                    observation_id, key, policy_id, organization_id, workspace_id,
                    asset_ref, window, canonical_json(payload), canonical_json(source_anchors),
                    collector_version, observation_hash, 1 if complete else 0, now_iso(),
                ),
            )
            row = conn.execute(
                hosted.sql_query("SELECT * FROM institutional_drift_observations WHERE observation_id = ?", 'SELECT * FROM institutional_drift_observations WHERE observation_id = %s'),
                (observation_id,),
            ).fetchone()
        return self._observation_obj(row)

    def create_or_resume_run(
        self,
        *,
        policy_id: str,
        organization_id: str,
        workspace_id: str | None,
        baseline_snapshot_id: str,
        baseline_snapshot_hash: str,
        observation_id: str,
        observation_window: str,
    ) -> dict[str, Any]:
        key = content_hash(
            {
                "organization_id": organization_id,
                "workspace_id": workspace_id,
                "policy_id": policy_id,
                "observation_window": observation_window,
                "baseline_snapshot_hash": baseline_snapshot_hash,
            }
        )
        with self._connect() as conn:
            existing = conn.execute(
                hosted.sql_query("SELECT * FROM institutional_drift_runs WHERE idempotency_key = ?", 'SELECT * FROM institutional_drift_runs WHERE idempotency_key = %s'),
                (key,),
            ).fetchone()
            if existing is not None:
                return dict(existing)
            run_id = f"institutional_drift_run_{uuid4().hex}"
            conn.execute(
                hosted.sql_query("""
                INSERT INTO institutional_drift_runs (
                  run_id, idempotency_key, policy_id, organization_id, workspace_id,
                  baseline_snapshot_id, observation_id, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
                """, """
                INSERT INTO institutional_drift_runs (
                  run_id, idempotency_key, policy_id, organization_id, workspace_id,
                  baseline_snapshot_id, observation_id, status, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'PENDING', %s)
                """),
                (run_id, key, policy_id, organization_id, workspace_id, baseline_snapshot_id, observation_id, now_iso()),
            )
            row = conn.execute(hosted.sql_query("SELECT * FROM institutional_drift_runs WHERE run_id = ?", 'SELECT * FROM institutional_drift_runs WHERE run_id = %s'), (run_id,)).fetchone()
        return dict(row)

    def claim_run_lease(self, *, run_id: str, lease_owner: str, lease_expires_at: str) -> dict[str, Any]:
        parse_utc(lease_expires_at)
        now = now_iso()
        with self._connect() as conn:
            row = conn.execute(hosted.sql_query("SELECT * FROM institutional_drift_runs WHERE run_id = ?", 'SELECT * FROM institutional_drift_runs WHERE run_id = %s'), (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            if row["lease_expires_at"] and row["lease_expires_at"] > now and row["status"] == "RUNNING":
                raise ValueError(LEASE_ALREADY_ACTIVE)
            conn.execute(
                hosted.sql_query("""
                UPDATE institutional_drift_runs
                SET status = 'RUNNING', lease_owner = ?, lease_expires_at = ?
                WHERE run_id = ?
                """, """
                UPDATE institutional_drift_runs
                SET status = 'RUNNING', lease_owner = %s, lease_expires_at = %s
                WHERE run_id = %s
                """),
                (lease_owner, lease_expires_at, run_id),
            )
            updated = conn.execute(hosted.sql_query("SELECT * FROM institutional_drift_runs WHERE run_id = ?", 'SELECT * FROM institutional_drift_runs WHERE run_id = %s'), (run_id,)).fetchone()
        return dict(updated)

    def persist_finding(
        self,
        *,
        run_id: str,
        organization_id: str,
        workspace_id: str | None,
        drift_family: str,
        drift_subtype: str,
        baseline_snapshot_id: str,
        observation_id: str,
        deterministic_differences: list[dict[str, Any]],
        semantic_hypotheses: list[dict[str, Any]],
        evidence_completeness: str,
        authority_impact: dict[str, Any],
        selected_learning_route: str,
        reason_codes: list[str],
        severity: str,
        urgency: str,
        lifecycle_state: str = "FINDING_CLASSIFIED",
    ) -> dict[str, Any]:
        payload = {
            "schema_version": "institutional_drift_finding.v1",
            "run_id": run_id,
            "drift_family": drift_family,
            "drift_subtype": drift_subtype,
            "baseline_snapshot_id": baseline_snapshot_id,
            "observation_id": observation_id,
            "deterministic_differences": deterministic_differences,
            "semantic_hypotheses": semantic_hypotheses,
            "authority_impact": authority_impact,
            "selected_learning_route": selected_learning_route,
            "reason_codes": reason_codes,
        }
        finding_hash = content_hash(payload)
        finding_id = f"institutional_drift_finding_{uuid4().hex}"
        with self._connect() as conn:
            conn.execute(
                hosted.sql_query("""
                INSERT INTO institutional_drift_findings (
                  finding_id, run_id, organization_id, workspace_id, drift_family,
                  drift_subtype, baseline_snapshot_id, observation_id,
                  deterministic_differences_json, semantic_hypotheses_json,
                  evidence_completeness, authority_impact_json,
                  selected_learning_route, reason_codes_json, severity, urgency,
                  lifecycle_state, finding_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO institutional_drift_findings (
                  finding_id, run_id, organization_id, workspace_id, drift_family,
                  drift_subtype, baseline_snapshot_id, observation_id,
                  deterministic_differences_json, semantic_hypotheses_json,
                  evidence_completeness, authority_impact_json,
                  selected_learning_route, reason_codes_json, severity, urgency,
                  lifecycle_state, finding_hash, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """),
                (
                    finding_id, run_id, organization_id, workspace_id, drift_family,
                    drift_subtype, baseline_snapshot_id, observation_id,
                    canonical_json(deterministic_differences),
                    canonical_json(semantic_hypotheses), evidence_completeness,
                    canonical_json(authority_impact), selected_learning_route,
                    canonical_json(reason_codes), severity, urgency, lifecycle_state,
                    finding_hash, now_iso(),
                ),
            )
            event_hash = content_hash({"finding_id": finding_id, "route": selected_learning_route, "reason_codes": reason_codes})
            conn.execute(
                hosted.sql_query("""
                INSERT INTO institutional_drift_routing_events (
                  event_id, finding_id, from_state, to_state, route,
                  reason_codes_json, actor_ref, event_hash, created_at
                ) VALUES (?, ?, NULL, ?, ?, ?, 'institutional_drift_service', ?, ?)
                """, """
                INSERT INTO institutional_drift_routing_events (
                  event_id, finding_id, from_state, to_state, route,
                  reason_codes_json, actor_ref, event_hash, created_at
                ) VALUES (%s, %s, NULL, %s, %s, %s, 'institutional_drift_service', %s, %s)
                """),
                (
                    f"institutional_drift_route_{uuid4().hex}", finding_id,
                    lifecycle_state, selected_learning_route, canonical_json(reason_codes),
                    event_hash, now_iso(),
                ),
            )
            signal_type = (
                "institutional_learning_signal"
                if selected_learning_route == "LOOP_B_INSTITUTIONAL"
                else "operational_learning_signal"
                if selected_learning_route == "LOOP_A_OPERATIONAL"
                else None
            )
            if signal_type:
                signal_payload = {"finding_id": finding_id, "signal_type": signal_type, "route": selected_learning_route}
                conn.execute(
                    hosted.sql_query("""
                    INSERT INTO institutional_drift_learning_signals (
                      signal_id, finding_id, signal_type, lifecycle_state,
                      candidate_ref, approval_ref, activation_ref, content_hash, created_at
                    ) VALUES (?, ?, ?, 'OPEN', NULL, NULL, NULL, ?, ?)
                    """, """
                    INSERT INTO institutional_drift_learning_signals (
                      signal_id, finding_id, signal_type, lifecycle_state,
                      candidate_ref, approval_ref, activation_ref, content_hash, created_at
                    ) VALUES (%s, %s, %s, 'OPEN', NULL, NULL, NULL, %s, %s)
                    """),
                    (
                        f"institutional_drift_signal_{uuid4().hex}", finding_id,
                        signal_type, content_hash(signal_payload), now_iso(),
                    ),
                )
            conn.execute(
                hosted.sql_query("""
                UPDATE institutional_drift_runs
                SET status = 'COMPLETED', completed_at = ?, lease_owner = NULL, lease_expires_at = NULL
                WHERE run_id = ?
                """, """
                UPDATE institutional_drift_runs
                SET status = 'COMPLETED', completed_at = %s, lease_owner = NULL, lease_expires_at = NULL
                WHERE run_id = %s
                """),
                (now_iso(), run_id),
            )
        return {**payload, "finding_id": finding_id, "finding_hash": finding_hash, "lifecycle_state": lifecycle_state}

    def record_recovery_verification(
        self,
        *,
        finding_id: str,
        remediation_ref: str,
        deployed_artifact_refs: list[str],
        pre_observation_hash: str,
        post_observation_hash: str,
        replay_comparison_result: str,
        non_interference_result: str,
        expected_outcome: str,
        observed_outcome: str,
        regression_result: str,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        status = (
            "PASS"
            if replay_comparison_result == "MATCH"
            and non_interference_result == "PASS"
            and expected_outcome == observed_outcome
            and regression_result == "NO_REGRESSION"
            else "ESCALATE"
        )
        payload = {
            "finding_id": finding_id,
            "remediation_ref": remediation_ref,
            "deployed_artifact_refs": deployed_artifact_refs,
            "pre_observation_hash": pre_observation_hash,
            "post_observation_hash": post_observation_hash,
            "replay_comparison_result": replay_comparison_result,
            "non_interference_result": non_interference_result,
            "expected_outcome": expected_outcome,
            "observed_outcome": observed_outcome,
            "regression_result": regression_result,
            "verification_status": status,
            "evidence": evidence,
        }
        verification_hash = content_hash(payload)
        verification_id = f"institutional_drift_recovery_{uuid4().hex}"
        with self._connect() as conn:
            conn.execute(
                hosted.sql_query("""
                INSERT INTO institutional_drift_recovery_verifications (
                  verification_id, finding_id, remediation_ref,
                  deployed_artifact_refs_json, pre_observation_hash,
                  post_observation_hash, replay_comparison_result,
                  non_interference_result, expected_outcome, observed_outcome,
                  regression_result, verification_status, evidence_json,
                  verification_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO institutional_drift_recovery_verifications (
                  verification_id, finding_id, remediation_ref,
                  deployed_artifact_refs_json, pre_observation_hash,
                  post_observation_hash, replay_comparison_result,
                  non_interference_result, expected_outcome, observed_outcome,
                  regression_result, verification_status, evidence_json,
                  verification_hash, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """),
                (
                    verification_id, finding_id, remediation_ref,
                    canonical_json(deployed_artifact_refs), pre_observation_hash,
                    post_observation_hash, replay_comparison_result,
                    non_interference_result, expected_outcome, observed_outcome,
                    regression_result, status, canonical_json(evidence),
                    verification_hash, now_iso(),
                ),
            )
        return {**payload, "verification_id": verification_id, "verification_hash": verification_hash}

    def audit_export(self, *, finding_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            finding = conn.execute(hosted.sql_query("SELECT * FROM institutional_drift_findings WHERE finding_id = ?", 'SELECT * FROM institutional_drift_findings WHERE finding_id = %s'), (finding_id,)).fetchone()
            if finding is None:
                raise KeyError(finding_id)
            observation = conn.execute(hosted.sql_query("SELECT * FROM institutional_drift_observations WHERE observation_id = ?", 'SELECT * FROM institutional_drift_observations WHERE observation_id = %s'), (finding["observation_id"],)).fetchone()
            baseline = conn.execute(hosted.sql_query("SELECT * FROM institutional_drift_baseline_snapshots WHERE baseline_snapshot_id = ?", 'SELECT * FROM institutional_drift_baseline_snapshots WHERE baseline_snapshot_id = %s'), (finding["baseline_snapshot_id"],)).fetchone()
            signals = conn.execute(hosted.sql_query("SELECT * FROM institutional_drift_learning_signals WHERE finding_id = ?", 'SELECT * FROM institutional_drift_learning_signals WHERE finding_id = %s'), (finding_id,)).fetchall()
            recoveries = conn.execute(hosted.sql_query("SELECT * FROM institutional_drift_recovery_verifications WHERE finding_id = ?", 'SELECT * FROM institutional_drift_recovery_verifications WHERE finding_id = %s'), (finding_id,)).fetchall()
        payload = {
            "schema_version": "institutional_drift_audit_export.v1",
            "finding": self._finding_dict(finding),
            "observation": self._observation_obj(observation).to_record(),
            "baseline": self._baseline_dict(baseline),
            "learning_signals": [dict(row) for row in signals],
            "recovery_verifications": [dict(row) for row in recoveries],
        }
        return {**payload, "audit_hash": content_hash(payload)}

    def dashboard(self, *, organization_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            policies = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM metadata_drift_monitor_policies
                WHERE organization_id = ?
                ORDER BY created_at DESC
                """, """
                SELECT *
                FROM metadata_drift_monitor_policies
                WHERE organization_id = %s
                ORDER BY created_at DESC
                """),
                (organization_id,),
            ).fetchall()
            findings = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM institutional_drift_findings
                WHERE organization_id = ?
                ORDER BY created_at DESC
                LIMIT 25
                """, """
                SELECT *
                FROM institutional_drift_findings
                WHERE organization_id = %s
                ORDER BY created_at DESC
                LIMIT 25
                """),
                (organization_id,),
            ).fetchall()
            schedules = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM metadata_drift_monitor_schedules
                WHERE organization_id = ?
                ORDER BY created_at DESC
                """, """
                SELECT *
                FROM metadata_drift_monitor_schedules
                WHERE organization_id = %s
                ORDER BY created_at DESC
                """),
                (organization_id,),
            ).fetchall()
            signals = conn.execute(
                hosted.sql_query("""
                SELECT signal.*
                FROM institutional_drift_learning_signals AS signal
                JOIN institutional_drift_findings AS finding
                  ON finding.finding_id = signal.finding_id
                WHERE finding.organization_id = ?
                ORDER BY signal.created_at DESC
                LIMIT 25
                """, """
                SELECT signal.*
                FROM institutional_drift_learning_signals AS signal
                JOIN institutional_drift_findings AS finding
                  ON finding.finding_id = signal.finding_id
                WHERE finding.organization_id = %s
                ORDER BY signal.created_at DESC
                LIMIT 25
                """),
                (organization_id,),
            ).fetchall()
        finding_records = [self._finding_dict(row) for row in findings]
        return {
            "schema_version": "institutional_drift_dashboard.v1",
            "organization_id": organization_id,
            "policy_counts": _count_by(policies, "governance_state"),
            "finding_counts": _count_by(findings, "selected_learning_route"),
            "latest_findings": finding_records,
            "active_schedules": [
                dict(row) for row in schedules if row["status"] == "ACTIVE"
            ],
            "open_learning_signals": [
                dict(row) for row in signals if row["lifecycle_state"] == "OPEN"
            ],
            "dashboard_hash": content_hash(
                {
                    "organization_id": organization_id,
                    "policies": [row["policy_hash"] for row in policies],
                    "findings": [item["finding_hash"] for item in finding_records],
                    "schedules": [row["schedule_hash"] for row in schedules],
                    "signals": [row["content_hash"] for row in signals],
                }
            ),
        }

    def _ensure_single_schedule(self, conn: sqlite3.Connection, policy: sqlite3.Row) -> None:
        existing = conn.execute(
            hosted.sql_query("""
            SELECT *
            FROM metadata_drift_monitor_schedules
            WHERE policy_id = ? AND status = 'ACTIVE'
            """, """
            SELECT *
            FROM metadata_drift_monitor_schedules
            WHERE policy_id = %s AND status = 'ACTIVE'
            """),
            (policy["policy_id"],),
        ).fetchall()
        if len(existing) > 1:
            raise ValueError("DUPLICATE_ACTIVE_SCHEDULE")
        if existing:
            return
        payload = {
            "policy_id": policy["policy_id"],
            "organization_id": policy["organization_id"],
            "workspace_id": policy["workspace_id"],
            "cadence": policy["cadence"],
            "cron_contract": "repeat=0;no_agent=true;authoritative=semantier_sqlite",
        }
        conn.execute(
            hosted.sql_query("""
            INSERT INTO metadata_drift_monitor_schedules (
              schedule_id, policy_id, organization_id, workspace_id, cadence,
              cron_ref, status, schedule_hash, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?)
            """, """
            INSERT INTO metadata_drift_monitor_schedules (
              schedule_id, policy_id, organization_id, workspace_id, cadence,
              cron_ref, status, schedule_hash, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, 'ACTIVE', %s, %s)
            """),
            (
                f"metadata_drift_schedule_{uuid4().hex}",
                policy["policy_id"], policy["organization_id"], policy["workspace_id"],
                policy["cadence"], f"cron:metadata-drift:{policy['policy_id']}",
                content_hash(payload), now_iso(),
            ),
        )

    def _record_policy_event(
        self,
        conn: sqlite3.Connection,
        *,
        policy_id: str,
        event_type: str,
        actor_ref: str,
        actor_role: str,
        from_state: str | None,
        to_state: str,
        evidence_hash: str,
        event_payload: dict[str, Any],
    ) -> dict[str, Any]:
        event_id = f"metadata_drift_policy_event_{uuid4().hex}"
        conn.execute(
            hosted.sql_query("""
            INSERT INTO metadata_drift_policy_events (
              event_id, policy_id, event_type, actor_ref, actor_role,
              from_state, to_state, evidence_hash, event_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, """
            INSERT INTO metadata_drift_policy_events (
              event_id, policy_id, event_type, actor_ref, actor_role,
              from_state, to_state, evidence_hash, event_json, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """),
            (
                event_id, policy_id, event_type, actor_ref, actor_role,
                from_state, to_state, evidence_hash, canonical_json(event_payload), now_iso(),
            ),
        )
        return {"event_id": event_id, "policy_id": policy_id, "event_type": event_type, "to_state": to_state}

    def _policy_row(self, conn: sqlite3.Connection, policy_id: str) -> sqlite3.Row:
        row = conn.execute(hosted.sql_query("SELECT * FROM metadata_drift_monitor_policies WHERE policy_id = ?", 'SELECT * FROM metadata_drift_monitor_policies WHERE policy_id = %s'), (policy_id,)).fetchone()
        if row is None:
            raise KeyError(policy_id)
        return row

    def _policy_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["asset_selector"] = json.loads(result.pop("asset_selector_json"))
        result["monitored_dimensions"] = json.loads(result.pop("monitored_dimensions_json"))
        result["severity_thresholds"] = json.loads(result.pop("severity_thresholds_json"))
        return result

    def _baseline_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["policy_version_refs"] = json.loads(result.pop("policy_version_refs_json"))
        result["source_hashes"] = json.loads(result.pop("source_hashes_json"))
        return result

    def _observation_obj(self, row: sqlite3.Row) -> DriftObservation:
        return DriftObservation(
            observation_id=row["observation_id"],
            idempotency_key=row["idempotency_key"],
            policy_id=row["policy_id"],
            organization_id=row["organization_id"],
            workspace_id=row["workspace_id"],
            asset_ref=row["asset_ref"],
            observation_window=row["observation_window"],
            payload=json.loads(row["normalized_payload_json"]),
            source_anchors=json.loads(row["source_anchors_json"]),
            collector_version=row["collector_version"],
            observation_hash=row["observation_hash"],
        )

    def _finding_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["deterministic_differences"] = json.loads(result.pop("deterministic_differences_json"))
        result["semantic_hypotheses"] = json.loads(result.pop("semantic_hypotheses_json"))
        result["authority_impact"] = json.loads(result.pop("authority_impact_json"))
        result["reason_codes"] = json.loads(result.pop("reason_codes_json"))
        return result


def _selector_matches(selector: dict[str, Any], asset_ref: str) -> bool:
    return selector.get("asset_ref") in {asset_ref, "*"} or asset_ref in selector.get("asset_refs", [])


def _effective(row: sqlite3.Row, as_of: str) -> bool:
    current = parse_utc(as_of)
    start = parse_utc(row["effective_from"])
    end = parse_utc(row["effective_to"]) if row["effective_to"] else None
    return current >= start and (end is None or current < end)


def _count_by(rows: list[sqlite3.Row], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return counts
