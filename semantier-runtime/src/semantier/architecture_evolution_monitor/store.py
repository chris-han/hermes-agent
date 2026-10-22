from __future__ import annotations

from storage import hosted_postgres as hosted

import json
import sqlite3
from typing import Any

from eos.artifact_hashing import canonical_json, content_hash
from eos.db import get_conn, now_iso

from .contracts import ReviewState, TERMINAL_REVIEW_STATES


class ReviewConflictError(ValueError):
    pass


class ArchitectureEvolutionMonitorStore:
    def ensure_schema(self, conn: sqlite3.Connection | None = None) -> None:
        close_conn = False
        if conn is None:
            conn = get_conn()
            close_conn = True
        try:
            if hosted.is_postgres(conn):
                hosted.require_tables(conn, 'architecture_assessments', 'architecture_handoff_grants', 'architecture_learning_classifications', 'architecture_legal_graph_pilot_evidence', 'architecture_metric_snapshots', 'architecture_monitor_scheduled_runs', 'architecture_monitor_schedules', 'architecture_review_decisions', 'architecture_review_packages', 'architecture_threshold_profiles')
                return
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS architecture_metric_snapshots (
                  snapshot_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  environment TEXT NOT NULL,
                  architecture_profile TEXT NOT NULL,
                  observed_from TEXT NOT NULL,
                  observed_to TEXT NOT NULL,
                  metric_definition_set_ref TEXT NOT NULL,
                  snapshot_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS architecture_threshold_profiles (
                  profile_ref TEXT PRIMARY KEY,
                  profile_id TEXT NOT NULL,
                  profile_version TEXT NOT NULL,
                  scope_json TEXT NOT NULL,
                  effective_from TEXT NOT NULL,
                  effective_to TEXT,
                  governance_state TEXT NOT NULL,
                  profile_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS architecture_assessments (
                  assessment_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  metric_snapshot_ref TEXT NOT NULL,
                  threshold_profile_ref TEXT NOT NULL,
                  assessment_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS architecture_learning_classifications (
                  classification_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  assessment_ref TEXT NOT NULL,
                  classification_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS architecture_review_packages (
                  review_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  state TEXT NOT NULL,
                  version INTEGER NOT NULL,
                  package_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS architecture_review_decisions (
                  decision_id TEXT PRIMARY KEY,
                  review_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  from_state TEXT NOT NULL,
                  to_state TEXT NOT NULL,
                  idempotency_key TEXT NOT NULL,
                  actor_ref TEXT NOT NULL,
                  decision_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  signature TEXT NOT NULL,
                  occurred_at TEXT NOT NULL,
                  UNIQUE(organization_id, review_id, idempotency_key)
                );

                CREATE TABLE IF NOT EXISTS architecture_handoff_grants (
                  grant_id TEXT PRIMARY KEY,
                  review_id TEXT NOT NULL,
                  organization_id TEXT NOT NULL,
                  grant_type TEXT NOT NULL,
                  status TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  revoked_at TEXT
                );

                CREATE TABLE IF NOT EXISTS architecture_monitor_schedules (
                  scope_id TEXT NOT NULL,
                  schedule_ref TEXT NOT NULL,
                  schedule_json TEXT NOT NULL,
                  cron_job_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY(scope_id, schedule_ref)
                );

                CREATE TABLE IF NOT EXISTS architecture_monitor_scheduled_runs (
                  scheduled_run_id TEXT PRIMARY KEY,
                  scope_id TEXT NOT NULL,
                  schedule_ref TEXT NOT NULL,
                  snapshot_ref TEXT NOT NULL,
                  assessment_ref TEXT NOT NULL,
                  classification_ref TEXT NOT NULL,
                  review_ref TEXT,
                  run_json TEXT NOT NULL,
                  output_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS architecture_legal_graph_pilot_evidence (
                  evidence_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  evidence_json TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                """
            )
        finally:
            if close_conn:
                conn.close()

    def put_snapshot(self, organization_id: str, snapshot: dict[str, Any]) -> None:
        with get_conn() as conn:
            self.ensure_schema(conn)
            conn.execute(
                hosted.sql_query("""
                INSERT OR REPLACE INTO architecture_metric_snapshots (
                  snapshot_id, organization_id, environment, architecture_profile,
                  observed_from, observed_to, metric_definition_set_ref,
                  snapshot_json, content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO architecture_metric_snapshots (
                  snapshot_id, organization_id, environment, architecture_profile,
                  observed_from, observed_to, metric_definition_set_ref,
                  snapshot_json, content_hash, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (snapshot_id) DO UPDATE SET organization_id=excluded.organization_id, environment=excluded.environment, architecture_profile=excluded.architecture_profile, observed_from=excluded.observed_from, observed_to=excluded.observed_to, metric_definition_set_ref=excluded.metric_definition_set_ref, snapshot_json=excluded.snapshot_json, content_hash=excluded.content_hash, created_at=excluded.created_at"""),
                (
                    snapshot["snapshot_id"],
                    organization_id,
                    snapshot["environment"],
                    snapshot["architecture_profile"],
                    snapshot["observed_from"],
                    snapshot["observed_to"],
                    snapshot["metric_definition_set_ref"],
                    canonical_json(snapshot),
                    snapshot["content_hash"],
                    now_iso(),
                ),
            )

    def put_profile(self, profile: dict[str, Any]) -> None:
        with get_conn() as conn:
            self.ensure_schema(conn)
            conn.execute(
                hosted.sql_query("""
                INSERT OR REPLACE INTO architecture_threshold_profiles (
                  profile_ref, profile_id, profile_version, scope_json,
                  effective_from, effective_to, governance_state,
                  profile_json, content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO architecture_threshold_profiles (
                  profile_ref, profile_id, profile_version, scope_json,
                  effective_from, effective_to, governance_state,
                  profile_json, content_hash, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (profile_ref) DO UPDATE SET profile_id=excluded.profile_id, profile_version=excluded.profile_version, scope_json=excluded.scope_json, effective_from=excluded.effective_from, effective_to=excluded.effective_to, governance_state=excluded.governance_state, profile_json=excluded.profile_json, content_hash=excluded.content_hash, created_at=excluded.created_at"""),
                (
                    profile["content_hash"],
                    profile["profile_id"],
                    profile["profile_version"],
                    canonical_json(profile["scope"]),
                    profile["effective_from"],
                    profile.get("effective_to"),
                    profile["governance_state"],
                    canonical_json(profile),
                    profile["content_hash"],
                    now_iso(),
                ),
            )

    def put_assessment(self, organization_id: str, assessment: dict[str, Any]) -> None:
        with get_conn() as conn:
            self.ensure_schema(conn)
            conn.execute(
                hosted.sql_query("""
                INSERT OR REPLACE INTO architecture_assessments (
                  assessment_id, organization_id, metric_snapshot_ref,
                  threshold_profile_ref, assessment_json, content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO architecture_assessments (
                  assessment_id, organization_id, metric_snapshot_ref,
                  threshold_profile_ref, assessment_json, content_hash, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (assessment_id) DO UPDATE SET organization_id=excluded.organization_id, metric_snapshot_ref=excluded.metric_snapshot_ref, threshold_profile_ref=excluded.threshold_profile_ref, assessment_json=excluded.assessment_json, content_hash=excluded.content_hash, created_at=excluded.created_at"""),
                (
                    assessment["assessment_id"],
                    organization_id,
                    assessment["metric_snapshot_ref"],
                    assessment["threshold_profile_ref"],
                    canonical_json(assessment),
                    assessment["content_hash"],
                    now_iso(),
                ),
            )

    def put_classification(self, organization_id: str, classification: dict[str, Any]) -> None:
        with get_conn() as conn:
            self.ensure_schema(conn)
            conn.execute(
                hosted.sql_query("""
                INSERT OR REPLACE INTO architecture_learning_classifications (
                  classification_id, organization_id, assessment_ref,
                  classification_json, content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO architecture_learning_classifications (
                  classification_id, organization_id, assessment_ref,
                  classification_json, content_hash, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (classification_id) DO UPDATE SET organization_id=excluded.organization_id, assessment_ref=excluded.assessment_ref, classification_json=excluded.classification_json, content_hash=excluded.content_hash, created_at=excluded.created_at"""),
                (
                    classification["classification_id"],
                    organization_id,
                    classification["assessment_ref"],
                    canonical_json(classification),
                    classification["content_hash"],
                    now_iso(),
                ),
            )

    def put_review_package(self, organization_id: str, package: dict[str, Any]) -> None:
        now = now_iso()
        with get_conn() as conn:
            self.ensure_schema(conn)
            conn.execute(
                hosted.sql_query("""
                INSERT OR REPLACE INTO architecture_review_packages (
                  review_id, organization_id, state, version,
                  package_json, content_hash, created_at, updated_at
                ) VALUES (
                  ?, ?, ?, ?,
                  ?, ?,
                  COALESCE((SELECT created_at FROM architecture_review_packages WHERE review_id = ?), ?),
                  ?
                )
                """, """
                INSERT INTO architecture_review_packages (
                  review_id, organization_id, state, version,
                  package_json, content_hash, created_at, updated_at
                ) VALUES (
                  %s, %s, %s, %s,
                  %s, %s,
                  COALESCE((SELECT created_at FROM architecture_review_packages WHERE review_id = %s), %s),
                  %s
                ) ON CONFLICT (review_id) DO UPDATE SET organization_id=excluded.organization_id, state=excluded.state, version=excluded.version, package_json=excluded.package_json, content_hash=excluded.content_hash, created_at=excluded.created_at, updated_at=excluded.updated_at"""),
                (
                    package["review_id"],
                    organization_id,
                    package["state"],
                    int(package["version"]),
                    canonical_json(package),
                    package["content_hash"],
                    package["review_id"],
                    now,
                    now,
                ),
            )

    def get_review_package(self, organization_id: str, review_id: str) -> dict[str, Any] | None:
        with get_conn() as conn:
            self.ensure_schema(conn)
            row = conn.execute(
                hosted.sql_query("""
                SELECT package_json FROM architecture_review_packages
                WHERE organization_id = ? AND review_id = ?
                """, """
                SELECT package_json FROM architecture_review_packages
                WHERE organization_id = %s AND review_id = %s
                """),
                (organization_id, review_id),
            ).fetchone()
        return json.loads(row["package_json"]) if row else None

    def list_review_packages(self, organization_id: str) -> list[dict[str, Any]]:
        with get_conn() as conn:
            self.ensure_schema(conn)
            rows = conn.execute(
                hosted.sql_query("""
                SELECT package_json FROM architecture_review_packages
                WHERE organization_id = ?
                ORDER BY updated_at DESC, review_id ASC
                """, """
                SELECT package_json FROM architecture_review_packages
                WHERE organization_id = %s
                ORDER BY updated_at DESC, review_id ASC
                """),
                (organization_id,),
            ).fetchall()
        return [json.loads(row["package_json"]) for row in rows]

    def get_classification_by_ref(
        self,
        organization_id: str,
        classification_ref: str,
    ) -> dict[str, Any] | None:
        with get_conn() as conn:
            self.ensure_schema(conn)
            row = conn.execute(
                hosted.sql_query("""
                SELECT classification_json
                FROM architecture_learning_classifications
                WHERE organization_id = ? AND content_hash = ?
                """, """
                SELECT classification_json
                FROM architecture_learning_classifications
                WHERE organization_id = %s AND content_hash = %s
                """),
                (organization_id, classification_ref),
            ).fetchone()
        return json.loads(row["classification_json"]) if row else None

    def transition_review(
        self,
        *,
        organization_id: str,
        review_id: str,
        expected_version: int,
        expected_hash: str,
        to_state: ReviewState,
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        with get_conn() as conn:
            self.ensure_schema(conn)
            existing_decision = conn.execute(
                hosted.sql_query("""
                SELECT decision_json FROM architecture_review_decisions
                WHERE organization_id = ? AND review_id = ? AND idempotency_key = ?
                """, """
                SELECT decision_json FROM architecture_review_decisions
                WHERE organization_id = %s AND review_id = %s AND idempotency_key = %s
                """),
                (organization_id, review_id, decision["idempotency_key"]),
            ).fetchone()
            if existing_decision:
                return json.loads(existing_decision["decision_json"])

            row = conn.execute(
                hosted.sql_query("""
                SELECT state, version, content_hash, package_json
                FROM architecture_review_packages
                WHERE organization_id = ? AND review_id = ?
                """, """
                SELECT state, version, content_hash, package_json
                FROM architecture_review_packages
                WHERE organization_id = %s AND review_id = %s
                """),
                (organization_id, review_id),
            ).fetchone()
            if row is None:
                raise KeyError(review_id)
            current_state = ReviewState(row["state"])
            if current_state in TERMINAL_REVIEW_STATES and to_state != ReviewState.OPEN:
                raise ReviewConflictError("terminal review decisions cannot be overwritten")
            if int(row["version"]) != expected_version or row["content_hash"] != expected_hash:
                raise ReviewConflictError("review package version/hash conflict")
            package = json.loads(row["package_json"])
            package["state"] = to_state.value
            package["version"] = int(row["version"]) + 1
            package["content_hash"] = f"sha256:{content_hash({k: v for k, v in package.items() if k != 'content_hash'})}"
            decision = dict(decision)
            decision["from_state"] = current_state.value
            decision["to_state"] = to_state.value
            decision["content_hash"] = f"sha256:{content_hash({k: v for k, v in decision.items() if k != 'content_hash'})}"
            conn.execute(
                hosted.sql_query("""
                INSERT INTO architecture_review_decisions (
                  decision_id, review_id, organization_id, from_state, to_state,
                  idempotency_key, actor_ref, decision_json, content_hash,
                  signature, occurred_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO architecture_review_decisions (
                  decision_id, review_id, organization_id, from_state, to_state,
                  idempotency_key, actor_ref, decision_json, content_hash,
                  signature, occurred_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """),
                (
                    decision["decision_id"],
                    review_id,
                    organization_id,
                    decision["from_state"],
                    decision["to_state"],
                    decision["idempotency_key"],
                    decision["actor_ref"],
                    canonical_json(decision),
                    decision["content_hash"],
                    decision["signature"],
                    decision["occurred_at"],
                ),
            )
            conn.execute(
                hosted.sql_query("""
                UPDATE architecture_review_packages
                SET state = ?, version = ?, package_json = ?, content_hash = ?, updated_at = ?
                WHERE organization_id = ? AND review_id = ?
                """, """
                UPDATE architecture_review_packages
                SET state = %s, version = %s, package_json = %s, content_hash = %s, updated_at = %s
                WHERE organization_id = %s AND review_id = %s
                """),
                (
                    package["state"],
                    package["version"],
                    canonical_json(package),
                    package["content_hash"],
                    now_iso(),
                    organization_id,
                    review_id,
                ),
            )
        return decision

    def put_handoff_grant(
        self,
        *,
        organization_id: str,
        review_id: str,
        grant_id: str,
        grant_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        now = now_iso()
        grant = {
            "schema_version": "architecture_handoff_grant.v1",
            "grant_id": grant_id,
            "review_id": review_id,
            "organization_id": organization_id,
            "grant_type": grant_type,
            "status": "ACTIVE",
            "payload": payload,
            "created_at": now,
            "revoked_at": None,
        }
        grant["content_hash"] = f"sha256:{content_hash(grant)}"
        with get_conn() as conn:
            self.ensure_schema(conn)
            conn.execute(
                hosted.sql_query("""
                INSERT OR REPLACE INTO architecture_handoff_grants (
                  grant_id, review_id, organization_id, grant_type, status,
                  payload_json, content_hash, created_at, revoked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO architecture_handoff_grants (
                  grant_id, review_id, organization_id, grant_type, status,
                  payload_json, content_hash, created_at, revoked_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (grant_id) DO UPDATE SET review_id=excluded.review_id, organization_id=excluded.organization_id, grant_type=excluded.grant_type, status=excluded.status, payload_json=excluded.payload_json, content_hash=excluded.content_hash, created_at=excluded.created_at, revoked_at=excluded.revoked_at"""),
                (
                    grant["grant_id"],
                    review_id,
                    organization_id,
                    grant_type,
                    grant["status"],
                    canonical_json(grant),
                    grant["content_hash"],
                    now,
                    None,
                ),
            )
        return grant

    def revoke_handoff_grants(
        self,
        *,
        organization_id: str,
        review_id: str,
    ) -> list[dict[str, Any]]:
        now = now_iso()
        revoked: list[dict[str, Any]] = []
        with get_conn() as conn:
            self.ensure_schema(conn)
            rows = conn.execute(
                hosted.sql_query("""
                SELECT grant_id, payload_json
                FROM architecture_handoff_grants
                WHERE organization_id = ? AND review_id = ? AND status = 'ACTIVE'
                """, """
                SELECT grant_id, payload_json
                FROM architecture_handoff_grants
                WHERE organization_id = %s AND review_id = %s AND status = 'ACTIVE'
                """),
                (organization_id, review_id),
            ).fetchall()
            for row in rows:
                grant = json.loads(row["payload_json"])
                grant["status"] = "REVOKED"
                grant["revoked_at"] = now
                grant["content_hash"] = f"sha256:{content_hash({k: v for k, v in grant.items() if k != 'content_hash'})}"
                conn.execute(
                    hosted.sql_query("""
                    UPDATE architecture_handoff_grants
                    SET status = 'REVOKED', payload_json = ?, content_hash = ?, revoked_at = ?
                    WHERE grant_id = ?
                    """, """
                    UPDATE architecture_handoff_grants
                    SET status = 'REVOKED', payload_json = %s, content_hash = %s, revoked_at = %s
                    WHERE grant_id = %s
                    """),
                    (canonical_json(grant), grant["content_hash"], now, row["grant_id"]),
                )
                revoked.append(grant)
        return revoked

    def list_handoff_grants(self, organization_id: str, review_id: str) -> list[dict[str, Any]]:
        with get_conn() as conn:
            self.ensure_schema(conn)
            rows = conn.execute(
                hosted.sql_query("""
                SELECT payload_json
                FROM architecture_handoff_grants
                WHERE organization_id = ? AND review_id = ?
                ORDER BY created_at ASC, grant_id ASC
                """, """
                SELECT payload_json
                FROM architecture_handoff_grants
                WHERE organization_id = %s AND review_id = %s
                ORDER BY created_at ASC, grant_id ASC
                """),
                (organization_id, review_id),
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def put_schedule(
        self,
        *,
        scope_id: str,
        schedule: dict[str, Any],
        cron_job: dict[str, Any],
    ) -> dict[str, Any]:
        now = now_iso()
        payload = {
            "schema_version": "architecture_monitor_schedule.v1",
            **schedule,
            "cron_job_ref": cron_job.get("id"),
        }
        payload["content_hash"] = f"sha256:{content_hash(payload)}"
        with get_conn() as conn:
            self.ensure_schema(conn)
            conn.execute(
                hosted.sql_query("""
                INSERT INTO architecture_monitor_schedules (
                  scope_id, schedule_ref, schedule_json, cron_job_json,
                  content_hash, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope_id, schedule_ref) DO UPDATE SET
                  schedule_json = excluded.schedule_json,
                  cron_job_json = excluded.cron_job_json,
                  content_hash = excluded.content_hash,
                  updated_at = excluded.updated_at
                """, """
                INSERT INTO architecture_monitor_schedules (
                  scope_id, schedule_ref, schedule_json, cron_job_json,
                  content_hash, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(scope_id, schedule_ref) DO UPDATE SET
                  schedule_json = excluded.schedule_json,
                  cron_job_json = excluded.cron_job_json,
                  content_hash = excluded.content_hash,
                  updated_at = excluded.updated_at
                """),
                (
                    scope_id,
                    payload["schedule_ref"],
                    canonical_json(payload),
                    canonical_json(cron_job),
                    payload["content_hash"],
                    now,
                    now,
                ),
            )
        return payload

    def get_schedule(self, scope_id: str, schedule_ref: str) -> dict[str, Any] | None:
        with get_conn() as conn:
            self.ensure_schema(conn)
            row = conn.execute(
                hosted.sql_query("""
                SELECT schedule_json
                FROM architecture_monitor_schedules
                WHERE scope_id = ? AND schedule_ref = ?
                """, """
                SELECT schedule_json
                FROM architecture_monitor_schedules
                WHERE scope_id = %s AND schedule_ref = %s
                """),
                (scope_id, schedule_ref),
            ).fetchone()
        return json.loads(row["schedule_json"]) if row else None

    def put_scheduled_run(self, run: dict[str, Any]) -> None:
        with get_conn() as conn:
            self.ensure_schema(conn)
            conn.execute(
                hosted.sql_query("""
                INSERT OR REPLACE INTO architecture_monitor_scheduled_runs (
                  scheduled_run_id, scope_id, schedule_ref, snapshot_ref,
                  assessment_ref, classification_ref, review_ref, run_json,
                  output_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO architecture_monitor_scheduled_runs (
                  scheduled_run_id, scope_id, schedule_ref, snapshot_ref,
                  assessment_ref, classification_ref, review_ref, run_json,
                  output_hash, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (scheduled_run_id) DO UPDATE SET scope_id=excluded.scope_id, schedule_ref=excluded.schedule_ref, snapshot_ref=excluded.snapshot_ref, assessment_ref=excluded.assessment_ref, classification_ref=excluded.classification_ref, review_ref=excluded.review_ref, run_json=excluded.run_json, output_hash=excluded.output_hash, created_at=excluded.created_at"""),
                (
                    run["scheduled_run_id"],
                    run["scope_id"],
                    run["schedule_ref"],
                    run["snapshot_ref"],
                    run["assessment_ref"],
                    run["classification_ref"],
                    run.get("review_ref"),
                    canonical_json(run),
                    run["output_hash"],
                    now_iso(),
                ),
            )

    def list_scheduled_runs(self, scope_id: str, schedule_ref: str) -> list[dict[str, Any]]:
        with get_conn() as conn:
            self.ensure_schema(conn)
            rows = conn.execute(
                hosted.sql_query("""
                SELECT run_json
                FROM architecture_monitor_scheduled_runs
                WHERE scope_id = ? AND schedule_ref = ?
                ORDER BY created_at ASC, scheduled_run_id ASC
                """, """
                SELECT run_json
                FROM architecture_monitor_scheduled_runs
                WHERE scope_id = %s AND schedule_ref = %s
                ORDER BY created_at ASC, scheduled_run_id ASC
                """),
                (scope_id, schedule_ref),
            ).fetchall()
        return [json.loads(row["run_json"]) for row in rows]

    def put_pilot_evidence(self, evidence: dict[str, Any]) -> None:
        with get_conn() as conn:
            self.ensure_schema(conn)
            conn.execute(
                hosted.sql_query("""
                INSERT OR REPLACE INTO architecture_legal_graph_pilot_evidence (
                  evidence_id, organization_id, evidence_json, content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """, """
                INSERT INTO architecture_legal_graph_pilot_evidence (
                  evidence_id, organization_id, evidence_json, content_hash, created_at
                ) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (evidence_id) DO UPDATE SET organization_id=excluded.organization_id, evidence_json=excluded.evidence_json, content_hash=excluded.content_hash, created_at=excluded.created_at"""),
                (
                    evidence["content_hash"],
                    evidence["organization_id"],
                    canonical_json(evidence),
                    evidence["content_hash"],
                    now_iso(),
                ),
            )

    def list_pilot_evidence(self, organization_id: str) -> list[dict[str, Any]]:
        with get_conn() as conn:
            self.ensure_schema(conn)
            rows = conn.execute(
                hosted.sql_query("""
                SELECT evidence_json
                FROM architecture_legal_graph_pilot_evidence
                WHERE organization_id = ?
                ORDER BY created_at ASC, evidence_id ASC
                """, """
                SELECT evidence_json
                FROM architecture_legal_graph_pilot_evidence
                WHERE organization_id = %s
                ORDER BY created_at ASC, evidence_id ASC
                """),
                (organization_id,),
            ).fetchall()
        return [json.loads(row["evidence_json"]) for row in rows]
