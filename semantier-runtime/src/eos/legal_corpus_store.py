"""Authoritative SQLite store for governed legal corpus lifecycle objects."""

from __future__ import annotations

from storage import hosted_postgres as hosted

import sqlite3
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4
from datetime import datetime, timezone
from urllib.parse import urlparse

from eos.artifact_hashing import canonical_json, content_hash
from eos.authority_validation import validate_effectivity_window, validate_source_provenance
from eos.db import get_conn, now_iso
from eos.legal_candidate_compilation import LegalSemanticCandidate, validate_candidate
from eos.legal_text_normalization import LegalSectionAnchor


LEGAL_DEMO_ARTIFACT_NOT_CERTIFIABLE = "LEGAL_DEMO_ARTIFACT_NOT_CERTIFIABLE"
LEGAL_DEMO_APPROVAL_NOT_ACTIVATABLE = "LEGAL_DEMO_APPROVAL_NOT_ACTIVATABLE"
LEGAL_ACTIVATION_EVIDENCE_REQUIRED = "LEGAL_ACTIVATION_EVIDENCE_REQUIRED"
LEGAL_ISSUE_MODEL_ACTIVE_REGISTRATION_FORBIDDEN = "LEGAL_ISSUE_MODEL_ACTIVE_REGISTRATION_FORBIDDEN"
LEGAL_ISSUE_MODEL_CERTIFICATION_REQUIRED = "LEGAL_ISSUE_MODEL_CERTIFICATION_REQUIRED"
LEGAL_ISSUE_MODEL_APPROVER_ROLE_REQUIRED = "LEGAL_ISSUE_MODEL_APPROVER_ROLE_REQUIRED"
LEGAL_SOURCE_QUARANTINED = "LEGAL_SOURCE_QUARANTINED"
LEGAL_UNSUPPORTED_VERSION_EDGE = "LEGAL_UNSUPPORTED_VERSION_EDGE"
LEGAL_VERSION_AUTO_ACTIVATION_FORBIDDEN = "LEGAL_VERSION_AUTO_ACTIVATION_FORBIDDEN"
LEGAL_REFRESH_SCHEDULE_INACTIVE = "LEGAL_REFRESH_SCHEDULE_INACTIVE"
LEGAL_REVIEW_ROLE_REQUIRED = "LEGAL_REVIEW_ROLE_REQUIRED"
LEGAL_REVIEW_DIMENSIONS_REQUIRED = "LEGAL_REVIEW_DIMENSIONS_REQUIRED"
LEGAL_ACTION_STALE_OBJECT = "LEGAL_ACTION_STALE_OBJECT"
LEGAL_ACTION_SNAPSHOT_MISMATCH = "LEGAL_ACTION_SNAPSHOT_MISMATCH"
LEGAL_ACTION_MUTATION_HASH_MISMATCH = "LEGAL_ACTION_MUTATION_HASH_MISMATCH"
LEGAL_MONITOR_POLICY_STATES = {
    "DRAFT",
    "APPROVED",
    "SUSPENDED",
    "REVOKED",
    "SUPERSEDED",
    "EXPIRED",
}
LEGAL_MONITOR_POLICY_RESOLUTION_ERRORS = {
    "POLICY_NOT_FOUND",
    "POLICY_SCOPE_MISMATCH",
    "POLICY_WORKSPACE_REQUIRED",
    "POLICY_CROSS_ORG_REFERENCE",
    "POLICY_SOURCE_REF_INVALID",
    "POLICY_HASH_MISMATCH",
    "POLICY_STATE_NOT_SCHEDULABLE",
    "POLICY_NOT_YET_EFFECTIVE",
    "POLICY_EXPIRED",
    "POLICY_SUPERSEDED",
    "POLICY_SAME_SCOPE_AMBIGUITY",
    "POLICY_LINEAGE_CONFLICT",
    "POLICY_WINDOW_NOT_CANONICAL",
}
LEGAL_MONITOR_COMPARISON_CLASSES = {
    "SEMANTICALLY_UNCHANGED",
    "REPRESENTATION_CHANGED",
    "CONTENT_CHANGE_DETECTED",
    "AUTHORITY_STATE_CHANGED",
    "SOURCE_MOVED",
    "SOURCE_UNAVAILABLE",
    "SOURCE_ACCESS_DENIED",
    "RETRY_PENDING",
    "UNKNOWN_REVIEW_REQUIRED",
}
LEGAL_REGULATORY_NOTIFICATION_EVENTS = {
    "LEGAL_SOURCE_SCAN_COMPLETED",
    "LEGAL_SOURCE_UNAVAILABLE",
    "LEGAL_SOURCE_MOVED",
    "LEGAL_SOURCE_REPRESENTATION_CHANGED",
    "LEGAL_SOURCE_CONTENT_CHANGED",
    "LEGAL_SOURCE_AUTHORITY_STATE_CHANGED",
    "LEGAL_SOURCE_REVIEW_REQUIRED",
    "LEGAL_SOURCE_REVIEW_OVERDUE",
    "LEGAL_SOURCE_BECAME_EFFECTIVE",
    "LEGAL_SOURCE_REPEALED",
    "LEGAL_RUNTIME_IMPACT_DETECTED",
    "LEGAL_BUNDLE_ACTIVATED",
}
LEGAL_REGULATORY_NOTIFICATION_SEVERITIES = {"INFO", "WARNING", "ERROR", "CRITICAL"}

LEGAL_ACTION_FAILURE_REASON_CODES = {
    LEGAL_ACTION_STALE_OBJECT,
    LEGAL_ACTION_SNAPSHOT_MISMATCH,
    LEGAL_ACTION_MUTATION_HASH_MISMATCH,
}

LEGAL_VERSION_EDGE_TYPES = {
    "AMENDS",
    "SUPERSEDES",
    "REPEALS",
    "RESTATES",
    "DERIVES_FROM",
}


def _parse_aware_utc(value: str, *, field_name: str) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"{field_name} is required")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return parsed.astimezone(timezone.utc).replace(microsecond=0)


def _format_utc_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_scheduled_window(
    *,
    scheduled_at: str,
    schedule_anchor_utc: str,
    interval_seconds: int,
) -> str:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    scheduled = _parse_aware_utc(scheduled_at, field_name="scheduled_at")
    anchor = _parse_aware_utc(schedule_anchor_utc, field_name="schedule_anchor_utc")
    elapsed_seconds = int((scheduled - anchor).total_seconds())
    window_index = elapsed_seconds // interval_seconds
    start = datetime.fromtimestamp(
        anchor.timestamp() + (window_index * interval_seconds),
        tz=timezone.utc,
    )
    end = datetime.fromtimestamp(start.timestamp() + interval_seconds, tz=timezone.utc)
    return f"{_format_utc_z(start)}/{_format_utc_z(end)}"


def legal_scan_item_idempotency_key(
    *,
    organization_id: str,
    source_ref: str,
    canonical_window: str,
    monitor_policy_version: str,
) -> str:
    return content_hash(
        {
            "legal_scan_item_key": "\n".join(
                [organization_id, source_ref, canonical_window, monitor_policy_version]
            )
        }
    )


def _impact_posture(*, dependency_types: set[str], has_authority_edges: bool) -> str:
    blocking_types = {
        "active_case",
        "active_session",
        "runtime_policy",
        "rego_policy_projection",
        "external_verification",
    }
    review_types = {
        "legal_claim",
        "source_anchor",
        "t3_doctrine",
        "t4_policy",
        "skill",
        "workflow",
        "fixture",
        "uat_case",
        "replay_test",
    }
    normalized = {str(value or "").strip().lower() for value in dependency_types}
    if normalized & blocking_types:
        return "BLOCK_NEW_EXECUTIONS"
    if has_authority_edges or normalized & review_types or normalized:
        return "REVIEW_NEW_EXECUTIONS"
    return "CONTINUE_PINNED"


@dataclass(frozen=True)
class CuratedKnowledgeArtifact:
    artifact_ref: str
    lifecycle_state: str
    content_hash: str


@dataclass(frozen=True)
class LegalSourceVersion:
    version_id: str
    source_id: str
    version_identity: str
    content_hash: str
    lifecycle_state: str


@dataclass(frozen=True)
class LegalClaimCandidate:
    candidate_ref: str
    source_version_id: str
    source_span_hash: str
    lifecycle_state: str


@dataclass(frozen=True)
class CertifiedLegalClaim:
    claim_ref: str
    source_version_id: str
    certification_event_id: str
    content_hash: str


@dataclass(frozen=True)
class KnowledgeContextVersion:
    knowledge_context_version_id: str
    source_registry_snapshot_id: str
    content_hash: str
    lifecycle_state: str


@dataclass(frozen=True)
class AuthorityBundleVersion:
    authority_bundle_version_id: str
    source_registry_snapshot_id: str
    knowledge_context_version_id: str
    bundle_hash: str
    lifecycle_state: str


@dataclass(frozen=True)
class LegalIssueModelVersion:
    issue_model_version_id: str
    issue_model_id: str
    organization_id: str
    model_hash: str
    lifecycle_state: str


class LegalCorpusStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path is not None else None

    def _connect(self) -> sqlite3.Connection:
        if hosted.enabled():
            return get_conn(self._db_path)
        if self._db_path is None:
            return get_conn()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _source_for_policy(
        self,
        conn: sqlite3.Connection,
        *,
        organization_id: str,
        source_id: str | None,
        source_ref: str | None,
    ) -> sqlite3.Row:
        resolved_ref = str(source_ref or source_id or "").strip()
        resolved_source_id = str(source_id or resolved_ref or "").strip()
        if not resolved_ref or not resolved_source_id:
            raise ValueError("source_ref is required")
        if resolved_ref != resolved_source_id:
            raise ValueError("POLICY_SOURCE_REF_INVALID")
        row = conn.execute(
            hosted.sql_query("""
            SELECT *
            FROM legal_sources
            WHERE source_id = ?
            """, """
            SELECT *
            FROM legal_sources
            WHERE source_id = %s
            """),
            (resolved_source_id,),
        ).fetchone()
        if row is None:
            raise ValueError("POLICY_SOURCE_REF_INVALID")
        if row["organization_id"] != organization_id:
            raise ValueError("POLICY_CROSS_ORG_REFERENCE")
        return row

    def _monitor_policy_payload(
        self,
        *,
        source_id: str,
        organization_id: str,
        workspace_id: str | None,
        source_domain: str,
        source_ref: str,
        cadence: str,
        interval_seconds: int,
        schedule_anchor_utc: str,
        priority: int,
        monitor_policy_version: str,
        official_domain_allowlist: list[str],
        redirect_policy: str,
        max_artifact_bytes: int,
        normalization_profile: str,
        comparison_profile: str,
        retry_policy: dict[str, Any],
        governance_state: str,
        effective_from: str,
        effective_to: str | None,
        supersedes_policy_version_ref: str | None,
    ) -> dict[str, Any]:
        return {
            "source_id": source_id,
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "source_domain": source_domain,
            "source_ref": source_ref,
            "cadence": cadence,
            "interval_seconds": interval_seconds,
            "schedule_anchor_utc": _format_utc_z(
                _parse_aware_utc(schedule_anchor_utc, field_name="schedule_anchor_utc")
            ),
            "priority": priority,
            "monitor_policy_version": monitor_policy_version,
            "official_domain_allowlist": sorted(str(item).strip() for item in official_domain_allowlist if str(item).strip()),
            "redirect_policy": redirect_policy,
            "max_artifact_bytes": max_artifact_bytes,
            "normalization_profile": normalization_profile,
            "comparison_profile": comparison_profile,
            "retry_policy": retry_policy,
            "governance_state": governance_state,
            "effective_from": _format_utc_z(
                _parse_aware_utc(effective_from, field_name="effective_from")
            ),
            "effective_to": _format_utc_z(
                _parse_aware_utc(effective_to, field_name="effective_to")
            )
            if effective_to
            else None,
            "supersedes_policy_version_ref": supersedes_policy_version_ref,
        }

    def create_monitor_policy(
        self,
        *,
        organization_id: str,
        source_id: str | None = None,
        source_ref: str | None = None,
        workspace_id: str | None = None,
        source_domain: str = "legal_corpus",
        cadence: str,
        interval_seconds: int,
        schedule_anchor_utc: str,
        priority: int = 100,
        monitor_policy_version: str,
        official_domain_allowlist: list[str] | None = None,
        redirect_policy: str = "same_host",
        max_artifact_bytes: int = 10_000_000,
        normalization_profile: str = "legal_text.v1",
        comparison_profile: str = "stable_anchor.v1",
        retry_policy: dict[str, Any] | None = None,
        governance_state: str = "DRAFT",
        effective_from: str,
        effective_to: str | None = None,
        supersedes_policy_version_ref: str | None = None,
        created_by: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if max_artifact_bytes <= 0:
            raise ValueError("max_artifact_bytes must be positive")
        governance_state = str(governance_state or "").strip().upper()
        if governance_state not in LEGAL_MONITOR_POLICY_STATES:
            raise ValueError("POLICY_STATE_NOT_SCHEDULABLE")
        self._require_nonempty(
            {
                "organization_id": organization_id,
                "cadence": cadence,
                "schedule_anchor_utc": schedule_anchor_utc,
                "monitor_policy_version": monitor_policy_version,
                "created_by": created_by,
                "effective_from": effective_from,
            }
        )
        now = now_iso()
        with self._connect() as conn:
            source = self._source_for_policy(
                conn,
                organization_id=organization_id,
                source_id=source_id,
                source_ref=source_ref,
            )
            if workspace_id and source["workspace_id"] and workspace_id != source["workspace_id"]:
                raise ValueError("POLICY_SCOPE_MISMATCH")
            if supersedes_policy_version_ref:
                prior = conn.execute(
                    hosted.sql_query("""
                    SELECT *
                    FROM legal_monitor_policies
                    WHERE policy_version_ref = ?
                    """, """
                    SELECT *
                    FROM legal_monitor_policies
                    WHERE policy_version_ref = %s
                    """),
                    (supersedes_policy_version_ref,),
                ).fetchone()
                if prior is None or prior["organization_id"] != organization_id or prior["source_ref"] != source["source_id"]:
                    raise ValueError("POLICY_LINEAGE_CONFLICT")
            payload = self._monitor_policy_payload(
                source_id=source["source_id"],
                organization_id=organization_id,
                workspace_id=workspace_id,
                source_domain=source_domain,
                source_ref=source["source_id"],
                cadence=cadence,
                interval_seconds=interval_seconds,
                schedule_anchor_utc=schedule_anchor_utc,
                priority=priority,
                monitor_policy_version=monitor_policy_version,
                official_domain_allowlist=official_domain_allowlist or [],
                redirect_policy=redirect_policy,
                max_artifact_bytes=max_artifact_bytes,
                normalization_profile=normalization_profile,
                comparison_profile=comparison_profile,
                retry_policy=retry_policy or {},
                governance_state=governance_state,
                effective_from=effective_from,
                effective_to=effective_to,
                supersedes_policy_version_ref=supersedes_policy_version_ref,
            )
            policy_hash = content_hash(payload)
            policy_version_ref = f"legal_monitor_policy_{uuid4().hex}"
            conn.execute(
                hosted.sql_query("""
                INSERT INTO legal_monitor_policies (
                  policy_version_ref, source_id, organization_id, workspace_id, source_domain,
                  source_ref, cadence, interval_seconds, schedule_anchor_utc, priority,
                  monitor_policy_version, official_domain_allowlist_json, redirect_policy,
                  max_artifact_bytes, normalization_profile, comparison_profile, retry_policy_json,
                  governance_state, effective_from, effective_to, supersedes_policy_version_ref,
                  created_by, created_at, policy_hash, policy_payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO legal_monitor_policies (
                  policy_version_ref, source_id, organization_id, workspace_id, source_domain,
                  source_ref, cadence, interval_seconds, schedule_anchor_utc, priority,
                  monitor_policy_version, official_domain_allowlist_json, redirect_policy,
                  max_artifact_bytes, normalization_profile, comparison_profile, retry_policy_json,
                  governance_state, effective_from, effective_to, supersedes_policy_version_ref,
                  created_by, created_at, policy_hash, policy_payload_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """),
                (
                    policy_version_ref,
                    payload["source_id"],
                    payload["organization_id"],
                    payload["workspace_id"],
                    payload["source_domain"],
                    payload["source_ref"],
                    payload["cadence"],
                    payload["interval_seconds"],
                    payload["schedule_anchor_utc"],
                    payload["priority"],
                    payload["monitor_policy_version"],
                    canonical_json(payload["official_domain_allowlist"]),
                    payload["redirect_policy"],
                    payload["max_artifact_bytes"],
                    payload["normalization_profile"],
                    payload["comparison_profile"],
                    canonical_json(payload["retry_policy"]),
                    payload["governance_state"],
                    payload["effective_from"],
                    payload["effective_to"],
                    payload["supersedes_policy_version_ref"],
                    created_by,
                    now,
                    policy_hash,
                    canonical_json(payload),
                ),
            )
            event_id = f"legal_monitor_policy_event_{uuid4().hex}"
            conn.execute(
                hosted.sql_query("""
                INSERT INTO legal_monitor_policy_events (
                  event_id, organization_id, policy_version_ref, event_type, actor_ref,
                  reason, prior_policy_version_ref, event_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO legal_monitor_policy_events (
                  event_id, organization_id, policy_version_ref, event_type, actor_ref,
                  reason, prior_policy_version_ref, event_json, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """),
                (
                    event_id,
                    organization_id,
                    policy_version_ref,
                    f"POLICY_{governance_state}",
                    created_by,
                    reason,
                    supersedes_policy_version_ref,
                    canonical_json({"policy_hash": policy_hash, "reason": reason}),
                    now,
                ),
            )
        result = dict(payload)
        result.update(
            {
                "policy_version_ref": policy_version_ref,
                "policy_hash": policy_hash,
                "created_at": now,
                "event_id": event_id,
            }
        )
        return result

    def _policy_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["official_domain_allowlist"] = json.loads(result.pop("official_domain_allowlist_json") or "[]")
        result["retry_policy"] = json.loads(result.pop("retry_policy_json") or "{}")
        result["policy_payload"] = json.loads(result.pop("policy_payload_json") or "{}")
        return result

    def resolve_monitor_policy(
        self,
        *,
        organization_id: str,
        source_ref: str,
        workspace_id: str | None,
        scheduled_at: str,
        require_workspace: bool = False,
    ) -> dict[str, Any]:
        scheduled = _parse_aware_utc(scheduled_at, field_name="scheduled_at")
        as_of = _format_utc_z(scheduled)
        with self._connect() as conn:
            source = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_sources
                WHERE source_id = ?
                """, """
                SELECT *
                FROM legal_sources
                WHERE source_id = %s
                """),
                (source_ref,),
            ).fetchone()
            if source is None:
                return {"ok": False, "error_code": "POLICY_SOURCE_REF_INVALID"}
            if source["organization_id"] != organization_id:
                return {"ok": False, "error_code": "POLICY_CROSS_ORG_REFERENCE"}
            rows = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_monitor_policies
                WHERE organization_id = ?
                  AND source_ref = ?
                """, """
                SELECT *
                FROM legal_monitor_policies
                WHERE organization_id = %s
                  AND source_ref = %s
                """),
                (organization_id, source_ref),
            ).fetchall()
        if not rows:
            return {"ok": False, "error_code": "POLICY_NOT_FOUND"}

        workspace_rows = [row for row in rows if row["workspace_id"] == workspace_id and workspace_id is not None]
        org_rows = [row for row in rows if row["workspace_id"] is None]
        other_workspace_rows = [row for row in rows if row["workspace_id"] not in {None, workspace_id}]
        if require_workspace and workspace_id is None and workspace_rows:
            return {"ok": False, "error_code": "POLICY_WORKSPACE_REQUIRED"}
        if workspace_id is not None and not workspace_rows and other_workspace_rows and not org_rows:
            return {"ok": False, "error_code": "POLICY_SCOPE_MISMATCH"}
        scoped_rows = workspace_rows or org_rows
        if not scoped_rows:
            return {"ok": False, "error_code": "POLICY_NOT_FOUND"}

        hash_valid_rows: list[sqlite3.Row] = []
        for row in scoped_rows:
            payload = json.loads(row["policy_payload_json"] or "{}")
            if content_hash(payload) != row["policy_hash"]:
                return {"ok": False, "error_code": "POLICY_HASH_MISMATCH"}
            hash_valid_rows.append(row)

        effective_rows: list[sqlite3.Row] = []
        for row in hash_valid_rows:
            if row["governance_state"] != "APPROVED":
                continue
            effective_from = _parse_aware_utc(row["effective_from"], field_name="effective_from")
            effective_to = (
                _parse_aware_utc(row["effective_to"], field_name="effective_to")
                if row["effective_to"]
                else None
            )
            if scheduled < effective_from:
                continue
            if effective_to is not None and scheduled >= effective_to:
                continue
            effective_rows.append(row)
        if not effective_rows:
            if any(row["governance_state"] != "APPROVED" for row in hash_valid_rows):
                return {"ok": False, "error_code": "POLICY_STATE_NOT_SCHEDULABLE"}
            if any(scheduled < _parse_aware_utc(row["effective_from"], field_name="effective_from") for row in hash_valid_rows):
                return {"ok": False, "error_code": "POLICY_NOT_YET_EFFECTIVE"}
            return {"ok": False, "error_code": "POLICY_EXPIRED"}

        superseded_refs = {
            row["supersedes_policy_version_ref"]
            for row in effective_rows
            if row["supersedes_policy_version_ref"]
        }
        candidates = [row for row in effective_rows if row["policy_version_ref"] not in superseded_refs]
        if not candidates:
            return {"ok": False, "error_code": "POLICY_SUPERSEDED"}
        if len(candidates) > 1:
            return {"ok": False, "error_code": "POLICY_SAME_SCOPE_AMBIGUITY"}
        policy = self._policy_dict(candidates[0])
        window = canonical_scheduled_window(
            scheduled_at=as_of,
            schedule_anchor_utc=policy["schedule_anchor_utc"],
            interval_seconds=int(policy["interval_seconds"]),
        )
        return {"ok": True, "policy": policy, "canonical_scheduled_window": window}

    def create_or_resume_scan_run(
        self,
        *,
        organization_id: str,
        monitor_policy_version: str,
        scheduled_window: str,
        trigger: str,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        self._require_nonempty(
            {
                "organization_id": organization_id,
                "monitor_policy_version": monitor_policy_version,
                "scheduled_window": scheduled_window,
                "trigger": trigger,
            }
        )
        now = now_iso()
        scan_run_id = f"legal_scan_run_{uuid4().hex}"
        with self._connect() as conn:
            hosted.begin_write(conn)
            existing = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_scan_runs
                WHERE organization_id = ?
                  AND monitor_policy_version = ?
                  AND scheduled_window = ?
                  AND workspace_id IS ?
                """, """
                SELECT *
                FROM legal_scan_runs
                WHERE organization_id = %s
                  AND monitor_policy_version = %s
                  AND scheduled_window = %s
                  AND workspace_id IS NOT DISTINCT FROM %s
                """),
                (organization_id, monitor_policy_version, scheduled_window, workspace_id),
            ).fetchone()
            if existing is not None:
                return dict(existing)
            conn.execute(
                hosted.sql_query("""
                INSERT INTO legal_scan_runs (
                  scan_run_id, organization_id, workspace_id, monitor_policy_version,
                  scheduled_window, trigger, status, started_at, aggregate_counts_json
                ) VALUES (?, ?, ?, ?, ?, ?, 'RUNNING', ?, '{}')
                """, """
                INSERT INTO legal_scan_runs (
                  scan_run_id, organization_id, workspace_id, monitor_policy_version,
                  scheduled_window, trigger, status, started_at, aggregate_counts_json
                ) VALUES (%s, %s, %s, %s, %s, %s, 'RUNNING', %s, '{}')
                """),
                (
                    scan_run_id,
                    organization_id,
                    workspace_id,
                    monitor_policy_version,
                    scheduled_window,
                    trigger,
                    now,
                ),
            )
        return {
            "scan_run_id": scan_run_id,
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "monitor_policy_version": monitor_policy_version,
            "scheduled_window": scheduled_window,
            "trigger": trigger,
            "status": "RUNNING",
            "started_at": now,
            "completed_at": None,
            "aggregate_counts_json": "{}",
        }

    def create_scan_item(
        self,
        *,
        scan_run_id: str,
        organization_id: str,
        source_id: str,
        monitor_policy_version: str,
        scheduled_window: str,
    ) -> dict[str, Any]:
        key = legal_scan_item_idempotency_key(
            organization_id=organization_id,
            source_ref=source_id,
            canonical_window=scheduled_window,
            monitor_policy_version=monitor_policy_version,
        )
        now = now_iso()
        scan_item_id = f"legal_scan_item_{uuid4().hex}"
        with self._connect() as conn:
            hosted.begin_write(conn)
            run = conn.execute(
                hosted.sql_query("SELECT * FROM legal_scan_runs WHERE scan_run_id = ?", 'SELECT * FROM legal_scan_runs WHERE scan_run_id = %s'),
                (scan_run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(f"Unknown scan_run_id: {scan_run_id}")
            source = self._source_for_policy(
                conn,
                organization_id=organization_id,
                source_id=source_id,
                source_ref=source_id,
            )
            if run["organization_id"] != organization_id:
                raise ValueError("scan run organization mismatch")
            existing = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_scan_items
                WHERE organization_id = ? AND idempotency_key = ?
                """, """
                SELECT *
                FROM legal_scan_items
                WHERE organization_id = %s AND idempotency_key = %s
                """),
                (organization_id, key),
            ).fetchone()
            if existing is not None:
                return dict(existing)
            conn.execute(
                hosted.sql_query("""
                INSERT INTO legal_scan_items (
                  scan_item_id, scan_run_id, organization_id, source_id, monitor_policy_version,
                  scheduled_window, idempotency_key, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?)
                """, """
                INSERT INTO legal_scan_items (
                  scan_item_id, scan_run_id, organization_id, source_id, monitor_policy_version,
                  scheduled_window, idempotency_key, status, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'PENDING', %s, %s)
                """),
                (
                    scan_item_id,
                    scan_run_id,
                    organization_id,
                    source["source_id"],
                    monitor_policy_version,
                    scheduled_window,
                    key,
                    now,
                    now,
                ),
            )
        return {
            "scan_item_id": scan_item_id,
            "scan_run_id": scan_run_id,
            "organization_id": organization_id,
            "source_id": source_id,
            "monitor_policy_version": monitor_policy_version,
            "scheduled_window": scheduled_window,
            "idempotency_key": key,
            "status": "PENDING",
            "attempt_count": 0,
        }

    def claim_scan_item(
        self,
        *,
        scan_item_id: str,
        lease_owner: str,
        lease_expires_at: str,
        as_of: str | None = None,
    ) -> dict[str, Any]:
        now = _format_utc_z(_parse_aware_utc(as_of or now_iso(), field_name="as_of"))
        lease_until = _format_utc_z(_parse_aware_utc(lease_expires_at, field_name="lease_expires_at"))
        with self._connect() as conn:
            hosted.begin_write(conn)
            row = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_scan_items
                WHERE scan_item_id = ?
                """, """
                SELECT *
                FROM legal_scan_items
                WHERE scan_item_id = %s
                """),
                (scan_item_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown scan_item_id: {scan_item_id}")
            if row["status"] not in {"PENDING", "RETRY_PENDING", "CLAIMED"}:
                raise ValueError("scan item is not claimable")
            if row["status"] == "CLAIMED" and row["lease_expires_at"] and row["lease_expires_at"] > now:
                raise ValueError("scan item lease is active")
            conn.execute(
                hosted.sql_query("""
                UPDATE legal_scan_items
                SET status = 'CLAIMED',
                    lease_owner = ?,
                    lease_expires_at = ?,
                    attempt_count = attempt_count + 1,
                    updated_at = ?
                WHERE scan_item_id = ?
                """, """
                UPDATE legal_scan_items
                SET status = 'CLAIMED',
                    lease_owner = %s,
                    lease_expires_at = %s,
                    attempt_count = attempt_count + 1,
                    updated_at = %s
                WHERE scan_item_id = %s
                """),
                (lease_owner, lease_until, now_iso(), scan_item_id),
            )
            updated = conn.execute(
                hosted.sql_query("SELECT * FROM legal_scan_items WHERE scan_item_id = ?", 'SELECT * FROM legal_scan_items WHERE scan_item_id = %s'),
                (scan_item_id,),
            ).fetchone()
        return dict(updated)

    def complete_scan_item(
        self,
        *,
        scan_item_id: str,
        status: str,
        result_class: str,
        error_code: str | None = None,
        error_detail: str | None = None,
        observed_version_id: str | None = None,
        refresh_check_id: str | None = None,
        lease_owner: str | None = None,
        expected_attempt: int | None = None,
        refresh_result: dict[str, Any] | None = None,
        checked_uri: str | None = None,
        check_status: str | None = None,
        checked_at: str | None = None,
        observed_content: bytes | None = None,
    ) -> dict[str, Any]:
        status = str(status or "").strip().upper()
        if status not in {"SUCCEEDED", "FAILED", "RETRY_PENDING"}:
            raise ValueError("unsupported scan item status")
        with self._connect() as conn:
            hosted.begin_write(conn)
            row = conn.execute(
                hosted.sql_query("SELECT * FROM legal_scan_items WHERE scan_item_id = ?", 'SELECT * FROM legal_scan_items WHERE scan_item_id = %s'),
                (scan_item_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown scan_item_id: {scan_item_id}")
            if lease_owner is not None:
                if (row["status"] != "CLAIMED" or row["lease_owner"] != lease_owner
                    or row["attempt_count"] != expected_attempt
                    or _parse_aware_utc(row["lease_expires_at"], field_name="lease_expires_at") <= _parse_aware_utc(now_iso(), field_name="now")):
                    raise ValueError("SCAN_LEASE_OWNERSHIP_LOST")
            if refresh_result is not None:
                refresh_check_id = f"legal_refresh_{uuid4().hex}"
                checked_at = _format_utc_z(_parse_aware_utc(checked_at or now_iso(), field_name="checked_at"))
                observed_hash = content_hash({"sha256_bytes": observed_content.hex()}) if observed_content else None
                conn.execute(
                    hosted.sql_query("INSERT INTO source_refresh_checks (refresh_check_id, source_id, checked_uri, checked_at, "
                    "status, observed_content_hash, result_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", 'INSERT INTO source_refresh_checks (refresh_check_id, source_id, checked_uri, checked_at, status, observed_content_hash, result_json, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)'),
                    (refresh_check_id, row["source_id"], checked_uri, checked_at, check_status,
                     observed_hash, canonical_json(refresh_result), now_iso()),
                )
            conn.execute(
                hosted.sql_query("""
                UPDATE legal_scan_items
                SET status = ?,
                    result_class = ?,
                    error_code = ?,
                    error_detail = ?,
                    observed_version_id = ?,
                    refresh_check_id = ?,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = ?
                WHERE scan_item_id = ?
                """, """
                UPDATE legal_scan_items
                SET status = %s,
                    result_class = %s,
                    error_code = %s,
                    error_detail = %s,
                    observed_version_id = %s,
                    refresh_check_id = %s,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = %s
                WHERE scan_item_id = %s
                """),
                (
                    status,
                    result_class,
                    error_code,
                    error_detail,
                    observed_version_id,
                    refresh_check_id,
                    now_iso(),
                    scan_item_id,
                ),
            )
            if (status == "SUCCEEDED" and lease_owner is not None and refresh_result
                and result_class in {"SEMANTICALLY_UNCHANGED", "REPRESENTATION_CHANGED"}):
                observation = conn.execute(
                    hosted.sql_query("SELECT event_json FROM source_ingestion_events WHERE ingestion_event_id = ? "
                    "AND organization_id = ? AND source_id = ? AND correlation_id = ? "
                    "AND event_type = 'ACQUIRED_OBSERVATION'", "SELECT event_json FROM source_ingestion_events WHERE ingestion_event_id = %s AND organization_id = %s AND source_id = %s AND correlation_id = %s AND event_type = 'ACQUIRED_OBSERVATION'"),
                    (refresh_result.get("artifact_id"), row["organization_id"], row["source_id"], scan_item_id),
                ).fetchone()
                policy = conn.execute(
                    hosted.sql_query("SELECT * FROM legal_monitor_policies WHERE policy_version_ref = ? "
                    "AND organization_id = ? AND source_id = ? AND monitor_policy_version = ?", 'SELECT * FROM legal_monitor_policies WHERE policy_version_ref = %s AND organization_id = %s AND source_id = %s AND monitor_policy_version = %s'),
                    (refresh_result.get("policy_version_ref"), row["organization_id"],
                     row["source_id"], row["monitor_policy_version"]),
                ).fetchone()
                run_scope = conn.execute(hosted.sql_query("SELECT workspace_id FROM legal_scan_runs WHERE scan_run_id = ?", 'SELECT workspace_id FROM legal_scan_runs WHERE scan_run_id = %s'),
                                         (row["scan_run_id"],)).fetchone()
                current_policy = self.resolve_monitor_policy(
                    organization_id=row["organization_id"], source_ref=row["source_id"],
                    workspace_id=run_scope["workspace_id"],
                    scheduled_at=checked_at,
                )
                if (observation is not None and policy is not None
                    and current_policy["ok"]
                    and current_policy["policy"]["policy_version_ref"] == policy["policy_version_ref"]
                    and policy["governance_state"] == "APPROVED"
                    and content_hash(json.loads(policy["policy_payload_json"])) == policy["policy_hash"]):
                    payload = json.loads(observation["event_json"])
                    window_end = _parse_aware_utc(row["scheduled_window"].split("/")[1], field_name="window_end")
                    if payload["content_hash"] == observed_hash == refresh_result.get("acquired_content_hash"):
                        schedules = conn.execute(
                            hosted.sql_query("SELECT * FROM legal_source_refresh_schedules WHERE source_id = ? "
                            "AND lifecycle_state = 'ACTIVE' AND interval_seconds = ?", "SELECT * FROM legal_source_refresh_schedules WHERE source_id = %s AND lifecycle_state = 'ACTIVE' AND interval_seconds = %s"),
                            (row["source_id"], policy["interval_seconds"]),
                        ).fetchall()
                        for schedule in schedules:
                            due = _parse_aware_utc(schedule["next_check_at"], field_name="next_check_at")
                            retrieved = _parse_aware_utc(payload["retrieved_at"], field_name="retrieved_at")
                            if due < window_end and due <= retrieved:
                                conn.execute(
                                    hosted.sql_query("UPDATE legal_source_refresh_schedules SET next_check_at = ?, retry_after = NULL "
                                    "WHERE schedule_id = ? AND next_check_at = ? AND lifecycle_state = 'ACTIVE'", "UPDATE legal_source_refresh_schedules SET next_check_at = %s, retry_after = NULL WHERE schedule_id = %s AND next_check_at = %s AND lifecycle_state = 'ACTIVE'"),
                                    (_format_utc_z(window_end), schedule["schedule_id"], schedule["next_check_at"]),
                                )
            updated = conn.execute(
                hosted.sql_query("SELECT * FROM legal_scan_items WHERE scan_item_id = ?", 'SELECT * FROM legal_scan_items WHERE scan_item_id = %s'),
                (scan_item_id,),
            ).fetchone()
        return dict(updated)

    def completed_scan_items_at(self, *, organization_id: str, workspace_id: str | None,
                                scheduled_at: str) -> list[dict[str, Any]]:
        """Return completed work in this scope/window; never schedule or refetch."""
        _parse_aware_utc(scheduled_at, field_name="scheduled_at")
        with self._connect() as conn:
            rows = conn.execute(
                hosted.sql_query("SELECT i.* FROM legal_scan_items i JOIN legal_scan_runs r ON r.scan_run_id = i.scan_run_id "
                "WHERE i.organization_id = ? AND r.workspace_id IS ? AND r.completed_at IS NOT NULL "
                "AND julianday(substr(i.scheduled_window, 1, instr(i.scheduled_window, '/') - 1)) <= julianday(?) "
                "AND julianday(substr(i.scheduled_window, instr(i.scheduled_window, '/') + 1)) > julianday(?) "
                "ORDER BY i.source_id, i.scan_item_id", "SELECT i.* FROM legal_scan_items i JOIN legal_scan_runs r ON r.scan_run_id = i.scan_run_id WHERE i.organization_id = %s AND r.workspace_id IS NOT DISTINCT FROM %s AND r.completed_at IS NOT NULL AND split_part(i.scheduled_window, '/', 1)::timestamptz <= %s::timestamptz AND split_part(i.scheduled_window, '/', 2)::timestamptz > %s::timestamptz ORDER BY i.source_id, i.scan_item_id"),
                (organization_id, workspace_id, scheduled_at, scheduled_at),
            ).fetchall()
        return [dict(row) for row in rows]

    def finalize_scan_run(self, *, scan_run_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            run = conn.execute(
                hosted.sql_query("SELECT * FROM legal_scan_runs WHERE scan_run_id = ?", 'SELECT * FROM legal_scan_runs WHERE scan_run_id = %s'),
                (scan_run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(f"Unknown scan_run_id: {scan_run_id}")
            rows = conn.execute(
                hosted.sql_query("""
                SELECT status, COUNT(*) AS count
                FROM legal_scan_items
                WHERE scan_run_id = ?
                GROUP BY status
                """, """
                SELECT status, COUNT(*) AS count
                FROM legal_scan_items
                WHERE scan_run_id = %s
                GROUP BY status
                """),
                (scan_run_id,),
            ).fetchall()
            counts = {row["status"]: row["count"] for row in rows}
            if run["completed_at"]:
                return dict(run)
            if counts.get("CLAIMED") or counts.get("PENDING"):
                return dict(run)
            if counts.get("FAILED") and counts.get("SUCCEEDED"):
                status = "PARTIAL"
            elif counts.get("FAILED") and not counts.get("SUCCEEDED"):
                status = "FAILED"
            elif counts and not (set(counts) - {"SUCCEEDED"}):
                status = "SUCCEEDED"
            else:
                status = "PARTIAL"
            conn.execute(
                hosted.sql_query("""
                UPDATE legal_scan_runs
                SET status = ?, completed_at = ?, aggregate_counts_json = ?
                WHERE scan_run_id = ?
                """, """
                UPDATE legal_scan_runs
                SET status = %s, completed_at = %s, aggregate_counts_json = %s
                WHERE scan_run_id = %s
                """),
                (status, now_iso(), canonical_json(counts), scan_run_id),
            )
            updated = conn.execute(
                hosted.sql_query("SELECT * FROM legal_scan_runs WHERE scan_run_id = ?", 'SELECT * FROM legal_scan_runs WHERE scan_run_id = %s'),
                (scan_run_id,),
            ).fetchone()
        return dict(updated)

    def list_scan_runs(
        self,
        *,
        organization_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        with self._connect() as conn:
            rows = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_scan_runs
                WHERE organization_id = ?
                ORDER BY started_at DESC, scan_run_id DESC
                LIMIT ? OFFSET ?
                """, """
                SELECT *
                FROM legal_scan_runs
                WHERE organization_id = %s
                ORDER BY started_at DESC, scan_run_id DESC
                LIMIT %s OFFSET %s
                """),
                (organization_id, limit, offset),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_scan_run(
        self,
        *,
        organization_id: str,
        scan_run_id: str,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            run = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_scan_runs
                WHERE organization_id = ? AND scan_run_id = ?
                """, """
                SELECT *
                FROM legal_scan_runs
                WHERE organization_id = %s AND scan_run_id = %s
                """),
                (organization_id, scan_run_id),
            ).fetchone()
            if run is None:
                raise KeyError(f"Unknown scan_run_id: {scan_run_id}")
            items = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_scan_items
                WHERE scan_run_id = ?
                ORDER BY created_at, scan_item_id
                """, """
                SELECT *
                FROM legal_scan_items
                WHERE scan_run_id = %s
                ORDER BY created_at, scan_item_id
                """),
                (scan_run_id,),
            ).fetchall()
        return {"scan_run": dict(run), "scan_items": [dict(row) for row in items]}

    def retry_scan_item(
        self,
        *,
        organization_id: str,
        scan_item_id: str,
        actor_ref: str,
    ) -> dict[str, Any]:
        self._require_nonempty({"organization_id": organization_id, "scan_item_id": scan_item_id, "actor_ref": actor_ref})
        with self._connect() as conn:
            hosted.begin_write(conn)
            row = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_scan_items
                WHERE organization_id = ? AND scan_item_id = ?
                """, """
                SELECT *
                FROM legal_scan_items
                WHERE organization_id = %s AND scan_item_id = %s
                """),
                (organization_id, scan_item_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown scan_item_id: {scan_item_id}")
            if row["status"] not in {"FAILED", "RETRY_PENDING"}:
                raise ValueError("scan item is not retryable")
            policies = conn.execute(
                hosted.sql_query("SELECT retry_policy_json FROM legal_monitor_policies WHERE organization_id = ? "
                "AND source_id = ? AND monitor_policy_version = ?", 'SELECT retry_policy_json FROM legal_monitor_policies WHERE organization_id = %s AND source_id = %s AND monitor_policy_version = %s'),
                (organization_id, row["source_id"], row["monitor_policy_version"]),
            ).fetchall()
            max_attempts = min((int(json.loads(p[0]).get("max_attempts", 3)) for p in policies), default=3)
            if row["attempt_count"] >= max_attempts:
                raise ValueError("RETRY_LIMIT_EXHAUSTED")
            conn.execute(hosted.sql_query("UPDATE legal_scan_runs SET status = 'RUNNING', completed_at = NULL WHERE scan_run_id = ?", "UPDATE legal_scan_runs SET status = 'RUNNING', completed_at = NULL WHERE scan_run_id = %s"),
                         (row["scan_run_id"],))
            conn.execute(
                hosted.sql_query("""
                UPDATE legal_scan_items
                SET status = 'RETRY_PENDING',
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    error_code = NULL,
                    error_detail = NULL,
                    updated_at = ?
                WHERE scan_item_id = ?
                """, """
                UPDATE legal_scan_items
                SET status = 'RETRY_PENDING',
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    error_code = NULL,
                    error_detail = NULL,
                    updated_at = %s
                WHERE scan_item_id = %s
                """),
                (now_iso(), scan_item_id),
            )
            updated = conn.execute(
                hosted.sql_query("SELECT * FROM legal_scan_items WHERE scan_item_id = ?", 'SELECT * FROM legal_scan_items WHERE scan_item_id = %s'),
                (scan_item_id,),
            ).fetchone()
            conn.execute(
                hosted.sql_query("""
                INSERT INTO legal_audit_events (
                  audit_event_id, event_family, object_ref, object_type, lifecycle_state,
                  attempted_endpoint, actor_ref, organization_id, reason_code,
                  route_policy_id, correlation_id, event_json, created_at
                ) VALUES (?, 'LEGAL_SOURCE_SCAN_RETRY_REQUESTED', ?, 'legal_scan_item',
                  'RETRY_PENDING', NULL, ?, ?, NULL, NULL, NULL, ?, ?)
                """, """
                INSERT INTO legal_audit_events (
                  audit_event_id, event_family, object_ref, object_type, lifecycle_state,
                  attempted_endpoint, actor_ref, organization_id, reason_code,
                  route_policy_id, correlation_id, event_json, created_at
                ) VALUES (%s, 'LEGAL_SOURCE_SCAN_RETRY_REQUESTED', %s, 'legal_scan_item',
                  'RETRY_PENDING', NULL, %s, %s, NULL, NULL, NULL, %s, %s)
                """),
                (
                    f"legal_audit_{uuid4().hex}",
                    scan_item_id,
                    actor_ref,
                    organization_id,
                    canonical_json({"scan_item_id": scan_item_id}),
                    now_iso(),
                ),
            )
        return dict(updated)

    def get_source_status_projection(
        self,
        *,
        organization_id: str,
        source_id: str,
        as_of: str,
    ) -> dict[str, Any]:
        validate_effectivity_window(
            {"effective_from": as_of, "effective_to": None},
            prefix="legal source status as_of",
        )
        with self._connect() as conn:
            source = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_sources
                WHERE organization_id = ? AND source_id = ?
                """, """
                SELECT *
                FROM legal_sources
                WHERE organization_id = %s AND source_id = %s
                """),
                (organization_id, source_id),
            ).fetchone()
            if source is None:
                raise KeyError(f"Unknown source_id: {source_id}")
            schedule = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_source_refresh_schedules
                WHERE source_id = ? AND lifecycle_state = 'ACTIVE'
                ORDER BY next_check_at ASC
                LIMIT 1
                """, """
                SELECT *
                FROM legal_source_refresh_schedules
                WHERE source_id = %s AND lifecycle_state = 'ACTIVE'
                ORDER BY next_check_at ASC
                LIMIT 1
                """),
                (source_id,),
            ).fetchone()
            latest_check = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM source_refresh_checks
                WHERE source_id = ?
                ORDER BY checked_at DESC, created_at DESC
                LIMIT 1
                """, """
                SELECT *
                FROM source_refresh_checks
                WHERE source_id = %s
                ORDER BY checked_at DESC, created_at DESC
                LIMIT 1
                """),
                (source_id,),
            ).fetchone()
            versions = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_source_versions
                WHERE source_id = ?
                ORDER BY created_at DESC
                """, """
                SELECT *
                FROM legal_source_versions
                WHERE source_id = %s
                ORDER BY created_at DESC
                """),
                (source_id,),
            ).fetchall()
            active_bundle = conn.execute(
                """
                SELECT *
                FROM legal_authority_bundle_versions
                WHERE lifecycle_state = 'ACTIVE'
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()
        result_json = json.loads(latest_check["result_json"]) if latest_check else {}
        comparison = result_json.get("comparison") if isinstance(result_json.get("comparison"), dict) else {}
        active_versions = [dict(row) for row in versions if row["lifecycle_state"] == "ACTIVE"]
        pending_versions = [
            dict(row)
            for row in versions
            if row["lifecycle_state"] in {"REGISTERED", "PENDING_REVIEW", "CERTIFIED"}
        ]
        latest_version = dict(versions[0]) if versions else None
        acquisition_basis = False
        if latest_check and result_json.get("artifact_id") and result_json.get("baseline_version_id"):
            try:
                observation = self.read_acquired_observation(artifact_id=result_json["artifact_id"],
                    organization_id=organization_id, source_id=source_id)
                baseline_artifact = self.read_preserved_artifact(artifact_id=result_json["baseline_artifact_id"],
                    organization_id=organization_id, source_id=source_id)
                acquisition_basis = (observation["content_hash"] == latest_check["observed_content_hash"]
                    and observation["baseline"]["version_id"] == result_json["baseline_version_id"]
                    and baseline_artifact["version_id"] == result_json["baseline_version_id"]
                    and comparison.get("prior_version_id") == result_json["baseline_version_id"]
                    and (latest_check["status"] != "UNCHANGED" or
                         baseline_artifact["content_hash"] == observation["content_hash"]))
            except (ValueError, KeyError):
                pass
        source_health = "UNKNOWN"
        if latest_check:
            if latest_check["status"] == "UNCHANGED":
                source_health = "HEALTHY" if (acquisition_basis and latest_check["observed_content_hash"]
                    and result_json.get("artifact_id") and result_json.get("baseline_version_id")
                    and result_json.get("acquired_content_hash") == latest_check["observed_content_hash"]
                    and comparison.get("comparison_class") == "SEMANTICALLY_UNCHANGED") else "ACQUISITION_BASIS_MISSING"
            elif latest_check["status"] == "CHANGE_DETECTED":
                source_health = str(comparison.get("comparison_class") or "CHANGE_DETECTED")
            else:
                source_health = latest_check["status"]
        if schedule and schedule["next_check_at"] <= as_of:
            source_health = "DUE" if source_health in {"UNKNOWN", "HEALTHY"} else source_health
        return {
            "source_id": source_id,
            "source_status": {
                "canonical_title": source["canonical_title"],
                "identity_state": source["governance_state"],
                "availability_state": source_health,
                "last_check": dict(latest_check) if latest_check else None,
                "next_due_check": schedule["next_check_at"] if schedule else None,
                "raw_hash": latest_version["content_hash"] if latest_version else None,
                "normalized_hash": latest_version["normalized_hash"] if latest_version else None,
                "latest_comparison_class": comparison.get("comparison_class"),
            },
            "knowledge_status": {
                "active_version_ids": [row["version_id"] for row in active_versions],
                "pending_version_ids": [row["version_id"] for row in pending_versions],
                "review_required": bool(comparison.get("candidate_required")),
                "changed_anchors": list(comparison.get("changed_anchors") or []),
            },
            "runtime_status": {
                "active_authority_bundle_version_id": active_bundle["authority_bundle_version_id"] if active_bundle else None,
                "posture": "REVIEW_NEW_EXECUTIONS"
                if comparison.get("candidate_required")
                else "CONTINUE_PINNED",
                "activation_ready": False,
            },
        }

    def latest_source_acquisition_uri(self, *, source_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                hosted.sql_query("""
                SELECT source_provenance_json
                FROM legal_source_versions
                WHERE source_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """, """
                SELECT source_provenance_json
                FROM legal_source_versions
                WHERE source_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """),
                (source_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown source_id: {source_id}")
        provenance = json.loads(row["source_provenance_json"] or "{}")
        source_ref = str(provenance.get("source_ref") or "").strip()
        parsed = urlparse(source_ref)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return source_ref
        return None

    def create_regulatory_notification(
        self,
        *,
        organization_id: str,
        event_type: str,
        severity: str,
        notification: dict[str, Any],
        source_id: str | None = None,
        refresh_check_id: str | None = None,
        candidate_id: str | None = None,
        impact_report_ref: str | None = None,
        recipient_role: str | None = None,
        recipient_user_id: str | None = None,
        audit_event_ref: str | None = None,
        correlation_id: str | None = None,
        causation_id: str | None = None,
    ) -> dict[str, Any]:
        event_type = str(event_type or "").strip().upper()
        severity = str(severity or "").strip().upper()
        if event_type not in LEGAL_REGULATORY_NOTIFICATION_EVENTS:
            raise ValueError("unsupported regulatory notification event_type")
        if severity not in LEGAL_REGULATORY_NOTIFICATION_SEVERITIES:
            raise ValueError("unsupported regulatory notification severity")
        if not recipient_role and not recipient_user_id:
            raise ValueError("recipient_role or recipient_user_id is required")
        self._require_nonempty({"organization_id": organization_id})
        alert_id = f"legal_alert_{uuid4().hex}"
        now = now_iso()
        with self._connect() as conn:
            if source_id:
                source = conn.execute(
                    hosted.sql_query("SELECT organization_id FROM legal_sources WHERE source_id = ?", 'SELECT organization_id FROM legal_sources WHERE source_id = %s'),
                    (source_id,),
                ).fetchone()
                if source is None:
                    raise KeyError(f"Unknown source_id: {source_id}")
                if source["organization_id"] != organization_id:
                    raise ValueError("notification source organization mismatch")
            if refresh_check_id:
                self._assert_ref_exists(conn, "source_refresh_checks", "refresh_check_id", refresh_check_id)
            if candidate_id:
                self._assert_ref_exists(conn, "legal_semantic_candidates", "candidate_id", candidate_id)
            conn.execute(
                hosted.sql_query("""
                INSERT INTO legal_regulatory_notifications (
                  alert_id, organization_id, source_id, refresh_check_id, candidate_id,
                  impact_report_ref, event_type, severity, recipient_role,
                  recipient_user_id, audit_event_ref, correlation_id, causation_id,
                  notification_json, lifecycle_state, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
                """, """
                INSERT INTO legal_regulatory_notifications (
                  alert_id, organization_id, source_id, refresh_check_id, candidate_id,
                  impact_report_ref, event_type, severity, recipient_role,
                  recipient_user_id, audit_event_ref, correlation_id, causation_id,
                  notification_json, lifecycle_state, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'OPEN', %s)
                """),
                (
                    alert_id,
                    organization_id,
                    source_id,
                    refresh_check_id,
                    candidate_id,
                    impact_report_ref,
                    event_type,
                    severity,
                    recipient_role,
                    recipient_user_id,
                    audit_event_ref,
                    correlation_id,
                    causation_id,
                    canonical_json(notification),
                    now,
                ),
            )
        return {
            "alert_id": alert_id,
            "organization_id": organization_id,
            "source_id": source_id,
            "event_type": event_type,
            "severity": severity,
            "recipient_role": recipient_role,
            "recipient_user_id": recipient_user_id,
            "lifecycle_state": "OPEN",
            "created_at": now,
        }

    def acknowledge_regulatory_alert(
        self,
        *,
        organization_id: str,
        alert_id: str,
        acknowledged_by: str,
        acknowledgement: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._require_nonempty(
            {
                "organization_id": organization_id,
                "alert_id": alert_id,
                "acknowledged_by": acknowledged_by,
            }
        )
        acknowledgement_id = f"legal_alert_ack_{uuid4().hex}"
        audit_event_id = f"legal_audit_{uuid4().hex}"
        now = now_iso()
        with self._connect() as conn:
            alert = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_regulatory_notifications
                WHERE organization_id = ? AND alert_id = ?
                """, """
                SELECT *
                FROM legal_regulatory_notifications
                WHERE organization_id = %s AND alert_id = %s
                """),
                (organization_id, alert_id),
            ).fetchone()
            if alert is None:
                raise KeyError(f"Unknown alert_id: {alert_id}")
            existing = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_regulatory_notification_acknowledgements
                WHERE alert_id = ? AND acknowledged_by = ?
                """, """
                SELECT *
                FROM legal_regulatory_notification_acknowledgements
                WHERE alert_id = %s AND acknowledged_by = %s
                """),
                (alert_id, acknowledged_by),
            ).fetchone()
            if existing is not None:
                return dict(existing)
            conn.execute(
                hosted.sql_query("""
                INSERT INTO legal_audit_events (
                  audit_event_id, event_family, object_ref, object_type, lifecycle_state,
                  attempted_endpoint, actor_ref, organization_id, reason_code,
                  route_policy_id, correlation_id, event_json, created_at
                ) VALUES (?, 'LEGAL_REGULATORY_ALERT_ACKNOWLEDGED', ?, 'legal_regulatory_notification',
                  'ACKNOWLEDGED', NULL, ?, ?, NULL, NULL, ?, ?, ?)
                """, """
                INSERT INTO legal_audit_events (
                  audit_event_id, event_family, object_ref, object_type, lifecycle_state,
                  attempted_endpoint, actor_ref, organization_id, reason_code,
                  route_policy_id, correlation_id, event_json, created_at
                ) VALUES (%s, 'LEGAL_REGULATORY_ALERT_ACKNOWLEDGED', %s, 'legal_regulatory_notification',
                  'ACKNOWLEDGED', NULL, %s, %s, NULL, NULL, %s, %s, %s)
                """),
                (
                    audit_event_id,
                    alert_id,
                    acknowledged_by,
                    organization_id,
                    alert["correlation_id"],
                    canonical_json(acknowledgement or {}),
                    now,
                ),
            )
            conn.execute(
                hosted.sql_query("""
                INSERT INTO legal_regulatory_notification_acknowledgements (
                  acknowledgement_id, alert_id, organization_id, acknowledged_by,
                  acknowledgement_json, audit_event_ref, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO legal_regulatory_notification_acknowledgements (
                  acknowledgement_id, alert_id, organization_id, acknowledged_by,
                  acknowledgement_json, audit_event_ref, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """),
                (
                    acknowledgement_id,
                    alert_id,
                    organization_id,
                    acknowledged_by,
                    canonical_json(acknowledgement or {}),
                    audit_event_id,
                    now,
                ),
            )
            conn.execute(
                hosted.sql_query("""
                UPDATE legal_regulatory_notifications
                SET lifecycle_state = 'ACKNOWLEDGED'
                WHERE alert_id = ?
                """, """
                UPDATE legal_regulatory_notifications
                SET lifecycle_state = 'ACKNOWLEDGED'
                WHERE alert_id = %s
                """),
                (alert_id,),
            )
        return {
            "acknowledgement_id": acknowledgement_id,
            "alert_id": alert_id,
            "organization_id": organization_id,
            "acknowledged_by": acknowledged_by,
            "audit_event_ref": audit_event_id,
            "created_at": now,
        }

    def list_regulatory_notifications(
        self,
        *,
        organization_id: str,
        recipient_user_id: str | None = None,
        recipient_role: str | None = None,
        lifecycle_state: str | None = "OPEN",
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        self._require_nonempty({"organization_id": organization_id})
        if limit <= 0:
            raise ValueError("limit must be positive")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        params: list[Any] = [organization_id]
        clauses = [hosted.sql_query("organization_id = ?", 'organization_id = %s')]
        if lifecycle_state:
            clauses.append(hosted.sql_query("lifecycle_state = ?", 'lifecycle_state = %s'))
            params.append(str(lifecycle_state).strip().upper())
        recipient_clauses: list[str] = []
        if recipient_user_id:
            recipient_clauses.append(hosted.sql_query("recipient_user_id = ?", 'recipient_user_id = %s'))
            params.append(str(recipient_user_id).strip())
        if recipient_role:
            recipient_clauses.append(hosted.sql_query("(recipient_user_id IS NULL AND recipient_role = ?)", '(recipient_user_id IS NULL AND recipient_role = %s)'))
            params.append(str(recipient_role).strip())
        if recipient_clauses:
            clauses.append(f"({' OR '.join(recipient_clauses)})")
        params.extend([limit, offset])
        with self._connect() as conn:
            rows = conn.execute(
                hosted.sql_query(f"""
                SELECT *
                FROM legal_regulatory_notifications
                WHERE {' AND '.join(clauses)}
                ORDER BY created_at DESC, alert_id DESC
                LIMIT ? OFFSET ?
                """, f"\n                SELECT *\n                FROM legal_regulatory_notifications\n                WHERE {' AND '.join(clauses)}\n                ORDER BY created_at DESC, alert_id DESC\n                LIMIT %s OFFSET %s\n                "),
                params,
            ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["notification"] = json.loads(item.pop("notification_json") or "{}")
            results.append(item)
        return results

    def register_source(
        self,
        *,
        organization_id: str,
        workspace_id: str | None = None,
        source_id: str | None = None,
        canonical_title: str,
        issuer: str,
        authority_tier: str,
        jurisdiction: str,
        expected_status: str,
        document_number: str | None = None,
        governance_state: str = "LISTED",
    ) -> dict[str, Any]:
        self._require_nonempty(
            {
                "organization_id": organization_id,
                "canonical_title": canonical_title,
                "issuer": issuer,
                "authority_tier": authority_tier,
                "jurisdiction": jurisdiction,
                "expected_status": expected_status,
                "governance_state": governance_state,
            }
        )
        if jurisdiction == "UNSPECIFIED":
            raise ValueError("legal source jurisdiction must not be UNSPECIFIED")
        source_id = str(source_id or "").strip() or f"legal_source_{uuid4().hex}"
        if not source_id.isascii():
            raise ValueError("legal source source_id must be ASCII-stable")
        now = now_iso()
        row_hash = content_hash(
            {
                "organization_id": organization_id,
                "workspace_id": workspace_id,
                "canonical_title": canonical_title,
                "issuer": issuer,
                "authority_tier": authority_tier,
                "jurisdiction": jurisdiction,
                "document_number": document_number,
                "expected_status": expected_status,
                "governance_state": governance_state,
            }
        )
        try:
            with self._connect() as conn:
                conn.execute(
                    hosted.sql_query("""
                    INSERT INTO legal_sources (
                      source_id, organization_id, workspace_id, canonical_title, issuer, authority_tier, jurisdiction,
                      document_number, expected_status, governance_state, created_at, content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, """
                    INSERT INTO legal_sources (
                      source_id, organization_id, workspace_id, canonical_title, issuer, authority_tier, jurisdiction,
                      document_number, expected_status, governance_state, created_at, content_hash
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """),
                    (
                        source_id,
                        organization_id,
                        workspace_id,
                        canonical_title,
                        issuer,
                        authority_tier,
                        jurisdiction,
                        document_number,
                        expected_status,
                        governance_state,
                        now,
                        row_hash,
                    ),
                )
        except hosted.integrity_errors() as exc:
            raise ValueError("duplicate legal source identity") from exc
        return self.get_source(source_id)

    def promote_source_to_organization_scope(
        self,
        *,
        organization_id: str,
        source_id: str,
        expected_current_workspace_id: str,
        expected_current_content_hash: str,
        actor_ref: str,
        reason: str,
    ) -> dict[str, Any]:
        """Promote one exact workspace-scoped legal source to organization-wide visibility.

        This is a compare-and-swap authority transition on the source owner row only.
        Version, artifact, section, provenance, effectivity, and governance records are
        not rewritten. The source row content hash is recomputed because workspace scope
        participates in source identity hashing.
        """
        self._require_nonempty(
            {
                "organization_id": organization_id,
                "source_id": source_id,
                "expected_current_workspace_id": expected_current_workspace_id,
                "expected_current_content_hash": expected_current_content_hash,
                "actor_ref": actor_ref,
                "reason": reason,
            }
        )
        with self._connect() as conn:
            row = conn.execute(
                hosted.sql_query("SELECT * FROM legal_sources WHERE source_id = ?", 'SELECT * FROM legal_sources WHERE source_id = %s'),
                (source_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown source_id: {source_id}")
            current = dict(row)
            if current["organization_id"] != organization_id:
                raise ValueError("LEGAL_SOURCE_ORGANIZATION_MISMATCH")

            promoted_hash = content_hash(
                {
                    "organization_id": current["organization_id"],
                    "workspace_id": None,
                    "canonical_title": current["canonical_title"],
                    "issuer": current["issuer"],
                    "authority_tier": current["authority_tier"],
                    "jurisdiction": current["jurisdiction"],
                    "document_number": current["document_number"],
                    "expected_status": current["expected_status"],
                    "governance_state": current["governance_state"],
                }
            )
            if current["workspace_id"] is None:
                if current["content_hash"] != promoted_hash:
                    raise ValueError("LEGAL_SOURCE_CONTENT_HASH_MISMATCH")
                return current
            if current["workspace_id"] != expected_current_workspace_id:
                raise ValueError("LEGAL_SOURCE_SCOPE_MISMATCH")
            if current["content_hash"] != expected_current_content_hash:
                raise ValueError("LEGAL_SOURCE_CONTENT_HASH_MISMATCH")
            updated = conn.execute(
                hosted.sql_query("""
                UPDATE legal_sources
                SET workspace_id = NULL, content_hash = ?
                WHERE source_id = ? AND organization_id = ?
                  AND workspace_id = ? AND content_hash = ?
                """, """
                UPDATE legal_sources
                SET workspace_id = NULL, content_hash = %s
                WHERE source_id = %s AND organization_id = %s
                  AND workspace_id = %s AND content_hash = %s
                """),
                (
                    promoted_hash,
                    source_id,
                    organization_id,
                    expected_current_workspace_id,
                    expected_current_content_hash,
                ),
            )
            if updated.rowcount != 1:
                raise ValueError("LEGAL_SOURCE_SCOPE_CAS_FAILED")
        return self.get_source(source_id)

    def create_source_version(
        self,
        *,
        source_id: str,
        version_identity: str,
        source_hash: str,
        effective_from: str,
        source_provenance: dict[str, Any],
        normalized_hash: str | None = None,
        effective_to: str | None = None,
        promulgated_at: str | None = None,
        lifecycle_state: str = "REGISTERED",
    ) -> LegalSourceVersion:
        self._require_nonempty(
            {
                "source_id": source_id,
                "version_identity": version_identity,
                "source_hash": source_hash,
                "lifecycle_state": lifecycle_state,
            }
        )
        effective_from, effective_to = validate_effectivity_window(
            {"effective_from": effective_from, "effective_to": effective_to},
            prefix="legal source version",
        )
        validate_source_provenance(
            {"source_provenance": source_provenance},
            prefix="legal source version",
        )
        version_id = f"legal_source_version_{uuid4().hex}"
        now = now_iso()
        try:
            with self._connect() as conn:
                if conn.execute(
                    hosted.sql_query("SELECT 1 FROM legal_sources WHERE source_id = ?", 'SELECT 1 FROM legal_sources WHERE source_id = %s'),
                    (source_id,),
                ).fetchone() is None:
                    raise KeyError(f"Unknown source_id: {source_id}")
                conn.execute(
                    hosted.sql_query("""
                    INSERT INTO legal_source_versions (
                      version_id, source_id, version_identity, content_hash, normalized_hash,
                      effective_from, effective_to, promulgated_at, lifecycle_state,
                      source_provenance_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, """
                    INSERT INTO legal_source_versions (
                      version_id, source_id, version_identity, content_hash, normalized_hash,
                      effective_from, effective_to, promulgated_at, lifecycle_state,
                      source_provenance_json, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """),
                    (
                        version_id,
                        source_id,
                        version_identity,
                        source_hash,
                        normalized_hash,
                        effective_from,
                        effective_to,
                        promulgated_at,
                        lifecycle_state,
                        canonical_json(source_provenance),
                        now,
                    ),
                )
        except hosted.integrity_errors() as exc:
            raise ValueError("duplicate or ambiguous legal source version") from exc
        return LegalSourceVersion(
            version_id=version_id,
            source_id=source_id,
            version_identity=version_identity,
            content_hash=source_hash,
            lifecycle_state=lifecycle_state,
        )

    def record_certification_event(
        self,
        *,
        object_ref: str,
        object_type: str,
        decision: str,
        reviewer_ref: str,
        reviewer_role: str,
        evidence_hash: str,
        event_payload: dict[str, Any],
    ) -> dict[str, Any]:
        if object_type in {"CuratedKnowledgeArtifact", "DemoPromotionRecord"}:
            raise ValueError(LEGAL_DEMO_ARTIFACT_NOT_CERTIFIABLE)
        self._validate_certification_review(
            object_type=object_type,
            reviewer_role=reviewer_role,
            event_payload=event_payload,
        )
        self._require_nonempty(
            {
                "object_ref": object_ref,
                "object_type": object_type,
                "decision": decision,
                "reviewer_ref": reviewer_ref,
                "reviewer_role": reviewer_role,
                "evidence_hash": evidence_hash,
            }
        )
        event_id = f"legal_certification_{uuid4().hex}"
        with self._connect() as conn:
            conn.execute(
                hosted.sql_query("""
                INSERT INTO legal_certification_events (
                  certification_event_id, object_ref, object_type, decision, reviewer_ref,
                  reviewer_role, evidence_hash, event_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO legal_certification_events (
                  certification_event_id, object_ref, object_type, decision, reviewer_ref,
                  reviewer_role, evidence_hash, event_json, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """),
                (
                    event_id,
                    object_ref,
                    object_type,
                    decision,
                    reviewer_ref,
                    reviewer_role,
                    evidence_hash,
                    canonical_json(event_payload),
                    now_iso(),
                ),
            )
        return {"certification_event_id": event_id, "object_ref": object_ref, "decision": decision}

    def supersede_active_bundle(
        self,
        *,
        prior_authority_bundle_version_id: str,
        new_source_registry_snapshot_id: str,
        new_knowledge_context_version_id: str,
        validation_result_ref: str,
        replay_dry_run_hash: str,
        approver_refs: list[str],
        content_hashes: dict[str, str],
        idempotency_key: str,
    ) -> dict[str, Any]:
        new_bundle = self.materialize_authority_bundle_version(
            source_registry_snapshot_id=new_source_registry_snapshot_id,
            knowledge_context_version_id=new_knowledge_context_version_id,
            validation_result_ref=validation_result_ref,
            replay_dry_run_hash=replay_dry_run_hash,
            lifecycle_state="CERTIFIED",
        )
        activation = self.record_activation_event(
            source_registry_snapshot_id=new_source_registry_snapshot_id,
            knowledge_context_version_id=new_knowledge_context_version_id,
            authority_bundle_version_id=new_bundle.authority_bundle_version_id,
            activation_performed=True,
            approver_refs=approver_refs,
            content_hashes=content_hashes,
            idempotency_key=idempotency_key,
        )
        with self._connect() as conn:
            self._assert_ref_exists(
                conn,
                "legal_authority_bundle_versions",
                "authority_bundle_version_id",
                prior_authority_bundle_version_id,
            )
            conn.execute(
                hosted.sql_query("""
                UPDATE legal_authority_bundle_versions
                SET lifecycle_state = 'SUPERSEDED'
                WHERE authority_bundle_version_id = ?
                """, """
                UPDATE legal_authority_bundle_versions
                SET lifecycle_state = 'SUPERSEDED'
                WHERE authority_bundle_version_id = %s
                """),
                (prior_authority_bundle_version_id,),
            )
            conn.execute(
                hosted.sql_query("""
                UPDATE legal_authority_bundle_versions
                SET lifecycle_state = 'ACTIVE'
                WHERE authority_bundle_version_id = ?
                """, """
                UPDATE legal_authority_bundle_versions
                SET lifecycle_state = 'ACTIVE'
                WHERE authority_bundle_version_id = %s
                """),
                (new_bundle.authority_bundle_version_id,),
            )
        return {
            "prior_authority_bundle_version_id": prior_authority_bundle_version_id,
            "new_authority_bundle_version_id": new_bundle.authority_bundle_version_id,
            "activation_event_id": activation["activation_event_id"],
        }

    def preserve_artifact(
        self,
        *,
        version_id: str,
        artifact_kind: str,
        uri: str,
        content: bytes,
        mime_type: str,
        storage_ref: str,
        http_metadata: dict[str, Any] | None = None,
        redirect_metadata: list[dict[str, Any]] | None = None,
        retrieved_at: str | None = None,
    ) -> dict[str, Any]:
        self._require_nonempty(
            {
                "version_id": version_id,
                "artifact_kind": artifact_kind,
                "uri": uri,
                "mime_type": mime_type,
                "storage_ref": storage_ref,
            }
        )
        if not content:
            raise ValueError("artifact content is required")
        artifact_hash = content_hash({"sha256_bytes": content.hex()})
        artifact_id = f"legal_artifact_{uuid4().hex}"
        retrieved_at = _format_utc_z(_parse_aware_utc(retrieved_at or now_iso(), field_name="retrieved_at"))
        with self._connect() as conn:
            if conn.execute(
                hosted.sql_query("SELECT 1 FROM legal_source_versions WHERE version_id = ?", 'SELECT 1 FROM legal_source_versions WHERE version_id = %s'),
                (version_id,),
            ).fetchone() is None:
                raise KeyError(f"Unknown version_id: {version_id}")
            existing = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_source_artifacts
                WHERE version_id = ? AND artifact_kind = ? AND content_hash = ?
                """, """
                SELECT *
                FROM legal_source_artifacts
                WHERE version_id = %s AND artifact_kind = %s AND content_hash = %s
                """),
                (version_id, artifact_kind, artifact_hash),
            ).fetchone()
            if existing is not None:
                row = dict(existing)
                row["idempotent_replay"] = True
                return row
            source = conn.execute(
                hosted.sql_query("SELECT src.source_id, src.organization_id FROM legal_sources src "
                "JOIN legal_source_versions v ON v.source_id = src.source_id WHERE v.version_id = ?", 'SELECT src.source_id, src.organization_id FROM legal_sources src JOIN legal_source_versions v ON v.source_id = src.source_id WHERE v.version_id = %s'),
                (version_id,),
            ).fetchone()
            conn.execute(
                hosted.sql_query("INSERT INTO source_ingestion_events "
                "(ingestion_event_id, source_id, version_id, event_type, event_json, organization_id, created_at) "
                "VALUES (?, ?, ?, 'ARTIFACT_BYTES_PRESERVED', ?, ?, ?)", "INSERT INTO source_ingestion_events (ingestion_event_id, source_id, version_id, event_type, event_json, organization_id, created_at) VALUES (%s, %s, %s, 'ARTIFACT_BYTES_PRESERVED', %s, %s, %s)"),
                (artifact_id, source["source_id"], version_id,
                 canonical_json({"artifact_id": artifact_id, "content_hex": content.hex(),
                                 "content_hash": artifact_hash}), source["organization_id"], retrieved_at),
            )
            conn.execute(
                hosted.sql_query("""
                INSERT INTO legal_source_artifacts (
                  artifact_id, version_id, artifact_kind, uri, retrieved_at,
                  http_metadata_json, mime_type, redirect_metadata_json,
                  content_hash, storage_ref, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO legal_source_artifacts (
                  artifact_id, version_id, artifact_kind, uri, retrieved_at,
                  http_metadata_json, mime_type, redirect_metadata_json,
                  content_hash, storage_ref, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """),
                (
                    artifact_id,
                    version_id,
                    artifact_kind,
                    uri,
                    retrieved_at,
                    canonical_json(http_metadata or {}),
                    mime_type,
                    canonical_json(redirect_metadata or []),
                    artifact_hash,
                    storage_ref,
                    now_iso(),
                ),
            )
        return {
            "artifact_id": artifact_id,
            "version_id": version_id,
            "artifact_kind": artifact_kind,
            "uri": uri,
            "retrieved_at": retrieved_at,
            "mime_type": mime_type,
            "content_hash": artifact_hash,
            "storage_ref": storage_ref,
        }

    def read_preserved_artifact(self, *, artifact_id: str, organization_id: str,
                                source_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                hosted.sql_query("SELECT a.*, e.event_json FROM legal_source_artifacts a JOIN source_ingestion_events e "
                "ON e.ingestion_event_id = a.artifact_id AND e.event_type = 'ARTIFACT_BYTES_PRESERVED' "
                "WHERE a.artifact_id = ? AND e.organization_id = ? AND e.source_id = ?", "SELECT a.*, e.event_json FROM legal_source_artifacts a JOIN source_ingestion_events e ON e.ingestion_event_id = a.artifact_id AND e.event_type = 'ARTIFACT_BYTES_PRESERVED' WHERE a.artifact_id = %s AND e.organization_id = %s AND e.source_id = %s"),
                (artifact_id, organization_id, source_id),
            ).fetchone()
        if row is None:
            raise KeyError("BASELINE_BYTES_MISSING")
        artifact = dict(row)
        artifact["content_hex"] = json.loads(artifact.pop("event_json"))["content_hex"]
        if content_hash({"sha256_bytes": artifact["content_hex"]}) != artifact["content_hash"]:
            raise ValueError("BASELINE_HASH_MISMATCH")
        artifact["http_metadata"] = json.loads(artifact.pop("http_metadata_json"))
        return artifact

    def resolve_refresh_baseline(self, *, organization_id: str, source_id: str,
                                 as_of: str) -> dict[str, Any]:
        """Pin the unique effective ACTIVE version, never the latest observation."""
        when = _parse_aware_utc(as_of, field_name="as_of")
        with self._connect() as conn:
            rows = conn.execute(
                hosted.sql_query("SELECT v.* FROM legal_source_versions v JOIN legal_sources src ON src.source_id = v.source_id "
                "WHERE src.organization_id = ? AND v.source_id = ? AND v.lifecycle_state = 'ACTIVE'", "SELECT v.* FROM legal_source_versions v JOIN legal_sources src ON src.source_id = v.source_id WHERE src.organization_id = %s AND v.source_id = %s AND v.lifecycle_state = 'ACTIVE'"),
                (organization_id, source_id),
            ).fetchall()
            effective = [dict(row) for row in rows
                         if _parse_aware_utc(row["effective_from"], field_name="effective_from") <= when
                         and (not row["effective_to"] or when < _parse_aware_utc(row["effective_to"], field_name="effective_to"))]
            if len(effective) != 1:
                raise ValueError("BASELINE_AMBIGUOUS" if len(effective) > 1 else "BASELINE_MISSING")
            version = effective[0]
            artifacts = conn.execute(
                hosted.sql_query("SELECT a.*, e.event_json FROM legal_source_artifacts a JOIN source_ingestion_events e "
                "ON e.ingestion_event_id = a.artifact_id AND e.event_type = 'ARTIFACT_BYTES_PRESERVED' "
                "WHERE a.version_id = ? AND a.content_hash = ? ORDER BY a.artifact_id", "SELECT a.*, e.event_json FROM legal_source_artifacts a JOIN source_ingestion_events e ON e.ingestion_event_id = a.artifact_id AND e.event_type = 'ARTIFACT_BYTES_PRESERVED' WHERE a.version_id = %s AND a.content_hash = %s ORDER BY a.artifact_id"),
                (version["version_id"], version["content_hash"]),
            ).fetchall()
        if not artifacts:
            raise ValueError("BASELINE_BYTES_MISSING")
        artifact = dict(artifacts[0])
        raw = bytes.fromhex(json.loads(artifact.pop("event_json"))["content_hex"])
        if content_hash({"sha256_bytes": raw.hex()}) != version["content_hash"]:
            raise ValueError("BASELINE_HASH_MISMATCH")
        uri = str(json.loads(version["source_provenance_json"]).get("source_ref") or "")
        if urlparse(uri).scheme not in {"http", "https"}:
            raise ValueError("ACQUISITION_URI_MISSING")
        return {"version_id": version["version_id"], "artifact_id": artifact["artifact_id"],
                "content_hash": version["content_hash"], "normalized_hash": version["normalized_hash"],
                "uri": uri}

    def pin_refresh_basis(self, *, organization_id: str, source_id: str, scan_item_id: str,
                          lease_owner: str, as_of: str, policy_version_ref: str) -> dict[str, Any]:
        with self._connect() as conn:
            existing = conn.execute(
                hosted.sql_query("SELECT event_json FROM source_ingestion_events WHERE organization_id = ? AND source_id = ? "
                "AND correlation_id = ? AND event_type = 'REFRESH_BASIS_PINNED'", "SELECT event_json FROM source_ingestion_events WHERE organization_id = %s AND source_id = %s AND correlation_id = %s AND event_type = 'REFRESH_BASIS_PINNED'"),
                (organization_id, source_id, scan_item_id),
            ).fetchone()
        if existing:
            payload = json.loads(existing[0])
            if payload["policy_version_ref"] != policy_version_ref:
                raise ValueError("POLICY_CHANGED_DURING_RUN")
            return payload["baseline"]
        baseline = self.resolve_refresh_baseline(organization_id=organization_id, source_id=source_id, as_of=as_of)
        with self._connect() as conn:
            hosted.begin_write(conn)
            item = conn.execute(hosted.sql_query("SELECT * FROM legal_scan_items WHERE scan_item_id = ?", 'SELECT * FROM legal_scan_items WHERE scan_item_id = %s'), (scan_item_id,)).fetchone()
            if item is None or item["status"] != "CLAIMED" or item["lease_owner"] != lease_owner:
                raise ValueError("SCAN_LEASE_OWNERSHIP_LOST")
            # The claim is exclusive; append the immutable basis before any network acquisition.
            conn.execute(
                hosted.sql_query("INSERT INTO source_ingestion_events (ingestion_event_id, source_id, event_type, event_json, "
                "actor_ref, organization_id, correlation_id, created_at) VALUES (?, ?, 'REFRESH_BASIS_PINNED', ?, ?, ?, ?, ?)", "INSERT INTO source_ingestion_events (ingestion_event_id, source_id, event_type, event_json, actor_ref, organization_id, correlation_id, created_at) VALUES (%s, %s, 'REFRESH_BASIS_PINNED', %s, %s, %s, %s, %s)"),
                (f"legal_basis_{uuid4().hex}", source_id,
                 canonical_json({"baseline": baseline, "policy_version_ref": policy_version_ref}),
                 lease_owner, organization_id, scan_item_id, now_iso()),
            )
        return baseline

    def preserve_acquired_observation(
        self, *, source_id: str, organization_id: str, correlation_id: str, actor_ref: str,
        uri: str, content: bytes, mime_type: str, http_metadata: dict[str, Any],
        redirect_metadata: list[dict[str, Any]], baseline: dict[str, Any],
    ) -> dict[str, Any]:
        """Append actual acquisition evidence without assigning legal effectivity/provenance."""
        artifact_id = f"legal_observation_{uuid4().hex}"
        payload = {"artifact_id": artifact_id, "source_id": source_id, "uri": uri,
                   "content_hex": content.hex(), "content_hash": content_hash({"sha256_bytes": content.hex()}),
                   "mime_type": mime_type, "http_metadata": http_metadata,
                   "redirect_metadata": redirect_metadata, "retrieved_at": now_iso(), "baseline": baseline,
                   "event_type": "ACQUIRED_OBSERVATION", "version_id": None}
        with self._connect() as conn:
            hosted.begin_write(conn)
            self._source_for_policy(conn, organization_id=organization_id, source_id=source_id, source_ref=source_id)
            item = conn.execute(hosted.sql_query("SELECT * FROM legal_scan_items WHERE scan_item_id = ? AND organization_id = ? AND source_id = ?", 'SELECT * FROM legal_scan_items WHERE scan_item_id = %s AND organization_id = %s AND source_id = %s'),
                                (correlation_id, organization_id, source_id)).fetchone()
            if (item is None or item["status"] != "CLAIMED" or item["lease_owner"] != actor_ref
                or _parse_aware_utc(item["lease_expires_at"], field_name="lease_expires_at") <= _parse_aware_utc(now_iso(), field_name="now")):
                raise ValueError("SCAN_LEASE_OWNERSHIP_LOST")
            existing = conn.execute(
                hosted.sql_query("SELECT event_json FROM source_ingestion_events WHERE organization_id = ? AND source_id = ? "
                "AND correlation_id = ? AND event_type = 'ACQUIRED_OBSERVATION'", "SELECT event_json FROM source_ingestion_events WHERE organization_id = %s AND source_id = %s AND correlation_id = %s AND event_type = 'ACQUIRED_OBSERVATION'"),
                (organization_id, source_id, correlation_id),
            ).fetchone()
            if existing:
                return json.loads(existing["event_json"])
            conn.execute(
                hosted.sql_query("INSERT INTO source_ingestion_events (ingestion_event_id, source_id, event_type, event_json, "
                "actor_ref, organization_id, correlation_id, created_at) VALUES (?, ?, 'ACQUIRED_OBSERVATION', ?, ?, ?, ?, ?)", "INSERT INTO source_ingestion_events (ingestion_event_id, source_id, event_type, event_json, actor_ref, organization_id, correlation_id, created_at) VALUES (%s, %s, 'ACQUIRED_OBSERVATION', %s, %s, %s, %s, %s)"),
                (artifact_id, source_id, canonical_json(payload), actor_ref, organization_id, correlation_id, payload["retrieved_at"]),
            )
        return payload

    def read_acquired_observation(self, *, artifact_id: str, organization_id: str,
                                  source_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                hosted.sql_query("SELECT event_json FROM source_ingestion_events WHERE ingestion_event_id = ? "
                "AND organization_id = ? AND source_id = ? AND event_type = 'ACQUIRED_OBSERVATION'", "SELECT event_json FROM source_ingestion_events WHERE ingestion_event_id = %s AND organization_id = %s AND source_id = %s AND event_type = 'ACQUIRED_OBSERVATION'"),
                (artifact_id, organization_id, source_id),
            ).fetchone()
        if row is None:
            raise KeyError("ACQUISITION_EVIDENCE_MISSING")
        payload = json.loads(row["event_json"])
        if content_hash({"sha256_bytes": payload["content_hex"]}) != payload["content_hash"]:
            raise ValueError("ACQUISITION_HASH_MISMATCH")
        return payload

    def scan_item_observation(self, *, organization_id: str, source_id: str,
                              scan_item_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                hosted.sql_query("SELECT ingestion_event_id FROM source_ingestion_events WHERE organization_id = ? "
                "AND source_id = ? AND correlation_id = ? AND event_type = 'ACQUIRED_OBSERVATION'", "SELECT ingestion_event_id FROM source_ingestion_events WHERE organization_id = %s AND source_id = %s AND correlation_id = %s AND event_type = 'ACQUIRED_OBSERVATION'"),
                (organization_id, source_id, scan_item_id),
            ).fetchone()
        return self.read_acquired_observation(artifact_id=row[0], organization_id=organization_id,
                                             source_id=source_id) if row else None

    def quarantine_acquisition(
        self,
        *,
        source_id: str,
        reason_code: str,
        uri: str,
        event_payload: dict[str, Any],
        actor_ref: str,
        organization_id: str,
        correlation_id: str,
    ) -> dict[str, Any]:
        event_id = f"legal_ingestion_{uuid4().hex}"
        with self._connect() as conn:
            conn.execute(
                hosted.sql_query("""
                INSERT INTO source_ingestion_events (
                  ingestion_event_id, source_id, version_id, event_type, event_json,
                  actor_ref, organization_id, correlation_id, created_at
                ) VALUES (?, ?, NULL, 'QUARANTINED', ?, ?, ?, ?, ?)
                """, """
                INSERT INTO source_ingestion_events (
                  ingestion_event_id, source_id, version_id, event_type, event_json,
                  actor_ref, organization_id, correlation_id, created_at
                ) VALUES (%s, %s, NULL, 'QUARANTINED', %s, %s, %s, %s, %s)
                """),
                (
                    event_id,
                    source_id,
                    canonical_json(
                        {
                            "reason_code": reason_code,
                            "uri": uri,
                            "payload": event_payload,
                        }
                    ),
                    actor_ref,
                    organization_id,
                    correlation_id,
                    now_iso(),
                ),
            )
        return {
            "ingestion_event_id": event_id,
            "event_type": "QUARANTINED",
            "reason_code": reason_code,
            "result_class": event_payload.get("result_class", "SOURCE_UNAVAILABLE"),
        }

    def record_refresh_check(
        self,
        *,
        source_id: str,
        checked_uri: str,
        status: str,
        result: dict[str, Any],
        observed_content: bytes | None = None,
        checked_at: str | None = None,
    ) -> dict[str, Any]:
        self._require_nonempty(
            {
                "source_id": source_id,
                "checked_uri": checked_uri,
                "status": status,
            }
        )
        observed_hash = (
            content_hash({"sha256_bytes": observed_content.hex()})
            if observed_content is not None
            else None
        )
        refresh_check_id = f"legal_refresh_{uuid4().hex}"
        checked_at = _format_utc_z(_parse_aware_utc(checked_at or now_iso(), field_name="checked_at"))
        with self._connect() as conn:
            if conn.execute(
                hosted.sql_query("SELECT 1 FROM legal_sources WHERE source_id = ?", 'SELECT 1 FROM legal_sources WHERE source_id = %s'),
                (source_id,),
            ).fetchone() is None:
                raise KeyError(f"Unknown source_id: {source_id}")
            conn.execute(
                hosted.sql_query("""
                INSERT INTO source_refresh_checks (
                  refresh_check_id, source_id, checked_uri, checked_at, status,
                  observed_content_hash, result_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO source_refresh_checks (
                  refresh_check_id, source_id, checked_uri, checked_at, status,
                  observed_content_hash, result_json, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """),
                (
                    refresh_check_id,
                    source_id,
                    checked_uri,
                    checked_at,
                    status,
                    observed_hash,
                    canonical_json(result),
                    now_iso(),
                ),
            )
        return {
            "refresh_check_id": refresh_check_id,
            "source_id": source_id,
            "status": status,
            "observed_content_hash": observed_hash,
            "checked_at": checked_at,
        }

    def schedule_refresh(
        self,
        *,
        source_id: str,
        interval_seconds: int,
        next_check_at: str,
        operator_ref: str,
        lifecycle_state: str = "ACTIVE",
        priority: int = 100,
        retry_after: str | None = None,
        last_result_class: str | None = None,
    ) -> dict[str, Any]:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if priority < 0:
            raise ValueError("priority must be non-negative")
        self._require_nonempty(
            {
                "source_id": source_id,
                "next_check_at": next_check_at,
                "operator_ref": operator_ref,
                "lifecycle_state": lifecycle_state,
            }
        )
        # Reuse effectivity parser for timezone-aware scheduling validation.
        validate_effectivity_window(
            {"effective_from": next_check_at, "effective_to": None},
            prefix="legal refresh schedule",
        )
        if retry_after:
            validate_effectivity_window(
                {"effective_from": retry_after, "effective_to": None},
                prefix="legal refresh retry_after",
            )
        schedule_id = f"legal_refresh_schedule_{uuid4().hex}"
        with self._connect() as conn:
            self._assert_ref_exists(conn, "legal_sources", "source_id", source_id)
            existing = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_source_refresh_schedules
                WHERE source_id = ? AND lifecycle_state = ?
                """, """
                SELECT *
                FROM legal_source_refresh_schedules
                WHERE source_id = %s AND lifecycle_state = %s
                """),
                (source_id, lifecycle_state),
            ).fetchone()
            if existing is not None:
                conn.execute(
                    hosted.sql_query("""
                    UPDATE legal_source_refresh_schedules
                    SET interval_seconds = ?, next_check_at = ?, operator_ref = ?,
                        priority = ?, retry_after = ?, last_result_class = ?
                    WHERE schedule_id = ?
                    """, """
                    UPDATE legal_source_refresh_schedules
                    SET interval_seconds = %s, next_check_at = %s, operator_ref = %s,
                        priority = %s, retry_after = %s, last_result_class = %s
                    WHERE schedule_id = %s
                    """),
                    (
                        interval_seconds,
                        next_check_at,
                        operator_ref,
                        priority,
                        retry_after,
                        last_result_class,
                        existing["schedule_id"],
                    ),
                )
                schedule_id = existing["schedule_id"]
            else:
                conn.execute(
                    hosted.sql_query("""
                    INSERT INTO legal_source_refresh_schedules (
                      schedule_id, source_id, interval_seconds, next_check_at,
                      operator_ref, lifecycle_state, priority, retry_after,
                      last_result_class, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, """
                    INSERT INTO legal_source_refresh_schedules (
                      schedule_id, source_id, interval_seconds, next_check_at,
                      operator_ref, lifecycle_state, priority, retry_after,
                      last_result_class, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """),
                    (
                        schedule_id,
                        source_id,
                        interval_seconds,
                        next_check_at,
                        operator_ref,
                        lifecycle_state,
                        priority,
                        retry_after,
                        last_result_class,
                        now_iso(),
                    ),
                )
        return {
            "schedule_id": schedule_id,
            "source_id": source_id,
            "interval_seconds": interval_seconds,
            "next_check_at": next_check_at,
            "lifecycle_state": lifecycle_state,
            "priority": priority,
            "retry_after": retry_after,
            "last_result_class": last_result_class,
        }

    def due_refresh_sources(
        self,
        *,
        organization_id: str,
        as_of: str,
        limit: int | None = None,
        offset: int = 0,
        include_retry_pending: bool = True,
    ) -> list[dict[str, Any]]:
        validate_effectivity_window(
            {"effective_from": as_of, "effective_to": None},
            prefix="legal refresh as_of",
        )
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        with self._connect() as conn:
            params: list[Any] = [organization_id, as_of]
            retry_clause = ""
            if include_retry_pending:
                retry_clause = hosted.sql_query("AND (sched.retry_after IS NULL OR sched.retry_after <= ?)", 'AND (sched.retry_after IS NULL OR sched.retry_after <= %s)')
                params.append(as_of)
            if limit is not None:
                limit_clause = hosted.sql_query("LIMIT ? OFFSET ?", 'LIMIT %s OFFSET %s')
                params.extend([limit, offset])
            else:
                limit_clause = ""
            rows = conn.execute(
                hosted.sql_query(f"""
                SELECT sched.*, src.organization_id
                FROM legal_source_refresh_schedules sched
                JOIN legal_sources src ON src.source_id = sched.source_id
                WHERE src.organization_id = ?
                  AND sched.lifecycle_state = 'ACTIVE'
                  AND sched.next_check_at <= ?
                  {retry_clause}
                ORDER BY sched.priority ASC, sched.next_check_at ASC, sched.source_id ASC
                {limit_clause}
                """, f"\n                SELECT sched.*, src.organization_id\n                FROM legal_source_refresh_schedules sched\n                JOIN legal_sources src ON src.source_id = sched.source_id\n                WHERE src.organization_id = %s\n                  AND sched.lifecycle_state = 'ACTIVE'\n                  AND sched.next_check_at <= %s\n                  {retry_clause}\n                ORDER BY sched.priority ASC, sched.next_check_at ASC, sched.source_id ASC\n                {limit_clause}\n                "),
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def run_operator_refresh_check(
        self,
        *,
        source_id: str,
        checked_uri: str,
        observed_content: bytes,
        active_content_hash: str | None,
        operator_ref: str,
    ) -> dict[str, Any]:
        observed_hash = content_hash({"sha256_bytes": observed_content.hex()})
        status = "UNCHANGED" if active_content_hash and observed_hash == active_content_hash else "CHANGE_DETECTED"
        result = {
            "operator_ref": operator_ref,
            "review_state": "NO_CHANGE" if status == "UNCHANGED" else "PENDING_REVIEW",
            "activation_performed": False,
        }
        return self.record_refresh_check(
            source_id=source_id,
            checked_uri=checked_uri,
            status=status,
            observed_content=observed_content,
            result=result,
        )

    def create_version_edge(
        self,
        *,
        from_version_id: str,
        to_version_id: str,
        edge_type: str,
        rationale: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        edge_type = str(edge_type or "").strip().upper()
        if edge_type not in LEGAL_VERSION_EDGE_TYPES:
            raise ValueError(LEGAL_UNSUPPORTED_VERSION_EDGE)
        edge_id = f"legal_version_edge_{uuid4().hex}"
        with self._connect() as conn:
            for version_id in {from_version_id, to_version_id}:
                if conn.execute(
                    hosted.sql_query("SELECT 1 FROM legal_source_versions WHERE version_id = ?", 'SELECT 1 FROM legal_source_versions WHERE version_id = %s'),
                    (version_id,),
                ).fetchone() is None:
                    raise KeyError(f"Unknown version_id: {version_id}")
            existing = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_source_version_edges
                WHERE from_version_id = ? AND to_version_id = ? AND edge_type = ?
                """, """
                SELECT *
                FROM legal_source_version_edges
                WHERE from_version_id = %s AND to_version_id = %s AND edge_type = %s
                """),
                (from_version_id, to_version_id, edge_type),
            ).fetchone()
            if existing is not None:
                row = dict(existing)
                row["idempotent_replay"] = True
                return row
            conn.execute(
                hosted.sql_query("""
                INSERT INTO legal_source_version_edges (
                  edge_id, from_version_id, to_version_id, edge_type, rationale_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO legal_source_version_edges (
                  edge_id, from_version_id, to_version_id, edge_type, rationale_json, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """),
                (
                    edge_id,
                    from_version_id,
                    to_version_id,
                    edge_type,
                    canonical_json(rationale or {}),
                    now_iso(),
                ),
            )
        return {
            "edge_id": edge_id,
            "from_version_id": from_version_id,
            "to_version_id": to_version_id,
            "edge_type": edge_type,
        }

    def activate_source_version_directly(self, *, version_id: str) -> None:
        raise ValueError(LEGAL_VERSION_AUTO_ACTIVATION_FORBIDDEN)

    def persist_sections(
        self,
        *,
        version_id: str,
        sections: list[LegalSectionAnchor],
    ) -> list[dict[str, Any]]:
        self._require_nonempty({"version_id": version_id})
        persisted: list[dict[str, Any]] = []
        with self._connect() as conn:
            if conn.execute(
                hosted.sql_query("SELECT 1 FROM legal_source_versions WHERE version_id = ?", 'SELECT 1 FROM legal_source_versions WHERE version_id = %s'),
                (version_id,),
            ).fetchone() is None:
                raise KeyError(f"Unknown version_id: {version_id}")
            for section in sections:
                section_id = f"legal_section_{uuid4().hex}"
                conn.execute(
                    hosted.sql_query("""
                    INSERT INTO legal_source_sections (
                      section_id, version_id, stable_locator, parent_section_id,
                      section_type, ordinal, title, text_hash, local_hash, anchor_json, created_at
                    ) VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(version_id, stable_locator) DO UPDATE SET
                      section_type = excluded.section_type,
                      ordinal = excluded.ordinal,
                      title = excluded.title,
                      text_hash = excluded.text_hash,
                      local_hash = excluded.local_hash,
                      anchor_json = excluded.anchor_json
                    """, """
                    INSERT INTO legal_source_sections (
                      section_id, version_id, stable_locator, parent_section_id,
                      section_type, ordinal, title, text_hash, local_hash, anchor_json, created_at
                    ) VALUES (%s, %s, %s, NULL, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT(version_id, stable_locator) DO UPDATE SET
                      section_type = excluded.section_type,
                      ordinal = excluded.ordinal,
                      title = excluded.title,
                      text_hash = excluded.text_hash,
                      local_hash = excluded.local_hash,
                      anchor_json = excluded.anchor_json
                    """),
                    (
                        section_id,
                        version_id,
                        section.stable_locator,
                        section.section_type,
                        section.ordinal,
                        section.title,
                        section.text_hash,
                        section.local_hash,
                        canonical_json({**section.anchor, "retained_text": section.text}),
                        now_iso(),
                    ),
                )
                row = conn.execute(
                    hosted.sql_query("""
                    SELECT *
                    FROM legal_source_sections
                    WHERE version_id = ? AND stable_locator = ?
                    """, """
                    SELECT *
                    FROM legal_source_sections
                    WHERE version_id = %s AND stable_locator = %s
                    """),
                    (version_id, section.stable_locator),
                ).fetchone()
                if row is not None:
                    persisted.append(dict(row))
        return persisted

    def compare_source_versions(
        self,
        *,
        prior_version_id: str,
        observed_version_id: str | None = None,
        acquired_observation: dict[str, Any] | None = None,
        normalized_text: str | None = None,
        baseline_normalized_text: str | None = None,
    ) -> dict[str, Any]:
        if (observed_version_id is None) == (acquired_observation is None):
            raise ValueError("exactly one observed version or acquired observation is required")
        with self._connect() as conn:
            prior = conn.execute(
                hosted.sql_query("SELECT * FROM legal_source_versions WHERE version_id = ?", 'SELECT * FROM legal_source_versions WHERE version_id = %s'),
                (prior_version_id,),
            ).fetchone()
            observed = conn.execute(
                hosted.sql_query("SELECT * FROM legal_source_versions WHERE version_id = ?", 'SELECT * FROM legal_source_versions WHERE version_id = %s'),
                (observed_version_id,),
            ).fetchone()
            if prior is None:
                raise KeyError(f"Unknown prior_version_id: {prior_version_id}")
            if acquired_observation is not None:
                from dataclasses import asdict
                from eos.legal_text_normalization import generate_legal_section_anchors
                acquired_observation = self.read_acquired_observation(
                    artifact_id=acquired_observation["artifact_id"],
                    organization_id=acquired_observation["organization_id"], source_id=prior["source_id"])
                observed = {"source_id": acquired_observation["source_id"],
                            "content_hash": acquired_observation["content_hash"],
                            "normalized_hash": content_hash({"text": normalized_text}) if normalized_text else None}
            elif observed is None:
                raise KeyError(f"Unknown observed_version_id: {observed_version_id}")
            if prior["source_id"] != observed["source_id"]:
                raise ValueError("source versions must belong to the same legal source")
            prior_sections = conn.execute(
                hosted.sql_query("""
                SELECT stable_locator, local_hash, text_hash, section_type
                FROM legal_source_sections
                WHERE version_id = ?
                ORDER BY stable_locator
                """, """
                SELECT stable_locator, local_hash, text_hash, section_type
                FROM legal_source_sections
                WHERE version_id = %s
                ORDER BY stable_locator
                """),
                (prior_version_id,),
            ).fetchall()
            observed_sections = conn.execute(
                hosted.sql_query("""
                SELECT stable_locator, local_hash, text_hash, section_type
                FROM legal_source_sections
                WHERE version_id = ?
                ORDER BY stable_locator
                """, """
                SELECT stable_locator, local_hash, text_hash, section_type
                FROM legal_source_sections
                WHERE version_id = %s
                ORDER BY stable_locator
                """),
                (observed_version_id,),
            ).fetchall()
            authority_edges = conn.execute(
                hosted.sql_query("""
                SELECT edge_type
                FROM legal_source_version_edges
                WHERE (
                    from_version_id = ? AND to_version_id = ?
                  ) OR (
                    from_version_id = ? AND to_version_id = ?
                  )
                """, """
                SELECT edge_type
                FROM legal_source_version_edges
                WHERE (
                    from_version_id = %s AND to_version_id = %s
                  ) OR (
                    from_version_id = %s AND to_version_id = %s
                  )
                """),
                (prior_version_id, observed_version_id, observed_version_id, prior_version_id),
            ).fetchall()
        if acquired_observation is not None:
            observed_sections = [asdict(anchor) for anchor in generate_legal_section_anchors(normalized_text or "")]
            if prior["content_hash"] != observed["content_hash"] and (not prior["normalized_hash"] or not prior_sections or not normalized_text):
                raise ValueError("BASELINE_NORMALIZATION_MISSING")
            if prior["content_hash"] != observed["content_hash"]:
                if content_hash({"text": baseline_normalized_text}) != prior["normalized_hash"]:
                    raise ValueError("BASELINE_NORMALIZATION_HASH_MISMATCH")
                expected_sections = {anchor.stable_locator: anchor.local_hash
                                     for anchor in generate_legal_section_anchors(baseline_normalized_text or "")}
                if expected_sections != {row["stable_locator"]: row["local_hash"] for row in prior_sections}:
                    raise ValueError("BASELINE_ANCHOR_MISMATCH")
        prior_by_locator = {row["stable_locator"]: dict(row) for row in prior_sections}
        observed_by_locator = {row["stable_locator"]: dict(row) for row in observed_sections}
        prior_locators = set(prior_by_locator)
        observed_locators = set(observed_by_locator)
        added = [
            observed_by_locator[locator]
            for locator in sorted(observed_locators - prior_locators)
        ]
        removed = [
            prior_by_locator[locator]
            for locator in sorted(prior_locators - observed_locators)
        ]
        modified = [
            {
                "stable_locator": locator,
                "before_hash": prior_by_locator[locator]["local_hash"],
                "after_hash": observed_by_locator[locator]["local_hash"],
                "section_type": observed_by_locator[locator]["section_type"],
            }
            for locator in sorted(prior_locators & observed_locators)
            if prior_by_locator[locator]["local_hash"] != observed_by_locator[locator]["local_hash"]
        ]
        authority_delta = {} if acquired_observation is not None else {
            "effective_from_changed": prior["effective_from"] != observed["effective_from"],
            "effective_to_changed": prior["effective_to"] != observed["effective_to"],
            "promulgated_at_changed": prior["promulgated_at"] != observed["promulgated_at"],
            "lifecycle_state_changed": prior["lifecycle_state"] != observed["lifecycle_state"],
            "authority_edge_types": sorted({row["edge_type"] for row in authority_edges}),
        }
        has_authority_delta = any(
            value
            for key, value in authority_delta.items()
            if key != "authority_edge_types"
        ) or bool(authority_delta.get("authority_edge_types"))
        section_diff = {"added": added, "removed": removed, "modified": modified}
        has_section_delta = bool(added or removed or modified)
        normalized_equal = (
            prior["normalized_hash"]
            and observed["normalized_hash"]
            and prior["normalized_hash"] == observed["normalized_hash"]
        )
        normalized_changed = (
            prior["normalized_hash"]
            and observed["normalized_hash"]
            and prior["normalized_hash"] != observed["normalized_hash"]
        )
        if acquired_observation is not None and prior["content_hash"] == observed["content_hash"]:
            comparison_class = "SEMANTICALLY_UNCHANGED"
            section_diff = {"added": [], "removed": [], "modified": []}
            added, removed, modified = [], [], []
        elif has_authority_delta:
            comparison_class = "AUTHORITY_STATE_CHANGED"
        elif normalized_changed or has_section_delta:
            comparison_class = "CONTENT_CHANGE_DETECTED"
        elif prior["content_hash"] != observed["content_hash"] and normalized_equal:
            comparison_class = "REPRESENTATION_CHANGED"
        elif (prior["content_hash"] and observed["content_hash"] and prior["content_hash"] == observed["content_hash"]) or normalized_equal:
            comparison_class = "SEMANTICALLY_UNCHANGED"
        else:
            comparison_class = "UNKNOWN_REVIEW_REQUIRED"
        changed_anchors = [
            row["stable_locator"]
            for row in [*added, *removed, *modified]
        ]
        return {
            "prior_version_id": prior_version_id,
            "observed_version_id": observed_version_id,
            "prior_content_hash": prior["content_hash"],
            "observed_content_hash": observed["content_hash"],
            "prior_normalized_hash": prior["normalized_hash"],
            "observed_normalized_hash": observed["normalized_hash"],
            "authority_comparison": "UNAVAILABLE" if acquired_observation is not None else "GOVERNED_VERSIONS",
            "source_id": prior["source_id"],
            "comparison_class": comparison_class,
            "section_diff": section_diff,
            "changed_anchors": changed_anchors,
            "authority_delta": authority_delta,
            "candidate_required": comparison_class in {
                "CONTENT_CHANGE_DETECTED",
                "AUTHORITY_STATE_CHANGED",
                "UNKNOWN_REVIEW_REQUIRED",
            },
            "activation_performed": False,
        }

    def persist_semantic_candidates(
        self,
        *,
        version_id: str,
        section_id: str | None,
        candidates: list[LegalSemanticCandidate],
    ) -> list[dict[str, Any]]:
        self._require_nonempty({"version_id": version_id})
        persisted: list[dict[str, Any]] = []
        with self._connect() as conn:
            if conn.execute(
                hosted.sql_query("SELECT 1 FROM legal_source_versions WHERE version_id = ?", 'SELECT 1 FROM legal_source_versions WHERE version_id = %s'),
                (version_id,),
            ).fetchone() is None:
                raise KeyError(f"Unknown version_id: {version_id}")
            if section_id and conn.execute(
                hosted.sql_query("SELECT 1 FROM legal_source_sections WHERE section_id = ?", 'SELECT 1 FROM legal_source_sections WHERE section_id = %s'),
                (section_id,),
            ).fetchone() is None:
                raise KeyError(f"Unknown section_id: {section_id}")
            for candidate in candidates:
                validated = validate_candidate(candidate)
                candidate_id = f"legal_candidate_{uuid4().hex}"
                conn.execute(
                    hosted.sql_query("""
                    INSERT INTO legal_semantic_candidates (
                      candidate_id, version_id, section_id, candidate_type, candidate_json,
                      source_span_json, extraction_method, review_state, candidate_hash,
                      rejection_reason, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, """
                    INSERT INTO legal_semantic_candidates (
                      candidate_id, version_id, section_id, candidate_type, candidate_json,
                      source_span_json, extraction_method, review_state, candidate_hash,
                      rejection_reason, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """),
                    (
                        candidate_id,
                        version_id,
                        section_id,
                        validated.candidate_type,
                        canonical_json(validated.candidate),
                        canonical_json(validated.source_span),
                        validated.extraction_method,
                        validated.review_state,
                        validated.candidate_hash,
                        validated.rejection_reason,
                        now_iso(),
                    ),
                )
                persisted.append(
                    {
                        "candidate_id": candidate_id,
                        "version_id": version_id,
                        "section_id": section_id,
                        "candidate_type": validated.candidate_type,
                        "review_state": validated.review_state,
                        "candidate_hash": validated.candidate_hash,
                        "rejection_reason": validated.rejection_reason,
                    }
                )
        return persisted

    def register_source_dependency_link(
        self,
        *,
        source_ref: str,
        dependency_ref: str,
        dependency_type: str,
        dependency: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._require_nonempty(
            {
                "source_ref": source_ref,
                "dependency_ref": dependency_ref,
                "dependency_type": dependency_type,
            }
        )
        link_id = f"source_dependency_{uuid4().hex}"
        with self._connect() as conn:
            existing = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM source_dependency_links
                WHERE source_ref = ? AND dependency_ref = ? AND dependency_type = ?
                """, """
                SELECT *
                FROM source_dependency_links
                WHERE source_ref = %s AND dependency_ref = %s AND dependency_type = %s
                """),
                (source_ref, dependency_ref, dependency_type),
            ).fetchone()
            if existing is not None:
                row = dict(existing)
                row["idempotent_replay"] = True
                return row
            conn.execute(
                hosted.sql_query("""
                INSERT INTO source_dependency_links (
                  dependency_link_id, source_ref, dependency_ref, dependency_type,
                  dependency_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO source_dependency_links (
                  dependency_link_id, source_ref, dependency_ref, dependency_type,
                  dependency_json, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """),
                (
                    link_id,
                    source_ref,
                    dependency_ref,
                    dependency_type,
                    canonical_json(dependency or {}),
                    now_iso(),
                ),
            )
        return {
            "dependency_link_id": link_id,
            "source_ref": source_ref,
            "dependency_ref": dependency_ref,
            "dependency_type": dependency_type,
        }

    def read_finding_candidate_trace(
        self, *, organization_id: str, workspace_id: str,
        finding_ref: str, finding_hash: str, pattern_ref: str, pattern_hash: str,
    ) -> tuple[dict[str, Any], ...]:
        """Optional current candidate context, never the saved decision's legal basis.

        Exact scoped pins only. No database creation, migration, approval or
        inference. Invalid candidate/source joins are omitted.
        """
        if not hosted.enabled() and (self._db_path is None or not self._db_path.is_file()):
            return ()
        reader = self._connect() if hosted.enabled() else sqlite3.connect(self._db_path.resolve().as_uri() + "?mode=ro", uri=True)
        with reader as conn:
            if hosted.is_postgres(conn):
                conn.execute('SET TRANSACTION READ ONLY')
            else:
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA query_only=ON")
            rows = conn.execute(
                hosted.sql_query("""
                SELECT c.*, s.source_id, s.canonical_title,
                       v.version_identity,
                       sec.stable_locator, sec.text_hash, sec.local_hash, sec.anchor_json
                FROM legal_semantic_candidates c
                JOIN legal_source_versions v ON v.version_id=c.version_id
                JOIN legal_sources s ON s.source_id=v.source_id
                JOIN legal_source_sections sec ON sec.section_id=c.section_id AND sec.version_id=v.version_id
                WHERE s.organization_id=? AND (s.workspace_id=? OR s.workspace_id IS NULL)
                  AND c.candidate_type='issue_dependency' AND c.review_state='UNRESOLVED'
                  AND json_valid(c.candidate_json)
                  AND json_extract(c.candidate_json, '$.finding_ref')=?
                  AND json_extract(c.candidate_json, '$.finding_hash')=?
                  AND json_extract(c.candidate_json, '$.pattern_ref')=?
                  AND json_extract(c.candidate_json, '$.pattern_hash')=?
                ORDER BY s.source_id, c.version_id, sec.stable_locator, c.candidate_id
                """, """
                SELECT c.*, s.source_id, s.canonical_title,
                       v.version_identity,
                       sec.stable_locator, sec.text_hash, sec.local_hash, sec.anchor_json
                FROM legal_semantic_candidates c
                JOIN legal_source_versions v ON v.version_id=c.version_id
                JOIN legal_sources s ON s.source_id=v.source_id
                JOIN legal_source_sections sec ON sec.section_id=c.section_id AND sec.version_id=v.version_id
                WHERE s.organization_id=%s AND (s.workspace_id=%s OR s.workspace_id IS NULL)
                  AND c.candidate_type='issue_dependency' AND c.review_state='UNRESOLVED'
                ORDER BY s.source_id, c.version_id, sec.stable_locator, c.candidate_id
                """),
                (organization_id, workspace_id) if hosted.is_postgres(conn) else (organization_id, workspace_id, finding_ref, finding_hash, pattern_ref, pattern_hash),
            ).fetchall()
        result = []
        for row in rows:
            try:
                candidate = json.loads(row["candidate_json"])
                span = json.loads(row["source_span_json"])
                quote = json.loads(row["anchor_json"]).get("retained_text")
                roles = candidate.get("dependency_roles")
                digest = content_hash(dict(candidate_type=row["candidate_type"], candidate=candidate,
                    source_span=span, extraction_method=row["extraction_method"]))
                if (
                    candidate.get('finding_ref') != finding_ref
                    or candidate.get('finding_hash') != finding_hash
                    or candidate.get('pattern_ref') != pattern_ref
                    or candidate.get('pattern_hash') != pattern_hash
                    or digest != row["candidate_hash"]
                    or not isinstance(quote, str) or not quote
                    or content_hash({"text": quote}) != row["text_hash"]
                    or candidate.get("proposed_section_id") != row["section_id"]
                    or span.get("stable_locator") != row["stable_locator"]
                    or span.get("local_hash") != row["local_hash"]
                    or not isinstance(roles, list) or not roles
                    or any(role not in ("rule_authority", "scope_authority", "effectivity_authority",
                        "precedence_authority", "exception_authority") for role in roles)
                ):
                    continue
                projected = dict(
                    candidate_ref=row["candidate_id"], candidate_hash="sha256:" + digest,
                    finding_ref=finding_ref, finding_hash=finding_hash,
                    pattern_ref=pattern_ref, pattern_hash=pattern_hash,
                    source_ref=row["source_id"], source_title=row["canonical_title"],
                    version_ref=row["version_id"], version_identity=row["version_identity"],
                    section_ref=row["section_id"], section_locator=row["stable_locator"],
                    quote=quote, quote_hash="sha256:" + row["text_hash"],
                    proposed_roles=tuple(roles), review_state="UNRESOLVED",
                    applicability="UNKNOWN", test_only=candidate.get("test_only") is True,
                    applicability_review=candidate.get("applicability_review"),
                    review_context_status="AVAILABLE" if candidate.get("applicability_review") is not None else "UNAVAILABLE",
                )
                from contracts.tender_decision_basis import TenderProvisionCandidateV1
                try:
                    projected = TenderProvisionCandidateV1.model_validate(projected).model_dump(mode="json")
                except ValueError:
                    projected["applicability_review"] = None
                    projected["review_context_status"] = "INVALID"
                result.append(projected)
            except (ValueError, TypeError, AttributeError):
                continue
        return tuple(result)

    def list_change_candidates(
        self,
        *,
        organization_id: str,
        review_state: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        params: list[Any] = [organization_id]
        review_clause = ""
        if review_state:
            review_clause = hosted.sql_query("AND c.review_state = ?", 'AND c.review_state = %s')
            params.append(str(review_state).strip().upper())
        params.extend([limit, offset])
        with self._connect() as conn:
            rows = conn.execute(
                hosted.sql_query(f"""
                SELECT
                  c.*, v.source_id, v.version_identity,
                  v.lifecycle_state AS version_lifecycle_state,
                  s.organization_id, s.canonical_title,
                  sec.stable_locator, sec.section_type
                FROM legal_semantic_candidates c
                JOIN legal_source_versions v ON v.version_id = c.version_id
                JOIN legal_sources s ON s.source_id = v.source_id
                LEFT JOIN legal_source_sections sec ON sec.section_id = c.section_id
                WHERE s.organization_id = ?
                  {review_clause}
                ORDER BY c.created_at DESC, c.candidate_id DESC
                LIMIT ? OFFSET ?
                """, f'\n                SELECT\n                  c.*, v.source_id, v.version_identity,\n                  v.lifecycle_state AS version_lifecycle_state,\n                  s.organization_id, s.canonical_title,\n                  sec.stable_locator, sec.section_type\n                FROM legal_semantic_candidates c\n                JOIN legal_source_versions v ON v.version_id = c.version_id\n                JOIN legal_sources s ON s.source_id = v.source_id\n                LEFT JOIN legal_source_sections sec ON sec.section_id = c.section_id\n                WHERE s.organization_id = %s\n                  {review_clause}\n                ORDER BY c.created_at DESC, c.candidate_id DESC\n                LIMIT %s OFFSET %s\n                '),
                params,
            ).fetchall()
        return [self._candidate_row_to_dict(row) for row in rows]

    def compute_candidate_impact(
        self,
        *,
        organization_id: str,
        candidate_id: str,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                hosted.sql_query("""
                SELECT
                  c.*, v.source_id, v.version_identity,
                  v.lifecycle_state AS version_lifecycle_state,
                  s.organization_id, s.canonical_title,
                  sec.stable_locator, sec.section_type
                FROM legal_semantic_candidates c
                JOIN legal_source_versions v ON v.version_id = c.version_id
                JOIN legal_sources s ON s.source_id = v.source_id
                LEFT JOIN legal_source_sections sec ON sec.section_id = c.section_id
                WHERE s.organization_id = ? AND c.candidate_id = ?
                """, """
                SELECT
                  c.*, v.source_id, v.version_identity,
                  v.lifecycle_state AS version_lifecycle_state,
                  s.organization_id, s.canonical_title,
                  sec.stable_locator, sec.section_type
                FROM legal_semantic_candidates c
                JOIN legal_source_versions v ON v.version_id = c.version_id
                JOIN legal_sources s ON s.source_id = v.source_id
                LEFT JOIN legal_source_sections sec ON sec.section_id = c.section_id
                WHERE s.organization_id = %s AND c.candidate_id = %s
                """),
                (organization_id, candidate_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown candidate_id: {candidate_id}")
            candidate = self._candidate_row_to_dict(row)
            edge_rows = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_source_version_edges
                WHERE from_version_id = ? OR to_version_id = ?
                ORDER BY edge_type, from_version_id, to_version_id
                """, """
                SELECT *
                FROM legal_source_version_edges
                WHERE from_version_id = %s OR to_version_id = %s
                ORDER BY edge_type, from_version_id, to_version_id
                """),
                (candidate["version_id"], candidate["version_id"]),
            ).fetchall()
            dependency_refs = {
                candidate_id,
                candidate["version_id"],
                candidate["source_id"],
            }
            if candidate.get("stable_locator"):
                dependency_refs.add(str(candidate["stable_locator"]))
            placeholders = ",".join(hosted.sql_query("?", '%s') for _ in dependency_refs)
            dependency_rows = conn.execute(
                f"""
                SELECT *
                FROM source_dependency_links
                WHERE source_ref IN ({placeholders})
                ORDER BY dependency_type, dependency_ref, source_ref
                """,
                tuple(sorted(dependency_refs)),
            ).fetchall()
        dependencies = [
            {
                **dict(dep_row),
                "dependency": json.loads(dep_row["dependency_json"] or "{}"),
            }
            for dep_row in dependency_rows
        ]
        authority_edges = [
            {
                **dict(edge_row),
                "rationale": json.loads(edge_row["rationale_json"] or "{}"),
            }
            for edge_row in edge_rows
        ]
        posture = _impact_posture(
            dependency_types={row["dependency_type"] for row in dependencies},
            has_authority_edges=bool(authority_edges),
        )
        impact_report_ref = content_hash(
            {
                "candidate_id": candidate_id,
                "dependency_link_ids": [row["dependency_link_id"] for row in dependencies],
                "edge_ids": [row["edge_id"] for row in authority_edges],
                "posture": posture,
            }
        )
        return {
            "impact_report_ref": impact_report_ref,
            "candidate": candidate,
            "legal_anchors": [candidate["source_span"]],
            "authority_edges": authority_edges,
            "dependencies": dependencies,
            "posture": posture,
            "activation_performed": False,
        }

    def _candidate_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["candidate"] = json.loads(result.pop("candidate_json") or "{}")
        result["source_span"] = json.loads(result.pop("source_span_json") or "{}")
        return result

    def record_activation_event(
        self,
        *,
        source_registry_snapshot_id: str,
        knowledge_context_version_id: str,
        authority_bundle_version_id: str,
        activation_performed: bool,
        approver_refs: list[str],
        content_hashes: dict[str, str],
        idempotency_key: str,
    ) -> dict[str, Any]:
        if not activation_performed:
            raise ValueError(LEGAL_ACTIVATION_EVIDENCE_REQUIRED)
        if not approver_refs or not content_hashes:
            raise ValueError(LEGAL_ACTIVATION_EVIDENCE_REQUIRED)
        self._require_nonempty(
            {
                "source_registry_snapshot_id": source_registry_snapshot_id,
                "knowledge_context_version_id": knowledge_context_version_id,
                "authority_bundle_version_id": authority_bundle_version_id,
                "idempotency_key": idempotency_key,
            }
        )
        event_id = f"legal_activation_{uuid4().hex}"
        with self._connect() as conn:
            self._assert_ref_exists(
                conn,
                "legal_source_registry_snapshots",
                "snapshot_id",
                source_registry_snapshot_id,
            )
            self._assert_ref_exists(
                conn,
                "legal_knowledge_context_versions",
                "knowledge_context_version_id",
                knowledge_context_version_id,
            )
            self._assert_ref_exists(
                conn,
                "legal_authority_bundle_versions",
                "authority_bundle_version_id",
                authority_bundle_version_id,
            )
            existing = conn.execute(
                hosted.sql_query("""
                SELECT activation_event_id, activation_performed
                FROM legal_activation_events
                WHERE idempotency_key = ?
                """, """
                SELECT activation_event_id, activation_performed
                FROM legal_activation_events
                WHERE idempotency_key = %s
                """),
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                return {
                    "activation_event_id": existing["activation_event_id"],
                    "activation_performed": bool(existing["activation_performed"]),
                    "idempotent_replay": True,
                }
            conn.execute(
                hosted.sql_query("""
                INSERT INTO legal_activation_events (
                  activation_event_id, source_registry_snapshot_id, knowledge_context_version_id,
                  authority_bundle_version_id, activation_performed, approver_refs_json,
                  content_hashes_json, idempotency_key, created_at
                ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)
                """, """
                INSERT INTO legal_activation_events (
                  activation_event_id, source_registry_snapshot_id, knowledge_context_version_id,
                  authority_bundle_version_id, activation_performed, approver_refs_json,
                  content_hashes_json, idempotency_key, created_at
                ) VALUES (%s, %s, %s, %s, 1, %s, %s, %s, %s)
                """),
                (
                    event_id,
                    source_registry_snapshot_id,
                    knowledge_context_version_id,
                    authority_bundle_version_id,
                    canonical_json(approver_refs),
                    canonical_json(content_hashes),
                    idempotency_key,
                    now_iso(),
                ),
            )
        return {"activation_event_id": event_id, "activation_performed": True}

    def materialize_source_registry_snapshot(
        self,
        *,
        source_version_ids: list[str],
        certification_event_ids: list[str],
        lifecycle_state: str = "CERTIFIED",
    ) -> dict[str, Any]:
        if not source_version_ids:
            raise ValueError("source_version_ids are required")
        if not certification_event_ids:
            raise ValueError("certification_event_ids are required")
        snapshot_id = f"legal_snapshot_{uuid4().hex}"
        snapshot_hash = content_hash(
            {
                "source_version_ids": sorted(source_version_ids),
                "certification_event_ids": sorted(certification_event_ids),
                "lifecycle_state": lifecycle_state,
            }
        )
        with self._connect() as conn:
            for version_id in source_version_ids:
                self._assert_ref_exists(conn, "legal_source_versions", "version_id", version_id)
            for event_id in certification_event_ids:
                self._assert_ref_exists(
                    conn,
                    "legal_certification_events",
                    "certification_event_id",
                    event_id,
                )
            conn.execute(
                hosted.sql_query("""
                INSERT INTO legal_source_registry_snapshots (
                  snapshot_id, source_version_ids_json, snapshot_hash,
                  certification_event_ids_json, lifecycle_state, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO legal_source_registry_snapshots (
                  snapshot_id, source_version_ids_json, snapshot_hash,
                  certification_event_ids_json, lifecycle_state, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """),
                (
                    snapshot_id,
                    canonical_json(sorted(source_version_ids)),
                    snapshot_hash,
                    canonical_json(sorted(certification_event_ids)),
                    lifecycle_state,
                    now_iso(),
                ),
            )
        return {
            "snapshot_id": snapshot_id,
            "source_version_ids": sorted(source_version_ids),
            "snapshot_hash": snapshot_hash,
            "lifecycle_state": lifecycle_state,
        }

    def materialize_knowledge_context_version(
        self,
        *,
        source_registry_snapshot_id: str,
        ontology_version_ref: str | None = None,
        claim_version_refs: list[str] | None = None,
        lifecycle_state: str = "CERTIFIED",
    ) -> KnowledgeContextVersion:
        claim_version_refs = claim_version_refs or []
        with self._connect() as conn:
            self._assert_ref_exists(
                conn,
                "legal_source_registry_snapshots",
                "snapshot_id",
                source_registry_snapshot_id,
            )
            content = {
                "source_registry_snapshot_id": source_registry_snapshot_id,
                "ontology_version_ref": ontology_version_ref,
                "claim_version_refs": sorted(claim_version_refs),
                "lifecycle_state": lifecycle_state,
            }
            context_hash = content_hash(content)
            context_id = f"legal_kv_{uuid4().hex}"
            conn.execute(
                hosted.sql_query("""
                INSERT INTO legal_knowledge_context_versions (
                  knowledge_context_version_id, source_registry_snapshot_id,
                  ontology_version_ref, claim_version_refs_json, content_hash,
                  lifecycle_state, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO legal_knowledge_context_versions (
                  knowledge_context_version_id, source_registry_snapshot_id,
                  ontology_version_ref, claim_version_refs_json, content_hash,
                  lifecycle_state, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """),
                (
                    context_id,
                    source_registry_snapshot_id,
                    ontology_version_ref,
                    canonical_json(sorted(claim_version_refs)),
                    context_hash,
                    lifecycle_state,
                    now_iso(),
                ),
            )
        return KnowledgeContextVersion(
            knowledge_context_version_id=context_id,
            source_registry_snapshot_id=source_registry_snapshot_id,
            content_hash=context_hash,
            lifecycle_state=lifecycle_state,
        )

    def materialize_authority_bundle_version(
        self,
        *,
        source_registry_snapshot_id: str,
        knowledge_context_version_id: str,
        validation_result_ref: str,
        replay_dry_run_hash: str,
        lifecycle_state: str = "CERTIFIED",
    ) -> AuthorityBundleVersion:
        self._require_nonempty(
            {
                "source_registry_snapshot_id": source_registry_snapshot_id,
                "knowledge_context_version_id": knowledge_context_version_id,
                "validation_result_ref": validation_result_ref,
                "replay_dry_run_hash": replay_dry_run_hash,
                "lifecycle_state": lifecycle_state,
            }
        )
        with self._connect() as conn:
            self._assert_ref_exists(
                conn,
                "legal_source_registry_snapshots",
                "snapshot_id",
                source_registry_snapshot_id,
            )
            self._assert_ref_exists(
                conn,
                "legal_knowledge_context_versions",
                "knowledge_context_version_id",
                knowledge_context_version_id,
            )
            bundle_hash = content_hash(
                {
                    "source_registry_snapshot_id": source_registry_snapshot_id,
                    "knowledge_context_version_id": knowledge_context_version_id,
                    "validation_result_ref": validation_result_ref,
                    "replay_dry_run_hash": replay_dry_run_hash,
                    "lifecycle_state": lifecycle_state,
                }
            )
            bundle_id = f"legal_bundle_{uuid4().hex}"
            conn.execute(
                hosted.sql_query("""
                INSERT INTO legal_authority_bundle_versions (
                  authority_bundle_version_id, source_registry_snapshot_id,
                  knowledge_context_version_id, bundle_hash, validation_result_ref,
                  replay_dry_run_hash, lifecycle_state, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO legal_authority_bundle_versions (
                  authority_bundle_version_id, source_registry_snapshot_id,
                  knowledge_context_version_id, bundle_hash, validation_result_ref,
                  replay_dry_run_hash, lifecycle_state, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """),
                (
                    bundle_id,
                    source_registry_snapshot_id,
                    knowledge_context_version_id,
                    bundle_hash,
                    validation_result_ref,
                    replay_dry_run_hash,
                    lifecycle_state,
                    now_iso(),
                ),
            )
        return AuthorityBundleVersion(
            authority_bundle_version_id=bundle_id,
            source_registry_snapshot_id=source_registry_snapshot_id,
            knowledge_context_version_id=knowledge_context_version_id,
            bundle_hash=bundle_hash,
            lifecycle_state=lifecycle_state,
        )

    def register_issue_model_version(
        self,
        *,
        organization_id: str,
        issue_model_id: str,
        ontology_version_ref: str,
        resolution_rules_version_ref: str,
        required_issues: list[dict[str, Any]],
        lifecycle_state: str = "DRAFT",
        effective_from: str,
        effective_to: str | None = None,
        created_by: str,
        proposal_ref: str | None = None,
        supersedes_issue_model_version_id: str | None = None,
        precedence_basis: dict[str, Any] | None = None,
    ) -> LegalIssueModelVersion:
        self._require_nonempty(
            {
                "organization_id": organization_id,
                "issue_model_id": issue_model_id,
                "ontology_version_ref": ontology_version_ref,
                "resolution_rules_version_ref": resolution_rules_version_ref,
                "effective_from": effective_from,
                "created_by": created_by,
            }
        )
        if not required_issues:
            raise ValueError("required_issues are required")
        lifecycle_state = str(lifecycle_state or "").strip().upper()
        if lifecycle_state == "ACTIVE":
            raise ValueError(LEGAL_ISSUE_MODEL_ACTIVE_REGISTRATION_FORBIDDEN)
        if lifecycle_state not in {"DRAFT", "PROPOSED"}:
            raise ValueError("unsupported issue model lifecycle_state")
        if supersedes_issue_model_version_id:
            with self._connect() as conn:
                prior = conn.execute(
                    hosted.sql_query("""
                    SELECT *
                    FROM legal_issue_model_versions
                    WHERE issue_model_version_id = ?
                    """, """
                    SELECT *
                    FROM legal_issue_model_versions
                    WHERE issue_model_version_id = %s
                    """),
                    (supersedes_issue_model_version_id,),
                ).fetchone()
            if prior is None or prior["organization_id"] != organization_id or prior["issue_model_id"] != issue_model_id:
                raise ValueError("LEGAL_ISSUE_MODEL_SUPERSESSION_MISMATCH")
        payload = {
            "schema_version": "legal_issue_model.v1",
            "issue_model_id": issue_model_id,
            "organization_id": organization_id,
            "ontology_version_ref": ontology_version_ref,
            "resolution_rules_version_ref": resolution_rules_version_ref,
            "proposal_ref": proposal_ref,
            "supersedes_issue_model_version_id": supersedes_issue_model_version_id,
            "required_issues": sorted(
                required_issues,
                key=lambda issue: str(issue.get("issue_id") or ""),
            ),
            "lifecycle_state": lifecycle_state,
            "effective_from": _format_utc_z(
                _parse_aware_utc(effective_from, field_name="effective_from")
            ),
            "effective_to": _format_utc_z(
                _parse_aware_utc(effective_to, field_name="effective_to")
            )
            if effective_to
            else None,
        }
        if precedence_basis is not None:
            from contracts.ontology_validation import LegalPrecedenceBasisV2
            basis = LegalPrecedenceBasisV2.model_validate(precedence_basis)
            if (basis.organization_id, basis.issue_model_id, basis.ontology_version_ref) != (
                organization_id, issue_model_id, ontology_version_ref
            ) or resolution_rules_version_ref != "legal_resolution_rules.v2":
                raise ValueError("LEGAL_PRECEDENCE_SCOPE_MISMATCH")
            payload["precedence_basis"] = basis.model_dump(mode="json")
        model_hash = content_hash(payload)
        issue_model_version_id = f"legal_issue_model_{uuid4().hex}"
        with self._connect() as conn:
            conn.execute(
                hosted.sql_query("""
                INSERT INTO legal_issue_model_versions (
                  issue_model_version_id, issue_model_id, organization_id,
                  ontology_version_ref, resolution_rules_version_ref, model_json,
                  model_hash, lifecycle_state, effective_from, effective_to,
                  created_by, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO legal_issue_model_versions (
                  issue_model_version_id, issue_model_id, organization_id,
                  ontology_version_ref, resolution_rules_version_ref, model_json,
                  model_hash, lifecycle_state, effective_from, effective_to,
                  created_by, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """),
                (
                    issue_model_version_id,
                    issue_model_id,
                    organization_id,
                    ontology_version_ref,
                    resolution_rules_version_ref,
                    canonical_json(payload),
                    model_hash,
                    lifecycle_state,
                    payload["effective_from"],
                    payload["effective_to"],
                    created_by,
                    now_iso(),
                ),
            )
            event_id = f"legal_issue_model_event_{uuid4().hex}"
            conn.execute(
                hosted.sql_query("""
                INSERT INTO legal_issue_model_events (
                  event_id, issue_model_version_id, event_type, actor_ref,
                  actor_role, from_state, to_state, evidence_hash, event_json, created_at
                ) VALUES (?, ?, 'PROPOSE', ?, 'author', NULL, ?, ?, ?, ?)
                """, """
                INSERT INTO legal_issue_model_events (
                  event_id, issue_model_version_id, event_type, actor_ref,
                  actor_role, from_state, to_state, evidence_hash, event_json, created_at
                ) VALUES (%s, %s, 'PROPOSE', %s, 'author', NULL, %s, %s, %s, %s)
                """),
                (
                    event_id,
                    issue_model_version_id,
                    created_by,
                    lifecycle_state,
                    model_hash,
                    canonical_json({"proposal_ref": proposal_ref, "model_hash": model_hash}),
                    now_iso(),
                ),
            )
        return LegalIssueModelVersion(
            issue_model_version_id=issue_model_version_id,
            issue_model_id=issue_model_id,
            organization_id=organization_id,
            model_hash=model_hash,
            lifecycle_state=lifecycle_state,
        )

    def certify_issue_model_version(
        self,
        *,
        issue_model_version_id: str,
        reviewer_ref: str,
        reviewer_role: str,
        evidence_hash: str,
        event_payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_nonempty(
            {
                "issue_model_version_id": issue_model_version_id,
                "reviewer_ref": reviewer_ref,
                "reviewer_role": reviewer_role,
                "evidence_hash": evidence_hash,
            }
        )
        now = now_iso()
        event_id = f"legal_issue_model_event_{uuid4().hex}"
        with self._connect() as conn:
            row = conn.execute(
                hosted.sql_query("SELECT * FROM legal_issue_model_versions WHERE issue_model_version_id = ?", 'SELECT * FROM legal_issue_model_versions WHERE issue_model_version_id = %s'),
                (issue_model_version_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown issue_model_version_id: {issue_model_version_id}")
            if row["lifecycle_state"] not in {"DRAFT", "PROPOSED"}:
                raise ValueError("LEGAL_ISSUE_MODEL_NOT_CERTIFIABLE")
            conn.execute(
                hosted.sql_query("""
                INSERT INTO legal_certification_events (
                  certification_event_id, object_ref, object_type, decision,
                  reviewer_ref, reviewer_role, evidence_hash, event_json, created_at
                ) VALUES (?, ?, 'LegalIssueModel', 'CERTIFY', ?, ?, ?, ?, ?)
                """, """
                INSERT INTO legal_certification_events (
                  certification_event_id, object_ref, object_type, decision,
                  reviewer_ref, reviewer_role, evidence_hash, event_json, created_at
                ) VALUES (%s, %s, 'LegalIssueModel', 'CERTIFY', %s, %s, %s, %s, %s)
                """),
                (
                    event_id,
                    issue_model_version_id,
                    reviewer_ref,
                    reviewer_role,
                    evidence_hash,
                    canonical_json(event_payload),
                    now,
                ),
            )
            conn.execute(
                hosted.sql_query("""
                INSERT INTO legal_issue_model_events (
                  event_id, issue_model_version_id, event_type, actor_ref,
                  actor_role, from_state, to_state, evidence_hash, event_json, created_at
                ) VALUES (?, ?, 'CERTIFY', ?, ?, ?, 'CERTIFIED', ?, ?, ?)
                """, """
                INSERT INTO legal_issue_model_events (
                  event_id, issue_model_version_id, event_type, actor_ref,
                  actor_role, from_state, to_state, evidence_hash, event_json, created_at
                ) VALUES (%s, %s, 'CERTIFY', %s, %s, %s, 'CERTIFIED', %s, %s, %s)
                """),
                (
                    f"legal_issue_model_event_{uuid4().hex}",
                    issue_model_version_id,
                    reviewer_ref,
                    reviewer_role,
                    row["lifecycle_state"],
                    evidence_hash,
                    canonical_json(event_payload),
                    now,
                ),
            )
            conn.execute(
                hosted.sql_query("""
                UPDATE legal_issue_model_versions
                SET lifecycle_state = 'CERTIFIED'
                WHERE issue_model_version_id = ?
                """, """
                UPDATE legal_issue_model_versions
                SET lifecycle_state = 'CERTIFIED'
                WHERE issue_model_version_id = %s
                """),
                (issue_model_version_id,),
            )
        return {
            "certification_event_id": event_id,
            "issue_model_version_id": issue_model_version_id,
            "decision": "CERTIFY",
        }

    def activate_issue_model_version(
        self,
        *,
        issue_model_version_id: str,
        approver_ref: str,
        approver_role: str,
        certification_event_id: str,
        activation_evidence_hash: str,
        supersedes_issue_model_version_id: str | None = None,
    ) -> dict[str, Any]:
        self._require_nonempty(
            {
                "issue_model_version_id": issue_model_version_id,
                "approver_ref": approver_ref,
                "approver_role": approver_role,
                "certification_event_id": certification_event_id,
                "activation_evidence_hash": activation_evidence_hash,
            }
        )
        now = now_iso()
        with self._connect() as conn:
            row = conn.execute(
                hosted.sql_query("SELECT * FROM legal_issue_model_versions WHERE issue_model_version_id = ?", 'SELECT * FROM legal_issue_model_versions WHERE issue_model_version_id = %s'),
                (issue_model_version_id,),
            ).fetchone()
            cert = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_certification_events
                WHERE certification_event_id = ?
                  AND object_ref = ?
                  AND object_type = 'LegalIssueModel'
                  AND decision = 'CERTIFY'
                """, """
                SELECT *
                FROM legal_certification_events
                WHERE certification_event_id = %s
                  AND object_ref = %s
                  AND object_type = 'LegalIssueModel'
                  AND decision = 'CERTIFY'
                """),
                (certification_event_id, issue_model_version_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown issue_model_version_id: {issue_model_version_id}")
            if cert is None or row["lifecycle_state"] != "CERTIFIED":
                raise ValueError(LEGAL_ISSUE_MODEL_CERTIFICATION_REQUIRED)
            prior_ref = supersedes_issue_model_version_id or json.loads(row["model_json"]).get(
                "supersedes_issue_model_version_id"
            )
            if prior_ref:
                prior = conn.execute(
                    hosted.sql_query("""
                    SELECT *
                    FROM legal_issue_model_versions
                    WHERE issue_model_version_id = ?
                    """, """
                    SELECT *
                    FROM legal_issue_model_versions
                    WHERE issue_model_version_id = %s
                    """),
                    (prior_ref,),
                ).fetchone()
                if prior is None or prior["organization_id"] != row["organization_id"] or prior["issue_model_id"] != row["issue_model_id"]:
                    raise ValueError("LEGAL_ISSUE_MODEL_SUPERSESSION_MISMATCH")
                conn.execute(
                    hosted.sql_query("""
                    UPDATE legal_issue_model_versions
                    SET lifecycle_state = 'SUPERSEDED'
                    WHERE issue_model_version_id = ?
                    """, """
                    UPDATE legal_issue_model_versions
                    SET lifecycle_state = 'SUPERSEDED'
                    WHERE issue_model_version_id = %s
                    """),
                    (prior_ref,),
                )
                conn.execute(
                    hosted.sql_query("""
                    INSERT INTO legal_issue_model_events (
                      event_id, issue_model_version_id, event_type, actor_ref,
                      actor_role, from_state, to_state, evidence_hash, event_json, created_at
                    ) VALUES (?, ?, 'SUPERSEDE', ?, ?, ?, 'SUPERSEDED', ?, ?, ?)
                    """, """
                    INSERT INTO legal_issue_model_events (
                      event_id, issue_model_version_id, event_type, actor_ref,
                      actor_role, from_state, to_state, evidence_hash, event_json, created_at
                    ) VALUES (%s, %s, 'SUPERSEDE', %s, %s, %s, 'SUPERSEDED', %s, %s, %s)
                    """),
                    (
                        f"legal_issue_model_event_{uuid4().hex}",
                        prior_ref,
                        approver_ref,
                        approver_role,
                        prior["lifecycle_state"],
                        activation_evidence_hash,
                        canonical_json({"superseded_by": issue_model_version_id}),
                        now,
                    ),
                )
            conn.execute(
                hosted.sql_query("""
                UPDATE legal_issue_model_versions
                SET lifecycle_state = 'ACTIVE'
                WHERE issue_model_version_id = ?
                """, """
                UPDATE legal_issue_model_versions
                SET lifecycle_state = 'ACTIVE'
                WHERE issue_model_version_id = %s
                """),
                (issue_model_version_id,),
            )
            activation_event_id = f"legal_issue_model_activation_{uuid4().hex}"
            conn.execute(
                hosted.sql_query("""
                INSERT INTO legal_issue_model_events (
                  event_id, issue_model_version_id, event_type, actor_ref,
                  actor_role, from_state, to_state, evidence_hash, event_json, created_at
                ) VALUES (?, ?, 'ACTIVATE', ?, ?, 'CERTIFIED', 'ACTIVE', ?, ?, ?)
                """, """
                INSERT INTO legal_issue_model_events (
                  event_id, issue_model_version_id, event_type, actor_ref,
                  actor_role, from_state, to_state, evidence_hash, event_json, created_at
                ) VALUES (%s, %s, 'ACTIVATE', %s, %s, 'CERTIFIED', 'ACTIVE', %s, %s, %s)
                """),
                (
                    activation_event_id,
                    issue_model_version_id,
                    approver_ref,
                    approver_role,
                    activation_evidence_hash,
                    canonical_json(
                        {
                            "certification_event_id": certification_event_id,
                            "supersedes_issue_model_version_id": prior_ref,
                        }
                    ),
                    now,
                ),
            )
        return {
            "activation_event_id": activation_event_id,
            "issue_model_version_id": issue_model_version_id,
            "activation_performed": True,
        }

    def ensure_runtime_pin_artifact(
        self,
        *,
        artifact_ref: str,
        artifact_type: str,
        payload: dict[str, Any],
        lifecycle_state: str = "ACTIVE",
    ) -> dict[str, Any]:
        self._require_nonempty({"artifact_ref": artifact_ref, "artifact_type": artifact_type})
        artifact_hash = content_hash(payload)
        with self._connect() as conn:
            existing = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_runtime_pin_artifacts
                WHERE artifact_ref = ?
                """, """
                SELECT *
                FROM legal_runtime_pin_artifacts
                WHERE artifact_ref = %s
                """),
                (artifact_ref,),
            ).fetchone()
            if existing is not None:
                if existing["artifact_type"] != artifact_type or existing["artifact_hash"] != artifact_hash:
                    raise ValueError("LEGAL_RUNTIME_PIN_ARTIFACT_COLLISION")
                return dict(existing)
            conn.execute(
                hosted.sql_query("""
                INSERT INTO legal_runtime_pin_artifacts (
                  artifact_ref, artifact_type, artifact_hash, payload_json,
                  lifecycle_state, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO legal_runtime_pin_artifacts (
                  artifact_ref, artifact_type, artifact_hash, payload_json,
                  lifecycle_state, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """),
                (
                    artifact_ref,
                    artifact_type,
                    artifact_hash,
                    canonical_json(payload),
                    lifecycle_state,
                    now_iso(),
                ),
            )
        return {
            "artifact_ref": artifact_ref,
            "artifact_type": artifact_type,
            "artifact_hash": artifact_hash,
            "payload_json": canonical_json(payload),
            "lifecycle_state": lifecycle_state,
        }

    def build_replay_manifest(
        self,
        *,
        activation_event_id: str,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            activation = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_activation_events
                WHERE activation_event_id = ?
                """, """
                SELECT *
                FROM legal_activation_events
                WHERE activation_event_id = %s
                """),
                (activation_event_id,),
            ).fetchone()
            if activation is None:
                raise KeyError(f"Unknown activation_event_id: {activation_event_id}")
            snapshot = conn.execute(
                hosted.sql_query("SELECT * FROM legal_source_registry_snapshots WHERE snapshot_id = ?", 'SELECT * FROM legal_source_registry_snapshots WHERE snapshot_id = %s'),
                (activation["source_registry_snapshot_id"],),
            ).fetchone()
            context = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_knowledge_context_versions
                WHERE knowledge_context_version_id = ?
                """, """
                SELECT *
                FROM legal_knowledge_context_versions
                WHERE knowledge_context_version_id = %s
                """),
                (activation["knowledge_context_version_id"],),
            ).fetchone()
            bundle = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_authority_bundle_versions
                WHERE authority_bundle_version_id = ?
                """, """
                SELECT *
                FROM legal_authority_bundle_versions
                WHERE authority_bundle_version_id = %s
                """),
                (activation["authority_bundle_version_id"],),
            ).fetchone()
        manifest = {
            "schema": "legal_corpus_replay_manifest.v1",
            "activation_event_id": activation_event_id,
            "activation_performed": bool(activation["activation_performed"]),
            "source_registry_snapshot_id": activation["source_registry_snapshot_id"],
            "source_registry_snapshot_hash": snapshot["snapshot_hash"],
            "knowledge_context_version_id": activation["knowledge_context_version_id"],
            "knowledge_context_hash": context["content_hash"],
            "authority_bundle_version_id": activation["authority_bundle_version_id"],
            "authority_bundle_hash": bundle["bundle_hash"],
            "content_hashes": json.loads(activation["content_hashes_json"]),
        }
        return {**manifest, "manifest_hash": content_hash(manifest)}

    def build_pipeline2_evidence_contract(
        self,
        *,
        organization_id: str,
    ) -> dict[str, Any]:
        inventory = self.list_inventory(organization_id=organization_id)
        active_bundle = self._latest_activation_bundle()
        contract = {
            "schema": "legal_corpus_pipeline2_evidence_contract.v1",
            "organization_id": organization_id,
            "corpus_object_refs": {
                "source_refs": [source["source_id"] for source in inventory["sources"]],
                "source_version_refs": [version["version_id"] for version in inventory["versions"]],
                "candidate_refs": [
                    candidate["candidate_id"]
                    for candidate in inventory["semantic_candidates"]
                ],
            },
            "source_anchor_contract": {
                "anchor_ref_fields": ["version_id", "stable_locator", "local_hash"],
                "span_ref_fields": ["stable_locator", "local_hash", "char_start", "char_end"],
            },
            "authority_path_contract": {
                "edge_ref_fields": ["from_version_id", "to_version_id", "edge_type"],
                "allowed_edge_types": sorted(LEGAL_VERSION_EDGE_TYPES),
            },
            "issue_model_refs": {
                "candidate_types": sorted(
                    {
                        candidate["candidate_type"]
                        for candidate in inventory["semantic_candidates"]
                    }
                )
            },
            "active_pins": active_bundle,
            "highlighted_subgraph_binding": {
                "schema": "renderer_neutral_legal_subgraph_binding.v1",
                "node_ref_fields": ["node_type", "object_ref", "label"],
                "edge_ref_fields": ["edge_type", "from_ref", "to_ref", "source_anchor_ref"],
            },
            "pipeline1_boundary": {
                "does_not_include": [
                    "request_grounding",
                    "legal_query_ir_execution",
                    "advice_assembly",
                    "release_gating",
                    "controlled_rendering",
                    "advisory_replay_comparison",
                    "LEGAL_ADVISORY_EVIDENCE_PACKAGE",
                    "session_linked_decision_trace_page",
                ]
            },
        }
        return {**contract, "contract_hash": content_hash(contract)}

    def build_acceptance_evidence_export(
        self,
        *,
        organization_id: str,
        test_run_id: str,
        user_signoff: dict[str, Any] | None = None,
        automated_verification_summary: list[dict[str, Any]] | None = None,
        screenshots: list[dict[str, Any]] | None = None,
        uat_cases: list[dict[str, Any]] | None = None,
        workspace_id: str | None = None,
        code_revision: str | None = None,
        evidence_bundle_ref: str | None = None,
        demo_operator: str | None = None,
        domain_reviewer: str | None = None,
        product_acceptor: str | None = None,
    ) -> dict[str, Any]:
        self._require_nonempty(
            {"organization_id": organization_id, "test_run_id": test_run_id}
        )
        inventory = self.list_inventory(organization_id=organization_id)
        dashboard = self.dashboard_metrics(organization_id=organization_id)
        latest_activation = self._latest_activation_bundle()
        normalized_uat_cases = uat_cases or _default_uat_case_matrix()
        uat_summary = _summarize_uat_cases(normalized_uat_cases)
        signoff = user_signoff or {}
        decision = str(signoff.get("decision") or "PENDING").strip() or "PENDING"
        acceptance_record = {
            "schema": "legal_corpus_pipeline1_acceptance_record.v1",
            "test_run_id": test_run_id,
            "executed_at": now_iso(),
            "code_revision": code_revision,
            "database_profile": "sqlite",
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "source_registry_snapshot": {
                "id": (latest_activation or {}).get("source_registry_snapshot_id"),
                "hash": (latest_activation or {})
                .get("content_hashes", {})
                .get("source_registry_snapshot"),
                "activation_event_ref": (latest_activation or {}).get(
                    "activation_event_id"
                ),
            },
            "knowledge_version": {
                "id": (latest_activation or {}).get("knowledge_context_version_id"),
                "hash": (latest_activation or {})
                .get("content_hashes", {})
                .get("knowledge_context"),
                "activation_event_ref": (latest_activation or {}).get(
                    "activation_event_id"
                ),
            },
            "authority_bundle": {
                "id": (latest_activation or {}).get("authority_bundle_version_id"),
                "hash": (latest_activation or {})
                .get("content_hashes", {})
                .get("authority_bundle"),
                "activation_event_ref": (latest_activation or {}).get(
                    "activation_event_id"
                ),
            },
            "activation_performed": bool(
                (latest_activation or {}).get("activation_performed")
            ),
            "automated_tests": _summarize_automated_verification(
                automated_verification_summary or []
            ),
            "uat_cases": uat_summary,
            "unresolved_critical_findings": int(
                signoff.get("unresolved_critical_findings") or uat_summary["pending"]
            ),
            "evidence_bundle_ref": evidence_bundle_ref,
            "demo_operator": demo_operator,
            "domain_reviewer": domain_reviewer,
            "product_acceptor": product_acceptor,
            "decision": decision,
            "comments": str(signoff.get("comments") or ""),
        }
        export = {
            "schema": "legal_corpus_pipeline1_acceptance_evidence.v1",
            "test_run_id": test_run_id,
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "generated_at": now_iso(),
            "source_inventory": inventory,
            "dashboard": dashboard,
            "active_pins": latest_activation,
            "pipeline2_evidence_contract": self.build_pipeline2_evidence_contract(
                organization_id=organization_id
            ),
            "automated_verification_summary": automated_verification_summary or [],
            "screenshots": screenshots or [],
            "uat_cases": normalized_uat_cases,
            "uat_summary": uat_summary,
            "acceptance_record": {
                **acceptance_record,
                "record_hash": content_hash(acceptance_record),
            },
            "user_signoff": signoff,
        }
        return {**export, "evidence_hash": content_hash(export)}

    def persist_acceptance_evidence_export(
        self,
        export: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(export, dict):
            raise ValueError("export must be an object")
        test_run_id = str(export.get("test_run_id") or "").strip()
        organization_id = str(export.get("organization_id") or "").strip()
        workspace_id = str(export.get("workspace_id") or "").strip() or None
        evidence_hash = str(export.get("evidence_hash") or "").strip()
        record_hash = str(
            (export.get("acceptance_record") or {}).get("record_hash") or ""
        ).strip()
        self._require_nonempty(
            {
                "test_run_id": test_run_id,
                "organization_id": organization_id,
                "evidence_hash": evidence_hash,
                "record_hash": record_hash,
            }
        )
        export_id = f"legal_acceptance_export_{uuid4().hex}"
        with self._connect() as conn:
            existing = conn.execute(
                hosted.sql_query("""
                SELECT acceptance_export_id, evidence_hash, record_hash
                FROM legal_acceptance_evidence_exports
                WHERE test_run_id = ? AND organization_id = ? AND evidence_hash = ?
                """, """
                SELECT acceptance_export_id, evidence_hash, record_hash
                FROM legal_acceptance_evidence_exports
                WHERE test_run_id = %s AND organization_id = %s AND evidence_hash = %s
                """),
                (test_run_id, organization_id, evidence_hash),
            ).fetchone()
            if existing is not None:
                return {
                    "acceptance_export_id": existing["acceptance_export_id"],
                    "evidence_hash": existing["evidence_hash"],
                    "record_hash": existing["record_hash"],
                    "idempotent_replay": True,
                }
            conn.execute(
                hosted.sql_query("""
                INSERT INTO legal_acceptance_evidence_exports (
                  acceptance_export_id, test_run_id, organization_id, workspace_id,
                  evidence_hash, record_hash, evidence_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO legal_acceptance_evidence_exports (
                  acceptance_export_id, test_run_id, organization_id, workspace_id,
                  evidence_hash, record_hash, evidence_json, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """),
                (
                    export_id,
                    test_run_id,
                    organization_id,
                    workspace_id,
                    evidence_hash,
                    record_hash,
                    canonical_json(export),
                    now_iso(),
                ),
            )
        return {
            "acceptance_export_id": export_id,
            "evidence_hash": evidence_hash,
            "record_hash": record_hash,
        }

    def list_acceptance_evidence_exports(
        self,
        *,
        organization_id: str,
    ) -> list[dict[str, Any]]:
        self._require_nonempty({"organization_id": organization_id})
        with self._connect() as conn:
            rows = conn.execute(
                hosted.sql_query("""
                SELECT acceptance_export_id, test_run_id, organization_id, workspace_id,
                       evidence_hash, record_hash, created_at
                FROM legal_acceptance_evidence_exports
                WHERE organization_id = ?
                ORDER BY created_at DESC
                """, """
                SELECT acceptance_export_id, test_run_id, organization_id, workspace_id,
                       evidence_hash, record_hash, created_at
                FROM legal_acceptance_evidence_exports
                WHERE organization_id = %s
                ORDER BY created_at DESC
                """),
                (organization_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _latest_activation_bundle(self) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT activation_event_id, source_registry_snapshot_id,
                       knowledge_context_version_id, authority_bundle_version_id,
                       activation_performed, content_hashes_json, created_at
                FROM legal_activation_events
                WHERE activation_performed = 1
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        payload = dict(row)
        payload["activation_performed"] = bool(payload["activation_performed"])
        payload["content_hashes"] = json.loads(payload.pop("content_hashes_json"))
        return payload

    def record_denial_event(
        self,
        *,
        event_family: str = "legal_action_failure.v1",
        object_ref: str,
        object_type: str,
        lifecycle_state: str | None,
        attempted_endpoint: str,
        actor_ref: str,
        organization_id: str,
        reason_code: str,
        route_policy_id: str,
        correlation_id: str,
        event_payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_nonempty(
            {
                "event_family": event_family,
                "object_ref": object_ref,
                "object_type": object_type,
                "attempted_endpoint": attempted_endpoint,
                "actor_ref": actor_ref,
                "organization_id": organization_id,
                "reason_code": reason_code,
                "route_policy_id": route_policy_id,
                "correlation_id": correlation_id,
            }
        )
        event_id = f"legal_audit_{uuid4().hex}"
        with self._connect() as conn:
            conn.execute(
                hosted.sql_query("""
                INSERT INTO legal_audit_events (
                  audit_event_id, event_family, object_ref, object_type, lifecycle_state,
                  attempted_endpoint, actor_ref, organization_id, reason_code, route_policy_id,
                  correlation_id, event_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO legal_audit_events (
                  audit_event_id, event_family, object_ref, object_type, lifecycle_state,
                  attempted_endpoint, actor_ref, organization_id, reason_code, route_policy_id,
                  correlation_id, event_json, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """),
                (
                    event_id,
                    event_family,
                    object_ref,
                    object_type,
                    lifecycle_state,
                    attempted_endpoint,
                    actor_ref,
                    organization_id,
                    reason_code,
                    route_policy_id,
                    correlation_id,
                    canonical_json(event_payload),
                    now_iso(),
                ),
            )
        return {
            "audit_event_id": event_id,
            "event_family": event_family,
            "reason_code": reason_code,
            "correlation_id": correlation_id,
        }

    def record_action_failure(
        self,
        *,
        reason_code: str,
        route_policy_id: str,
        attempted_action: str,
        object_ref: str,
        object_type: str,
        actor_ref: str,
        organization_id: str,
        correlation_id: str,
        lifecycle_state: str | None = None,
        expected_version: str | None = None,
        actual_version: str | None = None,
        expected_object_hash: str | None = None,
        actual_object_hash: str | None = None,
        expected_snapshot_hash: str | None = None,
        actual_snapshot_hash: str | None = None,
        proposed_mutation_hash: str | None = None,
        retryable: bool = True,
    ) -> dict[str, Any]:
        if reason_code not in LEGAL_ACTION_FAILURE_REASON_CODES:
            raise ValueError(f"unsupported legal action failure reason: {reason_code}")
        payload = {
            "schema_version": "legal_action_failure.v1",
            "reason_code": reason_code,
            "route_policy_id": route_policy_id,
            "attempted_action": attempted_action,
            "object_ref": object_ref,
            "object_type": object_type,
            "expected_version": expected_version,
            "actual_version": actual_version,
            "expected_object_hash": expected_object_hash,
            "actual_object_hash": actual_object_hash,
            "expected_snapshot_hash": expected_snapshot_hash,
            "actual_snapshot_hash": actual_snapshot_hash,
            "proposed_mutation_hash": proposed_mutation_hash,
            "actor_ref": actor_ref,
            "organization_id": organization_id,
            "correlation_id": correlation_id,
            "occurred_at": now_iso(),
            "retryable": retryable,
        }
        audit = self.record_denial_event(
            event_family="legal_action_failure.v1",
            object_ref=object_ref,
            object_type=object_type,
            lifecycle_state=lifecycle_state,
            attempted_endpoint=attempted_action,
            actor_ref=actor_ref,
            organization_id=organization_id,
            reason_code=reason_code,
            route_policy_id=route_policy_id,
            correlation_id=correlation_id,
            event_payload=payload,
        )
        return {
            "legal_action_failure": payload,
            "append_event_ref": audit["audit_event_id"],
        }

    def list_inventory(self, *, organization_id: str) -> dict[str, Any]:
        self._require_nonempty({"organization_id": organization_id})
        with self._connect() as conn:
            source_rows = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM legal_sources
                WHERE organization_id = ?
                ORDER BY authority_tier, issuer, canonical_title
                """, """
                SELECT *
                FROM legal_sources
                WHERE organization_id = %s
                ORDER BY authority_tier, issuer, canonical_title
                """),
                (organization_id,),
            ).fetchall()
            version_rows = conn.execute(
                hosted.sql_query("""
                SELECT v.*
                FROM legal_source_versions v
                JOIN legal_sources s ON s.source_id = v.source_id
                WHERE s.organization_id = ?
                ORDER BY v.created_at
                """, """
                SELECT v.*
                FROM legal_source_versions v
                JOIN legal_sources s ON s.source_id = v.source_id
                WHERE s.organization_id = %s
                ORDER BY v.created_at
                """),
                (organization_id,),
            ).fetchall()
            artifact_rows = conn.execute(
                hosted.sql_query("""
                SELECT a.*
                FROM legal_source_artifacts a
                JOIN legal_source_versions v ON v.version_id = a.version_id
                JOIN legal_sources s ON s.source_id = v.source_id
                WHERE s.organization_id = ?
                ORDER BY a.created_at
                """, """
                SELECT a.*
                FROM legal_source_artifacts a
                JOIN legal_source_versions v ON v.version_id = a.version_id
                JOIN legal_sources s ON s.source_id = v.source_id
                WHERE s.organization_id = %s
                ORDER BY a.created_at
                """),
                (organization_id,),
            ).fetchall()
            section_rows = conn.execute(
                hosted.sql_query("""
                SELECT sec.*
                FROM legal_source_sections sec
                JOIN legal_source_versions v ON v.version_id = sec.version_id
                JOIN legal_sources s ON s.source_id = v.source_id
                WHERE s.organization_id = ?
                ORDER BY sec.version_id, sec.stable_locator, sec.ordinal
                """, """
                SELECT sec.*
                FROM legal_source_sections sec
                JOIN legal_source_versions v ON v.version_id = sec.version_id
                JOIN legal_sources s ON s.source_id = v.source_id
                WHERE s.organization_id = %s
                ORDER BY sec.version_id, sec.stable_locator, sec.ordinal
                """),
                (organization_id,),
            ).fetchall()
            refresh_rows = conn.execute(
                hosted.sql_query("""
                SELECT r.*
                FROM source_refresh_checks r
                JOIN legal_sources s ON s.source_id = r.source_id
                WHERE s.organization_id = ?
                ORDER BY r.checked_at
                """, """
                SELECT r.*
                FROM source_refresh_checks r
                JOIN legal_sources s ON s.source_id = r.source_id
                WHERE s.organization_id = %s
                ORDER BY r.checked_at
                """),
                (organization_id,),
            ).fetchall()
            schedule_rows = conn.execute(
                hosted.sql_query("""
                SELECT sched.*
                FROM legal_source_refresh_schedules sched
                JOIN legal_sources s ON s.source_id = sched.source_id
                WHERE s.organization_id = ?
                ORDER BY sched.next_check_at
                """, """
                SELECT sched.*
                FROM legal_source_refresh_schedules sched
                JOIN legal_sources s ON s.source_id = sched.source_id
                WHERE s.organization_id = %s
                ORDER BY sched.next_check_at
                """),
                (organization_id,),
            ).fetchall()
            edge_rows = conn.execute(
                hosted.sql_query("""
                SELECT e.*
                FROM legal_source_version_edges e
                JOIN legal_source_versions v ON v.version_id = e.from_version_id
                JOIN legal_sources s ON s.source_id = v.source_id
                WHERE s.organization_id = ?
                ORDER BY e.created_at
                """, """
                SELECT e.*
                FROM legal_source_version_edges e
                JOIN legal_source_versions v ON v.version_id = e.from_version_id
                JOIN legal_sources s ON s.source_id = v.source_id
                WHERE s.organization_id = %s
                ORDER BY e.created_at
                """),
                (organization_id,),
            ).fetchall()
            candidate_rows = conn.execute(
                hosted.sql_query("""
                SELECT c.*
                FROM legal_semantic_candidates c
                JOIN legal_source_versions v ON v.version_id = c.version_id
                JOIN legal_sources s ON s.source_id = v.source_id
                WHERE s.organization_id = ?
                ORDER BY c.created_at
                """, """
                SELECT c.*
                FROM legal_semantic_candidates c
                JOIN legal_source_versions v ON v.version_id = c.version_id
                JOIN legal_sources s ON s.source_id = v.source_id
                WHERE s.organization_id = %s
                ORDER BY c.created_at
                """),
                (organization_id,),
            ).fetchall()
        sources = [dict(row) for row in source_rows]
        versions = [{**dict(row), "source_hash": row["content_hash"]} for row in version_rows]
        artifacts = [dict(row) for row in artifact_rows]
        sections = [dict(row) for row in section_rows]
        refresh_checks = [dict(row) for row in refresh_rows]
        refresh_schedules = [dict(row) for row in schedule_rows]
        version_edges = [dict(row) for row in edge_rows]
        candidates = [dict(row) for row in candidate_rows]
        return {
            "organization_id": organization_id,
            "sources": sources,
            "versions": versions,
            "artifacts": artifacts,
            "sections": sections,
            "refresh_checks": refresh_checks,
            "refresh_schedules": refresh_schedules,
            "version_edges": version_edges,
            "semantic_candidates": candidates,
            "metrics": {
                "source_count": len(sources),
                "version_count": len(versions),
                "artifact_count": len(artifacts),
                "section_count": len(sections),
                "refresh_check_count": len(refresh_checks),
                "refresh_schedule_count": len(refresh_schedules),
                "version_edge_count": len(version_edges),
                "semantic_candidate_count": len(candidates),
                "unresolved_candidate_count": sum(
                    1 for candidate in candidates if candidate["review_state"] == "UNRESOLVED"
                ),
                "rejected_candidate_count": sum(
                    1 for candidate in candidates if candidate["review_state"] == "REJECTED"
                ),
                "registered_source_count": sum(
                    1 for source in sources if source["governance_state"] != "LISTED"
                ),
            },
        }

    def dashboard_metrics(
        self,
        *,
        organization_id: str,
        expected_source_count: int | None = None,
    ) -> dict[str, Any]:
        inventory = self.list_inventory(organization_id=organization_id)
        sources = inventory["sources"]
        versions = inventory["versions"]
        artifacts = inventory["artifacts"]
        refresh_checks = inventory["refresh_checks"]
        candidates = inventory["semantic_candidates"]
        edges = inventory["version_edges"]

        source_count = len(sources)
        version_count = len(versions)
        expected = expected_source_count or source_count or 1
        version_ids_with_artifacts = {artifact["version_id"] for artifact in artifacts}
        version_ids_with_sections = self._version_ids_with_sections(organization_id=organization_id)
        current_versions = [
            version for version in versions if str(version["lifecycle_state"]).upper() in {"REGISTERED", "CERTIFIED", "ACTIVE"}
        ]
        freshness_seconds = [
            _age_seconds(row["checked_at"])
            for row in refresh_checks
            if row.get("checked_at")
        ]
        candidate_count = len(candidates)
        certified_candidate_count = sum(
            1 for candidate in candidates if candidate["review_state"] == "CERTIFIED"
        )
        unanchored_claim_count = sum(
            1
            for candidate in candidates
            if not str(candidate["source_span_json"] or "").strip()
        )
        edge_types = {edge["edge_type"] for edge in edges}
        node_count = source_count + version_count + len(version_ids_with_sections) + candidate_count
        edge_count = len(edges) + candidate_count + len(artifacts)
        metrics = {
            "source_inventory_completion": _ratio(source_count, expected),
            "official_origin_ratio": _ratio(
                sum(1 for artifact in artifacts if str(artifact["artifact_kind"]).startswith("official")),
                len(artifacts),
            ),
            "raw_artifact_coverage": _ratio(len(version_ids_with_artifacts), version_count),
            "article_anchor_coverage": _ratio(len(version_ids_with_sections), version_count),
            "effectivity_completeness": _ratio(
                sum(1 for version in versions if version["effective_from"]),
                version_count,
            ),
            "version_lineage_completeness": _ratio(len(edges), max(0, version_count - source_count)),
            "source_freshness_seconds_p95": _p95(freshness_seconds),
            "refresh_age_seconds_max": max(freshness_seconds) if freshness_seconds else None,
            "unresolved_provenance": sum(
                1 for check in refresh_checks if check["status"] in {"QUARANTINED", "PROVENANCE_CONFLICT"}
            ),
            "open_change_candidates": sum(
                1 for check in refresh_checks if check["status"] == "CHANGE_DETECTED"
            ),
            "claim_extraction_coverage": _ratio(candidate_count, len(version_ids_with_sections)),
            "claim_certification_coverage": _ratio(certified_candidate_count, candidate_count),
            "unanchored_claim_count": unanchored_claim_count,
            "unresolved_impact_analysis_count": sum(
                1 for check in refresh_checks if check["status"] == "CHANGE_DETECTED"
            ),
            "node_count": node_count,
            "edge_count": edge_count,
            "edge_type_count": len(edge_types),
            "authority_chain_depth_p95": 0,
            "supersession_chain_depth_p95": _p95([1 for edge in edges if edge["edge_type"] == "SUPERSEDES"]),
            "exception_chain_depth_p95": 0,
            "issue_dependency_depth_p95": 0,
            "provenance_path_depth_p95": _p95([1 for artifact in artifacts]),
            "variable_length_traversal_ratio": 0.0,
            "recursive_cte_ratio": 0.0,
            "recursive_cte_rows_expanded_p95": 0,
            "resolution_latency_seconds_p95": None,
            "replay_equality_ratio": None,
            "active_version_count": len(
                [version for version in current_versions if version["lifecycle_state"] == "ACTIVE"]
            ),
        }
        return {
            "schema": "legal_corpus_observation_dashboard.v1",
            "organization_id": organization_id,
            "expected_source_count": expected_source_count,
            "metrics": metrics,
            "source_count": source_count,
            "version_count": version_count,
        }

    def _version_ids_with_sections(self, *, organization_id: str) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute(
                hosted.sql_query("""
                SELECT DISTINCT sec.version_id
                FROM legal_source_sections sec
                JOIN legal_source_versions v ON v.version_id = sec.version_id
                JOIN legal_sources s ON s.source_id = v.source_id
                WHERE s.organization_id = ?
                """, """
                SELECT DISTINCT sec.version_id
                FROM legal_source_sections sec
                JOIN legal_source_versions v ON v.version_id = sec.version_id
                JOIN legal_sources s ON s.source_id = v.source_id
                WHERE s.organization_id = %s
                """),
                (organization_id,),
            ).fetchall()
        return {row["version_id"] for row in rows}

    def resolve_certification_source_evidence(
        self, *, organization_id: str, workspace_id: str,
        source_id: str, version_id: str, artifact_id: str,
        artifact_hash: str, section_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        """Read one exact legal evidence chain for prerequisite production.

        This snapshots owner records, not an approval or activation decision.
        Certification must retain the returned exact evidence in its own pinned
        artifact; historical replay must not call this live registry reader.
        """
        if not section_ids or len(section_ids) != len(set(section_ids)):
            raise ValueError("LEGAL_SECTION_SELECTION_REQUIRED")
        with self._connect() as conn:
            conn.execute("BEGIN")
            source = conn.execute(
                hosted.sql_query("SELECT * FROM legal_sources WHERE source_id=? AND organization_id=? "
                "AND (workspace_id=? OR workspace_id IS NULL)", 'SELECT * FROM legal_sources WHERE source_id=%s AND organization_id=%s AND (workspace_id=%s OR workspace_id IS NULL)'),
                (source_id, organization_id, workspace_id),
            ).fetchone()
            if source is None:
                raise KeyError("LEGAL_SOURCE_NOT_FOUND")
            version = conn.execute(
                hosted.sql_query("SELECT * FROM legal_source_versions WHERE version_id=? AND source_id=?", 'SELECT * FROM legal_source_versions WHERE version_id=%s AND source_id=%s'),
                (version_id, source_id),
            ).fetchone()
            artifact = conn.execute(
                hosted.sql_query("SELECT * FROM legal_source_artifacts WHERE artifact_id=? AND version_id=?", 'SELECT * FROM legal_source_artifacts WHERE artifact_id=%s AND version_id=%s'),
                (artifact_id, version_id),
            ).fetchone()
            if version is None or artifact is None:
                raise KeyError("LEGAL_EVIDENCE_CHAIN_NOT_FOUND")
            if artifact_hash != "sha256:" + str(artifact["content_hash"]):
                raise ValueError("LEGAL_ARTIFACT_PIN_MISMATCH")
            sections = []
            for section_id in section_ids:
                section = conn.execute(
                    hosted.sql_query("SELECT * FROM legal_source_sections WHERE section_id=? AND version_id=?", 'SELECT * FROM legal_source_sections WHERE section_id=%s AND version_id=%s'),
                    (section_id, version_id),
                ).fetchone()
                if section is None:
                    raise KeyError("LEGAL_SECTION_NOT_FOUND")
                sections.append(dict(section))
            return {"source": dict(source), "version": dict(version),
                    "artifact": dict(artifact), "sections": sections}

    def backfill_retained_section_text(
        self, *, organization_id: str, workspace_id: str,
        source_id: str, version_id: str, artifact_id: str,
        artifact_hash: str, section_id: str, section: LegalSectionAnchor,
    ) -> dict[str, Any]:
        """Enrich missing text only; preserve the exact reviewed legal identity."""
        with self._connect() as conn:
            hosted.begin_write(conn)
            source = conn.execute(
                hosted.sql_query("SELECT 1 FROM legal_sources WHERE source_id=? AND organization_id=? "
                "AND (workspace_id=? OR workspace_id IS NULL)", 'SELECT 1 FROM legal_sources WHERE source_id=%s AND organization_id=%s AND (workspace_id=%s OR workspace_id IS NULL)'),
                (source_id, organization_id, workspace_id),
            ).fetchone()
            artifact = conn.execute(
                hosted.sql_query("SELECT a.content_hash FROM legal_source_artifacts a "
                "JOIN legal_source_versions v ON v.version_id=a.version_id "
                "WHERE a.artifact_id=? AND a.version_id=? AND v.source_id=?", 'SELECT a.content_hash FROM legal_source_artifacts a JOIN legal_source_versions v ON v.version_id=a.version_id WHERE a.artifact_id=%s AND a.version_id=%s AND v.source_id=%s'),
                (artifact_id, version_id, source_id),
            ).fetchone()
            if source is None or artifact is None:
                raise KeyError("LEGAL_EVIDENCE_CHAIN_NOT_FOUND")
            if artifact_hash != "sha256:" + artifact["content_hash"]:
                raise ValueError("LEGAL_ARTIFACT_PIN_MISMATCH")
            row = conn.execute(
                hosted.sql_query("SELECT * FROM legal_source_sections WHERE section_id=? AND version_id=?", 'SELECT * FROM legal_source_sections WHERE section_id=%s AND version_id=%s'),
                (section_id, version_id),
            ).fetchone()
            if row is None:
                raise KeyError("LEGAL_SECTION_NOT_FOUND")
            text_hash = content_hash({"text": section.text})
            local_hash = content_hash({
                "stable_locator": section.stable_locator,
                "section_type": section.section_type, "ordinal": section.ordinal,
                "title": section.title, "text_hash": text_hash,
            })
            if not section.text or text_hash != section.text_hash or local_hash != section.local_hash:
                raise ValueError("LEGAL_SECTION_TEXT_HASH_MISMATCH")
            for field in ("stable_locator", "section_type", "ordinal", "title", "text_hash", "local_hash"):
                if row[field] != getattr(section, field):
                    raise ValueError("LEGAL_SECTION_PIN_MISMATCH")
            anchor = json.loads(row["anchor_json"])
            retained = anchor.get("retained_text")
            if retained not in (None, "", section.text):
                raise ValueError("LEGAL_RETAINED_TEXT_CONFLICT")
            if retained != section.text:
                anchor["retained_text"] = section.text
                conn.execute(
                    hosted.sql_query("UPDATE legal_source_sections SET anchor_json=? WHERE section_id=? AND version_id=?", 'UPDATE legal_source_sections SET anchor_json=%s WHERE section_id=%s AND version_id=%s'),
                    (canonical_json(anchor), section_id, version_id),
                )
            return dict(conn.execute(
                hosted.sql_query("SELECT * FROM legal_source_sections WHERE section_id=?", 'SELECT * FROM legal_source_sections WHERE section_id=%s'), (section_id,),
            ).fetchone())

    def get_source(self, source_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                hosted.sql_query("SELECT * FROM legal_sources WHERE source_id = ?", 'SELECT * FROM legal_sources WHERE source_id = %s'),
                (source_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown source_id: {source_id}")
        return dict(row)

    def get_source_version(
        self,
        *,
        source_id: str,
        version_identity: str | None = None,
        content_hash: str | None = None,
    ) -> LegalSourceVersion:
        if not version_identity and not content_hash:
            raise ValueError("version_identity or content_hash is required")
        clauses = [hosted.sql_query("source_id = ?", 'source_id = %s')]
        params: list[Any] = [source_id]
        if version_identity:
            clauses.append(hosted.sql_query("version_identity = ?", 'version_identity = %s'))
            params.append(version_identity)
        if content_hash:
            clauses.append(hosted.sql_query("content_hash = ?", 'content_hash = %s'))
            params.append(content_hash)
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT *
                FROM legal_source_versions
                WHERE {' AND '.join(clauses)}
                ORDER BY created_at DESC
                LIMIT 1
                """,
                params,
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown source version for source_id: {source_id}")
        return LegalSourceVersion(
            version_id=row["version_id"],
            source_id=row["source_id"],
            version_identity=row["version_identity"],
            content_hash=row["content_hash"],
            lifecycle_state=row["lifecycle_state"],
        )

    @staticmethod
    def _require_nonempty(values: dict[str, Any]) -> None:
        missing = [key for key, value in values.items() if not str(value or "").strip()]
        if missing:
            raise ValueError(",".join(missing))

    @staticmethod
    def _validate_certification_review(
        *,
        object_type: str,
        reviewer_role: str,
        event_payload: dict[str, Any],
    ) -> None:
        review_dimensions = event_payload.get("review_dimensions")
        if object_type == "LegalSourceVersion":
            required_dimensions = {"authenticity", "effectivity", "hierarchy"}
        else:
            required_dimensions = {"semantics", "source_spans", "hierarchy"}
        if not isinstance(review_dimensions, dict) or not all(
            review_dimensions.get(dimension) is True for dimension in required_dimensions
        ):
            raise ValueError(LEGAL_REVIEW_DIMENSIONS_REQUIRED)

    @staticmethod
    def _assert_ref_exists(
        conn: sqlite3.Connection,
        table: str,
        column: str,
        value: str,
    ) -> None:
        row = conn.execute(
            hosted.sql_query(f"SELECT 1 FROM {table} WHERE {column} = ?", f'SELECT 1 FROM {table} WHERE {column} = %s'),
            (value,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown {column}: {value}")


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _p95(values: list[int | float]) -> int | float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return ordered[index]


def _age_seconds(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return 0
    return max(0, int((datetime.now(timezone.utc) - parsed).total_seconds()))


_MANDATORY_UAT_CASES = (
    "UAT-KB-001",
    "UAT-KB-002",
    "UAT-KB-003",
    "UAT-KB-004",
    "UAT-KB-005",
    "UAT-KB-006",
    "UAT-KB-007",
    "UAT-KB-008",
    "UAT-KB-009",
    "UAT-KB-010",
    "UAT-KB-011",
    "UAT-KB-011A",
    "UAT-KB-011B",
    "UAT-KB-011C",
    "UAT-KB-011D",
    "UAT-KB-011E",
    "UAT-KB-012",
    "UAT-KB-013",
    "UAT-KB-014",
    "UAT-KB-015",
    "UAT-KB-016",
    "UAT-KB-017",
    "UAT-KB-018",
    "UAT-KB-019",
    "UAT-KB-020",
    "UAT-KB-021",
    "UAT-KB-022",
    "UAT-KB-023",
    "UAT-KB-024",
)


def _default_uat_case_matrix() -> list[dict[str, Any]]:
    return [
        {
            "case_id": case_id,
            "status": "PENDING_EVIDENCE",
            "evidence_refs": [],
            "waiver_ref": None,
        }
        for case_id in _MANDATORY_UAT_CASES
    ]


def _summarize_uat_cases(cases: list[dict[str, Any]]) -> dict[str, int]:
    normalized = cases or []
    return {
        "total": len(normalized),
        "passed": sum(1 for case in normalized if case.get("status") == "PASS"),
        "failed": sum(1 for case in normalized if case.get("status") == "FAIL"),
        "waived": sum(1 for case in normalized if case.get("status") == "WAIVED"),
        "pending": sum(
            1 for case in normalized if case.get("status") == "PENDING_EVIDENCE"
        ),
    }


def _summarize_automated_verification(items: list[dict[str, Any]]) -> dict[str, int]:
    normalized = items or []
    failed = sum(
        1
        for item in normalized
        if str(item.get("result") or item.get("status") or "").lower()
        in {"failed", "fail", "error"}
    )
    passed = sum(
        1
        for item in normalized
        if str(item.get("result") or item.get("status") or "").lower()
        in {"passed", "pass", "ok"}
    )
    return {
        "passed": passed,
        "failed": failed,
        "total": len(normalized),
    }
