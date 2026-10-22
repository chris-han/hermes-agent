from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash as compute_content_hash
from eos.db import init_db
from runtime_paths import sqlite_db_path
from storage.hosted_postgres import enabled,connect_authority,require_tables,sql_query,begin_write
from storage.transaction_scope import transaction_scope,StoreMutationLock

FEEDBACK_TYPES = {
    "false_positive",
    "false_negative",
    "ambiguity",
    "conflict",
    "remediation_rejected",
    "user_edited_remediation",
    "missing_control",
    "weak_explanation",
}

DELTA_TYPES = {
    "not_same_as",
    "allowed_context_for",
    "prohibited_context_for",
    "split_cluster",
    "merge_cluster",
    "new_control_candidate",
}


def _db_path() -> Path:
    return Path(os.environ.get("SEMANTIER_EOS_DB_PATH", str(sqlite_db_path("eos.db", ".semantier/eos.db"))))


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_json(value: Any) -> str:
    return canonical_json(value if value is not None else {})


def _loads_json(value: str) -> Any:
    return json.loads(value) if value else None


def _hydrate(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    for field in (
        "user_disposition_json",
        "feedback_evidence_json",
        "source_span_json",
        "candidate_graph_ref_json",
        "evaluation_routing_json",
    ):
        if field in payload:
            payload[field.removesuffix("_json")] = _loads_json(payload[field])
    return payload


class RuntimeFeedbackStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        path = Path(db_path) if db_path else _db_path()
        if enabled():
            self._conn=connect_authority("eos",path)
            require_tables(self._conn,'runtime_feedback_candidate_deltas','runtime_feedback_events')
            self._conn.commit()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            prior_db_path = os.environ.get("SEMANTIER_EOS_DB_PATH")
            os.environ["SEMANTIER_EOS_DB_PATH"] = str(path)
            try:
                init_db()
            finally:
                if prior_db_path is None:
                    os.environ.pop("SEMANTIER_EOS_DB_PATH", None)
                else:
                    os.environ["SEMANTIER_EOS_DB_PATH"] = prior_db_path
            self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10.0)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = StoreMutationLock(self._conn)

    def create_feedback_event(
        self,
        *,
        organization_id: str,
        workspace_id: str | None,
        source_kind: str,
        source_run_id: str,
        source_finding_id: str | None,
        feedback_type: str,
        user_disposition: dict[str, Any],
        escalation_outcome: str | None,
        feedback_evidence: dict[str, Any],
        affected_term: str | None,
        affected_control_ref: str | None,
        affected_relation_ref: str | None,
        source_span: dict[str, Any],
        created_by: str,
    ) -> dict[str, Any]:
        normalized_type = feedback_type.strip().lower()
        if normalized_type not in FEEDBACK_TYPES:
            raise ValueError("unsupported feedback_type")
        if escalation_outcome not in {None, "not_escalated", "escalated", "resolved", "deferred"}:
            raise ValueError("unsupported escalation_outcome")
        now = _now_iso()
        event_id = f"rfe_{uuid4().hex}"
        row = {
            "feedback_event_id": event_id,
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "source_kind": source_kind,
            "source_run_id": source_run_id,
            "source_finding_id": source_finding_id,
            "feedback_type": normalized_type,
            "user_disposition": user_disposition,
            "escalation_outcome": escalation_outcome,
            "feedback_evidence": feedback_evidence,
            "affected_term": affected_term,
            "affected_control_ref": affected_control_ref,
            "affected_relation_ref": affected_relation_ref,
            "source_span": source_span,
            "created_by": created_by,
            "created_at": now,
        }
        row["content_hash"] = compute_content_hash(canonical_json(row))
        with self._lock:
            self._conn.execute(
                sql_query("""
                INSERT INTO runtime_feedback_events (
                  feedback_event_id, organization_id, workspace_id, source_kind,
                  source_run_id, source_finding_id, feedback_type,
                  user_disposition_json, escalation_outcome, feedback_evidence_json,
                  affected_term, affected_control_ref, affected_relation_ref,
                  source_span_json, created_by, created_at, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO runtime_feedback_events (\n                  feedback_event_id, organization_id, workspace_id, source_kind,\n                  source_run_id, source_finding_id, feedback_type,\n                  user_disposition_json, escalation_outcome, feedback_evidence_json,\n                  affected_term, affected_control_ref, affected_relation_ref,\n                  source_span_json, created_by, created_at, content_hash\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    event_id,
                    organization_id,
                    workspace_id,
                    source_kind,
                    source_run_id,
                    source_finding_id,
                    normalized_type,
                    _as_json(user_disposition),
                    escalation_outcome,
                    _as_json(feedback_evidence),
                    affected_term,
                    affected_control_ref,
                    affected_relation_ref,
                    _as_json(source_span),
                    created_by,
                    now,
                    row["content_hash"],
                ),
            )
            self._conn.commit()
        return row

    def create_candidate_delta(
        self,
        *,
        feedback_event_id: str,
        discovery_run_id: str,
        delta_kind: str,
        candidate_graph_ref: dict[str, Any],
        evaluation_routing: dict[str, Any],
    ) -> dict[str, Any]:
        normalized_kind = delta_kind.strip().lower()
        if normalized_kind not in DELTA_TYPES:
            raise ValueError("unsupported delta_kind")
        if self.get_feedback_event(feedback_event_id) is None:
            raise ValueError(f"runtime feedback event not found: {feedback_event_id}")
        now = _now_iso()
        delta_id = f"rfcd_{uuid4().hex}"
        row = {
            "candidate_delta_id": delta_id,
            "feedback_event_id": feedback_event_id,
            "discovery_run_id": discovery_run_id,
            "delta_kind": normalized_kind,
            "candidate_graph_ref": candidate_graph_ref,
            "evaluation_routing": evaluation_routing,
            "governance_state": "DISCOVERED",
            "created_at": now,
        }
        row["content_hash"] = compute_content_hash(canonical_json(row))
        with self._lock:
            self._conn.execute(
                sql_query("""
                INSERT INTO runtime_feedback_candidate_deltas (
                  candidate_delta_id, feedback_event_id, discovery_run_id, delta_kind,
                  candidate_graph_ref_json, evaluation_routing_json, governance_state,
                  created_at, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO runtime_feedback_candidate_deltas (\n                  candidate_delta_id, feedback_event_id, discovery_run_id, delta_kind,\n                  candidate_graph_ref_json, evaluation_routing_json, governance_state,\n                  created_at, content_hash\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    delta_id,
                    feedback_event_id,
                    discovery_run_id,
                    normalized_kind,
                    _as_json(candidate_graph_ref),
                    _as_json(evaluation_routing),
                    "DISCOVERED",
                    now,
                    row["content_hash"],
                ),
            )
            self._conn.commit()
        return row

    def get_feedback_event(self, feedback_event_id: str) -> dict[str, Any] | None:
        return _hydrate(
            self._conn.execute(
                sql_query("SELECT * FROM runtime_feedback_events WHERE feedback_event_id = ?", 'SELECT * FROM runtime_feedback_events WHERE feedback_event_id = %s'),
                (feedback_event_id,),
            ).fetchone()
        )

    def list_candidate_deltas(self, discovery_run_id: str | None = None) -> list[dict[str, Any]]:
        if discovery_run_id:
            rows = self._conn.execute(
                sql_query("""
                SELECT
                  d.*,
                  e.feedback_type,
                  e.source_kind,
                  e.source_run_id,
                  e.source_finding_id,
                  e.escalation_outcome,
                  e.affected_term,
                  e.affected_control_ref,
                  e.affected_relation_ref
                FROM runtime_feedback_candidate_deltas d
                JOIN runtime_feedback_events e
                  ON e.feedback_event_id = d.feedback_event_id
                WHERE d.discovery_run_id = ?
                ORDER BY d.created_at, d.candidate_delta_id
                """, '\n                SELECT\n                  d.*,\n                  e.feedback_type,\n                  e.source_kind,\n                  e.source_run_id,\n                  e.source_finding_id,\n                  e.escalation_outcome,\n                  e.affected_term,\n                  e.affected_control_ref,\n                  e.affected_relation_ref\n                FROM runtime_feedback_candidate_deltas d\n                JOIN runtime_feedback_events e\n                  ON e.feedback_event_id = d.feedback_event_id\n                WHERE d.discovery_run_id = %s\n                ORDER BY d.created_at, d.candidate_delta_id\n                '),
                (discovery_run_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT
                  d.*,
                  e.feedback_type,
                  e.source_kind,
                  e.source_run_id,
                  e.source_finding_id,
                  e.escalation_outcome,
                  e.affected_term,
                  e.affected_control_ref,
                  e.affected_relation_ref
                FROM runtime_feedback_candidate_deltas d
                JOIN runtime_feedback_events e
                  ON e.feedback_event_id = d.feedback_event_id
                ORDER BY d.created_at, d.candidate_delta_id
                """
            ).fetchall()
        return [_hydrate(row) for row in rows]
