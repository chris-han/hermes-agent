"""Canonical semantic feedback event persistence."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from storage.hosted_postgres import enabled,connect_authority,mapped_store,require_tables,sql_query,begin_write,integrity_errors
from eos.artifact_hashing import canonical_json, content_hash


SEMANTIC_FEEDBACK_EVENT_SCHEMA_VERSION = "semantic_feedback_event.v1"
FeedbackAction = Literal["confirm", "change", "dismiss"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _loads(value: str) -> Any:
    return json.loads(value) if value else None


@dataclass(frozen=True)
class SemanticFeedbackEvent:
    schema_version: str
    feedback_id: str
    run_id: str
    finding_id: str
    action: FeedbackAction
    original_result_ref: str
    corrected_result: dict[str, Any] | None
    reason_code: str | None
    justification: str | None
    evidence_ref: str
    graph_version_ref: str
    ontology_version_ref: str | None
    profile_ref: str
    profile_version: str
    skill_ref: str
    derivation_refs: list[str]
    reviewer: str
    occurred_at: str
    event_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feedback_id": self.feedback_id,
            "run_id": self.run_id,
            "finding_id": self.finding_id,
            "action": self.action,
            "original_result_ref": self.original_result_ref,
            "corrected_result": self.corrected_result,
            "reason_code": self.reason_code,
            "justification": self.justification,
            "evidence_ref": self.evidence_ref,
            "graph_version_ref": self.graph_version_ref,
            "ontology_version_ref": self.ontology_version_ref,
            "profile_ref": self.profile_ref,
            "profile_version": self.profile_version,
            "skill_ref": self.skill_ref,
            "derivation_refs": self.derivation_refs,
            "reviewer": self.reviewer,
            "occurred_at": self.occurred_at,
            "event_hash": self.event_hash,
        }


class SemanticFeedbackEventStore:
    """Append-only semantic feedback event table."""

    _process_locks: dict[Path, threading.RLock] = {}
    _process_locks_guard = threading.Lock()

    def __init__(self, workspace_root: str | Path, *, workspace_id: str) -> None:
        if not workspace_id:
            raise ValueError("workspace_id is required")
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_id = workspace_id
        self.root = self.workspace_root / ".semantier" / "semantic-feedback"
        if not enabled():self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "feedback.sqlite3"
        self._initialize()

    @property
    def _lock(self) -> threading.RLock:
        with self._process_locks_guard:
            return self._process_locks.setdefault(self.db_path, threading.RLock())

    def _connect(self) -> sqlite3.Connection:
        if enabled():
            binding=mapped_store('standalone',self.db_path)
            if binding.get('workspace_id',self.workspace_id)!=self.workspace_id:
                raise ValueError('foreign workspace auxiliary binding')
            return connect_authority('standalone',self.db_path)
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        if enabled():
            with self._connect() as connection:require_tables(connection,'semantic_feedback_events')
            return
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_feedback_events (
                    feedback_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    finding_id TEXT NOT NULL,
                    action TEXT NOT NULL CHECK (action IN ('confirm', 'change', 'dismiss')),
                    original_result_ref TEXT NOT NULL,
                    corrected_result_json TEXT,
                    reason_code TEXT,
                    justification TEXT,
                    evidence_ref TEXT NOT NULL,
                    graph_version_ref TEXT NOT NULL,
                    ontology_version_ref TEXT,
                    profile_ref TEXT NOT NULL,
                    profile_version TEXT NOT NULL,
                    skill_ref TEXT NOT NULL,
                    derivation_refs_json TEXT NOT NULL,
                    reviewer TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    event_hash TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_semantic_feedback_run ON semantic_feedback_events (workspace_id, run_id, finding_id)"
            )

    def create_event(
        self,
        *,
        run_id: str,
        finding_id: str,
        action: FeedbackAction,
        original_result_ref: str,
        evidence_ref: str,
        graph_version_ref: str,
        profile_ref: str,
        profile_version: str,
        skill_ref: str,
        reviewer: str,
        corrected_result: dict[str, Any] | None = None,
        reason_code: str | None = None,
        justification: str | None = None,
        ontology_version_ref: str | None = None,
        derivation_refs: list[str] | None = None,
    ) -> SemanticFeedbackEvent:
        if action not in {"confirm", "change", "dismiss"}:
            raise ValueError("feedback action must be confirm, change, or dismiss")
        required = {
            "run_id": run_id,
            "finding_id": finding_id,
            "original_result_ref": original_result_ref,
            "evidence_ref": evidence_ref,
            "graph_version_ref": graph_version_ref,
            "profile_ref": profile_ref,
            "profile_version": profile_version,
            "skill_ref": skill_ref,
            "reviewer": reviewer,
        }
        missing = sorted(name for name, value in required.items() if not value)
        if missing:
            raise ValueError(f"semantic feedback event missing fields: {missing}")
        if action == "change" and not corrected_result:
            raise ValueError("change feedback requires corrected_result")
        if action in {"change", "dismiss"} and not (justification or reason_code):
            raise ValueError("change/dismiss feedback requires justification or reason_code")
        occurred_at = _utc_now()
        payload = {
            "schema_version": SEMANTIC_FEEDBACK_EVENT_SCHEMA_VERSION,
            "feedback_id": f"sfe_{uuid.uuid4().hex}",
            "run_id": run_id,
            "finding_id": finding_id,
            "action": action,
            "original_result_ref": original_result_ref,
            "corrected_result": corrected_result,
            "reason_code": reason_code,
            "justification": justification,
            "evidence_ref": evidence_ref,
            "graph_version_ref": graph_version_ref,
            "ontology_version_ref": ontology_version_ref,
            "profile_ref": profile_ref,
            "profile_version": profile_version,
            "skill_ref": skill_ref,
            "derivation_refs": derivation_refs or [],
            "reviewer": reviewer,
            "occurred_at": occurred_at,
        }
        payload["event_hash"] = f"sha256:{content_hash(payload)}"
        event = SemanticFeedbackEvent(**payload)
        with self._lock, self._connect() as connection:
            if enabled():begin_write(connection)
            connection.execute(
                sql_query("""
                INSERT INTO semantic_feedback_events VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,'\n                INSERT INTO semantic_feedback_events (feedback_id,workspace_id,schema_version,run_id,finding_id,action,original_result_ref,corrected_result_json,reason_code,justification,evidence_ref,graph_version_ref,ontology_version_ref,profile_ref,profile_version,skill_ref,derivation_refs_json,reviewer,occurred_at,event_hash) VALUES (\n                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n                )\n                '),
                (
                    event.feedback_id,
                    self.workspace_id,
                    event.schema_version,
                    event.run_id,
                    event.finding_id,
                    event.action,
                    event.original_result_ref,
                    canonical_json(event.corrected_result) if event.corrected_result is not None else None,
                    event.reason_code,
                    event.justification,
                    event.evidence_ref,
                    event.graph_version_ref,
                    event.ontology_version_ref,
                    event.profile_ref,
                    event.profile_version,
                    event.skill_ref,
                    canonical_json(event.derivation_refs),
                    event.reviewer,
                    event.occurred_at,
                    event.event_hash,
                ),
            )
        return event

    def get_event(self, feedback_id: str) -> SemanticFeedbackEvent:
        with self._connect() as connection:
            row = connection.execute(
                sql_query("""
                SELECT * FROM semantic_feedback_events
                WHERE workspace_id = ? AND feedback_id = ?
                """,'\n                SELECT * FROM semantic_feedback_events\n                WHERE workspace_id = %s AND feedback_id = %s\n                '),
                (self.workspace_id, feedback_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"semantic feedback event not found: {feedback_id}")
        return self._hydrate(row)

    def list_for_finding(self, *, run_id: str, finding_id: str) -> list[SemanticFeedbackEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                sql_query("""
                SELECT * FROM semantic_feedback_events
                WHERE workspace_id = ? AND run_id = ? AND finding_id = ?
                ORDER BY occurred_at, feedback_id
                """,'\n                SELECT * FROM semantic_feedback_events\n                WHERE workspace_id = %s AND run_id = %s AND finding_id = %s\n                ORDER BY occurred_at, feedback_id\n                '),
                (self.workspace_id, run_id, finding_id),
            ).fetchall()
        return [self._hydrate(row) for row in rows]

    def list_for_original_result(self, original_result_ref: str) -> list[SemanticFeedbackEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                sql_query("""
                SELECT * FROM semantic_feedback_events
                WHERE workspace_id = ? AND original_result_ref = ?
                ORDER BY occurred_at, rowid
                """,'\n                SELECT * FROM semantic_feedback_events\n                WHERE workspace_id = %s AND original_result_ref = %s\n                ORDER BY occurred_at, event_sequence\n                '),
                (self.workspace_id, original_result_ref),
            ).fetchall()
        return [self._hydrate(row) for row in rows]

    def latest_for_evidence(
        self, *, run_id: str, original_result_ref: str, evidence_ref: str
    ) -> SemanticFeedbackEvent | None:
        with self._connect() as connection:
            row = connection.execute(
                sql_query("""
                SELECT * FROM semantic_feedback_events
                WHERE workspace_id = ? AND run_id = ?
                  AND original_result_ref = ? AND evidence_ref = ?
                ORDER BY occurred_at DESC, rowid DESC
                LIMIT 1
                """,'\n                SELECT * FROM semantic_feedback_events\n                WHERE workspace_id = %s AND run_id = %s\n                  AND original_result_ref = %s AND evidence_ref = %s\n                ORDER BY occurred_at DESC, event_sequence DESC\n                LIMIT 1\n                '),
                (self.workspace_id, run_id, original_result_ref, evidence_ref),
            ).fetchone()
        return self._hydrate(row) if row is not None else None

    def list_ai_extract_negative_examples(
        self, *, profile_ref: str, profile_version: str | None = None, limit: int = 50
    ) -> list[SemanticFeedbackEvent]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        query = sql_query("""
            SELECT * FROM semantic_feedback_events
            WHERE workspace_id = ? AND profile_ref = ?
        """,'\n            SELECT * FROM semantic_feedback_events\n            WHERE workspace_id = %s AND profile_ref = %s\n        ')
        parameters: tuple[object, ...] = (self.workspace_id, profile_ref)
        if profile_version is not None:
            query += sql_query(" AND profile_version = ?",' AND profile_version = %s')
            parameters += (profile_version,)
        query += sql_query(" ORDER BY occurred_at DESC, rowid DESC"," ORDER BY occurred_at DESC, event_sequence DESC")
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        latest_by_occurrence: dict[tuple[str, str, str], SemanticFeedbackEvent] = {}
        for row in rows:
            event = self._hydrate(row)
            key = (event.run_id, event.original_result_ref, event.evidence_ref)
            if key in latest_by_occurrence:
                continue
            latest_by_occurrence[key] = event
        return [
            event
            for event in latest_by_occurrence.values()
            if event.action == "dismiss"
            and (event.corrected_result or {}).get("decision_scope")
            == "evidence_occurrence"
        ][:limit]

    def _hydrate(self, row: sqlite3.Row) -> SemanticFeedbackEvent:
        return SemanticFeedbackEvent(
            schema_version=row["schema_version"],
            feedback_id=row["feedback_id"],
            run_id=row["run_id"],
            finding_id=row["finding_id"],
            action=row["action"],
            original_result_ref=row["original_result_ref"],
            corrected_result=_loads(row["corrected_result_json"]) if row["corrected_result_json"] else None,
            reason_code=row["reason_code"],
            justification=row["justification"],
            evidence_ref=row["evidence_ref"],
            graph_version_ref=row["graph_version_ref"],
            ontology_version_ref=row["ontology_version_ref"],
            profile_ref=row["profile_ref"],
            profile_version=row["profile_version"],
            skill_ref=row["skill_ref"],
            derivation_refs=_loads(row["derivation_refs_json"]),
            reviewer=row["reviewer"],
            occurred_at=row["occurred_at"],
            event_hash=row["event_hash"],
        )
