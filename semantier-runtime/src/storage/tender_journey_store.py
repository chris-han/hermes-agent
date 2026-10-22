"""Explicit local/hosted workflow persistence for the resumable Tender review journey.

The store deliberately shares the workspace ontology-build store. It owns only
journey coordination state; governed semantic artifacts remain in their existing stores.
"""

from __future__ import annotations

from storage import hosted_postgres as hosted
from storage.transaction_scope import transaction_scope

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from uuid import uuid4

from contracts.tender_journey import (
    TenderJourneyHeadV1,
    TenderJourneyTransitionV1,
    TenderJourneyV1,
)
from storage.ontology_build_store import ontology_build_store_path


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TenderJourneyStore:
    def __init__(self, workspace_root: str | Path) -> None:
        self.db_path = ontology_build_store_path(workspace_root)
        if hosted.enabled():
            self._connection = hosted.connect_authority('standalone', self.db_path)
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self) -> None:
        self._connection.close()

    def _ensure_schema(self) -> None:
        if hosted.enabled():
            hosted.require_tables(self._connection, 'tender_journey', 'tender_journey_transition', 'tender_journey_head', 'tender_journey_session_binding')
            self._connection.commit()
            return
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tender_journey (
                journey_id TEXT PRIMARY KEY,
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                journey_kind TEXT NOT NULL CHECK(journey_kind='TENDER_REVIEW'),
                created_by TEXT NOT NULL,
                lifecycle_status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tender_journey_transition (
                transition_id TEXT PRIMARY KEY,
                journey_id TEXT NOT NULL,
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                predecessor_revision INTEGER NOT NULL CHECK(predecessor_revision>=0),
                successor_revision INTEGER NOT NULL CHECK(successor_revision=predecessor_revision+1),
                from_phase TEXT NOT NULL,
                to_phase TEXT NOT NULL,
                cause_code TEXT NOT NULL,
                actor_ref TEXT NOT NULL,
                surface TEXT NOT NULL CHECK(surface IN ('CHAT','WORKBENCH','SYSTEM')),
                command_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                result_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(journey_id) REFERENCES tender_journey(journey_id),
                UNIQUE(journey_id, successor_revision),
                UNIQUE(journey_id, idempotency_key)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tender_journey_head (
                journey_id TEXT PRIMARY KEY,
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                revision INTEGER NOT NULL CHECK(revision>=0),
                current_phase TEXT NOT NULL,
                lifecycle_status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                latest_transition_id TEXT,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(journey_id) REFERENCES tender_journey(journey_id)
            )
            """
        )
        columns = {row["name"] for row in self._connection.execute(
            "PRAGMA table_info(tender_journey_transition)"
        )}
        if "result_json" not in columns:
            self._connection.execute("ALTER TABLE tender_journey_transition ADD COLUMN result_json TEXT")
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS tender_journey_active_scope_idx "
            "ON tender_journey(organization_id,workspace_id,lifecycle_status,created_at)"
        )
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS tender_journey_session_binding (
                organization_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                actor_ref TEXT NOT NULL,
                session_ref TEXT NOT NULL,
                journey_id TEXT NOT NULL REFERENCES tender_journey(journey_id),
                PRIMARY KEY (organization_id, workspace_id, actor_ref, session_ref)
            )
        """)
        self._connection.commit()

    def create(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        created_by: str,
        journey_id: str | None = None,
        session_ref: str | None = None,
        now: datetime | None = None,
    ) -> TenderJourneyHeadV1:
        created_at = now or _utc_now()
        journey_ref = journey_id or f"tender-journey:{uuid4().hex}"
        journey = TenderJourneyV1(
            journey_id=journey_ref,
            organization_id=organization_id,
            workspace_id=workspace_id,
            created_by=created_by,
            created_at=created_at,
        )
        head = TenderJourneyHeadV1(
            journey_id=journey_ref,
            organization_id=organization_id,
            workspace_id=workspace_id,
            revision=0,
            current_phase="NEEDS_POLICY",
            updated_at=created_at,
        )
        with transaction_scope(self._connection):
            self._connection.execute(
                hosted.sql_query("INSERT INTO tender_journey(journey_id,organization_id,workspace_id,journey_kind,created_by,lifecycle_status,created_at) VALUES(?,?,?,?,?,?,?)", 'INSERT INTO tender_journey(journey_id,organization_id,workspace_id,journey_kind,created_by,lifecycle_status,created_at) VALUES(%s,%s,%s,%s,%s,%s,%s)'),
                (
                    journey.journey_id,
                    journey.organization_id,
                    journey.workspace_id,
                    journey.journey_kind,
                    journey.created_by,
                    journey.lifecycle_status,
                    journey.created_at.isoformat(),
                ),
            )
            self._connection.execute(
                hosted.sql_query("INSERT INTO tender_journey_head(journey_id,organization_id,workspace_id,revision,current_phase,lifecycle_status,payload_json,latest_transition_id,updated_at) VALUES(?,?,?,?,?,?,?,?,?)", 'INSERT INTO tender_journey_head(journey_id,organization_id,workspace_id,revision,current_phase,lifecycle_status,payload_json,latest_transition_id,updated_at) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)'),
                (
                    head.journey_id,
                    head.organization_id,
                    head.workspace_id,
                    head.revision,
                    head.current_phase,
                    head.lifecycle_status,
                    head.model_dump_json(),
                    None,
                    head.updated_at.isoformat(),
                ),
            )
            if session_ref is not None:
                self._write_session_binding(organization_id=organization_id, workspace_id=workspace_id,
                                            actor_ref=created_by, session_ref=session_ref, journey_id=journey_ref)
        return head

    def get_head(
        self, *, journey_id: str, organization_id: str, workspace_id: str
    ) -> TenderJourneyHeadV1:
        row = self._connection.execute(
            hosted.sql_query("SELECT h.*, j.lifecycle_status AS indexed_lifecycle, j.organization_id AS indexed_org, "
            "j.workspace_id AS indexed_workspace, t.transition_id AS last_id, "
            "t.successor_revision AS last_revision, t.organization_id AS last_org, t.workspace_id AS last_workspace, "
            "t.payload_json AS last_payload, t.result_json AS last_result "
            "FROM tender_journey_head h LEFT JOIN tender_journey j ON j.journey_id=h.journey_id "
            "LEFT JOIN tender_journey_transition t ON t.transition_id=("
            "SELECT transition_id FROM tender_journey_transition WHERE journey_id=h.journey_id "
            "ORDER BY successor_revision DESC LIMIT 1) "
            "WHERE h.journey_id=? AND h.organization_id=? AND h.workspace_id=?", 'SELECT h.*, j.lifecycle_status AS indexed_lifecycle, j.organization_id AS indexed_org, j.workspace_id AS indexed_workspace, t.transition_id AS last_id, t.successor_revision AS last_revision, t.organization_id AS last_org, t.workspace_id AS last_workspace, t.payload_json AS last_payload, t.result_json AS last_result FROM tender_journey_head h LEFT JOIN tender_journey j ON j.journey_id=h.journey_id LEFT JOIN tender_journey_transition t ON t.transition_id=(SELECT transition_id FROM tender_journey_transition WHERE journey_id=h.journey_id ORDER BY successor_revision DESC LIMIT 1) WHERE h.journey_id=%s AND h.organization_id=%s AND h.workspace_id=%s'),
            (journey_id, organization_id, workspace_id),
        ).fetchone()
        if row is None:
            raise KeyError("JOURNEY_NOT_FOUND")
        try:
            head = TenderJourneyHeadV1.model_validate_json(row["payload_json"])
            if any(getattr(head, field) != row[field] for field in (
                "journey_id", "organization_id", "workspace_id", "revision", "current_phase",
                "lifecycle_status", "latest_transition_id",
            )) or (row["indexed_org"], row["indexed_workspace"], row["indexed_lifecycle"]) != (
                organization_id, workspace_id, head.lifecycle_status,
            ) or head.updated_at.isoformat() != row["updated_at"]:
                raise ValueError("head columns differ")
            if head.revision == 0:
                if row["last_id"] is not None or head.latest_transition_id is not None:
                    raise ValueError("unexpected initial transition")
            else:
                if (row["last_id"], row["last_revision"], row["last_org"], row["last_workspace"]) != (
                    head.latest_transition_id, head.revision, organization_id, workspace_id,
                ):
                    raise ValueError("latest transition differs")
                transition = TenderJourneyTransitionV1.model_validate_json(row["last_payload"])
                if (transition.journey_id, transition.transition_id, transition.successor_revision,
                    transition.to_phase, transition.blocker_code, transition.operation_receipt,
                    transition.created_at) != (head.journey_id, head.latest_transition_id, head.revision,
                    head.current_phase, head.blocker_code, head.operation_receipt, head.updated_at):
                    raise ValueError("latest transition payload differs")
                if row["last_result"] is not None and TenderJourneyHeadV1.model_validate_json(row["last_result"]) != head:
                    raise ValueError("latest result differs")
            return head
        except ValueError as exc:
            raise ValueError("JOURNEY_HEAD_INCONSISTENT") from exc

    def get_transition_result(self, *, journey_id: str, organization_id: str,
                              workspace_id: str, transition_id: str):
        self.get_head(journey_id=journey_id, organization_id=organization_id, workspace_id=workspace_id)
        row = self._connection.execute(
            hosted.sql_query("SELECT payload_json,result_json FROM tender_journey_transition "
            "WHERE journey_id=? AND organization_id=? AND workspace_id=? AND transition_id=?", 'SELECT payload_json,result_json FROM tender_journey_transition WHERE journey_id=%s AND organization_id=%s AND workspace_id=%s AND transition_id=%s'),
            (journey_id, organization_id, workspace_id, transition_id),
        ).fetchone()
        if row is None:
            raise KeyError("JOURNEY_TRANSITION_NOT_FOUND")
        if row["result_json"] is None:
            raise ValueError("JOURNEY_COMMAND_RESULT_UNAVAILABLE")
        transition = TenderJourneyTransitionV1.model_validate_json(row["payload_json"])
        head = TenderJourneyHeadV1.model_validate_json(row["result_json"])
        if ((head.journey_id, head.organization_id, head.workspace_id) != (journey_id, organization_id, workspace_id)
                or transition.journey_id != journey_id or transition.transition_id != transition_id
                or head.latest_transition_id != transition_id or head.revision != transition.successor_revision
                or head.current_phase != transition.to_phase or head.operation_receipt != transition.operation_receipt):
            raise ValueError("JOURNEY_TRANSITION_RESULT_MISMATCH")
        return transition, head

    def list_transitions(
        self, *, journey_id: str, organization_id: str, workspace_id: str
    ) -> tuple[TenderJourneyTransitionV1, ...]:
        rows = self._connection.execute(
            hosted.sql_query("SELECT payload_json FROM tender_journey_transition WHERE journey_id=? AND organization_id=? AND workspace_id=? ORDER BY successor_revision ASC", 'SELECT payload_json FROM tender_journey_transition WHERE journey_id=%s AND organization_id=%s AND workspace_id=%s ORDER BY successor_revision ASC'),
            (journey_id, organization_id, workspace_id),
        ).fetchall()
        return tuple(TenderJourneyTransitionV1.model_validate_json(row["payload_json"]) for row in rows)

    def list_active_heads(
        self, *, organization_id: str, workspace_id: str
    ) -> tuple[TenderJourneyHeadV1, ...]:
        rows = self._connection.execute(
            hosted.sql_query("SELECT h.journey_id FROM tender_journey_head h JOIN tender_journey j ON j.journey_id=h.journey_id WHERE h.organization_id=? AND h.workspace_id=? AND j.lifecycle_status='ACTIVE' ORDER BY j.created_at DESC, j.journey_id", "SELECT h.journey_id FROM tender_journey_head h JOIN tender_journey j ON j.journey_id=h.journey_id WHERE h.organization_id=%s AND h.workspace_id=%s AND j.lifecycle_status='ACTIVE' ORDER BY j.created_at DESC, j.journey_id"),
            (organization_id, workspace_id),
        ).fetchall()
        return tuple(self.get_head(journey_id=row["journey_id"], organization_id=organization_id, workspace_id=workspace_id) for row in rows)

    def list_resumable_heads(self, *, organization_id: str, workspace_id: str) -> tuple[TenderJourneyHeadV1, ...]:
        rows = self._connection.execute(
            hosted.sql_query("SELECT journey_id FROM tender_journey_head WHERE organization_id=? AND workspace_id=? "
            "AND lifecycle_status IN ('ACTIVE','PAUSED') ORDER BY journey_id", "SELECT journey_id FROM tender_journey_head WHERE organization_id=%s AND workspace_id=%s AND lifecycle_status IN ('ACTIVE','PAUSED') ORDER BY journey_id"),
            (organization_id, workspace_id),
        ).fetchall()
        return tuple(self.get_head(journey_id=row["journey_id"], organization_id=organization_id, workspace_id=workspace_id) for row in rows)

    def get_session_journey(self, *, organization_id: str, workspace_id: str, actor_ref: str,
                            session_ref: str) -> TenderJourneyHeadV1 | None:
        row = self._connection.execute(
            hosted.sql_query("SELECT journey_id FROM tender_journey_session_binding "
            "WHERE organization_id=? AND workspace_id=? AND actor_ref=? AND session_ref=?", 'SELECT journey_id FROM tender_journey_session_binding WHERE organization_id=%s AND workspace_id=%s AND actor_ref=%s AND session_ref=%s'),
            (organization_id, workspace_id, actor_ref, session_ref),
        ).fetchone()
        if row is None:
            return None
        return self.get_head(journey_id=row["journey_id"], organization_id=organization_id, workspace_id=workspace_id)

    def bind_session(self, *, organization_id: str, workspace_id: str, actor_ref: str,
                     session_ref: str, journey_id: str) -> TenderJourneyHeadV1:
        head = self.get_head(journey_id=journey_id, organization_id=organization_id, workspace_id=workspace_id)
        with transaction_scope(self._connection):
            self._write_session_binding(organization_id=organization_id, workspace_id=workspace_id,
                                        actor_ref=actor_ref, session_ref=session_ref, journey_id=journey_id)
        return head

    def _write_session_binding(self, *, organization_id, workspace_id, actor_ref, session_ref, journey_id):
        self._connection.execute(
            hosted.sql_query("INSERT INTO tender_journey_session_binding "
            "(organization_id,workspace_id,actor_ref,session_ref,journey_id) VALUES (?,?,?,?,?) "
            "ON CONFLICT(organization_id,workspace_id,actor_ref,session_ref) "
            "DO UPDATE SET journey_id=excluded.journey_id", 'INSERT INTO tender_journey_session_binding (organization_id,workspace_id,actor_ref,session_ref,journey_id) VALUES (%s,%s,%s,%s,%s) ON CONFLICT(organization_id,workspace_id,actor_ref,session_ref) DO UPDATE SET journey_id=excluded.journey_id'),
            (organization_id, workspace_id, actor_ref, session_ref, journey_id),
        )

    def create_for_session(self, *, organization_id: str, workspace_id: str, actor_ref: str,
                           session_ref: str, command_id: str) -> TenderJourneyHeadV1:
        from eos.artifact_hashing import content_hash

        scope = dict(organization_id=organization_id, workspace_id=workspace_id)
        journey_id = "tender-journey:" + content_hash({**scope, "actor_ref": actor_ref,
                                                    "session_ref": session_ref, "command_id": command_id})
        try:
            self.get_head(**scope, journey_id=journey_id)
        except KeyError:
            try:
                self.create(**scope, created_by=actor_ref, journey_id=journey_id, session_ref=session_ref)
            except hosted.integrity_errors() as exc:
                # Another delivery may have committed the same create command.
                # Preserve an unrelated insertion failure instead of hiding it.
                try:
                    self.get_head(**scope, journey_id=journey_id)
                except KeyError:
                    raise exc
        return self.get_head(**scope, journey_id=journey_id)

    def set_lifecycle(
        self, *, journey_id: str, organization_id: str, workspace_id: str,
        expected_revision: int, lifecycle_status: str,
        actor_ref: str = "system:journey", surface: str = "SYSTEM",
        now: datetime | None = None,
    ) -> TenderJourneyHeadV1:
        scope = dict(journey_id=journey_id, organization_id=organization_id, workspace_id=workspace_id)
        command_id = f"journey-lifecycle:{journey_id}:{expected_revision}:{lifecycle_status}"
        cause = f"JOURNEY_LIFECYCLE_{lifecycle_status}"
        replay = self.replay_command(**scope, idempotency_key=command_id, command={
            "predecessor_revision": expected_revision, "cause_code": cause,
            "actor_ref": actor_ref, "surface": surface, "command_id": command_id,
            "to_lifecycle_status": lifecycle_status,
        })
        if replay is not None:
            return replay
        current = self.get_head(**scope)
        if current.revision != expected_revision:
            raise ValueError("STALE_JOURNEY_REVISION")
        return self.append_transition(
            organization_id=organization_id, workspace_id=workspace_id, expected_revision=expected_revision,
            transition=TenderJourneyTransitionV1(
                transition_id=f"tender-journey-transition:{uuid4().hex}", journey_id=journey_id,
                predecessor_revision=expected_revision, successor_revision=expected_revision + 1,
                from_phase=current.current_phase, to_phase=current.current_phase,
                from_lifecycle_status=current.lifecycle_status, to_lifecycle_status=lifecycle_status,
                cause_code=cause, actor_ref=actor_ref, surface=surface,
                command_id=command_id, idempotency_key=command_id,
                blocker_code=current.blocker_code, operation_receipt=current.operation_receipt,
                created_at=now or _utc_now(),
            ),
        )

    def replay_command(
        self, *, journey_id: str, organization_id: str, workspace_id: str,
        idempotency_key: str, command: dict,
    ) -> TenderJourneyHeadV1 | None:
        # Authorize before reading command identity or reporting a collision.
        self.get_head(journey_id=journey_id, organization_id=organization_id, workspace_id=workspace_id)
        row = self._connection.execute(
            hosted.sql_query("SELECT payload_json,result_json FROM tender_journey_transition "
            "WHERE journey_id=? AND organization_id=? AND workspace_id=? AND idempotency_key=?", 'SELECT payload_json,result_json FROM tender_journey_transition WHERE journey_id=%s AND organization_id=%s AND workspace_id=%s AND idempotency_key=%s'),
            (journey_id, organization_id, workspace_id, idempotency_key),
        ).fetchone()
        if row is None:
            return None
        persisted = TenderJourneyTransitionV1.model_validate_json(row["payload_json"])
        if any(getattr(persisted, key) != value for key, value in command.items()):
            raise ValueError("IDEMPOTENCY_KEY_REUSED_WITH_DIFFERENT_TRANSITION")
        if row["result_json"] is None:
            raise ValueError("JOURNEY_COMMAND_RESULT_UNAVAILABLE")
        return TenderJourneyHeadV1.model_validate_json(row["result_json"])

    def append_transition(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        expected_revision: int,
        transition: TenderJourneyTransitionV1,
    ) -> TenderJourneyHeadV1:
        if transition.predecessor_revision != expected_revision:
            raise ValueError("STALE_JOURNEY_REVISION")
        hosted.begin_write(self._connection)
        try:
            replay = self.replay_command(
                journey_id=transition.journey_id, organization_id=organization_id,
                workspace_id=workspace_id, idempotency_key=transition.idempotency_key,
                command={name: getattr(transition, name) for name in type(transition).model_fields
                         if name not in {"transition_id", "created_at"}},
            )
            if replay is not None:
                self._connection.commit()
                return replay

            current = self.get_head(
                journey_id=transition.journey_id,
                organization_id=organization_id,
                workspace_id=workspace_id,
            )
            if current.revision != expected_revision or current.current_phase != transition.from_phase:
                raise ValueError("STALE_JOURNEY_REVISION")

            if transition.to_lifecycle_status is not None:
                if (current.lifecycle_status, transition.to_lifecycle_status) not in {
                    ("ACTIVE", "PAUSED"), ("PAUSED", "ACTIVE"),
                    ("ACTIVE", "CANCELLED"), ("PAUSED", "CANCELLED"),
                }:
                    raise ValueError("INVALID_JOURNEY_LIFECYCLE_TRANSITION")
                if (transition.from_lifecycle_status != current.lifecycle_status
                        or transition.artifact_bindings or transition.operation_receipt != current.operation_receipt
                        or transition.blocker_code != current.blocker_code):
                    raise ValueError("INVALID_JOURNEY_LIFECYCLE_TRANSITION")
            elif current.lifecycle_status != "ACTIVE":
                raise ValueError("JOURNEY_NOT_ACTIVE")

            bindings = {binding.role: binding for binding in current.artifact_bindings}
            if current.current_phase == "REVIEW_RESULTS" and transition.to_phase in {
                "CONTEXT_READY", "NEEDS_RUNTIME_DOCUMENT",
            }:
                for role in ("runtime_document", "runtime_document_preparation", "runtime_context", "inspection_result"):
                    bindings.pop(role, None)
            bindings.update({binding.role: binding for binding in transition.artifact_bindings})
            now = transition.created_at
            successor = TenderJourneyHeadV1(
                journey_id=current.journey_id,
                organization_id=current.organization_id,
                workspace_id=current.workspace_id,
                revision=transition.successor_revision,
                current_phase=transition.to_phase,
                lifecycle_status=transition.to_lifecycle_status or current.lifecycle_status,
                artifact_bindings=tuple(bindings.values()),
                blocker_code=transition.blocker_code,
                operation_receipt=transition.operation_receipt,
                latest_transition_id=transition.transition_id,
                updated_at=now,
            )
            self._connection.execute(
                hosted.sql_query("INSERT INTO tender_journey_transition(transition_id,journey_id,organization_id,workspace_id,predecessor_revision,successor_revision,from_phase,to_phase,cause_code,actor_ref,surface,command_id,idempotency_key,payload_json,result_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", 'INSERT INTO tender_journey_transition(transition_id,journey_id,organization_id,workspace_id,predecessor_revision,successor_revision,from_phase,to_phase,cause_code,actor_ref,surface,command_id,idempotency_key,payload_json,result_json,created_at) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'),
                (
                    transition.transition_id,
                    transition.journey_id,
                    organization_id,
                    workspace_id,
                    transition.predecessor_revision,
                    transition.successor_revision,
                    transition.from_phase,
                    transition.to_phase,
                    transition.cause_code,
                    transition.actor_ref,
                    transition.surface,
                    transition.command_id,
                    transition.idempotency_key,
                    transition.model_dump_json(),
                    successor.model_dump_json(),
                    transition.created_at.isoformat(),
                ),
            )
            if successor.lifecycle_status != current.lifecycle_status:
                self._connection.execute(
                    hosted.sql_query("UPDATE tender_journey SET lifecycle_status=? WHERE journey_id=? AND organization_id=? AND workspace_id=?", 'UPDATE tender_journey SET lifecycle_status=%s WHERE journey_id=%s AND organization_id=%s AND workspace_id=%s'),
                    (successor.lifecycle_status, current.journey_id, organization_id, workspace_id),
                )
            updated = self._connection.execute(
                hosted.sql_query("UPDATE tender_journey_head SET revision=?,current_phase=?,lifecycle_status=?,payload_json=?,latest_transition_id=?,updated_at=? WHERE journey_id=? AND organization_id=? AND workspace_id=? AND revision=?", 'UPDATE tender_journey_head SET revision=%s,current_phase=%s,lifecycle_status=%s,payload_json=%s,latest_transition_id=%s,updated_at=%s WHERE journey_id=%s AND organization_id=%s AND workspace_id=%s AND revision=%s'),
                (
                    successor.revision,
                    successor.current_phase,
                    successor.lifecycle_status,
                    successor.model_dump_json(),
                    successor.latest_transition_id,
                    successor.updated_at.isoformat(),
                    successor.journey_id,
                    organization_id,
                    workspace_id,
                    expected_revision,
                ),
            )
            if updated.rowcount != 1:
                raise ValueError("STALE_JOURNEY_REVISION")
            self._connection.commit()
            return successor
        except BaseException:
            self._connection.rollback()
            raise
