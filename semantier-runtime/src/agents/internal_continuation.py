from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
from storage import hosted_postgres as hosted
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.durable_protocol_versions import (
    CURRENT_CONTINUATION_PAYLOAD_VERSION,
    CURRENT_SCHEMA_VERSION,
    CURRENT_WORKFLOW_CONTRACT_VERSION,
    assert_row_supported,
    assert_supported_versions,
    ensure_version_columns,
)


PENDING = "PENDING"
DISPATCHED = "DISPATCHED"
REJECTED = "REJECTED"

ACCEPTED = "ACCEPTED"
ALREADY_ACCEPTED = "ALREADY_ACCEPTED"
REJECTED_STALE = "REJECTED_STALE"
REJECTED_AUTHORITY = "REJECTED_AUTHORITY"
UNAVAILABLE = "UNAVAILABLE"
MAX_CONTINUATION_PAYLOAD_BYTES = 16 * 1024


class ContinuationError(RuntimeError):
    pass


class ContinuationIdentityConflict(ContinuationError):
    pass


@dataclass(frozen=True)
class ContinuationRecord:
    continuation_id: str
    workspace_id: str
    workflow_id: str
    event_id: str
    session_key: str
    payload_hash: str
    payload_json: str
    state: str
    rejection_reason: str | None


@dataclass(frozen=True)
class ContinuationDispatchResult:
    continuation_id: str
    outcome: str
    state: str
    detail: str | None = None


def _canonical_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(encoded.encode("utf-8")) > MAX_CONTINUATION_PAYLOAD_BYTES:
        raise ValueError("bounded continuation payload exceeds 16 KiB")
    return encoded


def derive_continuation_id(
    *,
    workspace_id: str,
    workflow_id: str,
    event_id: str,
    session_key: str,
    payload_json: str,
) -> str:
    if not workspace_id.strip():
        raise ValueError("workspace_id is required")
    if not workflow_id.strip():
        raise ValueError("workflow_id is required")
    if not event_id.strip():
        raise ValueError("event_id is required")
    if not session_key.strip():
        raise ValueError("session_key is required")
    payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    raw = "|".join([workspace_id, workflow_id, event_id, session_key, payload_hash])
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"continuation_{digest}"


class ContinuationAdmissionStore:
    """Durable admission for one bounded internal continuation."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        if not hosted.enabled():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            self.ensure_schema(conn)

    def _connect(self) -> sqlite3.Connection:
        if hosted.enabled():
            return hosted.connect_authority('execution', self.db_path, autocommit=True)
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @staticmethod
    def ensure_schema(conn: sqlite3.Connection) -> None:
        if hosted.is_postgres(conn):
            hosted.require_tables(conn, 'semantier_internal_continuations')
            return
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantier_internal_continuations (
                continuation_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                workflow_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                session_key TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                state TEXT NOT NULL,
                rejection_reason TEXT,
                schema_version INTEGER NOT NULL DEFAULT 1,
                workflow_contract_version INTEGER NOT NULL DEFAULT 1,
                continuation_payload_version INTEGER NOT NULL DEFAULT 1,
                CHECK (state IN ('PENDING','DISPATCHED','REJECTED')),
                UNIQUE(workspace_id, workflow_id, event_id, session_key, payload_hash)
            )
            """
        )
        ensure_version_columns(conn, "continuation")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_semantier_internal_continuations_pending
            ON semantier_internal_continuations(state, session_key)
            """
        )

    @staticmethod
    def admit(
        conn: sqlite3.Connection,
        *,
        workspace_id: str,
        workflow_id: str,
        event_id: str,
        session_key: str,
        bounded_payload: dict[str, Any],
    ) -> ContinuationRecord:
        ContinuationAdmissionStore.ensure_schema(conn)
        payload_json = _canonical_payload(bounded_payload)
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        continuation_id = derive_continuation_id(
            workspace_id=workspace_id,
            workflow_id=workflow_id,
            event_id=event_id,
            session_key=session_key,
            payload_json=payload_json,
        )
        conn.execute(
            hosted.sql_query("""
            INSERT OR IGNORE INTO semantier_internal_continuations(
                continuation_id, workspace_id, workflow_id, event_id, session_key,
                payload_hash, payload_json, state,
                schema_version, workflow_contract_version,
                continuation_payload_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?)
            """, """
            INSERT INTO semantier_internal_continuations(
                continuation_id, workspace_id, workflow_id, event_id, session_key,
                payload_hash, payload_json, state,
                schema_version, workflow_contract_version,
                continuation_payload_version
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'PENDING', %s, %s, %s) ON CONFLICT DO NOTHING"""),
            (
                continuation_id,
                workspace_id,
                workflow_id,
                event_id,
                session_key,
                payload_hash,
                payload_json,
                CURRENT_SCHEMA_VERSION,
                CURRENT_WORKFLOW_CONTRACT_VERSION,
                CURRENT_CONTINUATION_PAYLOAD_VERSION,
            ),
        )
        row = conn.execute(
            hosted.sql_query("""
            SELECT *
            FROM semantier_internal_continuations
            WHERE continuation_id=?
            """, """
            SELECT *
            FROM semantier_internal_continuations
            WHERE continuation_id=%s
            """),
            (continuation_id,),
        ).fetchone()
        if row is None:
            raise ContinuationError("continuation admission was not persisted")
        record = ContinuationAdmissionStore._row_to_record(row)
        if (
            record.workspace_id != workspace_id
            or record.workflow_id != workflow_id
            or record.event_id != event_id
            or record.session_key != session_key
            or record.payload_hash != payload_hash
            or record.payload_json != payload_json
        ):
            raise ContinuationIdentityConflict("continuation identity collision")
        return record

    @staticmethod
    def _row_to_record(row) -> ContinuationRecord:
        if isinstance(row, sqlite3.Row):
            assert_row_supported("continuation", row)
            return ContinuationRecord(
                continuation_id=str(row["continuation_id"]),
                workspace_id=str(row["workspace_id"]),
                workflow_id=str(row["workflow_id"]),
                event_id=str(row["event_id"]),
                session_key=str(row["session_key"]),
                payload_hash=str(row["payload_hash"]),
                payload_json=str(row["payload_json"]),
                state=str(row["state"]),
                rejection_reason=(
                    str(row["rejection_reason"])
                    if row["rejection_reason"] is not None
                    else None
                ),
            )
        assert_supported_versions(
            "continuation",
            {
                "schema_version": int(row[9]),
                "workflow_contract_version": int(row[10]),
                "continuation_payload_version": int(row[11]),
            },
        )
        return ContinuationRecord(
            continuation_id=str(row[0]),
            workspace_id=str(row[1]),
            workflow_id=str(row[2]),
            event_id=str(row[3]),
            session_key=str(row[4]),
            payload_hash=str(row[5]),
            payload_json=str(row[6]),
            state=str(row[7]),
            rejection_reason=str(row[8]) if row[8] is not None else None,
        )

    def get(self, continuation_id: str) -> ContinuationRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                hosted.sql_query("""
                SELECT *
                FROM semantier_internal_continuations
                WHERE continuation_id=?
                """, """
                SELECT *
                FROM semantier_internal_continuations
                WHERE continuation_id=%s
                """),
                (continuation_id,),
            ).fetchone()
            return self._row_to_record(row) if row else None

    def mark_dispatched(self, *, continuation_id: str) -> ContinuationRecord:
        with self._connect() as conn:
            conn.execute(
                hosted.sql_query("""
                UPDATE semantier_internal_continuations
                SET state='DISPATCHED', rejection_reason=NULL
                WHERE continuation_id=? AND state='PENDING'
                """, """
                UPDATE semantier_internal_continuations
                SET state='DISPATCHED', rejection_reason=NULL
                WHERE continuation_id=%s AND state='PENDING'
                """),
                (continuation_id,),
            )
        record = self.get(continuation_id)
        if record is None:
            raise KeyError(continuation_id)
        return record

    def mark_rejected(
        self, *, continuation_id: str, reason: str
    ) -> ContinuationRecord:
        if not reason.strip():
            raise ValueError("reason is required")
        with self._connect() as conn:
            conn.execute(
                hosted.sql_query("""
                UPDATE semantier_internal_continuations
                SET state='REJECTED', rejection_reason=?
                WHERE continuation_id=? AND state='PENDING'
                """, """
                UPDATE semantier_internal_continuations
                SET state='REJECTED', rejection_reason=%s
                WHERE continuation_id=%s AND state='PENDING'
                """),
                (reason, continuation_id),
            )
        record = self.get(continuation_id)
        if record is None:
            raise KeyError(continuation_id)
        return record

    def pending(self) -> list[ContinuationRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM semantier_internal_continuations
                WHERE state='PENDING'
                ORDER BY continuation_id
                """
            ).fetchall()
            return [self._row_to_record(row) for row in rows]


class HermesGatewayInternalContinuationProvider:
    """Host-owned adapter over Hermes' existing internal MessageEvent path."""

    def __init__(self, runner: Any, *, recent_limit: int = 4096) -> None:
        self.runner = runner
        self.recent_limit = max(32, int(recent_limit))
        self._recent: OrderedDict[str, None] = OrderedDict()

    def _remember(self, continuation_id: str) -> None:
        self._recent[continuation_id] = None
        self._recent.move_to_end(continuation_id)
        while len(self._recent) > self.recent_limit:
            self._recent.popitem(last=False)

    async def dispatch(self, record: ContinuationRecord) -> str:
        runner = self.runner
        if not getattr(runner, "_running", False) or getattr(
            runner, "_draining", False
        ):
            return UNAVAILABLE

        entry = await runner.async_session_store.lookup_by_session_key(
            record.session_key
        )
        if entry is None or entry.origin is None:
            return REJECTED_STALE
        resolved_workspace = str(
            getattr(entry.origin, "workspace_owner_id", None) or ""
        ).strip()
        if resolved_workspace != record.workspace_id:
            return REJECTED_STALE

        if await runner.async_session_store.has_platform_message_id(
            entry.session_id,
            record.continuation_id,
        ):
            return ALREADY_ACCEPTED

        if record.continuation_id in self._recent:
            return ALREADY_ACCEPTED

        source = dataclasses.replace(entry.origin)
        try:
            authorized = runner._is_user_authorized(
                source,
                allow_adapter_delegation=False,
            )
        except Exception:
            return REJECTED_AUTHORITY
        if not authorized:
            return REJECTED_AUTHORITY

        adapter = runner._adapter_for_source(source)
        if adapter is None:
            return REJECTED_STALE

        from gateway.platforms.base import MessageEvent, MessageType

        content = (
            "[Semantier internal continuation]\n"
            + record.payload_json
        )
        event = MessageEvent(
            text=content,
            message_type=MessageType.TEXT,
            source=source,
            message_id=record.continuation_id,
            internal=True,
            allow_gateway_control=False,
            metadata={
                "semantier_internal_continuation": True,
                "continuation_id": record.continuation_id,
                "gateway_session_key": record.session_key,
                "gateway_session_id": entry.session_id,
                "gateway_session_strict": True,
            },
        )

        self._remember(record.continuation_id)
        try:
            await adapter.handle_message(event)
        except Exception:
            self._recent.pop(record.continuation_id, None)
            return UNAVAILABLE
        return ACCEPTED


class ContinuationDispatchService:
    def __init__(self, store: ContinuationAdmissionStore) -> None:
        self.store = store

    async def dispatch(
        self,
        *,
        continuation_id: str,
        provider: HermesGatewayInternalContinuationProvider,
    ) -> ContinuationDispatchResult:
        record = self.store.get(continuation_id)
        if record is None:
            raise KeyError(continuation_id)

        if record.state == DISPATCHED and not getattr(provider, 'revalidate_dispatched', False):
            return ContinuationDispatchResult(
                continuation_id=continuation_id,
                outcome=ALREADY_ACCEPTED,
                state=DISPATCHED,
                detail="already_dispatched",
            )
        if record.state == REJECTED:
            outcome = (
                REJECTED_AUTHORITY
                if record.rejection_reason == "authority_revalidation_failed"
                else REJECTED_STALE
            )
            return ContinuationDispatchResult(
                continuation_id=continuation_id,
                outcome=outcome,
                state=REJECTED,
                detail=record.rejection_reason,
            )

        outcome = await provider.dispatch(record)
        if outcome in {ACCEPTED, ALREADY_ACCEPTED}:
            final = self.store.mark_dispatched(continuation_id=continuation_id)
            return ContinuationDispatchResult(
                continuation_id=continuation_id,
                outcome=outcome,
                state=final.state,
            )

        if outcome == REJECTED_STALE:
            final = self.store.mark_rejected(
                continuation_id=continuation_id,
                reason="stale_or_unroutable_session",
            )
            return ContinuationDispatchResult(
                continuation_id=continuation_id,
                outcome=outcome,
                state=final.state,
                detail=final.rejection_reason,
            )

        if outcome == REJECTED_AUTHORITY:
            final = self.store.mark_rejected(
                continuation_id=continuation_id,
                reason="authority_revalidation_failed",
            )
            return ContinuationDispatchResult(
                continuation_id=continuation_id,
                outcome=outcome,
                state=final.state,
                detail=final.rejection_reason,
            )

        return ContinuationDispatchResult(
            continuation_id=continuation_id,
            outcome=UNAVAILABLE,
            state=PENDING,
            detail="runtime_unavailable_retry_later",
        )
