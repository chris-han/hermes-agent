from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash as compute_content_hash
from eos.db import init_db
from runtime_paths import sqlite_db_path
from storage.hosted_postgres import enabled,connect_authority,require_tables,sql_query,begin_write
from storage.transaction_scope import transaction_scope,StoreMutationLock

ActivationEventType = Literal[
    "ACTIVATED",
    "SUSPENDED",
    "RESUMED",
    "REVOKED",
    "SUPERSESSION_RECORDED",
    "SCOPE_REPLACEMENT_RECORDED",
    "EXPIRATION_RECORDED",
]

AUTHORITY_CHANGING_EVENTS = {"ACTIVATED", "SUSPENDED", "RESUMED", "REVOKED"}


def _db_path() -> Path:
    return Path(os.environ.get("SEMANTIER_EOS_DB_PATH", str(sqlite_db_path("eos.db", ".semantier/eos.db"))))


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _assert_ascii(value: str, field: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        raise ValueError(f"{field} is required")
    try:
        normalized.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field} must be an ASCII-stable identifier") from exc
    return normalized


def _assert_utc_timestamp(value: str, field: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        raise ValueError(f"{field} is required")
    parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{field} must be timezone-aware UTC ISO-8601")
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _event_payload(
    *,
    activation_event_id: str,
    activation_stream_id: str,
    activation_id: str,
    event_type: str,
    from_state: str | None,
    to_state: str | None,
    effective_at: str,
    recorded_at: str,
    event_sequence: int,
    actor: str,
    actor_authority: str,
    reason: str | None,
    prior_event_hash: str | None,
    created_at: str,
) -> dict[str, Any]:
    return {
        "activation_event_id": activation_event_id,
        "activation_stream_id": activation_stream_id,
        "activation_id": activation_id,
        "event_type": event_type,
        "from_state": from_state,
        "to_state": to_state,
        "effective_at": effective_at,
        "recorded_at": recorded_at,
        "event_sequence": event_sequence,
        "actor": actor,
        "actor_authority": actor_authority,
        "reason": reason,
        "prior_event_hash": prior_event_hash,
        "created_at": created_at,
    }


def compute_activation_event_hash(event: dict[str, Any]) -> str:
    payload = {key: value for key, value in event.items() if key != "content_hash"}
    return compute_content_hash(canonical_json(payload))


class GovernedRuleActivationEventStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        path = Path(db_path) if db_path else _db_path()
        if enabled():
            self._conn=connect_authority("eos",path)
            require_tables(self._conn,'governed_rule_activation_event')
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
        self._lock=StoreMutationLock(self._conn)

    def append_event(
        self,
        *,
        activation_stream_id: str,
        activation_id: str,
        event_type: ActivationEventType,
        from_state: str | None,
        to_state: str | None,
        effective_at: str,
        actor: str,
        actor_authority: str,
        reason: str | None = None,
        activation_event_id: str | None = None,
    ) -> dict[str, Any]:
        activation_stream_id = _assert_ascii(activation_stream_id, "activation_stream_id")
        activation_id = _assert_ascii(activation_id, "activation_id")
        event_type = _assert_ascii(event_type, "event_type")  # type: ignore[assignment]
        actor = _assert_ascii(actor, "actor")
        actor_authority = _assert_ascii(actor_authority, "actor_authority") if actor_authority else ""
        effective_at = _assert_utc_timestamp(effective_at, "effective_at")
        recorded_at = _now_iso()
        created_at = recorded_at
        with self._lock, transaction_scope(self._conn):
            prior = self._conn.execute(
                sql_query("""
                SELECT * FROM governed_rule_activation_event
                WHERE activation_stream_id = ?
                ORDER BY event_sequence DESC
                LIMIT 1
                """, '\n                SELECT * FROM governed_rule_activation_event\n                WHERE activation_stream_id = %s\n                ORDER BY event_sequence DESC\n                LIMIT 1\n                '),
                (activation_stream_id,),
            ).fetchone()
            event_sequence = int(prior["event_sequence"]) + 1 if prior else 1
            prior_event_hash = prior["content_hash"] if prior else None
            event_id = _assert_ascii(
                activation_event_id or f"grae_{uuid4().hex}",
                "activation_event_id",
            )
            payload = _event_payload(
                activation_event_id=event_id,
                activation_stream_id=activation_stream_id,
                activation_id=activation_id,
                event_type=event_type,
                from_state=from_state,
                to_state=to_state,
                effective_at=effective_at,
                recorded_at=recorded_at,
                event_sequence=event_sequence,
                actor=actor,
                actor_authority=actor_authority,
                reason=reason,
                prior_event_hash=prior_event_hash,
                created_at=created_at,
            )
            event_hash = compute_content_hash(canonical_json(payload))
            self._conn.execute(
                sql_query("""
                INSERT INTO governed_rule_activation_event (
                  activation_event_id, activation_stream_id, activation_id,
                  event_type, from_state, to_state, effective_at, recorded_at,
                  event_sequence, actor, actor_authority, reason,
                  prior_event_hash, content_hash, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO governed_rule_activation_event (\n                  activation_event_id, activation_stream_id, activation_id,\n                  event_type, from_state, to_state, effective_at, recorded_at,\n                  event_sequence, actor, actor_authority, reason,\n                  prior_event_hash, content_hash, created_at\n                )\n                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    event_id,
                    activation_stream_id,
                    activation_id,
                    event_type,
                    from_state,
                    to_state,
                    effective_at,
                    recorded_at,
                    event_sequence,
                    actor,
                    actor_authority,
                    reason,
                    prior_event_hash,
                    event_hash,
                    created_at,
                ),
            )
        return self.get_event(event_id)  # type: ignore[return-value]

    def get_event(self, activation_event_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            sql_query("SELECT * FROM governed_rule_activation_event WHERE activation_event_id = ?", 'SELECT * FROM governed_rule_activation_event WHERE activation_event_id = %s'),
            (_assert_ascii(activation_event_id, "activation_event_id"),),
        ).fetchone()
        return dict(row) if row else None

    def list_events(self, activation_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            sql_query("""
            SELECT * FROM governed_rule_activation_event
            WHERE activation_id = ?
            ORDER BY event_sequence
            """, '\n            SELECT * FROM governed_rule_activation_event\n            WHERE activation_id = %s\n            ORDER BY event_sequence\n            '),
            (_assert_ascii(activation_id, "activation_id"),),
        ).fetchall()
        return [dict(row) for row in rows]
