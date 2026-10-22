from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash as compute_content_hash
from eos.db import init_db
from runtime_paths import sqlite_db_path
from storage.hosted_postgres import enabled,connect_authority,require_tables,sql_query,begin_write
from storage.transaction_scope import transaction_scope,StoreMutationLock
from storage.governed_rule_activation_event_store import (
    AUTHORITY_CHANGING_EVENTS,
    compute_activation_event_hash,
)

ActivationScope = Literal["organization", "workspace"]


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


def _assert_optional_ascii(value: str | None, field: str) -> str | None:
    if value is None or not str(value).strip():
        return None
    return _assert_ascii(str(value), field)


def _assert_utc_timestamp(value: str, field: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        raise ValueError(f"{field} is required")
    parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{field} must be timezone-aware UTC ISO-8601")
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _assert_optional_utc_timestamp(value: str | None, field: str) -> str | None:
    if value is None or not str(value).strip():
        return None
    return _assert_utc_timestamp(str(value), field)


def _activation_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "activation_id": row["activation_id"],
        "rule_version_id": row["rule_version_id"],
        "organization_id": row["organization_id"],
        "activation_scope": row["activation_scope"],
        "workspace_id": row.get("workspace_id"),
        "activation_status": row["activation_status"],
        "effective_from": row["effective_from"],
        "effective_to": row.get("effective_to"),
        "activated_by": row["activated_by"],
        "actor_authority": row["actor_authority"],
        "approval_event_ref": row["approval_event_ref"],
        "source_activation_ref": row.get("source_activation_ref"),
        "activation_scope_hash": row["activation_scope_hash"],
        "supersedes_activation_id": row.get("supersedes_activation_id"),
        "created_at": row["created_at"],
    }


def _event_payload(event: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in event.items() if key != "content_hash"}


class GovernedRuleActivationStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        path = Path(db_path) if db_path else _db_path()
        if enabled():
            self._conn=connect_authority("eos",path)
            require_tables(self._conn,'governed_rule_activation','governed_rule_activation_event')
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

    def create_activation(
        self,
        *,
        rule_version_id: str,
        organization_id: str,
        activation_scope: ActivationScope,
        workspace_id: str | None,
        effective_from: str,
        effective_to: str | None = None,
        activated_by: str,
        actor_authority: str,
        approval_event_ref: str,
        source_activation_ref: str | None = None,
        supersedes_activation_id: str | None = None,
        activation_id: str | None = None,
    ) -> dict[str, Any]:
        rule_version_id = _assert_ascii(rule_version_id, "rule_version_id")
        organization_id = _assert_ascii(organization_id, "organization_id")
        activation_scope = _assert_ascii(activation_scope, "activation_scope")  # type: ignore[assignment]
        if activation_scope not in {"organization", "workspace"}:
            raise ValueError("activation_scope must be organization or workspace")
        workspace_id = _assert_optional_ascii(workspace_id, "workspace_id")
        if activation_scope == "workspace" and not workspace_id:
            raise ValueError("workspace-scoped governed rule activation requires workspace_id")
        if activation_scope == "organization":
            workspace_id = None
        effective_from = _assert_utc_timestamp(effective_from, "effective_from")
        effective_to = _assert_optional_utc_timestamp(effective_to, "effective_to")
        activated_by = _assert_ascii(activated_by, "activated_by")
        actor_authority = _assert_ascii(actor_authority, "actor_authority") if actor_authority else ""
        approval_event_ref = _assert_ascii(approval_event_ref, "approval_event_ref")
        source_activation_ref = _assert_optional_ascii(source_activation_ref, "source_activation_ref")
        supersedes_activation_id = _assert_optional_ascii(
            supersedes_activation_id,
            "supersedes_activation_id",
        )
        activation_id = _assert_ascii(activation_id or f"gra_{uuid4().hex}", "activation_id")
        scope_payload = {
            "organization_id": organization_id,
            "activation_scope": activation_scope,
            "workspace_id": workspace_id,
            "effective_from": effective_from,
            "effective_to": effective_to,
        }
        activation_scope_hash = compute_content_hash(canonical_json(scope_payload))
        created_at = _now_iso()
        activation = {
            "activation_id": activation_id,
            "rule_version_id": rule_version_id,
            "organization_id": organization_id,
            "activation_scope": activation_scope,
            "workspace_id": workspace_id,
            "activation_status": "ACTIVE",
            "effective_from": effective_from,
            "effective_to": effective_to,
            "activated_by": activated_by,
            "actor_authority": actor_authority,
            "approval_event_ref": approval_event_ref,
            "source_activation_ref": source_activation_ref,
            "activation_scope_hash": activation_scope_hash,
            "supersedes_activation_id": supersedes_activation_id,
            "created_at": created_at,
        }
        activation_hash = compute_content_hash(canonical_json(_activation_payload(activation)))
        event_id = f"grae_{uuid4().hex}"
        event = {
            "activation_event_id": event_id,
            "activation_stream_id": activation_id,
            "activation_id": activation_id,
            "event_type": "ACTIVATED",
            "from_state": None,
            "to_state": "ACTIVE",
            "effective_at": effective_from,
            "recorded_at": created_at,
            "event_sequence": 1,
            "actor": activated_by,
            "actor_authority": actor_authority,
            "reason": "initial activation",
            "prior_event_hash": None,
            "created_at": created_at,
        }
        event_hash = compute_content_hash(canonical_json(_event_payload(event)))
        with self._lock, transaction_scope(self._conn):
            existing = self._conn.execute(
                sql_query("""
                SELECT * FROM governed_rule_activation
                WHERE organization_id = ?
                  AND rule_version_id = ?
                  AND activation_scope = ?
                  AND COALESCE(workspace_id, '') = COALESCE(?, '')
                  AND effective_from = ?
                  AND COALESCE(effective_to, '') = COALESCE(?, '')
                  AND approval_event_ref = ?
                """, "\n                SELECT * FROM governed_rule_activation\n                WHERE organization_id = %s\n                  AND rule_version_id = %s\n                  AND activation_scope = %s\n                  AND COALESCE(workspace_id, '') = COALESCE(%s, '')\n                  AND effective_from = %s\n                  AND COALESCE(effective_to, '') = COALESCE(%s, '')\n                  AND approval_event_ref = %s\n                "),
                (
                    organization_id,
                    rule_version_id,
                    activation_scope,
                    workspace_id,
                    effective_from,
                    effective_to,
                    approval_event_ref,
                ),
            ).fetchone()
            if existing:
                existing_dict = dict(existing)
                if existing_dict["content_hash"] != activation_hash:
                    raise ValueError("conflicting governed rule activation idempotency payload")
                return existing_dict
            self._conn.execute(
                sql_query("""
                INSERT INTO governed_rule_activation (
                  activation_id, rule_version_id, organization_id, activation_scope,
                  workspace_id, activation_status, effective_from, effective_to,
                  activated_by, actor_authority, approval_event_ref,
                  source_activation_ref, activation_scope_hash,
                  supersedes_activation_id, content_hash, created_at
                )
                VALUES (?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, "\n                INSERT INTO governed_rule_activation (\n                  activation_id, rule_version_id, organization_id, activation_scope,\n                  workspace_id, activation_status, effective_from, effective_to,\n                  activated_by, actor_authority, approval_event_ref,\n                  source_activation_ref, activation_scope_hash,\n                  supersedes_activation_id, content_hash, created_at\n                )\n                VALUES (%s, %s, %s, %s, %s, 'ACTIVE', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                "),
                (
                    activation_id,
                    rule_version_id,
                    organization_id,
                    activation_scope,
                    workspace_id,
                    effective_from,
                    effective_to,
                    activated_by,
                    actor_authority,
                    approval_event_ref,
                    source_activation_ref,
                    activation_scope_hash,
                    supersedes_activation_id,
                    activation_hash,
                    created_at,
                ),
            )
            self._conn.execute(
                sql_query("""
                INSERT INTO governed_rule_activation_event (
                  activation_event_id, activation_stream_id, activation_id,
                  event_type, from_state, to_state, effective_at, recorded_at,
                  event_sequence, actor, actor_authority, reason,
                  prior_event_hash, content_hash, created_at
                )
                VALUES (?, ?, ?, 'ACTIVATED', NULL, 'ACTIVE', ?, ?, 1, ?, ?, ?, NULL, ?, ?)
                """, "\n                INSERT INTO governed_rule_activation_event (\n                  activation_event_id, activation_stream_id, activation_id,\n                  event_type, from_state, to_state, effective_at, recorded_at,\n                  event_sequence, actor, actor_authority, reason,\n                  prior_event_hash, content_hash, created_at\n                )\n                VALUES (%s, %s, %s, 'ACTIVATED', NULL, 'ACTIVE', %s, %s, 1, %s, %s, %s, NULL, %s, %s)\n                "),
                (
                    event_id,
                    activation_id,
                    activation_id,
                    effective_from,
                    created_at,
                    activated_by,
                    actor_authority,
                    "initial activation",
                    event_hash,
                    created_at,
                ),
            )
        return self.get_activation(activation_id)  # type: ignore[return-value]

    def get_activation(self, activation_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            sql_query("SELECT * FROM governed_rule_activation WHERE activation_id = ?", 'SELECT * FROM governed_rule_activation WHERE activation_id = %s'),
            (_assert_ascii(activation_id, "activation_id"),),
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

    def resolve_activation_state(self, *, activation_id: str, effective_at: str) -> str:
        activation = self.get_activation(activation_id)
        if activation is None:
            raise ValueError(f"governed rule activation not found: {activation_id}")
        if activation["activation_status"] != "ACTIVE":
            raise ValueError("activation row initial state must be ACTIVE")
        snapshot_time = _assert_utc_timestamp(effective_at, "effective_at")
        events = self.list_events(activation_id)
        if not events:
            raise ValueError("activation has no lifecycle events")
        self._verify_event_chain(activation, events)
        initial = events[0]
        if (
            initial["event_type"] != "ACTIVATED"
            or initial["from_state"] is not None
            or initial["to_state"] != "ACTIVE"
            or int(initial["event_sequence"]) != 1
            or initial["effective_at"] != activation["effective_from"]
        ):
            raise ValueError("activation initial event mismatch")

        selected = [event for event in events if event["effective_at"] <= snapshot_time]
        if not selected:
            return "PENDING"
        authority_events = [
            event for event in selected if event["event_type"] in AUTHORITY_CHANGING_EVENTS
        ]
        by_effective: dict[str, set[str]] = {}
        for event in authority_events:
            by_effective.setdefault(event["effective_at"], set()).add(event["event_type"])
        for event_types in by_effective.values():
            if len(event_types) > 1:
                raise ValueError("conflicting same-effective-time lifecycle events")
        latest = authority_events[-1] if authority_events else selected[0]
        state = latest["to_state"] or "ACTIVE"
        if state == "ACTIVE":
            if activation["effective_from"] > snapshot_time:
                return "PENDING"
            if activation["effective_to"] and snapshot_time >= activation["effective_to"]:
                return "EXPIRED"
            successor = self._successor_at(activation_id=activation_id, effective_at=snapshot_time)
            if successor:
                return "SUPERSEDED"
        return state

    def resolve_executable_activations(
        self,
        *,
        activation_ids: list[str],
        organization_id: str,
        workspace_id: str | None,
        effective_at: str,
    ) -> list[dict[str, Any]]:
        organization_id = _assert_ascii(organization_id, "organization_id")
        workspace_id = _assert_optional_ascii(workspace_id, "workspace_id")
        snapshot_time = _assert_utc_timestamp(effective_at, "effective_at")
        normalized_activation_ids = [_assert_ascii(item, "activation_id") for item in activation_ids]
        if len(normalized_activation_ids) != len(set(normalized_activation_ids)):
            raise ValueError("duplicate governed rule activation ids in resolver snapshot")
        executable: list[dict[str, Any]] = []
        for activation_id in normalized_activation_ids:
            activation = self.get_activation(activation_id)
            if activation is None:
                raise ValueError(f"governed rule activation not found: {activation_id}")
            if activation["organization_id"] != organization_id:
                raise ValueError("governed rule activation organization scope mismatch")
            if activation["activation_scope"] == "workspace" and activation["workspace_id"] != workspace_id:
                raise ValueError("governed rule activation workspace scope mismatch")
            state = self.resolve_activation_state(
                activation_id=activation["activation_id"],
                effective_at=snapshot_time,
            )
            if state != "ACTIVE":
                raise ValueError(f"governed rule activation is not executable: {state}")
            executable.append(activation)
        return executable

    def _successor_at(self, *, activation_id: str, effective_at: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            sql_query("""
            SELECT * FROM governed_rule_activation
            WHERE supersedes_activation_id = ? AND effective_from <= ?
            ORDER BY effective_from ASC, activation_id ASC
            LIMIT 1
            """, '\n            SELECT * FROM governed_rule_activation\n            WHERE supersedes_activation_id = %s AND effective_from <= %s\n            ORDER BY effective_from ASC, activation_id ASC\n            LIMIT 1\n            '),
            (activation_id, effective_at),
        ).fetchone()
        return dict(row) if row else None

    def _verify_event_chain(
        self,
        activation: dict[str, Any],
        events: list[dict[str, Any]],
    ) -> None:
        expected_prior = None
        stream_id = activation["activation_id"]
        for expected_sequence, event in enumerate(events, start=1):
            if event["activation_stream_id"] != stream_id:
                raise ValueError("activation stream mismatch")
            if int(event["event_sequence"]) != expected_sequence:
                raise ValueError("activation event sequence gap")
            if event["prior_event_hash"] != expected_prior:
                raise ValueError("activation event hash-chain mismatch")
            if compute_activation_event_hash(event) != event["content_hash"]:
                raise ValueError("activation event content hash mismatch")
            expected_prior = event["content_hash"]
