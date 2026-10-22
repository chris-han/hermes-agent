from __future__ import annotations

import hashlib
import json
import sqlite3
from storage import hosted_postgres as hosted
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


_EVENT_TYPES = frozenset(
    {
        "RESPONSIBILITY_BOUND",
        "DELEGATION_REQUESTED",
        "DELEGATION_ACCEPTED",
        "DELEGATION_STARTED",
        "WAIT_REGISTERED",
        "WAIT_SATISFIED",
        "RESULT_DELIVERED",
        "RESULT_ACCEPTED",
        "RESULT_REJECTED",
        "OBJECTION_RAISED",
        "OBJECTION_RESOLVED",
        "STEER_REQUESTED",
        "STEER_ACKNOWLEDGED",
        "STEER_APPLIED",
        "DELEGATION_STOPPED",
        "RESPONSIBILITY_CLOSED",
    }
)

_DELEGATION_EVENTS = frozenset(
    {
        "DELEGATION_ACCEPTED",
        "DELEGATION_STARTED",
        "RESULT_DELIVERED",
        "STEER_REQUESTED",
        "STEER_ACKNOWLEDGED",
        "STEER_APPLIED",
        "DELEGATION_STOPPED",
    }
)

_RESPONSIBILITY_EVENTS = frozenset(
    {
        "RESPONSIBILITY_BOUND",
        "WAIT_REGISTERED",
        "WAIT_SATISFIED",
        "RESULT_ACCEPTED",
        "RESULT_REJECTED",
        "OBJECTION_RAISED",
        "OBJECTION_RESOLVED",
        "RESPONSIBILITY_CLOSED",
    }
)


class CoordinationAssociationConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkRef:
    source_kind: str
    source_id: str
    source_version: str | None = None

    def __post_init__(self) -> None:
        if not self.source_kind.strip() or not self.source_id.strip():
            raise ValueError("work source_kind and source_id are required")

    @property
    def ref(self) -> str:
        base = f"{self.source_kind}:{self.source_id}"
        return f"{base}@{self.source_version}" if self.source_version else base


@dataclass(frozen=True)
class DelegationRef:
    delegation_id: str
    subagent_id: str
    owner_session_ref: str

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.delegation_id,
                self.subagent_id,
                self.owner_session_ref,
            )
        ):
            raise ValueError("delegation identifiers are required")


@dataclass(frozen=True)
class CoordinationContext:
    coordination_id: str
    workspace_id: str
    work_ref: WorkRef
    goal_ref: str
    goal_revision: str
    principal_ref: str
    opened_at: str
    closed_at: str | None = None


@dataclass(frozen=True)
class CoordinationReceipt:
    sequence: int
    receipt_id: str
    coordination_id: str
    workspace_id: str
    event_type: str
    event_ref: str
    responsibility_ref: str | None
    agent_identity_ref: str | None
    delegation_ref: DelegationRef | None
    basis_refs: tuple[str, ...]
    authority_ref: str | None
    prior_receipt_hash: str | None
    payload: dict[str, Any]
    produced_at: str
    receipt_hash: str


@dataclass(frozen=True)
class CoordinationProjection:
    coordination_id: str
    workspace_id: str
    work_ref: WorkRef
    goal_ref: str
    goal_revision: str
    principal_ref: str
    responsibility_refs: tuple[str, ...]
    delegation_ids: tuple[str, ...]
    delegation_to_responsibility: tuple[tuple[str, str | None], ...]
    pending_request_refs: tuple[str, ...]
    pending_wait_refs: tuple[str, ...]
    open_objection_refs: tuple[str, ...]
    delivered_result_refs: tuple[str, ...]
    accepted_result_refs: tuple[str, ...]
    closed_responsibility_refs: tuple[str, ...]
    steer_state_by_delegation: tuple[tuple[str, str], ...]
    latest_receipt_hash: str | None


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timezone-aware UTC timestamp required")
    return parsed.astimezone(timezone.utc)


def _hash(prefix: str, payload: Any, length: int = 32) -> str:
    digest = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:length]}"


def _ordered_unique(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        value = str(raw).strip()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return tuple(out)


def delegation_ref_from_hermes_attribution(
    attribution: Mapping[str, Any], *, owner_session_ref: str
) -> DelegationRef:
    """Project Hermes-native attribution into a Semantier association reference.

    This does not create responsibility, authority, or a new Hermes identity.
    The caller supplies the already-authoritative parent session reference.
    """

    delegation_id = str(attribution.get("delegation_id") or "").strip()
    subagent_id = str(attribution.get("subagent_id") or "").strip()
    return DelegationRef(
        delegation_id=delegation_id,
        subagent_id=subagent_id,
        owner_session_ref=owner_session_ref,
    )


class CoordinationAssociationStore:
    """Append-only associations across existing work/responsibility/execution owners.

    This store owns association receipts and their deterministic replay only.
    It cannot grant authority, create responsibility, assert domain effects, or
    substitute for Hermes execution state.
    """

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
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @staticmethod
    def ensure_schema(conn: sqlite3.Connection) -> None:
        if hosted.is_postgres(conn):
            hosted.require_tables(conn, 'semantier_coordination_contexts', 'semantier_coordination_receipts')
            return
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS semantier_coordination_contexts (
                coordination_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                work_source_kind TEXT NOT NULL,
                work_source_id TEXT NOT NULL,
                work_source_version TEXT,
                goal_ref TEXT NOT NULL,
                goal_revision TEXT NOT NULL,
                principal_ref TEXT NOT NULL,
                opened_at TEXT NOT NULL,
                closed_at TEXT,
                schema_version INTEGER NOT NULL DEFAULT 1,
                UNIQUE(
                    workspace_id,
                    work_source_kind,
                    work_source_id,
                    work_source_version,
                    goal_ref,
                    goal_revision
                )
            );

            CREATE TABLE IF NOT EXISTS semantier_coordination_receipts (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                receipt_id TEXT NOT NULL UNIQUE,
                coordination_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_ref TEXT NOT NULL,
                responsibility_ref TEXT,
                agent_identity_ref TEXT,
                delegation_id TEXT,
                subagent_id TEXT,
                owner_session_ref TEXT,
                basis_refs_json TEXT NOT NULL,
                authority_ref TEXT,
                prior_receipt_hash TEXT,
                payload_json TEXT NOT NULL,
                produced_at TEXT NOT NULL,
                receipt_hash TEXT NOT NULL,
                schema_version INTEGER NOT NULL DEFAULT 1,
                UNIQUE(coordination_id, event_type, event_ref),
                FOREIGN KEY(coordination_id)
                    REFERENCES semantier_coordination_contexts(coordination_id)
                    ON DELETE RESTRICT
            );

            CREATE INDEX IF NOT EXISTS idx_semantier_coordination_receipts_context
            ON semantier_coordination_receipts(coordination_id, sequence);
            """
        )

    @staticmethod
    def derive_context_id(
        *,
        workspace_id: str,
        work_ref: WorkRef,
        goal_ref: str,
        goal_revision: str,
    ) -> str:
        values = {
            "workspace_id": workspace_id.strip(),
            "work_ref": work_ref.ref,
            "goal_ref": goal_ref.strip(),
            "goal_revision": goal_revision.strip(),
        }
        if any(not value for value in values.values()):
            raise ValueError("coordination identity fields are required")
        return _hash("coord", values)

    def ensure_context(
        self,
        *,
        workspace_id: str,
        work_ref: WorkRef,
        goal_ref: str,
        goal_revision: str,
        principal_ref: str,
        opened_at: str,
    ) -> CoordinationContext:
        if not workspace_id.strip() or not principal_ref.strip():
            raise ValueError("workspace_id and principal_ref are required")
        _parse_utc(opened_at)
        coordination_id = self.derive_context_id(
            workspace_id=workspace_id,
            work_ref=work_ref,
            goal_ref=goal_ref,
            goal_revision=goal_revision,
        )
        values = (
            coordination_id,
            workspace_id,
            work_ref.source_kind,
            work_ref.source_id,
            work_ref.source_version,
            goal_ref,
            goal_revision,
            principal_ref,
            opened_at,
        )
        with self._connect() as conn:
            hosted.begin_write(conn)
            try:
                existing = conn.execute(
                    hosted.sql_query("SELECT * FROM semantier_coordination_contexts WHERE coordination_id=?", 'SELECT * FROM semantier_coordination_contexts WHERE coordination_id=%s'),
                    (coordination_id,),
                ).fetchone()
                if existing is None:
                    conn.execute(
                        hosted.sql_query("""
                        INSERT INTO semantier_coordination_contexts(
                            coordination_id,workspace_id,work_source_kind,work_source_id,
                            work_source_version,goal_ref,goal_revision,principal_ref,opened_at
                        ) VALUES (?,?,?,?,?,?,?,?,?)
                        """, """
                        INSERT INTO semantier_coordination_contexts(
                            coordination_id,workspace_id,work_source_kind,work_source_id,
                            work_source_version,goal_ref,goal_revision,principal_ref,opened_at
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """),
                        values,
                    )
                else:
                    candidate = self._context_payload_from_values(values)
                    observed = self._context_payload(existing)
                    if candidate != observed:
                        raise CoordinationAssociationConflict(
                            "coordination identity collision or conflicting retry"
                        )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        context = self.get_context(coordination_id)
        assert context is not None
        return context

    @staticmethod
    def _context_payload_from_values(values: tuple[Any, ...]) -> dict[str, Any]:
        return {
            "coordination_id": values[0],
            "workspace_id": values[1],
            "work_source_kind": values[2],
            "work_source_id": values[3],
            "work_source_version": values[4],
            "goal_ref": values[5],
            "goal_revision": values[6],
            "principal_ref": values[7],
            "opened_at": values[8],
        }

    @staticmethod
    def _context_payload(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "coordination_id": str(row["coordination_id"]),
            "workspace_id": str(row["workspace_id"]),
            "work_source_kind": str(row["work_source_kind"]),
            "work_source_id": str(row["work_source_id"]),
            "work_source_version": (
                str(row["work_source_version"])
                if row["work_source_version"] is not None
                else None
            ),
            "goal_ref": str(row["goal_ref"]),
            "goal_revision": str(row["goal_revision"]),
            "principal_ref": str(row["principal_ref"]),
            "opened_at": str(row["opened_at"]),
        }

    @staticmethod
    def _context(row: sqlite3.Row) -> CoordinationContext:
        return CoordinationContext(
            coordination_id=str(row["coordination_id"]),
            workspace_id=str(row["workspace_id"]),
            work_ref=WorkRef(
                source_kind=str(row["work_source_kind"]),
                source_id=str(row["work_source_id"]),
                source_version=(
                    str(row["work_source_version"])
                    if row["work_source_version"] is not None
                    else None
                ),
            ),
            goal_ref=str(row["goal_ref"]),
            goal_revision=str(row["goal_revision"]),
            principal_ref=str(row["principal_ref"]),
            opened_at=str(row["opened_at"]),
            closed_at=str(row["closed_at"]) if row["closed_at"] is not None else None,
        )

    def get_context(self, coordination_id: str) -> CoordinationContext | None:
        with self._connect() as conn:
            row = conn.execute(
                hosted.sql_query("SELECT * FROM semantier_coordination_contexts WHERE coordination_id=?", 'SELECT * FROM semantier_coordination_contexts WHERE coordination_id=%s'),
                (coordination_id,),
            ).fetchone()
        return self._context(row) if row is not None else None

    @staticmethod
    def _receipt_payload(
        *,
        receipt_id: str,
        coordination_id: str,
        workspace_id: str,
        event_type: str,
        event_ref: str,
        responsibility_ref: str | None,
        agent_identity_ref: str | None,
        delegation_ref: DelegationRef | None,
        basis_refs: tuple[str, ...],
        authority_ref: str | None,
        prior_receipt_hash: str | None,
        payload: dict[str, Any],
        produced_at: str,
    ) -> dict[str, Any]:
        return {
            "receipt_id": receipt_id,
            "coordination_id": coordination_id,
            "workspace_id": workspace_id,
            "event_type": event_type,
            "event_ref": event_ref,
            "responsibility_ref": responsibility_ref,
            "agent_identity_ref": agent_identity_ref,
            "delegation_id": delegation_ref.delegation_id if delegation_ref else None,
            "subagent_id": delegation_ref.subagent_id if delegation_ref else None,
            "owner_session_ref": delegation_ref.owner_session_ref if delegation_ref else None,
            "basis_refs": list(basis_refs),
            "authority_ref": authority_ref,
            "prior_receipt_hash": prior_receipt_hash,
            "payload": payload,
            "produced_at": produced_at,
        }

    @staticmethod
    def _receipt(row: sqlite3.Row) -> CoordinationReceipt:
        delegation_ref = None
        if row["delegation_id"] is not None:
            delegation_ref = DelegationRef(
                delegation_id=str(row["delegation_id"]),
                subagent_id=str(row["subagent_id"]),
                owner_session_ref=str(row["owner_session_ref"]),
            )
        return CoordinationReceipt(
            sequence=int(row["sequence"]),
            receipt_id=str(row["receipt_id"]),
            coordination_id=str(row["coordination_id"]),
            workspace_id=str(row["workspace_id"]),
            event_type=str(row["event_type"]),
            event_ref=str(row["event_ref"]),
            responsibility_ref=(
                str(row["responsibility_ref"])
                if row["responsibility_ref"] is not None
                else None
            ),
            agent_identity_ref=(
                str(row["agent_identity_ref"])
                if row["agent_identity_ref"] is not None
                else None
            ),
            delegation_ref=delegation_ref,
            basis_refs=tuple(json.loads(str(row["basis_refs_json"]))),
            authority_ref=(
                str(row["authority_ref"]) if row["authority_ref"] is not None else None
            ),
            prior_receipt_hash=(
                str(row["prior_receipt_hash"])
                if row["prior_receipt_hash"] is not None
                else None
            ),
            payload=dict(json.loads(str(row["payload_json"]))),
            produced_at=str(row["produced_at"]),
            receipt_hash=str(row["receipt_hash"]),
        )

    def list_receipts(self, coordination_id: str) -> tuple[CoordinationReceipt, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                hosted.sql_query("""
                SELECT * FROM semantier_coordination_receipts
                WHERE coordination_id=?
                ORDER BY sequence
                """, """
                SELECT * FROM semantier_coordination_receipts
                WHERE coordination_id=%s
                ORDER BY sequence
                """),
                (coordination_id,),
            ).fetchall()
        return tuple(self._receipt(row) for row in rows)

    def _has_open_objection(
        self, *, coordination_id: str, responsibility_ref: str
    ) -> bool:
        open_refs: set[str] = set()
        for receipt in self.list_receipts(coordination_id):
            if receipt.responsibility_ref != responsibility_ref:
                continue
            if receipt.event_type == "OBJECTION_RAISED":
                open_refs.add(receipt.event_ref)
            elif receipt.event_type == "OBJECTION_RESOLVED":
                objection_ref = str(receipt.payload.get("objection_ref") or "")
                open_refs.discard(objection_ref)
        return bool(open_refs)

    def _validate_event(
        self,
        *,
        coordination_id: str,
        event_type: str,
        responsibility_ref: str | None,
        agent_identity_ref: str | None,
        delegation_ref: DelegationRef | None,
        basis_refs: tuple[str, ...],
        authority_ref: str | None,
        payload: dict[str, Any],
    ) -> None:
        if event_type not in _EVENT_TYPES:
            raise ValueError(f"unsupported coordination event_type: {event_type}")
        if event_type in _DELEGATION_EVENTS and delegation_ref is None:
            raise ValueError(f"{event_type} requires delegation_ref")
        if event_type in _RESPONSIBILITY_EVENTS and not responsibility_ref:
            raise ValueError(f"{event_type} requires responsibility_ref")
        if event_type == "RESPONSIBILITY_BOUND":
            if not agent_identity_ref:
                raise ValueError("RESPONSIBILITY_BOUND requires agent_identity_ref")
            if not basis_refs:
                raise ValueError("RESPONSIBILITY_BOUND requires owner basis")
        if event_type == "OBJECTION_RAISED" and not basis_refs:
            raise ValueError("OBJECTION_RAISED requires evidence basis")
        if event_type == "OBJECTION_RESOLVED":
            if not basis_refs or not str(payload.get("objection_ref") or "").strip():
                raise ValueError("OBJECTION_RESOLVED requires objection_ref and review basis")
        if event_type in {"RESULT_ACCEPTED", "RESULT_REJECTED"} and not basis_refs:
            raise ValueError(f"{event_type} requires owner basis")
        if event_type == "STEER_APPLIED" and not basis_refs:
            raise ValueError("STEER_APPLIED requires applied goal/checkpoint basis")
        if event_type == "RESPONSIBILITY_CLOSED":
            if not authority_ref or not basis_refs:
                raise ValueError("RESPONSIBILITY_CLOSED requires owner authority and basis")
            if self._has_open_objection(
                coordination_id=coordination_id,
                responsibility_ref=str(responsibility_ref),
            ):
                raise ValueError("responsibility has open objection")

    def append_receipt(
        self,
        *,
        coordination_id: str,
        workspace_id: str,
        event_type: str,
        event_ref: str,
        produced_at: str,
        responsibility_ref: str | None = None,
        agent_identity_ref: str | None = None,
        delegation_ref: DelegationRef | None = None,
        basis_refs: tuple[str, ...] = (),
        authority_ref: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> CoordinationReceipt:
        context = self.get_context(coordination_id)
        if context is None:
            raise KeyError(coordination_id)
        if context.workspace_id != workspace_id:
            raise ValueError("coordination workspace mismatch")
        if not event_ref.strip():
            raise ValueError("event_ref is required")
        _parse_utc(produced_at)
        normalized_basis = _ordered_unique(basis_refs)
        normalized_payload = dict(payload or {})
        self._validate_event(
            coordination_id=coordination_id,
            event_type=event_type,
            responsibility_ref=responsibility_ref,
            agent_identity_ref=agent_identity_ref,
            delegation_ref=delegation_ref,
            basis_refs=normalized_basis,
            authority_ref=authority_ref,
            payload=normalized_payload,
        )
        receipt_id = _hash(
            "coordr",
            {
                "coordination_id": coordination_id,
                "event_type": event_type,
                "event_ref": event_ref,
            },
        )
        with self._connect() as conn:
            hosted.begin_write(conn)
            try:
                existing = conn.execute(
                    hosted.sql_query("SELECT * FROM semantier_coordination_receipts WHERE receipt_id=?", 'SELECT * FROM semantier_coordination_receipts WHERE receipt_id=%s'),
                    (receipt_id,),
                ).fetchone()
                if existing is not None:
                    observed = self._receipt(existing)
                    candidate = self._receipt_payload(
                        receipt_id=receipt_id,
                        coordination_id=coordination_id,
                        workspace_id=workspace_id,
                        event_type=event_type,
                        event_ref=event_ref,
                        responsibility_ref=responsibility_ref,
                        agent_identity_ref=agent_identity_ref,
                        delegation_ref=delegation_ref,
                        basis_refs=normalized_basis,
                        authority_ref=authority_ref,
                        prior_receipt_hash=observed.prior_receipt_hash,
                        payload=normalized_payload,
                        produced_at=produced_at,
                    )
                    if _canonical(candidate) != _canonical(
                        self._receipt_payload(
                            receipt_id=observed.receipt_id,
                            coordination_id=observed.coordination_id,
                            workspace_id=observed.workspace_id,
                            event_type=observed.event_type,
                            event_ref=observed.event_ref,
                            responsibility_ref=observed.responsibility_ref,
                            agent_identity_ref=observed.agent_identity_ref,
                            delegation_ref=observed.delegation_ref,
                            basis_refs=observed.basis_refs,
                            authority_ref=observed.authority_ref,
                            prior_receipt_hash=observed.prior_receipt_hash,
                            payload=observed.payload,
                            produced_at=observed.produced_at,
                        )
                    ):
                        raise CoordinationAssociationConflict(
                            "coordination receipt conflicting retry"
                        )
                    conn.execute("COMMIT")
                    return observed

                previous = conn.execute(
                    hosted.sql_query("""
                    SELECT receipt_hash FROM semantier_coordination_receipts
                    WHERE coordination_id=?
                    ORDER BY sequence DESC LIMIT 1
                    """, """
                    SELECT receipt_hash FROM semantier_coordination_receipts
                    WHERE coordination_id=%s
                    ORDER BY sequence DESC LIMIT 1
                    """),
                    (coordination_id,),
                ).fetchone()
                prior_hash = str(previous["receipt_hash"]) if previous is not None else None
                receipt_payload = self._receipt_payload(
                    receipt_id=receipt_id,
                    coordination_id=coordination_id,
                    workspace_id=workspace_id,
                    event_type=event_type,
                    event_ref=event_ref,
                    responsibility_ref=responsibility_ref,
                    agent_identity_ref=agent_identity_ref,
                    delegation_ref=delegation_ref,
                    basis_refs=normalized_basis,
                    authority_ref=authority_ref,
                    prior_receipt_hash=prior_hash,
                    payload=normalized_payload,
                    produced_at=produced_at,
                )
                receipt_hash = hashlib.sha256(
                    _canonical(receipt_payload).encode("utf-8")
                ).hexdigest()
                cursor = conn.execute(
                    hosted.sql_query("""
                    INSERT INTO semantier_coordination_receipts(
                        receipt_id,coordination_id,workspace_id,event_type,event_ref,
                        responsibility_ref,agent_identity_ref,delegation_id,subagent_id,
                        owner_session_ref,basis_refs_json,authority_ref,prior_receipt_hash,
                        payload_json,produced_at,receipt_hash
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, """
                    INSERT INTO semantier_coordination_receipts(
                        receipt_id,coordination_id,workspace_id,event_type,event_ref,
                        responsibility_ref,agent_identity_ref,delegation_id,subagent_id,
                        owner_session_ref,basis_refs_json,authority_ref,prior_receipt_hash,
                        payload_json,produced_at,receipt_hash
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING sequence"""),
                    (
                        receipt_id,
                        coordination_id,
                        workspace_id,
                        event_type,
                        event_ref,
                        responsibility_ref,
                        agent_identity_ref,
                        delegation_ref.delegation_id if delegation_ref else None,
                        delegation_ref.subagent_id if delegation_ref else None,
                        delegation_ref.owner_session_ref if delegation_ref else None,
                        _canonical(list(normalized_basis)),
                        authority_ref,
                        prior_hash,
                        _canonical(normalized_payload),
                        produced_at,
                        receipt_hash,
                    ),
                )
                sequence = int(cursor.fetchone()[0] if hosted.is_postgres(conn) else cursor.lastrowid)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return CoordinationReceipt(
            sequence=sequence,
            receipt_id=receipt_id,
            coordination_id=coordination_id,
            workspace_id=workspace_id,
            event_type=event_type,
            event_ref=event_ref,
            responsibility_ref=responsibility_ref,
            agent_identity_ref=agent_identity_ref,
            delegation_ref=delegation_ref,
            basis_refs=normalized_basis,
            authority_ref=authority_ref,
            prior_receipt_hash=prior_hash,
            payload=normalized_payload,
            produced_at=produced_at,
            receipt_hash=receipt_hash,
        )

    def verify_chain(self, coordination_id: str) -> bool:
        receipts = self.list_receipts(coordination_id)
        prior_hash: str | None = None
        for receipt in receipts:
            if receipt.prior_receipt_hash != prior_hash:
                raise CoordinationAssociationConflict(
                    "coordination receipt chain tamper detected"
                )
            payload = self._receipt_payload(
                receipt_id=receipt.receipt_id,
                coordination_id=receipt.coordination_id,
                workspace_id=receipt.workspace_id,
                event_type=receipt.event_type,
                event_ref=receipt.event_ref,
                responsibility_ref=receipt.responsibility_ref,
                agent_identity_ref=receipt.agent_identity_ref,
                delegation_ref=receipt.delegation_ref,
                basis_refs=receipt.basis_refs,
                authority_ref=receipt.authority_ref,
                prior_receipt_hash=receipt.prior_receipt_hash,
                payload=receipt.payload,
                produced_at=receipt.produced_at,
            )
            observed_hash = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
            if observed_hash != receipt.receipt_hash:
                raise CoordinationAssociationConflict(
                    "coordination receipt tamper detected"
                )
            prior_hash = receipt.receipt_hash
        return True

    def list_contexts(self, workspace_id: str) -> list[CoordinationContext]:
        """List only this workspace's explicit association identities."""
        with self._connect() as conn:
            rows = conn.execute(
                hosted.sql_query("SELECT coordination_id FROM semantier_coordination_contexts "
                "WHERE workspace_id=? ORDER BY opened_at DESC,coordination_id", 'SELECT coordination_id FROM semantier_coordination_contexts WHERE workspace_id=%s ORDER BY opened_at DESC,coordination_id'),
                (workspace_id,),
            ).fetchall()
        return [context for row in rows
                if (context := self.get_context(str(row["coordination_id"]))) is not None]

    def project(self, coordination_id: str) -> CoordinationProjection:
        context = self.get_context(coordination_id)
        if context is None:
            raise KeyError(coordination_id)
        self.verify_chain(coordination_id)
        receipts = self.list_receipts(coordination_id)

        responsibilities: list[str] = []
        delegations: dict[str, str | None] = {}
        pending_requests: list[str] = []
        pending_waits: list[str] = []
        open_objections: dict[str, str] = {}
        delivered_results: list[str] = []
        accepted_results: list[str] = []
        closed: list[str] = []
        steer_states: dict[str, str] = {}

        for receipt in receipts:
            responsibility_ref = receipt.responsibility_ref
            delegation_id = (
                receipt.delegation_ref.delegation_id
                if receipt.delegation_ref is not None
                else None
            )
            if receipt.event_type == "RESPONSIBILITY_BOUND" and responsibility_ref:
                if responsibility_ref not in responsibilities:
                    responsibilities.append(responsibility_ref)
            elif receipt.event_type == "DELEGATION_REQUESTED":
                pending_requests.append(receipt.event_ref)
            elif receipt.event_type in {"DELEGATION_ACCEPTED", "DELEGATION_STARTED"}:
                if delegation_id is not None:
                    delegations.setdefault(delegation_id, responsibility_ref)
                request_ref = str(receipt.payload.get("request_ref") or "")
                if request_ref in pending_requests:
                    pending_requests.remove(request_ref)
            elif receipt.event_type == "WAIT_REGISTERED":
                pending_waits.append(receipt.event_ref)
            elif receipt.event_type == "WAIT_SATISFIED":
                wait_ref = str(receipt.payload.get("wait_ref") or "")
                if wait_ref in pending_waits:
                    pending_waits.remove(wait_ref)
            elif receipt.event_type == "OBJECTION_RAISED" and responsibility_ref:
                open_objections[receipt.event_ref] = responsibility_ref
            elif receipt.event_type == "OBJECTION_RESOLVED":
                objection_ref = str(receipt.payload.get("objection_ref") or "")
                open_objections.pop(objection_ref, None)
            elif receipt.event_type == "RESULT_DELIVERED":
                delivered_results.append(receipt.event_ref)
                if delegation_id is not None:
                    delegations.setdefault(delegation_id, responsibility_ref)
            elif receipt.event_type == "RESULT_ACCEPTED":
                accepted_results.append(receipt.event_ref)
            elif receipt.event_type == "STEER_REQUESTED" and delegation_id is not None:
                steer_states[delegation_id] = "requested"
            elif receipt.event_type == "STEER_ACKNOWLEDGED" and delegation_id is not None:
                steer_states[delegation_id] = "acknowledged"
            elif receipt.event_type == "STEER_APPLIED" and delegation_id is not None:
                steer_states[delegation_id] = "applied"
            elif receipt.event_type == "RESPONSIBILITY_CLOSED" and responsibility_ref:
                if responsibility_ref not in closed:
                    closed.append(responsibility_ref)

        return CoordinationProjection(
            coordination_id=context.coordination_id,
            workspace_id=context.workspace_id,
            work_ref=context.work_ref,
            goal_ref=context.goal_ref,
            goal_revision=context.goal_revision,
            principal_ref=context.principal_ref,
            responsibility_refs=tuple(responsibilities),
            delegation_ids=tuple(delegations),
            delegation_to_responsibility=tuple(delegations.items()),
            pending_request_refs=tuple(pending_requests),
            pending_wait_refs=tuple(pending_waits),
            open_objection_refs=tuple(open_objections),
            delivered_result_refs=tuple(delivered_results),
            accepted_result_refs=tuple(accepted_results),
            closed_responsibility_refs=tuple(closed),
            steer_state_by_delegation=tuple(steer_states.items()),
            latest_receipt_hash=receipts[-1].receipt_hash if receipts else None,
        )
