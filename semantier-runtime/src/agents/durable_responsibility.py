from __future__ import annotations

import hashlib
import json
import sqlite3
from storage import hosted_postgres as hosted
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

READY = "READY"
RUNNING = "RUNNING"
WAIT_EXTERNAL = "WAIT_EXTERNAL"
WAIT_TIMER = "WAIT_TIMER"
REQUIRE_USER = "REQUIRE_USER"
RECONCILE = "RECONCILE"
TERMINAL = "TERMINAL"

DETERMINISTIC_CONTINUE = "DETERMINISTIC_CONTINUE"
AGENT_CONTINUE = "AGENT_CONTINUE"

EVENT = "EVENT"
TIMER = "TIMER"
USER = "USER"
RECONCILE_KIND = "RECONCILE"

_WAIT_STATES = {WAIT_EXTERNAL, WAIT_TIMER, REQUIRE_USER, RECONCILE}
_WAIT_KINDS = {EVENT, TIMER, USER, RECONCILE_KIND}
_LIFECYCLE_STATES = {READY, RUNNING, *_WAIT_STATES, TERMINAL}


class DurableResponsibilityError(RuntimeError):
    pass


class StaleResponsibilityClaim(DurableResponsibilityError):
    pass


@dataclass(frozen=True)
class ResponsibilityRecord:
    responsibility_id: str
    responsibility_kind: str
    workspace_id: str
    domain_ref: str
    workflow_id: str
    workflow_instance_id: str
    lifecycle_state: str
    continuation_disposition: str | None
    checkpoint_version: int
    checkpoint_hash: str
    current_wait_ref: str | None
    lease_owner: str | None
    lease_epoch: int
    lease_expires_at: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ResponsibilityCheckpoint:
    responsibility_id: str
    checkpoint_version: int
    lifecycle_state: str
    continuation_disposition: str | None
    domain_ref: str
    workflow_ref: str
    session_ref: str | None
    wait_ref: str | None
    accepted_event_cursor: str | None
    pending_effect_refs: tuple[str, ...]
    prior_checkpoint_hash: str | None
    checkpoint_hash: str
    produced_at: str


@dataclass(frozen=True)
class ResponsibilityClaim:
    responsibility_id: str
    owner_id: str
    lease_epoch: int
    lease_expires_at: str
    checkpoint_version: int


@dataclass(frozen=True)
class ResponsibilityWait:
    wait_id: str
    responsibility_id: str
    kind: str
    source_ref: str
    correlation_key: str
    expected_checkpoint_version: int
    state: str
    deadline_at: str | None
    created_at: str
    satisfied_at: str | None


@dataclass(frozen=True)
class WakeResult:
    accepted: bool
    reason: str
    wake_id: str | None = None


@dataclass(frozen=True)
class WaitSpec:
    kind: str
    source_ref: str
    correlation_key: str
    deadline_at: str | None = None


@dataclass(frozen=True)
class ContinuationDecision:
    disposition: str
    wait_kind: str | None = None
    wait_source_ref: str | None = None
    wait_correlation_key: str | None = None
    wait_deadline_at: str | None = None
    waits: tuple[WaitSpec, ...] = ()
    session_ref: str | None = None
    accepted_event_cursor: str | None = None
    pending_effect_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class DispatchResult:
    responsibility_id: str
    disposition: str
    checkpoint_version: int
    error: str | None = None


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timezone-aware UTC timestamp required")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _now_iso() -> str:
    return _iso(datetime.now(timezone.utc))


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_id(prefix: str, values: list[str], length: int = 32) -> str:
    raw = "|".join(values)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:length]}"


def _checkpoint_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


class DurableResponsibilityStore:
    """Semantier-owned durable responsibility continuity.

    This store owns work identity, bounded checkpoints, wait identity, wake
    admission and fenced worker ownership. Domain truth remains with the domain
    owner; native event/timer/effect/continuation stores remain authoritative for
    their respective artifacts.
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
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @staticmethod
    def ensure_schema(conn: sqlite3.Connection) -> None:
        if hosted.is_postgres(conn):
            hosted.require_tables(conn, 'semantier_responsibilities', 'semantier_responsibility_checkpoints', 'semantier_responsibility_events', 'semantier_responsibility_waits', 'semantier_responsibility_wakes')
            return
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS semantier_responsibilities (
                responsibility_id TEXT PRIMARY KEY,
                responsibility_kind TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                domain_ref TEXT NOT NULL,
                workflow_id TEXT NOT NULL,
                workflow_instance_id TEXT NOT NULL,
                lifecycle_state TEXT NOT NULL,
                continuation_disposition TEXT,
                checkpoint_version INTEGER NOT NULL DEFAULT 0,
                checkpoint_hash TEXT NOT NULL,
                current_wait_ref TEXT,
                lease_owner TEXT,
                lease_epoch INTEGER NOT NULL DEFAULT 0,
                lease_expires_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                schema_version INTEGER NOT NULL DEFAULT 1,
                UNIQUE(responsibility_kind, workspace_id, domain_ref, workflow_id, workflow_instance_id)
            );
            CREATE INDEX IF NOT EXISTS idx_semantier_responsibilities_ready
            ON semantier_responsibilities(lifecycle_state, lease_expires_at, updated_at);

            CREATE TABLE IF NOT EXISTS semantier_responsibility_checkpoints (
                responsibility_id TEXT NOT NULL,
                checkpoint_version INTEGER NOT NULL,
                lifecycle_state TEXT NOT NULL,
                continuation_disposition TEXT,
                domain_ref TEXT NOT NULL,
                workflow_ref TEXT NOT NULL,
                session_ref TEXT,
                wait_ref TEXT,
                accepted_event_cursor TEXT,
                pending_effect_refs_json TEXT NOT NULL,
                prior_checkpoint_hash TEXT,
                checkpoint_hash TEXT NOT NULL,
                produced_at TEXT NOT NULL,
                PRIMARY KEY(responsibility_id, checkpoint_version),
                FOREIGN KEY(responsibility_id) REFERENCES semantier_responsibilities(responsibility_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS semantier_responsibility_waits (
                wait_id TEXT PRIMARY KEY,
                responsibility_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                correlation_key TEXT NOT NULL,
                expected_checkpoint_version INTEGER NOT NULL,
                state TEXT NOT NULL,
                deadline_at TEXT,
                created_at TEXT NOT NULL,
                satisfied_at TEXT,
                FOREIGN KEY(responsibility_id) REFERENCES semantier_responsibilities(responsibility_id)
                    ON DELETE CASCADE,
                UNIQUE(responsibility_id, kind, source_ref, correlation_key, expected_checkpoint_version)
            );
            CREATE INDEX IF NOT EXISTS idx_semantier_responsibility_waits_pending
            ON semantier_responsibility_waits(responsibility_id, state, expected_checkpoint_version);

            CREATE TABLE IF NOT EXISTS semantier_responsibility_wakes (
                wake_id TEXT PRIMARY KEY,
                responsibility_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                correlation_key TEXT NOT NULL,
                dedupe_key TEXT NOT NULL UNIQUE,
                expected_checkpoint_version INTEGER NOT NULL,
                observed_at TEXT NOT NULL,
                accepted INTEGER NOT NULL,
                reason TEXT NOT NULL,
                FOREIGN KEY(responsibility_id) REFERENCES semantier_responsibilities(responsibility_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS semantier_responsibility_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                responsibility_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(responsibility_id) REFERENCES semantier_responsibilities(responsibility_id)
                    ON DELETE CASCADE
            );
            """
        )

    @staticmethod
    def derive_id(
        *,
        responsibility_kind: str,
        workspace_id: str,
        domain_ref: str,
        workflow_id: str,
        workflow_instance_id: str,
    ) -> str:
        values = [
            responsibility_kind.strip(),
            workspace_id.strip(),
            domain_ref.strip(),
            workflow_id.strip(),
            workflow_instance_id.strip(),
        ]
        if any(not value for value in values):
            raise ValueError("responsibility identity fields are required")
        return _hash_id("resp", values)

    def _event(
        self,
        conn: sqlite3.Connection,
        responsibility_id: str,
        event_type: str,
        payload: dict[str, Any],
        created_at: str,
    ) -> None:
        conn.execute(
            hosted.sql_query("""
            INSERT INTO semantier_responsibility_events(
                responsibility_id,event_type,payload_json,created_at
            ) VALUES (?,?,?,?)
            """, """
            INSERT INTO semantier_responsibility_events(
                responsibility_id,event_type,payload_json,created_at
            ) VALUES (%s,%s,%s,%s)
            """),
            (responsibility_id, event_type, _canonical(payload), created_at),
        )

    @staticmethod
    def _record(row: sqlite3.Row) -> ResponsibilityRecord:
        return ResponsibilityRecord(
            responsibility_id=str(row["responsibility_id"]),
            responsibility_kind=str(row["responsibility_kind"]),
            workspace_id=str(row["workspace_id"]),
            domain_ref=str(row["domain_ref"]),
            workflow_id=str(row["workflow_id"]),
            workflow_instance_id=str(row["workflow_instance_id"]),
            lifecycle_state=str(row["lifecycle_state"]),
            continuation_disposition=(
                str(row["continuation_disposition"])
                if row["continuation_disposition"] is not None
                else None
            ),
            checkpoint_version=int(row["checkpoint_version"]),
            checkpoint_hash=str(row["checkpoint_hash"]),
            current_wait_ref=(
                str(row["current_wait_ref"]) if row["current_wait_ref"] is not None else None
            ),
            lease_owner=str(row["lease_owner"]) if row["lease_owner"] is not None else None,
            lease_epoch=int(row["lease_epoch"]),
            lease_expires_at=(
                str(row["lease_expires_at"]) if row["lease_expires_at"] is not None else None
            ),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _wait(row: sqlite3.Row) -> ResponsibilityWait:
        return ResponsibilityWait(
            wait_id=str(row["wait_id"]),
            responsibility_id=str(row["responsibility_id"]),
            kind=str(row["kind"]),
            source_ref=str(row["source_ref"]),
            correlation_key=str(row["correlation_key"]),
            expected_checkpoint_version=int(row["expected_checkpoint_version"]),
            state=str(row["state"]),
            deadline_at=str(row["deadline_at"]) if row["deadline_at"] is not None else None,
            created_at=str(row["created_at"]),
            satisfied_at=(
                str(row["satisfied_at"]) if row["satisfied_at"] is not None else None
            ),
        )

    def ensure(
        self,
        *,
        responsibility_kind: str,
        workspace_id: str,
        domain_ref: str,
        workflow_id: str,
        workflow_instance_id: str,
        session_ref: str | None = None,
        now: str | None = None,
    ) -> ResponsibilityRecord:
        produced_at = now or _now_iso()
        _parse_utc(produced_at)
        responsibility_id = self.derive_id(
            responsibility_kind=responsibility_kind,
            workspace_id=workspace_id,
            domain_ref=domain_ref,
            workflow_id=workflow_id,
            workflow_instance_id=workflow_instance_id,
        )
        payload = {
            "responsibility_id": responsibility_id,
            "checkpoint_version": 0,
            "lifecycle_state": READY,
            "continuation_disposition": DETERMINISTIC_CONTINUE,
            "domain_ref": domain_ref,
            "workflow_ref": workflow_id,
            "session_ref": session_ref,
            "wait_ref": None,
            "accepted_event_cursor": None,
            "pending_effect_refs": [],
            "prior_checkpoint_hash": None,
            "produced_at": produced_at,
        }
        cp_hash = _checkpoint_hash(payload)
        with self._connect() as conn:
            hosted.begin_write(conn)
            try:
                inserted = conn.execute(
                    hosted.sql_query("""
                    INSERT OR IGNORE INTO semantier_responsibilities(
                        responsibility_id,responsibility_kind,workspace_id,domain_ref,
                        workflow_id,workflow_instance_id,lifecycle_state,
                        continuation_disposition,checkpoint_version,checkpoint_hash,
                        current_wait_ref,lease_owner,lease_epoch,lease_expires_at,
                        created_at,updated_at
                    ) VALUES (?,?,?,?,?,?,'READY',?,0,?,NULL,NULL,0,NULL,?,?)
                    """, """
                    INSERT INTO semantier_responsibilities(
                        responsibility_id,responsibility_kind,workspace_id,domain_ref,
                        workflow_id,workflow_instance_id,lifecycle_state,
                        continuation_disposition,checkpoint_version,checkpoint_hash,
                        current_wait_ref,lease_owner,lease_epoch,lease_expires_at,
                        created_at,updated_at
                    ) VALUES (%s,%s,%s,%s,%s,%s,'READY',%s,0,%s,NULL,NULL,0,NULL,%s,%s) ON CONFLICT DO NOTHING"""),
                    (
                        responsibility_id,
                        responsibility_kind,
                        workspace_id,
                        domain_ref,
                        workflow_id,
                        workflow_instance_id,
                        DETERMINISTIC_CONTINUE,
                        cp_hash,
                        produced_at,
                        produced_at,
                    ),
                )
                conn.execute(
                    hosted.sql_query("""
                    INSERT OR IGNORE INTO semantier_responsibility_checkpoints(
                        responsibility_id,checkpoint_version,lifecycle_state,
                        continuation_disposition,domain_ref,workflow_ref,session_ref,
                        wait_ref,accepted_event_cursor,pending_effect_refs_json,
                        prior_checkpoint_hash,checkpoint_hash,produced_at
                    ) VALUES (?,0,'READY',?,?,?,?,NULL,NULL,'[]',NULL,?,?)
                    """, """
                    INSERT INTO semantier_responsibility_checkpoints(
                        responsibility_id,checkpoint_version,lifecycle_state,
                        continuation_disposition,domain_ref,workflow_ref,session_ref,
                        wait_ref,accepted_event_cursor,pending_effect_refs_json,
                        prior_checkpoint_hash,checkpoint_hash,produced_at
                    ) VALUES (%s,0,'READY',%s,%s,%s,%s,NULL,NULL,'[]',NULL,%s,%s) ON CONFLICT DO NOTHING"""),
                    (
                        responsibility_id,
                        DETERMINISTIC_CONTINUE,
                        domain_ref,
                        workflow_id,
                        session_ref,
                        cp_hash,
                        produced_at,
                    ),
                )
                if inserted.rowcount == 1:
                    self._event(
                        conn,
                        responsibility_id,
                        "RESPONSIBILITY_ENSURED",
                        {"checkpoint_version": 0},
                        produced_at,
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        record = self.get(responsibility_id)
        assert record is not None
        return record

    def bind_execution(self, responsibility_id: str):
        """Publish the real workflow owner's execution identity without granting authority.

        Retries bind the immutable creation checkpoint, including after a crash.
        Ordinary autonomous workers acquire no fabricated workflow responsibility.
        """
        from agents.coordination_association import CoordinationAssociationStore, WorkRef

        record = self.get(responsibility_id)
        if record is None:
            raise KeyError(responsibility_id)
        with self._connect() as conn:
            basis = conn.execute(
                hosted.sql_query("SELECT checkpoint_hash, produced_at FROM semantier_responsibility_checkpoints "
                "WHERE responsibility_id=? AND checkpoint_version=0", 'SELECT checkpoint_hash, produced_at FROM semantier_responsibility_checkpoints WHERE responsibility_id=%s AND checkpoint_version=0'), (responsibility_id,),
            ).fetchone()
        if basis is None:
            raise DurableResponsibilityError("creation basis unavailable")
        associations = CoordinationAssociationStore(self.db_path)
        context = associations.ensure_context(
            workspace_id=record.workspace_id,
            work_ref=WorkRef("workflow_instance", record.workflow_instance_id, record.workflow_id),
            goal_ref=record.domain_ref, goal_revision=basis["checkpoint_hash"],
            principal_ref=f"workflow:{record.workflow_id}", opened_at=basis["produced_at"],
        )
        associations.append_receipt(
            coordination_id=context.coordination_id, workspace_id=record.workspace_id,
            event_type="RESPONSIBILITY_BOUND", event_ref=record.responsibility_id,
            responsibility_ref=record.responsibility_id,
            agent_identity_ref=f"workflow:{record.workflow_id}",
            basis_refs=(basis["checkpoint_hash"],), produced_at=basis["produced_at"],
        )
        return context

    def get(self, responsibility_id: str) -> ResponsibilityRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                hosted.sql_query("SELECT * FROM semantier_responsibilities WHERE responsibility_id=?", 'SELECT * FROM semantier_responsibilities WHERE responsibility_id=%s'),
                (responsibility_id,),
            ).fetchone()
        return self._record(row) if row else None

    def latest_checkpoint(self, responsibility_id: str) -> ResponsibilityCheckpoint:
        with self._connect() as conn:
            row = conn.execute(
                hosted.sql_query("""
                SELECT * FROM semantier_responsibility_checkpoints
                WHERE responsibility_id=?
                ORDER BY checkpoint_version DESC LIMIT 1
                """, """
                SELECT * FROM semantier_responsibility_checkpoints
                WHERE responsibility_id=%s
                ORDER BY checkpoint_version DESC LIMIT 1
                """),
                (responsibility_id,),
            ).fetchone()
        if row is None:
            raise KeyError(responsibility_id)
        return ResponsibilityCheckpoint(
            responsibility_id=str(row["responsibility_id"]),
            checkpoint_version=int(row["checkpoint_version"]),
            lifecycle_state=str(row["lifecycle_state"]),
            continuation_disposition=(
                str(row["continuation_disposition"])
                if row["continuation_disposition"] is not None
                else None
            ),
            domain_ref=str(row["domain_ref"]),
            workflow_ref=str(row["workflow_ref"]),
            session_ref=str(row["session_ref"]) if row["session_ref"] is not None else None,
            wait_ref=str(row["wait_ref"]) if row["wait_ref"] is not None else None,
            accepted_event_cursor=(
                str(row["accepted_event_cursor"])
                if row["accepted_event_cursor"] is not None
                else None
            ),
            pending_effect_refs=tuple(json.loads(str(row["pending_effect_refs_json"]))),
            prior_checkpoint_hash=(
                str(row["prior_checkpoint_hash"])
                if row["prior_checkpoint_hash"] is not None
                else None
            ),
            checkpoint_hash=str(row["checkpoint_hash"]),
            produced_at=str(row["produced_at"]),
        )

    def claim(
        self,
        *,
        responsibility_id: str,
        owner_id: str,
        now: str | None = None,
        lease_seconds: int = 180,
    ) -> ResponsibilityClaim:
        if not owner_id.strip():
            raise ValueError("owner_id is required")
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be >= 1")
        claimed_at = now or _now_iso()
        claimed_dt = _parse_utc(claimed_at)
        expires_at = _iso(claimed_dt + timedelta(seconds=lease_seconds))
        with self._connect() as conn:
            hosted.begin_write(conn)
            try:
                row = conn.execute(
                    hosted.sql_query("SELECT * FROM semantier_responsibilities WHERE responsibility_id=?", 'SELECT * FROM semantier_responsibilities WHERE responsibility_id=%s'),
                    (responsibility_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(responsibility_id)
                state = str(row["lifecycle_state"])
                if state == TERMINAL:
                    raise DurableResponsibilityError("terminal responsibility cannot be claimed")
                if state == RUNNING:
                    lease_expires_at = row["lease_expires_at"]
                    if lease_expires_at and _parse_utc(str(lease_expires_at)) > claimed_dt:
                        raise DurableResponsibilityError("responsibility already claimed")
                elif state != READY:
                    raise DurableResponsibilityError(
                        f"responsibility not ready: {state}"
                    )
                epoch = int(row["lease_epoch"]) + 1
                conn.execute(
                    hosted.sql_query("""
                    UPDATE semantier_responsibilities
                    SET lifecycle_state='RUNNING', lease_owner=?, lease_epoch=?,
                        lease_expires_at=?, updated_at=?
                    WHERE responsibility_id=?
                    """, """
                    UPDATE semantier_responsibilities
                    SET lifecycle_state='RUNNING', lease_owner=%s, lease_epoch=%s,
                        lease_expires_at=%s, updated_at=%s
                    WHERE responsibility_id=%s
                    """),
                    (owner_id, epoch, expires_at, claimed_at, responsibility_id),
                )
                self._event(
                    conn,
                    responsibility_id,
                    "RESPONSIBILITY_CLAIMED",
                    {"owner_id": owner_id, "lease_epoch": epoch},
                    claimed_at,
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return ResponsibilityClaim(
            responsibility_id=responsibility_id,
            owner_id=owner_id,
            lease_epoch=epoch,
            lease_expires_at=expires_at,
            checkpoint_version=int(row["checkpoint_version"]),
        )

    def assert_claim(
        self,
        claim: ResponsibilityClaim,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        owns_conn = conn is None
        db = conn or self._connect()
        try:
            row = db.execute(
                hosted.sql_query("""
                SELECT lifecycle_state,lease_owner,lease_epoch,checkpoint_version
                FROM semantier_responsibilities WHERE responsibility_id=?
                """, """
                SELECT lifecycle_state,lease_owner,lease_epoch,checkpoint_version
                FROM semantier_responsibilities WHERE responsibility_id=%s
                """),
                (claim.responsibility_id,),
            ).fetchone()
            if (
                row is None
                or str(row["lifecycle_state"]) != RUNNING
                or str(row["lease_owner"] or "") != claim.owner_id
                or int(row["lease_epoch"]) != claim.lease_epoch
                or int(row["checkpoint_version"]) != claim.checkpoint_version
            ):
                raise StaleResponsibilityClaim("responsibility claim is stale")
        finally:
            if owns_conn:
                db.close()

    def commit(
        self,
        claim: ResponsibilityClaim,
        *,
        continuation_disposition: str,
        lifecycle_state: str,
        session_ref: str | None = None,
        wait_ref: str | None = None,
        accepted_event_cursor: str | None = None,
        pending_effect_refs: tuple[str, ...] | list[str] = (),
        now: str | None = None,
    ) -> ResponsibilityRecord:
        if lifecycle_state not in _LIFECYCLE_STATES - {RUNNING}:
            raise ValueError("invalid lifecycle_state")
        produced_at = now or _now_iso()
        _parse_utc(produced_at)
        with self._connect() as conn:
            hosted.begin_write(conn)
            try:
                self.assert_claim(claim, conn=conn)
                current = conn.execute(
                    hosted.sql_query("SELECT * FROM semantier_responsibilities WHERE responsibility_id=?", 'SELECT * FROM semantier_responsibilities WHERE responsibility_id=%s'),
                    (claim.responsibility_id,),
                ).fetchone()
                next_version = int(current["checkpoint_version"]) + 1
                prior_hash = str(current["checkpoint_hash"])
                payload = {
                    "responsibility_id": claim.responsibility_id,
                    "checkpoint_version": next_version,
                    "lifecycle_state": lifecycle_state,
                    "continuation_disposition": continuation_disposition,
                    "domain_ref": str(current["domain_ref"]),
                    "workflow_ref": str(current["workflow_id"]),
                    "session_ref": session_ref,
                    "wait_ref": wait_ref,
                    "accepted_event_cursor": accepted_event_cursor,
                    "pending_effect_refs": sorted(set(pending_effect_refs)),
                    "prior_checkpoint_hash": prior_hash,
                    "produced_at": produced_at,
                }
                cp_hash = _checkpoint_hash(payload)
                conn.execute(
                    hosted.sql_query("""
                    INSERT INTO semantier_responsibility_checkpoints(
                        responsibility_id,checkpoint_version,lifecycle_state,
                        continuation_disposition,domain_ref,workflow_ref,session_ref,
                        wait_ref,accepted_event_cursor,pending_effect_refs_json,
                        prior_checkpoint_hash,checkpoint_hash,produced_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, """
                    INSERT INTO semantier_responsibility_checkpoints(
                        responsibility_id,checkpoint_version,lifecycle_state,
                        continuation_disposition,domain_ref,workflow_ref,session_ref,
                        wait_ref,accepted_event_cursor,pending_effect_refs_json,
                        prior_checkpoint_hash,checkpoint_hash,produced_at
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """),
                    (
                        claim.responsibility_id,
                        next_version,
                        lifecycle_state,
                        continuation_disposition,
                        current["domain_ref"],
                        current["workflow_id"],
                        session_ref,
                        wait_ref,
                        accepted_event_cursor,
                        _canonical(sorted(set(pending_effect_refs))),
                        prior_hash,
                        cp_hash,
                        produced_at,
                    ),
                )
                conn.execute(
                    hosted.sql_query("""
                    UPDATE semantier_responsibilities
                    SET lifecycle_state=?, continuation_disposition=?,
                        checkpoint_version=?, checkpoint_hash=?, current_wait_ref=?,
                        lease_owner=NULL, lease_expires_at=NULL, updated_at=?
                    WHERE responsibility_id=?
                    """, """
                    UPDATE semantier_responsibilities
                    SET lifecycle_state=%s, continuation_disposition=%s,
                        checkpoint_version=%s, checkpoint_hash=%s, current_wait_ref=%s,
                        lease_owner=NULL, lease_expires_at=NULL, updated_at=%s
                    WHERE responsibility_id=%s
                    """),
                    (
                        lifecycle_state,
                        continuation_disposition,
                        next_version,
                        cp_hash,
                        wait_ref,
                        produced_at,
                        claim.responsibility_id,
                    ),
                )
                self._event(
                    conn,
                    claim.responsibility_id,
                    "RESPONSIBILITY_CHECKPOINT_COMMITTED",
                    {
                        "checkpoint_version": next_version,
                        "lifecycle_state": lifecycle_state,
                        "continuation_disposition": continuation_disposition,
                    },
                    produced_at,
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        record = self.get(claim.responsibility_id)
        assert record is not None
        return record

    def register_wait(
        self,
        claim: ResponsibilityClaim,
        *,
        kind: str,
        source_ref: str,
        correlation_key: str,
        lifecycle_state: str,
        continuation_disposition: str,
        deadline_at: str | None = None,
        session_ref: str | None = None,
        now: str | None = None,
    ) -> ResponsibilityWait:
        waits = self.register_waits(
            claim,
            waits=(
                WaitSpec(
                    kind=kind,
                    source_ref=source_ref,
                    correlation_key=correlation_key,
                    deadline_at=deadline_at,
                ),
            ),
            lifecycle_state=lifecycle_state,
            continuation_disposition=continuation_disposition,
            session_ref=session_ref,
            now=now,
        )
        return waits[0]

    def register_waits(
        self,
        claim: ResponsibilityClaim,
        *,
        waits: tuple[WaitSpec, ...] | list[WaitSpec],
        lifecycle_state: str,
        continuation_disposition: str,
        session_ref: str | None = None,
        now: str | None = None,
    ) -> list[ResponsibilityWait]:
        if lifecycle_state not in _WAIT_STATES:
            raise ValueError("wait lifecycle_state required")
        specs = tuple(waits)
        if not specs:
            raise ValueError("at least one wait is required")
        for spec in specs:
            if spec.kind not in _WAIT_KINDS:
                raise ValueError("invalid wait kind")
            if not spec.source_ref.strip() or not spec.correlation_key.strip():
                raise ValueError("wait source and correlation are required")
            if spec.deadline_at is not None:
                _parse_utc(spec.deadline_at)

        created_at = now or _now_iso()
        _parse_utc(created_at)
        next_version = claim.checkpoint_version + 1
        wait_ids = [
            _hash_id(
                "wait",
                [
                    claim.responsibility_id,
                    spec.kind,
                    spec.source_ref,
                    spec.correlation_key,
                    str(next_version),
                ],
            )
            for spec in specs
        ]
        if len(set(wait_ids)) != len(wait_ids):
            raise ValueError("duplicate wait identity")

        with self._connect() as conn:
            hosted.begin_write(conn)
            try:
                self.assert_claim(claim, conn=conn)
                current = conn.execute(
                    hosted.sql_query("SELECT * FROM semantier_responsibilities WHERE responsibility_id=?", 'SELECT * FROM semantier_responsibilities WHERE responsibility_id=%s'),
                    (claim.responsibility_id,),
                ).fetchone()
                if current is None:
                    raise KeyError(claim.responsibility_id)
                prior_hash = str(current["checkpoint_hash"])

                for wait_id, spec in zip(wait_ids, specs, strict=True):
                    conn.execute(
                        hosted.sql_query("""
                        INSERT OR IGNORE INTO semantier_responsibility_waits(
                            wait_id,responsibility_id,kind,source_ref,correlation_key,
                            expected_checkpoint_version,state,deadline_at,created_at,satisfied_at
                        ) VALUES (?,?,?,?,?,?,'PENDING',?,?,NULL)
                        """, """
                        INSERT INTO semantier_responsibility_waits(
                            wait_id,responsibility_id,kind,source_ref,correlation_key,
                            expected_checkpoint_version,state,deadline_at,created_at,satisfied_at
                        ) VALUES (%s,%s,%s,%s,%s,%s,'PENDING',%s,%s,NULL) ON CONFLICT DO NOTHING"""),
                        (
                            wait_id,
                            claim.responsibility_id,
                            spec.kind,
                            spec.source_ref,
                            spec.correlation_key,
                            next_version,
                            spec.deadline_at,
                            created_at,
                        ),
                    )

                payload = {
                    "responsibility_id": claim.responsibility_id,
                    "checkpoint_version": next_version,
                    "lifecycle_state": lifecycle_state,
                    "continuation_disposition": continuation_disposition,
                    "domain_ref": str(current["domain_ref"]),
                    "workflow_ref": str(current["workflow_id"]),
                    "session_ref": session_ref,
                    "wait_ref": wait_ids[0],
                    "accepted_event_cursor": None,
                    "pending_effect_refs": [],
                    "prior_checkpoint_hash": prior_hash,
                    "produced_at": created_at,
                }
                cp_hash = _checkpoint_hash(payload)
                conn.execute(
                    hosted.sql_query("""
                    INSERT INTO semantier_responsibility_checkpoints(
                        responsibility_id,checkpoint_version,lifecycle_state,
                        continuation_disposition,domain_ref,workflow_ref,session_ref,
                        wait_ref,accepted_event_cursor,pending_effect_refs_json,
                        prior_checkpoint_hash,checkpoint_hash,produced_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, """
                    INSERT INTO semantier_responsibility_checkpoints(
                        responsibility_id,checkpoint_version,lifecycle_state,
                        continuation_disposition,domain_ref,workflow_ref,session_ref,
                        wait_ref,accepted_event_cursor,pending_effect_refs_json,
                        prior_checkpoint_hash,checkpoint_hash,produced_at
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """),
                    (
                        claim.responsibility_id,
                        next_version,
                        lifecycle_state,
                        continuation_disposition,
                        current["domain_ref"],
                        current["workflow_id"],
                        session_ref,
                        wait_ids[0],
                        None,
                        "[]",
                        prior_hash,
                        cp_hash,
                        created_at,
                    ),
                )
                conn.execute(
                    hosted.sql_query("""
                    UPDATE semantier_responsibilities
                    SET lifecycle_state=?, continuation_disposition=?,
                        checkpoint_version=?, checkpoint_hash=?, current_wait_ref=?,
                        lease_owner=NULL, lease_expires_at=NULL, updated_at=?
                    WHERE responsibility_id=?
                    """, """
                    UPDATE semantier_responsibilities
                    SET lifecycle_state=%s, continuation_disposition=%s,
                        checkpoint_version=%s, checkpoint_hash=%s, current_wait_ref=%s,
                        lease_owner=NULL, lease_expires_at=NULL, updated_at=%s
                    WHERE responsibility_id=%s
                    """),
                    (
                        lifecycle_state,
                        continuation_disposition,
                        next_version,
                        cp_hash,
                        wait_ids[0],
                        created_at,
                        claim.responsibility_id,
                    ),
                )
                self._event(
                    conn,
                    claim.responsibility_id,
                    "RESPONSIBILITY_CHECKPOINT_COMMITTED",
                    {
                        "checkpoint_version": next_version,
                        "lifecycle_state": lifecycle_state,
                        "continuation_disposition": continuation_disposition,
                    },
                    created_at,
                )

                placeholders = ",".join(hosted.sql_query("?", '%s') for _ in wait_ids)
                rows = conn.execute(
                    f"""
                    SELECT * FROM semantier_responsibility_waits
                    WHERE wait_id IN ({placeholders})
                    ORDER BY wait_id
                    """,
                    wait_ids,
                ).fetchall()
                conn.execute("COMMIT")
            except Exception:
                if hosted.in_transaction(conn):
                    conn.execute("ROLLBACK")
                raise
        return [self._wait(row) for row in rows]

    def pending_waits(self, responsibility_id: str) -> list[ResponsibilityWait]:
        with self._connect() as conn:
            rows = conn.execute(
                hosted.sql_query("""
                SELECT * FROM semantier_responsibility_waits
                WHERE responsibility_id=? AND state='PENDING'
                ORDER BY created_at,wait_id
                """, """
                SELECT * FROM semantier_responsibility_waits
                WHERE responsibility_id=%s AND state='PENDING'
                ORDER BY created_at,wait_id
                """),
                (responsibility_id,),
            ).fetchall()
        return [self._wait(row) for row in rows]


    @staticmethod
    def _wake_receipt(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "schema_version": "responsibility_wake_receipt.v1",
            "wake_id": str(row["wake_id"]),
            "responsibility_id": str(row["responsibility_id"]),
            "kind": str(row["kind"]),
            "source_ref": str(row["source_ref"]),
            "correlation_key": str(row["correlation_key"]),
            "checkpoint_version": int(row["expected_checkpoint_version"]),
            "checkpoint_hash": str(row["checkpoint_hash"]),
            "workspace_id": str(row["workspace_id"]),
            "produced_at": str(row["observed_at"]),
            "accepted": bool(row["accepted"]),
            "reason": str(row["reason"]),
            "external_effect": "UNCONFIRMED",
        }

    def decision_snapshot(
        self, responsibility_id: str, *, workspace_id: str, actor_ref: str,
    ) -> dict[str, Any]:
        """One consistent owner read; domain authority/effects are never inferred."""
        with self._connect() as conn:
            conn.execute("BEGIN")
            item = conn.execute(
                hosted.sql_query("SELECT * FROM semantier_responsibilities WHERE responsibility_id=?", 'SELECT * FROM semantier_responsibilities WHERE responsibility_id=%s'),
                (responsibility_id,),
            ).fetchone()
            if item is None:
                raise KeyError(responsibility_id)
            if item["workspace_id"] != workspace_id:
                raise PermissionError("Outside responsibility scope")
            checkpoint = conn.execute(
                hosted.sql_query("SELECT * FROM semantier_responsibility_checkpoints "
                "WHERE responsibility_id=? AND checkpoint_version=?", 'SELECT * FROM semantier_responsibility_checkpoints WHERE responsibility_id=%s AND checkpoint_version=%s'),
                (responsibility_id, item["checkpoint_version"]),
            ).fetchone()
            if checkpoint is None:
                raise DurableResponsibilityError("responsibility checkpoint missing")
            waits = conn.execute(
                hosted.sql_query("SELECT * FROM semantier_responsibility_waits WHERE responsibility_id=? "
                "ORDER BY created_at,wait_id", 'SELECT * FROM semantier_responsibility_waits WHERE responsibility_id=%s ORDER BY created_at,wait_id'), (responsibility_id,),
            ).fetchall()
            wakes = conn.execute(
                hosted.sql_query("SELECT w.*,c.checkpoint_hash,r.workspace_id FROM semantier_responsibility_wakes w "
                "JOIN semantier_responsibility_checkpoints c ON c.responsibility_id=w.responsibility_id "
                "AND c.checkpoint_version=w.expected_checkpoint_version "
                "JOIN semantier_responsibilities r ON r.responsibility_id=w.responsibility_id "
                "WHERE w.responsibility_id=? ORDER BY w.observed_at,w.wake_id", 'SELECT w.*,c.checkpoint_hash,r.workspace_id FROM semantier_responsibility_wakes w JOIN semantier_responsibility_checkpoints c ON c.responsibility_id=w.responsibility_id AND c.checkpoint_version=w.expected_checkpoint_version JOIN semantier_responsibilities r ON r.responsibility_id=w.responsibility_id WHERE w.responsibility_id=%s ORDER BY w.observed_at,w.wake_id'), (responsibility_id,),
            ).fetchall()
        basis = {
            "checkpoint_version": int(item["checkpoint_version"]),
            "checkpoint_hash": str(item["checkpoint_hash"]),
            "produced_at": str(checkpoint["produced_at"]),
            "observed_state_updated_at": str(item["updated_at"]),
        }
        pending = [w for w in waits if w["state"] == "PENDING"]
        waiting_basis = [{
            **asdict(self._wait(w)),
            "release_condition": {
                "kind": str(w["kind"]), "source_ref": str(w["source_ref"]),
                "correlation_key": str(w["correlation_key"]),
                "expected_checkpoint_version": int(w["expected_checkpoint_version"]),
                "deadline_at": w["deadline_at"],
            },
        } for w in pending]
        eligible = [{
            "operation": "signal_user_wait", "wait_id": str(w["wait_id"]),
            "responsibility_id": responsibility_id, "workspace_id": workspace_id,
            "actor_ref": actor_ref, **basis,
        } for w in pending if (
            w["kind"] == USER and w["source_ref"] == actor_ref
            and actor_ref.startswith("user:") and len(actor_ref) > 5
            and item["lifecycle_state"] == REQUIRE_USER
            and int(w["expected_checkpoint_version"]) == int(item["checkpoint_version"])
        )]
        observations = []
        for wake in wakes:
            matched = next((w for w in waits if (
                w["kind"] == wake["kind"] and w["source_ref"] == wake["source_ref"]
                and w["correlation_key"] == wake["correlation_key"]
                and w["expected_checkpoint_version"] == wake["expected_checkpoint_version"]
            )), None)
            observations.append({
                "wake_id": str(wake["wake_id"]), "responsibility_id": responsibility_id,
                "wait_id": str(matched["wait_id"]) if matched else None,
                "accepted": bool(wake["accepted"]),
                "local_wait_state": str(matched["state"]) if matched and wake["accepted"] else "UNCONFIRMED",
                "local_wait_observed_at": matched["satisfied_at"] if matched else None,
                "current_lifecycle_state": str(item["lifecycle_state"]),
                "external_effect": "UNCONFIRMED",
            })
        return {
            "schema_version": "responsibility_decision.v1",
            "observed_at": _now_iso(),
            "responsibility_id": responsibility_id, "workspace_id": workspace_id,
            "domain_ref": str(item["domain_ref"]), "workflow_id": str(item["workflow_id"]),
            "workflow_instance_id": str(item["workflow_instance_id"]),
            "session_ref": checkpoint["session_ref"],
            "lifecycle_state": str(item["lifecycle_state"]), "basis": basis,
            "waiting_basis": waiting_basis, "eligible_interventions": eligible,
            "receipts": [self._wake_receipt(w) for w in wakes],
            "observations": observations,
        }

    def signal_user_wait(
        self, *, responsibility_id: str, workspace_id: str, actor_ref: str,
        request_id: str, wait_id: str, expected_checkpoint_version: int,
        expected_checkpoint_hash: str, now: str | None = None,
    ) -> dict[str, Any]:
        """Signal only a producer's exact principal-addressed USER wait.

        This is a continuation signal, not domain approval or external execution.
        Scope, basis, eligibility and retry conflict are checked in the wake transaction.
        """
        if not request_id.strip() or len(request_id) > 200:
            raise ValueError("request_id required (maximum 200 characters)")
        if type(expected_checkpoint_version) is not int or expected_checkpoint_version < 0:
            raise ValueError("invalid checkpoint version")
        with self._connect() as conn:
            wait = conn.execute(
                hosted.sql_query("SELECT * FROM semantier_responsibility_waits "
                "WHERE responsibility_id=? AND wait_id=?", 'SELECT * FROM semantier_responsibility_waits WHERE responsibility_id=%s AND wait_id=%s'), (responsibility_id, wait_id),
            ).fetchone()
        if wait is None:
            raise DurableResponsibilityError("exact wait unavailable")
        key = _hash_id("user_signal", [workspace_id, actor_ref, responsibility_id, request_id], 64)
        result = self.accept_wake(
            responsibility_id=responsibility_id, kind=USER, source_ref=str(wait["source_ref"]),
            correlation_key=str(wait["correlation_key"]), dedupe_key=key,
            expected_checkpoint_version=expected_checkpoint_version, observed_at=now,
            user_basis=(workspace_id, actor_ref, wait_id, expected_checkpoint_hash),
        )
        with self._connect() as conn:
            receipt = conn.execute(
                hosted.sql_query("SELECT w.*,c.checkpoint_hash,r.workspace_id FROM semantier_responsibility_wakes w "
                "JOIN semantier_responsibility_checkpoints c ON c.responsibility_id=w.responsibility_id "
                "AND c.checkpoint_version=w.expected_checkpoint_version "
                "JOIN semantier_responsibilities r ON r.responsibility_id=w.responsibility_id "
                "WHERE w.wake_id=?", 'SELECT w.*,c.checkpoint_hash,r.workspace_id FROM semantier_responsibility_wakes w JOIN semantier_responsibility_checkpoints c ON c.responsibility_id=w.responsibility_id AND c.checkpoint_version=w.expected_checkpoint_version JOIN semantier_responsibilities r ON r.responsibility_id=w.responsibility_id WHERE w.wake_id=%s'), (result.wake_id,),
            ).fetchone()
        if receipt is None:
            raise DurableResponsibilityError("wake receipt unavailable")
        return self._wake_receipt(receipt)

    def accept_wake(
        self,
        *,
        responsibility_id: str,
        kind: str,
        source_ref: str,
        correlation_key: str,
        dedupe_key: str,
        expected_checkpoint_version: int,
        observed_at: str | None = None,
        user_basis: tuple[str, str, str, str] | None = None,
    ) -> WakeResult:
        if kind not in _WAIT_KINDS:
            raise ValueError("invalid wake kind")
        observed = observed_at or _now_iso()
        _parse_utc(observed)
        wake_id = _hash_id("wake", [responsibility_id, dedupe_key])
        with self._connect() as conn:
            hosted.begin_write(conn)
            try:
                if user_basis is not None:
                    workspace_id, actor_ref, wait_id, basis_hash = user_basis
                    current = conn.execute(
                        hosted.sql_query("SELECT * FROM semantier_responsibilities WHERE responsibility_id=?", 'SELECT * FROM semantier_responsibilities WHERE responsibility_id=%s'),
                        (responsibility_id,),
                    ).fetchone()
                    if current is None or current["workspace_id"] != workspace_id:
                        raise PermissionError("Outside responsibility scope")
                    wait = conn.execute(
                        hosted.sql_query("SELECT * FROM semantier_responsibility_waits "
                        "WHERE responsibility_id=? AND wait_id=?", 'SELECT * FROM semantier_responsibility_waits WHERE responsibility_id=%s AND wait_id=%s'), (responsibility_id, wait_id),
                    ).fetchone()
                    if (wait is None or wait["kind"] != USER or kind != USER
                            or not actor_ref.startswith("user:") or len(actor_ref) <= 5
                            or wait["source_ref"] != actor_ref or source_ref != actor_ref):
                        raise PermissionError("USER wait is not addressed to current principal")
                    pinned = conn.execute(
                        hosted.sql_query("SELECT checkpoint_hash FROM semantier_responsibility_checkpoints "
                        "WHERE responsibility_id=? AND checkpoint_version=?", 'SELECT checkpoint_hash FROM semantier_responsibility_checkpoints WHERE responsibility_id=%s AND checkpoint_version=%s'),
                        (responsibility_id, expected_checkpoint_version),
                    ).fetchone()
                    if (pinned is None or pinned["checkpoint_hash"] != basis_hash
                            or wait["expected_checkpoint_version"] != expected_checkpoint_version):
                        raise DurableResponsibilityError("decision basis changed")
                    retry = conn.execute(
                        hosted.sql_query("SELECT * FROM semantier_responsibility_wakes WHERE dedupe_key=?", 'SELECT * FROM semantier_responsibility_wakes WHERE dedupe_key=%s'),
                        (dedupe_key,),
                    ).fetchone()
                    if retry is not None:
                        if (retry["responsibility_id"] != responsibility_id or retry["kind"] != kind
                                or retry["source_ref"] != source_ref
                                or retry["correlation_key"] != correlation_key
                                or retry["expected_checkpoint_version"] != expected_checkpoint_version):
                            raise DurableResponsibilityError("conflicting intervention retry")
                    elif (current["checkpoint_hash"] != basis_hash
                            or current["checkpoint_version"] != expected_checkpoint_version
                            or current["lifecycle_state"] != REQUIRE_USER
                            or wait["state"] != "PENDING"):
                        raise DurableResponsibilityError("decision basis or eligibility changed")
                duplicate = conn.execute(
                    hosted.sql_query("SELECT wake_id FROM semantier_responsibility_wakes WHERE dedupe_key=?", 'SELECT wake_id FROM semantier_responsibility_wakes WHERE dedupe_key=%s'),
                    (dedupe_key,),
                ).fetchone()
                if duplicate:
                    conn.execute("ROLLBACK")
                    return WakeResult(False, "duplicate", str(duplicate["wake_id"]))

                item = conn.execute(
                    hosted.sql_query("SELECT * FROM semantier_responsibilities WHERE responsibility_id=?", 'SELECT * FROM semantier_responsibilities WHERE responsibility_id=%s'),
                    (responsibility_id,),
                ).fetchone()
                if item is None:
                    raise KeyError(responsibility_id)

                reason = "accepted"
                accepted = True
                wait_row = None
                if str(item["lifecycle_state"]) == TERMINAL:
                    accepted = False
                    reason = "terminal"
                elif int(item["checkpoint_version"]) != int(expected_checkpoint_version):
                    accepted = False
                    reason = "stale_checkpoint"
                else:
                    wait_row = conn.execute(
                        hosted.sql_query("""
                        SELECT * FROM semantier_responsibility_waits
                        WHERE responsibility_id=? AND kind=? AND source_ref=?
                          AND correlation_key=? AND expected_checkpoint_version=?
                          AND state='PENDING'
                        """, """
                        SELECT * FROM semantier_responsibility_waits
                        WHERE responsibility_id=%s AND kind=%s AND source_ref=%s
                          AND correlation_key=%s AND expected_checkpoint_version=%s
                          AND state='PENDING'
                        """),
                        (
                            responsibility_id,
                            kind,
                            source_ref,
                            correlation_key,
                            expected_checkpoint_version,
                        ),
                    ).fetchone()
                    if wait_row is None:
                        accepted = False
                        reason = "no_matching_wait"

                conn.execute(
                    hosted.sql_query("""
                    INSERT INTO semantier_responsibility_wakes(
                        wake_id,responsibility_id,kind,source_ref,correlation_key,
                        dedupe_key,expected_checkpoint_version,observed_at,accepted,reason
                    ) VALUES (?,?,?,?,?,?,?,?,?,?)
                    """, """
                    INSERT INTO semantier_responsibility_wakes(
                        wake_id,responsibility_id,kind,source_ref,correlation_key,
                        dedupe_key,expected_checkpoint_version,observed_at,accepted,reason
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """),
                    (
                        wake_id,
                        responsibility_id,
                        kind,
                        source_ref,
                        correlation_key,
                        dedupe_key,
                        expected_checkpoint_version,
                        observed,
                        1 if accepted else 0,
                        reason,
                    ),
                )
                if accepted and wait_row is not None:
                    conn.execute(
                        hosted.sql_query("""
                        UPDATE semantier_responsibility_waits
                        SET state='SATISFIED', satisfied_at=?
                        WHERE wait_id=? AND state='PENDING'
                        """, """
                        UPDATE semantier_responsibility_waits
                        SET state='SATISFIED', satisfied_at=%s
                        WHERE wait_id=%s AND state='PENDING'
                        """),
                        (observed, wait_row["wait_id"]),
                    )
                    conn.execute(
                        hosted.sql_query("""
                        UPDATE semantier_responsibility_waits
                        SET state='CANCELLED', satisfied_at=?
                        WHERE responsibility_id=?
                          AND expected_checkpoint_version=?
                          AND state='PENDING'
                          AND wait_id<>?
                        """, """
                        UPDATE semantier_responsibility_waits
                        SET state='CANCELLED', satisfied_at=%s
                        WHERE responsibility_id=%s
                          AND expected_checkpoint_version=%s
                          AND state='PENDING'
                          AND wait_id<>%s
                        """),
                        (
                            observed,
                            responsibility_id,
                            expected_checkpoint_version,
                            wait_row["wait_id"],
                        ),
                    )
                    conn.execute(
                        hosted.sql_query("""
                        UPDATE semantier_responsibilities
                        SET lifecycle_state='READY',
                            continuation_disposition=?,
                            current_wait_ref=NULL,
                            lease_owner=NULL,
                            lease_expires_at=NULL,
                            updated_at=?
                        WHERE responsibility_id=?
                        """, """
                        UPDATE semantier_responsibilities
                        SET lifecycle_state='READY',
                            continuation_disposition=%s,
                            current_wait_ref=NULL,
                            lease_owner=NULL,
                            lease_expires_at=NULL,
                            updated_at=%s
                        WHERE responsibility_id=%s
                        """),
                        (DETERMINISTIC_CONTINUE, observed, responsibility_id),
                    )
                self._event(
                    conn,
                    responsibility_id,
                    "WAKE_ACCEPTED" if accepted else "WAKE_REJECTED",
                    {
                        "wake_id": wake_id,
                        "kind": kind,
                        "source_ref": source_ref,
                        "reason": reason,
                    },
                    observed,
                )
                conn.execute("COMMIT")
            except Exception:
                if hosted.in_transaction(conn):
                    conn.execute("ROLLBACK")
                raise
        return WakeResult(accepted, reason, wake_id)

    def terminate(
        self,
        *,
        responsibility_id: str,
        reason: str,
        now: str | None = None,
        expected_lifecycle_state: str | None = None,
        expected_checkpoint_version: int | None = None,
    ) -> ResponsibilityRecord:
        """Terminalize owner-completed work without granting a stale worker authority."""
        if not reason.strip():
            raise ValueError("reason is required")
        if expected_lifecycle_state is not None and expected_lifecycle_state not in _LIFECYCLE_STATES:
            raise ValueError("invalid expected lifecycle state")
        if expected_checkpoint_version is not None and (
            type(expected_checkpoint_version) is not int or expected_checkpoint_version < 0
        ):
            raise ValueError("invalid expected checkpoint version")
        produced_at = now or _now_iso()
        produced_dt = _parse_utc(produced_at)
        with self._connect() as conn:
            hosted.begin_write(conn)
            try:
                current = conn.execute(
                    hosted.sql_query("SELECT * FROM semantier_responsibilities WHERE responsibility_id=?", 'SELECT * FROM semantier_responsibilities WHERE responsibility_id=%s'),
                    (responsibility_id,),
                ).fetchone()
                if current is None:
                    raise KeyError(responsibility_id)
                if (
                    expected_lifecycle_state is not None
                    and str(current["lifecycle_state"]) != expected_lifecycle_state
                ) or (
                    expected_checkpoint_version is not None
                    and int(current["checkpoint_version"]) != expected_checkpoint_version
                ):
                    raise DurableResponsibilityError("termination precondition changed")
                if str(current["lifecycle_state"]) == TERMINAL:
                    conn.execute("COMMIT")
                    record = self.get(responsibility_id)
                    assert record is not None
                    return record
                if str(current["lifecycle_state"]) == RUNNING:
                    expiry = current["lease_expires_at"]
                    if expiry and _parse_utc(str(expiry)) > produced_dt:
                        raise DurableResponsibilityError(
                            "active responsibility claim must finish or expire before owner terminalization"
                        )

                prior = conn.execute(
                    hosted.sql_query("""
                    SELECT * FROM semantier_responsibility_checkpoints
                    WHERE responsibility_id=?
                    ORDER BY checkpoint_version DESC LIMIT 1
                    """, """
                    SELECT * FROM semantier_responsibility_checkpoints
                    WHERE responsibility_id=%s
                    ORDER BY checkpoint_version DESC LIMIT 1
                    """),
                    (responsibility_id,),
                ).fetchone()
                if prior is None:
                    raise DurableResponsibilityError("responsibility checkpoint missing")
                next_version = int(current["checkpoint_version"]) + 1
                prior_hash = str(current["checkpoint_hash"])
                payload = {
                    "responsibility_id": responsibility_id,
                    "checkpoint_version": next_version,
                    "lifecycle_state": TERMINAL,
                    "continuation_disposition": TERMINAL,
                    "domain_ref": str(current["domain_ref"]),
                    "workflow_ref": str(current["workflow_id"]),
                    "session_ref": (
                        str(prior["session_ref"])
                        if prior["session_ref"] is not None
                        else None
                    ),
                    "wait_ref": None,
                    "accepted_event_cursor": (
                        str(prior["accepted_event_cursor"])
                        if prior["accepted_event_cursor"] is not None
                        else None
                    ),
                    "pending_effect_refs": sorted(
                        set(json.loads(str(prior["pending_effect_refs_json"])))
                    ),
                    "prior_checkpoint_hash": prior_hash,
                    "produced_at": produced_at,
                }
                cp_hash = _checkpoint_hash(payload)
                conn.execute(
                    hosted.sql_query("""
                    UPDATE semantier_responsibility_waits
                    SET state='CANCELLED', satisfied_at=?
                    WHERE responsibility_id=? AND state='PENDING'
                    """, """
                    UPDATE semantier_responsibility_waits
                    SET state='CANCELLED', satisfied_at=%s
                    WHERE responsibility_id=%s AND state='PENDING'
                    """),
                    (produced_at, responsibility_id),
                )
                conn.execute(
                    hosted.sql_query("""
                    INSERT INTO semantier_responsibility_checkpoints(
                        responsibility_id,checkpoint_version,lifecycle_state,
                        continuation_disposition,domain_ref,workflow_ref,session_ref,
                        wait_ref,accepted_event_cursor,pending_effect_refs_json,
                        prior_checkpoint_hash,checkpoint_hash,produced_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, """
                    INSERT INTO semantier_responsibility_checkpoints(
                        responsibility_id,checkpoint_version,lifecycle_state,
                        continuation_disposition,domain_ref,workflow_ref,session_ref,
                        wait_ref,accepted_event_cursor,pending_effect_refs_json,
                        prior_checkpoint_hash,checkpoint_hash,produced_at
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """),
                    (
                        responsibility_id,
                        next_version,
                        TERMINAL,
                        TERMINAL,
                        current["domain_ref"],
                        current["workflow_id"],
                        payload["session_ref"],
                        None,
                        payload["accepted_event_cursor"],
                        _canonical(payload["pending_effect_refs"]),
                        prior_hash,
                        cp_hash,
                        produced_at,
                    ),
                )
                conn.execute(
                    hosted.sql_query("""
                    UPDATE semantier_responsibilities
                    SET lifecycle_state='TERMINAL', continuation_disposition='TERMINAL',
                        checkpoint_version=?, checkpoint_hash=?, current_wait_ref=NULL,
                        lease_owner=NULL, lease_expires_at=NULL, updated_at=?
                    WHERE responsibility_id=?
                    """, """
                    UPDATE semantier_responsibilities
                    SET lifecycle_state='TERMINAL', continuation_disposition='TERMINAL',
                        checkpoint_version=%s, checkpoint_hash=%s, current_wait_ref=NULL,
                        lease_owner=NULL, lease_expires_at=NULL, updated_at=%s
                    WHERE responsibility_id=%s
                    """),
                    (next_version, cp_hash, produced_at, responsibility_id),
                )
                self._event(
                    conn,
                    responsibility_id,
                    "RESPONSIBILITY_TERMINATED",
                    {"reason": reason, "checkpoint_version": next_version},
                    produced_at,
                )
                conn.execute("COMMIT")
            except Exception:
                if hosted.in_transaction(conn):
                    conn.execute("ROLLBACK")
                raise
        record = self.get(responsibility_id)
        assert record is not None
        return record

    def release(
        self,
        claim: ResponsibilityClaim,
        *,
        reason: str,
        now: str | None = None,
    ) -> ResponsibilityRecord:
        released_at = now or _now_iso()
        _parse_utc(released_at)
        with self._connect() as conn:
            hosted.begin_write(conn)
            try:
                self.assert_claim(claim, conn=conn)
                conn.execute(
                    hosted.sql_query("""
                    UPDATE semantier_responsibilities
                    SET lifecycle_state='READY',
                        continuation_disposition=?,
                        lease_owner=NULL,
                        lease_expires_at=NULL,
                        updated_at=?
                    WHERE responsibility_id=?
                    """, """
                    UPDATE semantier_responsibilities
                    SET lifecycle_state='READY',
                        continuation_disposition=%s,
                        lease_owner=NULL,
                        lease_expires_at=NULL,
                        updated_at=%s
                    WHERE responsibility_id=%s
                    """),
                    (DETERMINISTIC_CONTINUE, released_at, claim.responsibility_id),
                )
                self._event(
                    conn,
                    claim.responsibility_id,
                    "RESPONSIBILITY_RELEASED",
                    {"reason": reason, "lease_epoch": claim.lease_epoch},
                    released_at,
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        record = self.get(claim.responsibility_id)
        assert record is not None
        return record

    def ready(self, *, now: str | None = None) -> list[ResponsibilityRecord]:
        observed = now or _now_iso()
        observed_dt = _parse_utc(observed)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM semantier_responsibilities
                WHERE lifecycle_state='READY'
                   OR lifecycle_state='RUNNING'
                ORDER BY updated_at,responsibility_id
                """
            ).fetchall()
        result = []
        for row in rows:
            state = str(row["lifecycle_state"])
            if state == READY:
                result.append(self._record(row))
                continue
            expiry = row["lease_expires_at"]
            if expiry and _parse_utc(str(expiry)) <= observed_dt:
                result.append(self._record(row))
        return result

    def events(self, responsibility_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                hosted.sql_query("""
                SELECT event_id,event_type,payload_json,created_at
                FROM semantier_responsibility_events
                WHERE responsibility_id=? ORDER BY event_id
                """, """
                SELECT event_id,event_type,payload_json,created_at
                FROM semantier_responsibility_events
                WHERE responsibility_id=%s ORDER BY event_id
                """),
                (responsibility_id,),
            ).fetchall()
        return [
            {
                "event_id": int(row["event_id"]),
                "event_type": str(row["event_type"]),
                "payload": json.loads(str(row["payload_json"])),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]


class DurableResponsibilityDispatcher:
    """Bounded executor over durable responsibilities."""

    def __init__(
        self,
        store: DurableResponsibilityStore,
        *,
        handlers: dict[str, Any],
        owner_id: str,
        lease_seconds: int = 180,
    ) -> None:
        self.store = store
        self.handlers = dict(handlers)
        self.owner_id = owner_id
        self.lease_seconds = lease_seconds

    def run_one(
        self,
        *,
        responsibility_id: str,
        now: str | None = None,
    ) -> DispatchResult:
        record = self.store.get(responsibility_id)
        if record is None:
            raise KeyError(responsibility_id)
        claim = self.store.claim(
            responsibility_id=responsibility_id,
            owner_id=self.owner_id,
            now=now,
            lease_seconds=self.lease_seconds,
        )
        handler = self.handlers.get(record.responsibility_kind)
        if handler is None:
            released = self.store.release(
                claim,
                reason="responsibility_handler_unavailable",
                now=now,
            )
            return DispatchResult(
                responsibility_id=responsibility_id,
                disposition=DETERMINISTIC_CONTINUE,
                checkpoint_version=released.checkpoint_version,
                error="responsibility_handler_unavailable",
            )

        checkpoint = self.store.latest_checkpoint(responsibility_id)
        try:
            decision = handler(record, checkpoint, claim)
        except Exception as exc:
            error = str(exc) or type(exc).__name__
            try:
                released = self.store.release(
                    claim,
                    reason=f"responsibility_handler_failed:{type(exc).__name__}",
                    now=now,
                )
                checkpoint_version = released.checkpoint_version
            except StaleResponsibilityClaim:
                current = self.store.get(responsibility_id)
                checkpoint_version = (
                    current.checkpoint_version
                    if current is not None
                    else claim.checkpoint_version
                )
            return DispatchResult(
                responsibility_id=responsibility_id,
                disposition=DETERMINISTIC_CONTINUE,
                checkpoint_version=checkpoint_version,
                error=error,
            )
        if not isinstance(decision, ContinuationDecision):
            self.store.release(
                claim,
                reason="invalid_continuation_decision",
                now=now,
            )
            raise DurableResponsibilityError(
                "handler returned invalid continuation decision"
            )

        if decision.disposition in _WAIT_STATES:
            wait_specs = decision.waits
            if not wait_specs and (
                decision.wait_kind
                and decision.wait_source_ref
                and decision.wait_correlation_key
            ):
                wait_specs = (
                    WaitSpec(
                        kind=decision.wait_kind,
                        source_ref=decision.wait_source_ref,
                        correlation_key=decision.wait_correlation_key,
                        deadline_at=decision.wait_deadline_at,
                    ),
                )
            if not wait_specs:
                self.store.release(claim, reason="wait_identity_required", now=now)
                raise DurableResponsibilityError(
                    "wait decision requires exact wait identity"
                )
            self.store.register_waits(
                claim,
                waits=wait_specs,
                lifecycle_state=decision.disposition,
                continuation_disposition=decision.disposition,
                session_ref=decision.session_ref,
                now=now,
            )
            current = self.store.get(responsibility_id)
            assert current is not None
            return DispatchResult(
                responsibility_id=responsibility_id,
                disposition=decision.disposition,
                checkpoint_version=current.checkpoint_version,
            )

        if decision.disposition == TERMINAL:
            current = self.store.commit(
                claim,
                continuation_disposition=TERMINAL,
                lifecycle_state=TERMINAL,
                session_ref=decision.session_ref,
                accepted_event_cursor=decision.accepted_event_cursor,
                pending_effect_refs=decision.pending_effect_refs,
                now=now,
            )
            return DispatchResult(
                responsibility_id=responsibility_id,
                disposition=TERMINAL,
                checkpoint_version=current.checkpoint_version,
            )

        if decision.disposition == DETERMINISTIC_CONTINUE:
            current = self.store.commit(
                claim,
                continuation_disposition=DETERMINISTIC_CONTINUE,
                lifecycle_state=READY,
                session_ref=decision.session_ref,
                accepted_event_cursor=decision.accepted_event_cursor,
                pending_effect_refs=decision.pending_effect_refs,
                now=now,
            )
            return DispatchResult(
                responsibility_id=responsibility_id,
                disposition=DETERMINISTIC_CONTINUE,
                checkpoint_version=current.checkpoint_version,
            )

        self.store.release(
            claim,
            reason=f"unsupported_continuation:{decision.disposition}",
            now=now,
        )
        raise DurableResponsibilityError(
            f"unsupported continuation disposition: {decision.disposition}"
        )

    def run_ready(self, *, now: str | None = None) -> list[DispatchResult]:
        return [
            self.run_one(
                responsibility_id=record.responsibility_id,
                now=now,
            )
            for record in self.store.ready(now=now)
        ]
