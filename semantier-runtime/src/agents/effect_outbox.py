from __future__ import annotations

import hashlib
import sqlite3
from storage import hosted_postgres as hosted
from dataclasses import dataclass
from pathlib import Path

from agents.durable_protocol_versions import (
    CURRENT_EFFECT_CONTRACT_VERSION,
    CURRENT_SCHEMA_VERSION,
    CURRENT_WORKFLOW_CONTRACT_VERSION,
    assert_row_supported,
    ensure_version_columns,
)


READY = "READY"
DISPATCHING = "DISPATCHING"
COMPLETED = "COMPLETED"
FAILED_SAFE = "FAILED_SAFE"
EFFECT_UNKNOWN = "EFFECT_UNKNOWN"


class EffectOutboxError(RuntimeError):
    pass


class InvalidEffectTransition(EffectOutboxError):
    pass


class UnsafeEffectRetry(EffectOutboxError):
    pass


@dataclass(frozen=True)
class EffectIdentity:
    effect_id: str
    workflow_id: str
    effect_type: str
    semantic_round: int
    idempotency_key: str
    request_hash: str


@dataclass(frozen=True)
class EffectRecord(EffectIdentity):
    canonical_payload: bytes
    state: str
    dispatch_attempt_id: str | None
    provider_correlation_ref: str | None
    receipt_ref: str | None
    failure_detail: str | None


def derive_effect_identity(
    *,
    workflow_id: str,
    effect_type: str,
    semantic_round: int,
    canonical_payload: bytes,
) -> EffectIdentity:
    if not workflow_id.strip():
        raise ValueError("workflow_id is required")
    if not effect_type.strip():
        raise ValueError("effect_type is required")
    if semantic_round < 1:
        raise ValueError("semantic_round must be >= 1")
    request_hash = hashlib.sha256(canonical_payload).hexdigest()
    stable = f"{workflow_id}|{effect_type}|{semantic_round}|{request_hash}"
    digest = hashlib.sha256(stable.encode("utf-8")).hexdigest()[:32]
    return EffectIdentity(
        effect_id=f"effect_{digest}",
        workflow_id=workflow_id,
        effect_type=effect_type,
        semantic_round=semantic_round,
        idempotency_key=f"{workflow_id}:{effect_type}:{semantic_round}:{request_hash}",
        request_hash=request_hash,
    )


class AtomicEffectOutbox:
    """Durable external-effect intent lifecycle.

    Business eligibility remains caller/domain-owned.  `enqueue_ready` accepts an
    existing SQLite connection so a domain mutation and effect intent can commit
    atomically in the caller's transaction.
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
            hosted.require_tables(conn, 'semantier_effect_outbox')
            return
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantier_effect_outbox (
                effect_id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                effect_type TEXT NOT NULL,
                semantic_round INTEGER NOT NULL,
                idempotency_key TEXT NOT NULL UNIQUE,
                request_hash TEXT NOT NULL,
                canonical_payload BLOB NOT NULL,
                state TEXT NOT NULL,
                dispatch_attempt_id TEXT,
                provider_correlation_ref TEXT,
                receipt_ref TEXT,
                failure_detail TEXT,
                schema_version INTEGER NOT NULL DEFAULT 1,
                workflow_contract_version INTEGER NOT NULL DEFAULT 1,
                effect_contract_version INTEGER NOT NULL DEFAULT 1,
                CHECK (state IN (
                    'READY','DISPATCHING','COMPLETED','FAILED_SAFE','EFFECT_UNKNOWN'
                ))
            )
            """
        )
        ensure_version_columns(conn, "effect")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_semantier_effect_outbox_ready
            ON semantier_effect_outbox(state, workflow_id, semantic_round)
            """
        )

    @staticmethod
    def enqueue_ready(
        conn: sqlite3.Connection,
        *,
        workflow_id: str,
        effect_type: str,
        semantic_round: int,
        canonical_payload: bytes,
    ) -> EffectIdentity:
        AtomicEffectOutbox.ensure_schema(conn)
        identity = derive_effect_identity(
            workflow_id=workflow_id,
            effect_type=effect_type,
            semantic_round=semantic_round,
            canonical_payload=canonical_payload,
        )
        conn.execute(
            hosted.sql_query("""
            INSERT OR IGNORE INTO semantier_effect_outbox(
                effect_id, workflow_id, effect_type, semantic_round,
                idempotency_key, request_hash, canonical_payload, state,
                schema_version, workflow_contract_version, effect_contract_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'READY', ?, ?, ?)
            """, """
            INSERT INTO semantier_effect_outbox(
                effect_id, workflow_id, effect_type, semantic_round,
                idempotency_key, request_hash, canonical_payload, state,
                schema_version, workflow_contract_version, effect_contract_version
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'READY', %s, %s, %s) ON CONFLICT DO NOTHING"""),
            (
                identity.effect_id,
                identity.workflow_id,
                identity.effect_type,
                identity.semantic_round,
                identity.idempotency_key,
                identity.request_hash,
                sqlite3.Binary(canonical_payload),
                CURRENT_SCHEMA_VERSION,
                CURRENT_WORKFLOW_CONTRACT_VERSION,
                CURRENT_EFFECT_CONTRACT_VERSION,
            ),
        )
        row = conn.execute(
            hosted.sql_query("SELECT idempotency_key FROM semantier_effect_outbox WHERE effect_id=?", 'SELECT idempotency_key FROM semantier_effect_outbox WHERE effect_id=%s'),
            (identity.effect_id,),
        ).fetchone()
        if row is None:
            raise EffectOutboxError("effect intent was not persisted")
        if str(row[0]) != identity.idempotency_key:
            raise EffectOutboxError("effect identity collision")
        return identity

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> EffectRecord:
        assert_row_supported("effect", row)
        return EffectRecord(
            effect_id=str(row["effect_id"]),
            workflow_id=str(row["workflow_id"]),
            effect_type=str(row["effect_type"]),
            semantic_round=int(row["semantic_round"]),
            idempotency_key=str(row["idempotency_key"]),
            request_hash=str(row["request_hash"]),
            canonical_payload=bytes(row["canonical_payload"]),
            state=str(row["state"]),
            dispatch_attempt_id=(
                str(row["dispatch_attempt_id"])
                if row["dispatch_attempt_id"] is not None
                else None
            ),
            provider_correlation_ref=(
                str(row["provider_correlation_ref"])
                if row["provider_correlation_ref"] is not None
                else None
            ),
            receipt_ref=str(row["receipt_ref"]) if row["receipt_ref"] is not None else None,
            failure_detail=(
                str(row["failure_detail"]) if row["failure_detail"] is not None else None
            ),
        )

    def get(self, effect_id: str) -> EffectRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                hosted.sql_query("SELECT * FROM semantier_effect_outbox WHERE effect_id=?", 'SELECT * FROM semantier_effect_outbox WHERE effect_id=%s'),
                (effect_id,),
            ).fetchone()
            return self._row_to_record(row) if row else None

    def pin_provider_correlation(
        self, *, effect_id: str, correlation_ref: str
    ) -> None:
        if not correlation_ref.strip():
            raise ValueError("correlation_ref is required")
        with self._connect() as conn:
            cur = conn.execute(
                hosted.sql_query("""
                UPDATE semantier_effect_outbox
                SET provider_correlation_ref=?
                WHERE effect_id=?
                  AND state IN ('DISPATCHING','EFFECT_UNKNOWN')
                  AND (
                    provider_correlation_ref IS NULL
                    OR provider_correlation_ref=?
                  )
                """, """
                UPDATE semantier_effect_outbox
                SET provider_correlation_ref=%s
                WHERE effect_id=%s
                  AND state IN ('DISPATCHING','EFFECT_UNKNOWN')
                  AND (
                    provider_correlation_ref IS NULL
                    OR provider_correlation_ref=%s
                  )
                """),
                (correlation_ref, effect_id, correlation_ref),
            )
            if cur.rowcount != 1:
                raise InvalidEffectTransition(
                    "provider correlation may only be pinned consistently "
                    "for DISPATCHING/EFFECT_UNKNOWN"
                )

    def claim_dispatch(self, *, effect_id: str, attempt_id: str) -> bool:
        if not attempt_id.strip():
            raise ValueError("attempt_id is required")
        with self._connect() as conn:
            hosted.begin_write(conn)
            cur = conn.execute(
                hosted.sql_query("""
                UPDATE semantier_effect_outbox
                SET state='DISPATCHING', dispatch_attempt_id=?,
                    receipt_ref=NULL, failure_detail=NULL
                WHERE effect_id=? AND state='READY'
                """, """
                UPDATE semantier_effect_outbox
                SET state='DISPATCHING', dispatch_attempt_id=%s,
                    receipt_ref=NULL, failure_detail=NULL
                WHERE effect_id=%s AND state='READY'
                """),
                (attempt_id, effect_id),
            )
            conn.commit()
            return cur.rowcount == 1

    def mark_completed(
        self, *, effect_id: str, attempt_id: str, receipt_ref: str
    ) -> None:
        if not receipt_ref.strip():
            raise ValueError("receipt_ref is required")
        with self._connect() as conn:
            cur = conn.execute(
                hosted.sql_query("""
                UPDATE semantier_effect_outbox
                SET state='COMPLETED', receipt_ref=?, failure_detail=NULL
                WHERE effect_id=? AND state='DISPATCHING'
                  AND dispatch_attempt_id=?
                """, """
                UPDATE semantier_effect_outbox
                SET state='COMPLETED', receipt_ref=%s, failure_detail=NULL
                WHERE effect_id=%s AND state='DISPATCHING'
                  AND dispatch_attempt_id=%s
                """),
                (receipt_ref, effect_id, attempt_id),
            )
            if cur.rowcount != 1:
                raise InvalidEffectTransition(
                    "completion requires the matching DISPATCHING attempt"
                )

    def mark_failed_safe(
        self, *, effect_id: str, attempt_id: str, failure_detail: str
    ) -> None:
        with self._connect() as conn:
            cur = conn.execute(
                hosted.sql_query("""
                UPDATE semantier_effect_outbox
                SET state='FAILED_SAFE', failure_detail=?
                WHERE effect_id=? AND state='DISPATCHING'
                  AND dispatch_attempt_id=?
                """, """
                UPDATE semantier_effect_outbox
                SET state='FAILED_SAFE', failure_detail=%s
                WHERE effect_id=%s AND state='DISPATCHING'
                  AND dispatch_attempt_id=%s
                """),
                (failure_detail, effect_id, attempt_id),
            )
            if cur.rowcount != 1:
                raise InvalidEffectTransition(
                    "safe failure requires the matching DISPATCHING attempt"
                )

    def mark_effect_unknown(
        self, *, effect_id: str, attempt_id: str, failure_detail: str
    ) -> None:
        with self._connect() as conn:
            cur = conn.execute(
                hosted.sql_query("""
                UPDATE semantier_effect_outbox
                SET state='EFFECT_UNKNOWN', failure_detail=?
                WHERE effect_id=? AND state='DISPATCHING'
                  AND dispatch_attempt_id=?
                """, """
                UPDATE semantier_effect_outbox
                SET state='EFFECT_UNKNOWN', failure_detail=%s
                WHERE effect_id=%s AND state='DISPATCHING'
                  AND dispatch_attempt_id=%s
                """),
                (failure_detail, effect_id, attempt_id),
            )
            if cur.rowcount != 1:
                raise InvalidEffectTransition(
                    "unknown effect requires the matching DISPATCHING attempt"
                )

    def retry_failed_safe(self, *, effect_id: str) -> None:
        with self._connect() as conn:
            cur = conn.execute(
                hosted.sql_query("""
                UPDATE semantier_effect_outbox
                SET state='READY', dispatch_attempt_id=NULL,
                    receipt_ref=NULL, failure_detail=NULL
                WHERE effect_id=? AND state='FAILED_SAFE'
                """, """
                UPDATE semantier_effect_outbox
                SET state='READY', dispatch_attempt_id=NULL,
                    receipt_ref=NULL, failure_detail=NULL
                WHERE effect_id=%s AND state='FAILED_SAFE'
                """),
                (effect_id,),
            )
            if cur.rowcount != 1:
                row = conn.execute(
                    hosted.sql_query("SELECT state FROM semantier_effect_outbox WHERE effect_id=?", 'SELECT state FROM semantier_effect_outbox WHERE effect_id=%s'),
                    (effect_id,),
                ).fetchone()
                if row and str(row["state"]) == EFFECT_UNKNOWN:
                    raise UnsafeEffectRetry(
                        "EFFECT_UNKNOWN requires G3 reconciliation before retry"
                    )
                raise InvalidEffectTransition(
                    "retry requires FAILED_SAFE"
                )

    def recover_dispatching_as_unknown(self) -> int:
        """Crash recovery: uncertain in-flight dispatches become EFFECT_UNKNOWN."""
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE semantier_effect_outbox
                SET state='EFFECT_UNKNOWN',
                    failure_detail=COALESCE(
                        failure_detail, 'recovered_after_uncertain_dispatch'
                    )
                WHERE state='DISPATCHING'
                """
            )
            return int(cur.rowcount)

    def reconcile_unknown_completed(
        self, *, effect_id: str, receipt_ref: str
    ) -> None:
        if not receipt_ref.strip():
            raise ValueError("receipt_ref is required")
        with self._connect() as conn:
            cur = conn.execute(
                hosted.sql_query("""
                UPDATE semantier_effect_outbox
                SET state='COMPLETED', receipt_ref=?, failure_detail=NULL
                WHERE effect_id=? AND state='EFFECT_UNKNOWN'
                """, """
                UPDATE semantier_effect_outbox
                SET state='COMPLETED', receipt_ref=%s, failure_detail=NULL
                WHERE effect_id=%s AND state='EFFECT_UNKNOWN'
                """),
                (receipt_ref, effect_id),
            )
            if cur.rowcount != 1:
                raise InvalidEffectTransition(
                    "G3 completion reconciliation requires EFFECT_UNKNOWN"
                )

    def reconcile_unknown_not_applied(self, *, effect_id: str) -> None:
        with self._connect() as conn:
            cur = conn.execute(
                hosted.sql_query("""
                UPDATE semantier_effect_outbox
                SET state='READY', dispatch_attempt_id=NULL,
                    receipt_ref=NULL,
                    failure_detail='g3_reconciled_not_applied'
                WHERE effect_id=? AND state='EFFECT_UNKNOWN'
                """, """
                UPDATE semantier_effect_outbox
                SET state='READY', dispatch_attempt_id=NULL,
                    receipt_ref=NULL,
                    failure_detail='g3_reconciled_not_applied'
                WHERE effect_id=%s AND state='EFFECT_UNKNOWN'
                """),
                (effect_id,),
            )
            if cur.rowcount != 1:
                raise InvalidEffectTransition(
                    "G3 not-applied reconciliation requires EFFECT_UNKNOWN"
                )


def _sandbox_calendar_request(*, ctx, workflow_id: str, idempotency_key: str, event: dict):
    """Bind a sandbox request to an existing governed principal and approval owner.

    This computes the approval descriptor; it does not grant approval. The host
    must record explicit confirmation through UnifiedApprovalStore separately.
    """
    import json
    import re
    from datetime import datetime, timezone

    if (not ctx.authenticated or not ctx.organization_id or not ctx.user_id
            or not ctx.workspace_id or ctx.workspace_id == 'public'):
        raise ValueError('SANDBOX_AUTHORITY_CONTEXT_REQUIRED')
    root, home = Path(ctx.workspace_root).resolve(), Path(ctx.hermes_home).resolve()
    if not home.is_relative_to(root):
        raise ValueError('SANDBOX_SCOPE_MISMATCH')
    for value in (workflow_id, idempotency_key):
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9._:-]{1,160}', value):
            raise ValueError('SANDBOX_REQUEST_ID_REQUIRED')
    if not isinstance(event, dict) or set(event) != {'title', 'participants', 'start', 'end'}:
        raise ValueError('INVALID_SANDBOX_EVENT')
    if not isinstance(event['title'], str) or not event['title'].strip() or len(event['title']) > 200:
        raise ValueError('INVALID_SANDBOX_EVENT')
    if (not isinstance(event['participants'], list) or not event['participants']
            or len(event['participants']) > 100 or any(
                not isinstance(p, str) or not p.strip() or len(p) > 200 for p in event['participants'])):
        raise ValueError('INVALID_SANDBOX_EVENT')
    times = []
    for field in ('start', 'end'):
        if not isinstance(event[field], str):
            raise ValueError('SANDBOX_UTC_TIME_REQUIRED')
        value = datetime.fromisoformat(event[field].replace('Z', '+00:00'))
        if value.utcoffset() is None:
            raise ValueError('SANDBOX_UTC_TIME_REQUIRED')
        times.append(value.astimezone(timezone.utc))
    if times[0] >= times[1]:
        raise ValueError('INVALID_SANDBOX_EVENT')
    normalized_event = {**event, 'start': times[0].isoformat().replace('+00:00', 'Z'),
                         'end': times[1].isoformat().replace('+00:00', 'Z')}
    payload = {'mode': 'SANDBOX', 'organization_id': ctx.organization_id,
        'workspace_id': ctx.workspace_id, 'principal_id': ctx.user_id,
        'workflow_id': workflow_id, 'idempotency_key': idempotency_key, 'event': normalized_event}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    payload_hash = hashlib.sha256(raw).hexdigest()
    key_basis = [ctx.organization_id, ctx.workspace_id, workflow_id, idempotency_key]
    key = 'sandbox-calendar:' + hashlib.sha256(json.dumps(key_basis).encode()).hexdigest()
    metadata = {'purpose': 'sandbox_calendar', 'organization_id': ctx.organization_id,
        'workspace_id': ctx.workspace_id, 'payload_hash': payload_hash}
    return key, metadata, raw


def sandbox_calendar_approval(*, ctx, workflow_id: str, idempotency_key: str, event: dict):
    key, metadata, _ = _sandbox_calendar_request(ctx=ctx, workflow_id=workflow_id,
        idempotency_key=idempotency_key, event=event)
    return key, metadata


def dispatch_approved_sandbox_calendar_event(
    *, ctx, outbox: AtomicEffectOutbox, approvals, workflow_id: str,
    idempotency_key: str, event: dict, confirmed: bool, send,
) -> dict:
    """Dispatch one explicitly approved sandbox event through the existing outbox.

    No live calendar transport is registered here. `send` is the host-owned
    sandbox boundary. Learner/model confirmation is insufficient without the
    exact principal/payload approval in the existing approval store.
    """
    import json
    import re
    from uuid import uuid4

    if confirmed is not True:
        return {'ok': False, 'error': 'EXPLICIT_CONFIRMATION_REQUIRED'}
    try:
        key, metadata, raw = _sandbox_calendar_request(ctx=ctx, workflow_id=workflow_id,
                                                idempotency_key=idempotency_key, event=event)
        home = Path(ctx.hermes_home).resolve()
        if not outbox.db_path.resolve().is_relative_to(home) or not approvals.db_path.resolve().is_relative_to(home):
            return {'ok': False, 'error': 'SANDBOX_STORE_SCOPE_MISMATCH'}
        records = approvals.get_approvals(key)
        if not any(record.approver == ctx.user_id and record.metadata == metadata for record in records):
            return {'ok': False, 'error': 'BOUND_SANDBOX_APPROVAL_REQUIRED'}
    except (ValueError, TypeError, AttributeError):
        return {'ok': False, 'error': 'INVALID_SANDBOX_REQUEST'}
    with outbox._connect() as conn:
        hosted.begin_write(conn)
        identity = outbox.enqueue_ready(conn, workflow_id=workflow_id,
            effect_type='sandbox_calendar_create', semantic_round=1, canonical_payload=raw)
        conn.commit()
    current = outbox.get(identity.effect_id)
    if current.state == COMPLETED:
        return {'ok': True, 'effect_id': identity.effect_id,
                'event_id': current.receipt_ref, 'replayed': True}
    attempt = 'sandbox-dispatch-' + uuid4().hex
    if not outbox.claim_dispatch(effect_id=identity.effect_id, attempt_id=attempt):
        return {'ok': False, 'effect_id': identity.effect_id, 'error': 'SANDBOX_EFFECT_NOT_READY'}
    try:
        response = send(json.loads(raw), identity.idempotency_key)
        if isinstance(response, dict) and response.get('ok') is False and response.get('not_applied') is True:
            outbox.mark_failed_safe(effect_id=identity.effect_id, attempt_id=attempt,
                                    failure_detail='SANDBOX_API_FAILED_SAFE')
            return {'ok': False, 'effect_id': identity.effect_id, 'error': 'SANDBOX_API_FAILED_SAFE'}
        event_id = response.get('event_id') if isinstance(response, dict) else None
        if response.get('ok') is not True or not isinstance(event_id, str) or not re.fullmatch(r'sandbox-event-[A-Za-z0-9._:-]{1,100}', event_id):
            raise ValueError('SANDBOX_RECEIPT_UNRESOLVED')
        outbox.mark_completed(effect_id=identity.effect_id, attempt_id=attempt, receipt_ref=event_id)
        return {'ok': True, 'effect_id': identity.effect_id, 'event_id': event_id, 'replayed': False}
    except Exception:
        outbox.mark_effect_unknown(effect_id=identity.effect_id, attempt_id=attempt,
                                   failure_detail='SANDBOX_EFFECT_UNKNOWN')
        return {'ok': False, 'effect_id': identity.effect_id, 'error': 'SANDBOX_EFFECT_UNKNOWN'}
