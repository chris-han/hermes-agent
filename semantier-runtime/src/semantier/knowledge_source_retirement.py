"""Source retirement orchestrator: durable deletion saga across stores.

Plan A M2 (D1, D6, D7): the orchestrator owns the saga state machine and
the cross-store transitions for retiring a source document. It is the
authoritative state machine for the lifecycle events emitted into
`source_lifecycle_events` and `artifact_lifecycle_events` /
`artifact_lifecycle_events_local`.

State machine (D1):

    requested
      -> file_staged
      -> metadata_tombstoned
      -> artifacts_tombstoned
      -> completed
    \\-> failed/retryable             (recoverable)
    \\-> failed/blocked_authoritative (terminal)

Each transition emits exactly one append-only event row. The orchestrator
is idempotent: calling `advance` with the current state is a no-op, and
calling `advance` with a forward state when the operation has already
advanced past it is a no-op. Resume reads the current state and advances
forward.

The orchestrator never opens new connections per stage; it routes through
the store-level methods which accept explicit `connection` arguments when
a transaction is required (D7).
"""
from __future__ import annotations

from storage import hosted_postgres as hosted

import json
from contextlib import contextmanager
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from storage.knowledge_extraction_store import KnowledgeExtractionStore
from semantier.governed_upload_ref import (
    KNOWLEDGE_UPLOAD_REF_PREFIX,
    decode_governed_upload_ref,
)
from semantier.workspace_retirement_stager import RepresentationMissing


def _utc_now() -> str:
    # Microsecond precision keeps lifecycle events sortable by insertion order
    # without relying on random UUIDs as the secondary sort key.
    return datetime.now(timezone.utc).isoformat()


class SourceRetirementState(str, Enum):
    REQUESTED = "requested"
    FILE_STAGED = "file_staged"
    METADATA_TOMBSTONED = "metadata_tombstoned"
    ARTIFACTS_TOMBSTONED = "artifacts_tombstoned"
    COMPLETED = "completed"
    FAILED_RETRYABLE = "failed/retryable"
    FAILED_BLOCKED_AUTHORITATIVE = "failed/blocked_authoritative"


# Forward transitions for the happy path.
_FORWARD = {
    None: {SourceRetirementState.REQUESTED},
    SourceRetirementState.REQUESTED: {
        SourceRetirementState.FILE_STAGED,
        SourceRetirementState.FAILED_RETRYABLE,
        SourceRetirementState.FAILED_BLOCKED_AUTHORITATIVE,
    },
    SourceRetirementState.FILE_STAGED: {
        SourceRetirementState.METADATA_TOMBSTONED,
        SourceRetirementState.FAILED_RETRYABLE,
    },
    SourceRetirementState.METADATA_TOMBSTONED: {
        SourceRetirementState.ARTIFACTS_TOMBSTONED,
        SourceRetirementState.FAILED_RETRYABLE,
    },
    SourceRetirementState.ARTIFACTS_TOMBSTONED: {
        SourceRetirementState.COMPLETED,
        SourceRetirementState.FAILED_RETRYABLE,
    },
    SourceRetirementState.COMPLETED: set(),
    SourceRetirementState.FAILED_RETRYABLE: set(),
    SourceRetirementState.FAILED_BLOCKED_AUTHORITATIVE: set(),
}

_TERMINAL = {
    SourceRetirementState.COMPLETED,
    SourceRetirementState.FAILED_BLOCKED_AUTHORITATIVE,
}


class SourceRetirementOrchestratorError(RuntimeError):
    """Base error for fail-closed orchestrator operations."""


class InvalidStateTransition(SourceRetirementOrchestratorError):
    """The requested transition is not valid for the current state."""


class TerminalStateReached(SourceRetirementOrchestratorError):
    """The operation has reached a terminal state and cannot be advanced."""


class AuthoritativeBlock(SourceRetirementOrchestratorError):
    """Plan A D5: the source has an authoritative descendant; retirement blocked."""


class SourceRetirementOrchestrator:
    """Owns the source-retirement saga state machine."""

    def __init__(
        self,
        *,
        extraction_store: KnowledgeExtractionStore,
        file_stager: Any | None = None,
        metadata_stage: Callable[[], Any] | None = None,
        artifact_stage: Callable[[], Any] | None = None,
    ) -> None:
        self._store = extraction_store
        self._file_stager = file_stager
        self._metadata_stage = metadata_stage
        self._artifact_stage = artifact_stage

    @contextmanager
    def _operation_lock(self):
        with self._store._lock:
            if not hosted.is_postgres(self._store._conn):
                yield
                return
            try:
                hosted.begin_write(self._store._conn)
                yield
                self._store._conn.commit()
            except Exception:
                self._store._conn.rollback()
                raise

    # ---------------------------------------------------------------- begin
    def begin(
        self,
        *,
        source_id: str,
        actor_id: str,
        operation_id: str | None = None,
        retirement_mode: str = "ordinary",
        domain_id: str | None = None,
        inventory_hash: str | None = None,
        organization_id: str | None = None,
        workspace_id: str | None = None,
    ) -> str:
        """Create a new operation in `requested` state. Returns operation_id."""
        operation_id = operation_id or f"op_{uuid.uuid4().hex}"
        if retirement_mode not in {"ordinary", "governed"}:
            raise ValueError("retirement_mode must be ordinary or governed")
        now = _utc_now()
        with self._operation_lock():
            self._store._conn.execute(
                hosted.sql_query("INSERT INTO source_retirement_operations ("
                " operation_id, source_id, actor_id, state, created_at, updated_at,"
                " retirement_mode, domain_id, inventory_hash, organization_id, workspace_id"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 'INSERT INTO source_retirement_operations ( operation_id, source_id, actor_id, state, created_at, updated_at, retirement_mode, domain_id, inventory_hash, organization_id, workspace_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'),
                (
                    operation_id, source_id, actor_id,
                    SourceRetirementState.REQUESTED.value, now, now,
                    retirement_mode, domain_id, inventory_hash,
                    organization_id, workspace_id,
                ),
            )
            self._store._conn.execute(
                hosted.sql_query("INSERT INTO source_lifecycle_events ("
                " event_id, source_id, operation_id, actor_id, previous_state,"
                " next_state, stage, reason, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", 'INSERT INTO source_lifecycle_events ( event_id, source_id, operation_id, actor_id, previous_state, next_state, stage, reason, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)'),
                (
                    f"evt_{uuid.uuid4().hex}",
                    source_id,
                    operation_id,
                    actor_id,
                    None,
                    SourceRetirementState.REQUESTED.value,
                    "begin",
                    None,
                    now,
                ),
            )
            self._store._conn.commit()
        return operation_id

    # -------------------------------------------------------------- advance
    def advance(
        self,
        *,
        operation_id: str,
        target_state: SourceRetirementState,
        file_stager: Any | None = None,
        metadata_stage: Callable[[], Any] | None = None,
        artifact_stage: Callable[[], Any] | None = None,
        reason: str | None = None,
    ) -> SourceRetirementState:
        """Idempotent state transition.

        - If the operation is already at or past `target_state` on the
          forward path, return the current state without emitting an event.
        - If the transition is invalid from the current state, raise
          `InvalidStateTransition`.
        - If the operation is in a terminal state, raise `TerminalStateReached`.
        - If a stage callable raises during the transition, transition to
          `failed/retryable`, record a failure event, and call
          `file_stager.restore(...)` if a staged file is recorded.
        """
        file_stager = file_stager or self._file_stager
        metadata_stage = metadata_stage or self._metadata_stage
        artifact_stage = artifact_stage or self._artifact_stage

        with self._operation_lock():
            row = self._store._conn.execute(
                hosted.sql_query("SELECT source_id, actor_id, state, staged_path, retirement_mode "
                "FROM source_retirement_operations "
                "WHERE operation_id = ?", 'SELECT source_id, actor_id, state, staged_path, retirement_mode FROM source_retirement_operations WHERE operation_id = %s'),
                (operation_id,),
            ).fetchone()
            if row is None:
                raise SourceRetirementOrchestratorError(
                    f"unknown operation_id: {operation_id}"
                )
            current = SourceRetirementState(row["state"])

            # Idempotent no-op: already at or past the target on the forward path.
            if _is_at_or_past(current, target_state):
                return current

            if current in _TERMINAL:
                raise TerminalStateReached(
                    f"operation {operation_id} is terminal at {current.value}; cannot advance to {target_state.value}"
                )

            if target_state not in _FORWARD[current]:
                raise InvalidStateTransition(
                    f"cannot transition from {current.value} to {target_state.value}"
                )

            previous_state = current
            now = _utc_now()
            actor_id = row["actor_id"]
            source_id = row["source_id"]
            staged_path = row["staged_path"]

            try:
                if target_state is SourceRetirementState.FILE_STAGED:
                    if file_stager is None:
                        raise SourceRetirementOrchestratorError(
                            "file_stager is required to advance to file_staged"
                        )
                    # Plan A D5: authoritative-block check runs BEFORE the file move.
                    # If any descendant has authority_state='authoritative', the saga
                    # terminates at failed/blocked_authoritative; no file is staged.
                    authoritative_count = self._count_authoritative_descendants(source_id)
                    if authoritative_count > 0 and row["retirement_mode"] != "governed":
                        raise AuthoritativeBlock(
                            f"source {source_id} has {authoritative_count} authoritative descendants; "
                            "requires_governed_retirement"
                        )
                    # Stage the file. The orchestrator tracks the original path
                    # so restore can put it back on metadata failure.
                    original_path = self._resolve_source_path(source_id)
                    scope = self._source_scope(source_id, required=False)
                    representations = (
                        self._store.list_active_source_representations(
                            source_id,
                            organization_id=scope[0],
                            workspace_id=scope[1],
                        )
                        if scope is not None
                        else []
                    )
                    representation_paths = [
                        representation["artifact_path"]
                        for representation in representations
                    ]
                    if hasattr(file_stager, "stage_many"):
                        staged_path = file_stager.stage_many(
                            source_id=source_id,
                            original_paths=[original_path, *representation_paths],
                        )
                    else:
                        staged_path = file_stager.stage(
                            source_id=source_id,
                            original_path=original_path,
                        )
                    self._store._conn.execute(
                        hosted.sql_query("UPDATE source_retirement_operations "
                        "SET staged_path = ?, original_path = ?, updated_at = ? "
                        "WHERE operation_id = ?", 'UPDATE source_retirement_operations SET staged_path = %s, original_path = %s, updated_at = %s WHERE operation_id = %s'),
                        (staged_path, original_path, now, operation_id),
                    )
                elif target_state is SourceRetirementState.METADATA_TOMBSTONED:
                    if metadata_stage is not None:
                        metadata_stage()
                    scope = self._source_scope(source_id, required=False)
                    if scope is not None:
                        self._store.tombstone_source_representations(
                            source_id,
                            operation_id=operation_id,
                            actor_id=actor_id,
                            organization_id=scope[0],
                            workspace_id=scope[1],
                        )
                elif target_state is SourceRetirementState.ARTIFACTS_TOMBSTONED:
                    if artifact_stage is None:
                        raise SourceRetirementOrchestratorError(
                            "artifact_stage is required to advance to artifacts_tombstoned"
                        )
                    artifact_stage()
                elif target_state is SourceRetirementState.COMPLETED:
                    if file_stager is None:
                        raise SourceRetirementOrchestratorError(
                            "file_stager is required to advance to completed"
                        )
                    file_stager.unlink(staged_path=staged_path)

                self._store._conn.execute(
                    hosted.sql_query("UPDATE source_retirement_operations SET state = ?, updated_at = ? "
                    "WHERE operation_id = ?", 'UPDATE source_retirement_operations SET state = %s, updated_at = %s WHERE operation_id = %s'),
                    (target_state.value, now, operation_id),
                )
                self._store._conn.execute(
                    hosted.sql_query("INSERT INTO source_lifecycle_events ("
                    " event_id, source_id, operation_id, actor_id, previous_state,"
                    " next_state, stage, reason, created_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", 'INSERT INTO source_lifecycle_events ( event_id, source_id, operation_id, actor_id, previous_state, next_state, stage, reason, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)'),
                    (
                        f"evt_{uuid.uuid4().hex}",
                        source_id,
                        operation_id,
                        actor_id,
                        previous_state.value,
                        target_state.value,
                        target_state.value,
                        reason,
                        now,
                    ),
                )
                self._store._conn.commit()
            except AuthoritativeBlock as exc:
                # Plan A D5: authoritative-block is a terminal non-recoverable state.
                self._store._conn.execute(
                    hosted.sql_query("UPDATE source_retirement_operations SET state = ?, updated_at = ?,"
                    " failure_reason = ?, failure_stage = ? WHERE operation_id = ?", 'UPDATE source_retirement_operations SET state = %s, updated_at = %s, failure_reason = %s, failure_stage = %s WHERE operation_id = %s'),
                    (
                        SourceRetirementState.FAILED_BLOCKED_AUTHORITATIVE.value,
                        now,
                        str(exc),
                        previous_state.value,
                        operation_id,
                    ),
                )
                self._store._conn.execute(
                    hosted.sql_query("INSERT INTO source_lifecycle_events ("
                    " event_id, source_id, operation_id, actor_id, previous_state,"
                    " next_state, stage, reason, created_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", 'INSERT INTO source_lifecycle_events ( event_id, source_id, operation_id, actor_id, previous_state, next_state, stage, reason, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)'),
                    (
                        f"evt_{uuid.uuid4().hex}",
                        source_id,
                        operation_id,
                        actor_id,
                        previous_state.value,
                        SourceRetirementState.FAILED_BLOCKED_AUTHORITATIVE.value,
                        previous_state.value,
                        str(exc),
                        now,
                    ),
                )
                self._store._conn.commit()
                # Re-raise so the route can return 409 requires_governed_retirement.
                raise
            except Exception as exc:
                # Classify the failure as retryable; restore the staged file if any.
                failure_stage = previous_state.value
                self._store._conn.execute(
                    hosted.sql_query("UPDATE source_retirement_operations SET state = ?, updated_at = ?,"
                    " failure_reason = ?, failure_stage = ? WHERE operation_id = ?", 'UPDATE source_retirement_operations SET state = %s, updated_at = %s, failure_reason = %s, failure_stage = %s WHERE operation_id = %s'),
                    (
                        SourceRetirementState.FAILED_RETRYABLE.value,
                        now,
                        str(exc),
                        failure_stage,
                        operation_id,
                    ),
                )
                self._store._conn.execute(
                    hosted.sql_query("INSERT INTO source_lifecycle_events ("
                    " event_id, source_id, operation_id, actor_id, previous_state,"
                    " next_state, stage, reason, created_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", 'INSERT INTO source_lifecycle_events ( event_id, source_id, operation_id, actor_id, previous_state, next_state, stage, reason, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)'),
                    (
                        f"evt_{uuid.uuid4().hex}",
                        source_id,
                        operation_id,
                        actor_id,
                        previous_state.value,
                        SourceRetirementState.FAILED_RETRYABLE.value,
                        failure_stage,
                        str(exc),
                        now,
                    ),
                )
                self._store._conn.commit()
                if (
                    target_state is SourceRetirementState.METADATA_TOMBSTONED
                    and file_stager is not None
                    and staged_path
                ):
                    try:
                        file_stager.restore(source_id=source_id, staged_path=staged_path)
                    except Exception:
                        # Restore failure must not mask the original failure.
                        pass
                if isinstance(exc, RepresentationMissing):
                    raise
                # Surface the failure as a state transition rather than re-raising;
                # callers can read the failed/retryable state and the failure event.
                return SourceRetirementState.FAILED_RETRYABLE

        return target_state

    # --------------------------------------------------------------- resume
    def resume(
        self,
        *,
        operation_id: str,
        file_stager: Any | None = None,
        metadata_stage: Callable[[], Any] | None = None,
        artifact_stage: Callable[[], Any] | None = None,
    ) -> SourceRetirementState:
        """Resume from the current state and advance to `completed`.

        Cross-store recovery: if the operation stopped at an intermediate
        state, resume picks the next forward stage and runs it. The
        implementation advances forward until either `completed` or a
        `failed/*` state is reached.
        """
        forward_chain = [
            SourceRetirementState.FILE_STAGED,
            SourceRetirementState.METADATA_TOMBSTONED,
            SourceRetirementState.ARTIFACTS_TOMBSTONED,
            SourceRetirementState.COMPLETED,
        ]
        current = self._current_state(operation_id)
        if current in _TERMINAL:
            return current
        if current is SourceRetirementState.FAILED_RETRYABLE:
            with self._operation_lock():
                row = self._store._conn.execute(
                    hosted.sql_query("SELECT failure_stage FROM source_retirement_operations "
                    "WHERE operation_id = ?", 'SELECT failure_stage FROM source_retirement_operations WHERE operation_id = %s'),
                    (operation_id,),
                ).fetchone()
                try:
                    reopened = SourceRetirementState(str(row["failure_stage"]))
                except (TypeError, ValueError) as exc:
                    raise SourceRetirementOrchestratorError(
                        "retryable operation has no deterministic failure checkpoint"
                    ) from exc
                if reopened not in {
                    SourceRetirementState.REQUESTED,
                    SourceRetirementState.FILE_STAGED,
                    SourceRetirementState.METADATA_TOMBSTONED,
                    SourceRetirementState.ARTIFACTS_TOMBSTONED,
                }:
                    raise SourceRetirementOrchestratorError(
                        "retryable operation has an invalid failure checkpoint"
                    )
                self._store._conn.execute(
                    hosted.sql_query("UPDATE source_retirement_operations SET state = ?, "
                    "failure_reason = NULL, failure_stage = NULL, updated_at = ? "
                    "WHERE operation_id = ?", 'UPDATE source_retirement_operations SET state = %s, failure_reason = NULL, failure_stage = NULL, updated_at = %s WHERE operation_id = %s'),
                    (reopened.value, _utc_now(), operation_id),
                )
                self._store._conn.commit()
            current = reopened
        for stage in forward_chain:
            if _is_at_or_past(current, stage):
                continue
            current = self.advance(
                operation_id=operation_id,
                target_state=stage,
                file_stager=file_stager,
                metadata_stage=metadata_stage,
                artifact_stage=artifact_stage,
            )
        return current

    # ------------------------------------------------------------- helpers
    def _count_authoritative_descendants(self, source_id: str) -> int:
        """Plan A D5: count lineage edges for source_id whose authority_state
        is 'authoritative'. The lineage-edge table carries the authority_state
        so the check is single-table (no cross-store join).
        """
        row = self._store._conn.execute(
            hosted.sql_query("SELECT COUNT(*) AS c FROM contextgraph_artifact_origins "
            "WHERE source_id = ? AND authority_state = 'authoritative'", "SELECT COUNT(*) AS c FROM contextgraph_artifact_origins WHERE source_id = %s AND authority_state = 'authoritative'"),
            (source_id,),
        ).fetchone()
        return int(row["c"])

    def _resolve_source_path(self, source_id: str) -> str:
        """Resolve the workspace-relative path of the source file.

        Falls back to  when no row is found; this is the worst
        case (the stager will treat  as the basename).
        """
        row = self._store._conn.execute(
            hosted.sql_query("SELECT source_ref, source_metadata_json, workspace_id "
            "FROM knowledge_builder_sources "
            "WHERE source_id = ?", 'SELECT source_ref, source_metadata_json, workspace_id FROM knowledge_builder_sources WHERE source_id = %s'),
            (source_id,),
        ).fetchone()
        if row is None:
            return source_id
        try:
            metadata = json.loads(row["source_metadata_json"] or "{}")
        except (TypeError, ValueError):
            metadata = {}
        source_path = str(metadata.get("original_path") or row["source_ref"])
        if source_path.startswith(KNOWLEDGE_UPLOAD_REF_PREFIX):
            source_path = decode_governed_upload_ref(
                source_path,
                expected_workspace_id=str(row["workspace_id"] or ""),
            )
        return source_path

    def _source_scope(
        self, source_id: str, *, required: bool = True
    ) -> tuple[str, str] | None:
        row = self._store._conn.execute(
            hosted.sql_query("SELECT organization_id, workspace_id FROM knowledge_builder_sources "
            "WHERE source_id = ?", 'SELECT organization_id, workspace_id FROM knowledge_builder_sources WHERE source_id = %s'),
            (source_id,),
        ).fetchone()
        if row is None or row["workspace_id"] is None:
            if not required:
                return None
            raise SourceRetirementOrchestratorError(
                f"source {source_id} is not bound to a governed workspace"
            )
        return str(row["organization_id"]), str(row["workspace_id"])

    def _store_org_id(self) -> str:
        """Plan A M3: the org_id of the active connection (single-tenant per session)."""
        row = self._store._conn.execute(
            hosted.sql_query("SELECT organization_id FROM knowledge_builder_sources LIMIT 1", 'SELECT organization_id FROM knowledge_builder_sources LIMIT 1')
        ).fetchone()
        return row["organization_id"] if row else ""

    def _store_workspace_id(self) -> str:
        row = self._store._conn.execute(
            hosted.sql_query("SELECT workspace_id FROM knowledge_builder_sources LIMIT 1", 'SELECT workspace_id FROM knowledge_builder_sources LIMIT 1')
        ).fetchone()
        return row["workspace_id"] if row else ""

    def _current_state(self, operation_id: str) -> SourceRetirementState:
        row = self._store._conn.execute(
            hosted.sql_query("SELECT state FROM source_retirement_operations WHERE operation_id = ?", 'SELECT state FROM source_retirement_operations WHERE operation_id = %s'),
            (operation_id,),
        ).fetchone()
        if row is None:
            raise SourceRetirementOrchestratorError(
                f"unknown operation_id: {operation_id}"
            )
        return SourceRetirementState(row["state"])


def _is_at_or_past(current: SourceRetirementState, target: SourceRetirementState) -> bool:
    forward_chain = [
        SourceRetirementState.REQUESTED,
        SourceRetirementState.FILE_STAGED,
        SourceRetirementState.METADATA_TOMBSTONED,
        SourceRetirementState.ARTIFACTS_TOMBSTONED,
        SourceRetirementState.COMPLETED,
    ]
    # Terminal failure states are not "past" forward stages.
    if current in {SourceRetirementState.FAILED_RETRYABLE,
                    SourceRetirementState.FAILED_BLOCKED_AUTHORITATIVE}:
        return False
    try:
        return forward_chain.index(current) >= forward_chain.index(target)
    except ValueError:
        return False
