from __future__ import annotations

from storage import hosted_postgres as hosted

import json
import sqlite3
from pathlib import Path
from typing import Any

from eos.db import get_conn

from .models import (
    EffectiveContextCheckpoint,
    EffectiveContextEvent,
    canonical_json_bytes,
    sha256_hex,
    utc_now_millis,
)


CHECKPOINT_NOT_FOUND = "CHECKPOINT_NOT_FOUND"
CHECKPOINT_AUTHORITY_PROOF_MISSING = "CHECKPOINT_AUTHORITY_PROOF_MISSING"
EVENT_HASH_CHAIN_MISMATCH = "EVENT_HASH_CHAIN_MISMATCH"
ECC_SCOPE_REQUIRED = "ECC_SCOPE_REQUIRED"


class EffectiveContextStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path is not None else None
        self.ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = get_conn(self._db_path)
        if not hosted.is_postgres(conn):
            conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            if hosted.is_postgres(conn):
                hosted.require_tables(conn, 'effective_context_checkpoints', 'effective_context_eval_results', 'effective_context_events', 'effective_context_heads', 'effective_context_idempotency', 'effective_context_payloads')
                return
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS effective_context_events (
                  event_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  session_id TEXT NOT NULL,
                  sequence INTEGER NOT NULL,
                  occurred_at TEXT NOT NULL,
                  actor_ref TEXT NOT NULL,
                  event_type TEXT NOT NULL,
                  state_effect TEXT NOT NULL,
                  input_refs_json TEXT NOT NULL,
                  output_refs_json TEXT NOT NULL,
                  cause_event_ids_json TEXT NOT NULL,
                  authority_context_ref TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  prior_event_hash TEXT,
                  event_json TEXT NOT NULL,
                  UNIQUE(organization_id, workspace_id, session_id, sequence)
                );
                CREATE INDEX IF NOT EXISTS idx_effective_context_events_session_type
                  ON effective_context_events(organization_id, workspace_id, session_id, event_type);
                CREATE INDEX IF NOT EXISTS idx_effective_context_events_occurred
                  ON effective_context_events(organization_id, workspace_id, occurred_at);

                CREATE TABLE IF NOT EXISTS effective_context_checkpoints (
                  checkpoint_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  session_id TEXT NOT NULL,
                  boundary_message_id TEXT NOT NULL,
                  boundary_event_sequence INTEGER NOT NULL,
                  knowledge_state_ref TEXT NOT NULL,
                  active_contract_ref TEXT NOT NULL,
                  authority_pins_ref TEXT NOT NULL,
                  transcript_snapshot_ref TEXT NOT NULL,
                  trajectory_from_sequence INTEGER NOT NULL,
                  trajectory_to_sequence INTEGER NOT NULL,
                  parent_checkpoint_id TEXT,
                  lineage_root_checkpoint_id TEXT NOT NULL,
                  delta_ref TEXT,
                  checkpoint_hash TEXT NOT NULL UNIQUE,
                  verification_status TEXT NOT NULL,
                  verification_reason TEXT,
                  created_at TEXT NOT NULL,
                  checkpoint_json TEXT NOT NULL,
                  UNIQUE(organization_id, workspace_id, session_id, boundary_message_id)
                );
                CREATE INDEX IF NOT EXISTS idx_effective_context_checkpoints_boundary
                  ON effective_context_checkpoints(organization_id, workspace_id, session_id, boundary_event_sequence);
                CREATE INDEX IF NOT EXISTS idx_effective_context_checkpoints_parent
                  ON effective_context_checkpoints(organization_id, workspace_id, parent_checkpoint_id);

                CREATE TABLE IF NOT EXISTS effective_context_payloads (
                  payload_ref TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  payload_type TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  byte_length INTEGER NOT NULL,
                  uri TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  UNIQUE(organization_id, workspace_id, content_hash)
                );

                CREATE TABLE IF NOT EXISTS effective_context_idempotency (
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  operation_type TEXT NOT NULL,
                  idempotency_key TEXT NOT NULL,
                  request_hash TEXT NOT NULL,
                  result_ref TEXT NOT NULL,
                  expires_at TEXT,
                  created_at TEXT NOT NULL,
                  PRIMARY KEY(organization_id, workspace_id, operation_type, idempotency_key)
                );
                CREATE INDEX IF NOT EXISTS idx_effective_context_idempotency_expires
                  ON effective_context_idempotency(expires_at);

                CREATE TABLE IF NOT EXISTS effective_context_heads (
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  session_id TEXT NOT NULL,
                  latest_sequence INTEGER NOT NULL,
                  prior_hash TEXT,
                  latest_checkpoint_id TEXT,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY(organization_id, workspace_id, session_id)
                );
                CREATE TABLE IF NOT EXISTS effective_context_eval_results (
                  result_id TEXT PRIMARY KEY,
                  organization_id TEXT NOT NULL,
                  workspace_id TEXT NOT NULL,
                  session_id TEXT NOT NULL,
                  checkpoint_id TEXT,
                  result_json TEXT NOT NULL,
                  result_hash TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_effective_context_eval_results_session
                  ON effective_context_eval_results(organization_id, workspace_id, session_id, created_at);
                """
            )
            columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(effective_context_checkpoints)").fetchall()
            }
            if "verification_reason" not in columns:
                conn.execute("ALTER TABLE effective_context_checkpoints ADD COLUMN verification_reason TEXT")

    def append_event(self, event: EffectiveContextEvent) -> EffectiveContextEvent:
        with self._connect() as conn:
            hosted.begin_write(conn)
            head = conn.execute(
                hosted.sql_query("""
                SELECT latest_sequence, prior_hash
                FROM effective_context_heads
                WHERE organization_id=? AND workspace_id=? AND session_id=?
                """, """
                SELECT latest_sequence, prior_hash
                FROM effective_context_heads
                WHERE organization_id=%s AND workspace_id=%s AND session_id=%s
                """),
                (event.organization_id, event.workspace_id, event.session_id),
            ).fetchone()
            latest_sequence = int(head["latest_sequence"]) if head else 0
            prior_hash = head["prior_hash"] if head else None
            sequence = latest_sequence + 1
            event = event.model_copy(update={"sequence": sequence, "prior_event_hash": prior_hash}).with_hash()
            self._insert_event(conn, event)
            conn.execute(
                hosted.sql_query("""
                INSERT INTO effective_context_heads
                  (organization_id, workspace_id, session_id, latest_sequence, prior_hash, latest_checkpoint_id, updated_at)
                VALUES (?, ?, ?, ?, ?, NULL, ?)
                ON CONFLICT(organization_id, workspace_id, session_id)
                DO UPDATE SET latest_sequence=excluded.latest_sequence,
                              prior_hash=excluded.prior_hash,
                              updated_at=excluded.updated_at
                """, """
                INSERT INTO effective_context_heads
                  (organization_id, workspace_id, session_id, latest_sequence, prior_hash, latest_checkpoint_id, updated_at)
                VALUES (%s, %s, %s, %s, %s, NULL, %s)
                ON CONFLICT(organization_id, workspace_id, session_id)
                DO UPDATE SET latest_sequence=excluded.latest_sequence,
                              prior_hash=excluded.prior_hash,
                              updated_at=excluded.updated_at
                """),
                (event.organization_id, event.workspace_id, event.session_id, event.sequence, event.content_hash, utc_now_millis()),
            )
            conn.commit()
            return event

    def _insert_event(self, conn: sqlite3.Connection, event: EffectiveContextEvent) -> None:
        if event.content_hash != sha256_hex(event.hash_payload()):
            raise ValueError(EVENT_HASH_CHAIN_MISMATCH)
        data = event.model_dump(mode="json")
        conn.execute(
            hosted.sql_query("""
            INSERT INTO effective_context_events
              (event_id, organization_id, workspace_id, session_id, sequence, occurred_at,
               actor_ref, event_type, state_effect, input_refs_json, output_refs_json,
               cause_event_ids_json, authority_context_ref, content_hash, prior_event_hash, event_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, """
            INSERT INTO effective_context_events
              (event_id, organization_id, workspace_id, session_id, sequence, occurred_at,
               actor_ref, event_type, state_effect, input_refs_json, output_refs_json,
               cause_event_ids_json, authority_context_ref, content_hash, prior_event_hash, event_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """),
            (
                event.event_id,
                event.organization_id,
                event.workspace_id,
                event.session_id,
                event.sequence,
                event.occurred_at,
                event.actor_ref,
                event.event_type.value,
                event.state_effect.value,
                json.dumps(data["input_refs"], sort_keys=True, separators=(",", ":")),
                json.dumps(data["output_refs"], sort_keys=True, separators=(",", ":")),
                json.dumps(data["cause_event_ids"], sort_keys=True, separators=(",", ":")),
                event.authority_context_ref,
                event.content_hash,
                event.prior_event_hash,
                json.dumps(data, sort_keys=True, separators=(",", ":")),
            ),
        )

    def list_events(self, organization_id: str, workspace_id: str, session_id: str) -> list[EffectiveContextEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                hosted.sql_query("""
                SELECT event_json
                FROM effective_context_events
                WHERE organization_id=? AND workspace_id=? AND session_id=?
                ORDER BY sequence
                """, """
                SELECT event_json
                FROM effective_context_events
                WHERE organization_id=%s AND workspace_id=%s AND session_id=%s
                ORDER BY sequence
                """),
                (organization_id, workspace_id, session_id),
            ).fetchall()
        return [EffectiveContextEvent.model_validate_json(row["event_json"]) for row in rows]

    def verify_event_chain(
        self,
        organization_id: str,
        workspace_id: str,
        session_id: str,
        *,
        through_sequence: int | None = None,
    ) -> bool:
        prior_hash = None
        expected_sequence = 1
        sequence_filter = ""
        params: tuple[Any, ...] = (organization_id, workspace_id, session_id)
        if through_sequence is not None:
            sequence_filter = hosted.sql_query("AND sequence <= ?", 'AND sequence <= %s')
            params = (*params, through_sequence)
        with self._connect() as conn:
            rows = conn.execute(
                hosted.sql_query(f"""
                SELECT event_json, content_hash, prior_event_hash, sequence,
                       organization_id, workspace_id, session_id, occurred_at,
                       actor_ref, event_type, state_effect, input_refs_json,
                       output_refs_json, cause_event_ids_json, authority_context_ref
                FROM effective_context_events
                WHERE organization_id=? AND workspace_id=? AND session_id=?
                  {sequence_filter}
                ORDER BY sequence
                """, f'\n                SELECT event_json, content_hash, prior_event_hash, sequence,\n                       organization_id, workspace_id, session_id, occurred_at,\n                       actor_ref, event_type, state_effect, input_refs_json,\n                       output_refs_json, cause_event_ids_json, authority_context_ref\n                FROM effective_context_events\n                WHERE organization_id=%s AND workspace_id=%s AND session_id=%s\n                  {sequence_filter}\n                ORDER BY sequence\n                '),
                params,
            ).fetchall()
        for row in rows:
            event = EffectiveContextEvent.model_validate_json(row["event_json"])
            if event.sequence != expected_sequence:
                return False
            if row["sequence"] != expected_sequence:
                return False
            if event.prior_event_hash != prior_hash:
                return False
            if row["prior_event_hash"] != prior_hash:
                return False
            recomputed_hash = sha256_hex(event.hash_payload())
            if event.content_hash != recomputed_hash:
                return False
            if row["content_hash"] != recomputed_hash:
                return False
            data = event.model_dump(mode="json")
            if any(
                (
                    row["organization_id"] != event.organization_id,
                    row["workspace_id"] != event.workspace_id,
                    row["session_id"] != event.session_id,
                    row["occurred_at"] != event.occurred_at,
                    row["actor_ref"] != event.actor_ref,
                    row["event_type"] != event.event_type.value,
                    row["state_effect"] != event.state_effect.value,
                    row["input_refs_json"] != json.dumps(data["input_refs"], sort_keys=True, separators=(",", ":")),
                    row["output_refs_json"] != json.dumps(data["output_refs"], sort_keys=True, separators=(",", ":")),
                    row["cause_event_ids_json"] != json.dumps(data["cause_event_ids"], sort_keys=True, separators=(",", ":")),
                    row["authority_context_ref"] != event.authority_context_ref,
                )
            ):
                return False
            prior_hash = event.content_hash
            expected_sequence += 1
        return True

    def put_payload(self, organization_id: str, workspace_id: str, payload_type: str, payload: dict[str, Any]) -> str:
        payload_hash = sha256_hex(payload)
        payload_ref = f"payload:{organization_id}:{workspace_id}:{payload_hash}"
        payload_bytes = canonical_json_bytes(payload)
        with self._connect() as conn:
            conn.execute(
                hosted.sql_query("""
                INSERT OR IGNORE INTO effective_context_payloads
                  (payload_ref, organization_id, workspace_id, payload_type, content_hash, byte_length, uri, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO effective_context_payloads
                  (payload_ref, organization_id, workspace_id, payload_type, content_hash, byte_length, uri, payload_json, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING"""),
                (
                    payload_ref,
                    organization_id,
                    workspace_id,
                    payload_type,
                    payload_hash,
                    len(payload_bytes),
                    f"workspaces/{workspace_id}/.semantier/effective_context/blobs/sha256/{payload_hash}",
                    payload_bytes.decode("utf-8"),
                    utc_now_millis(),
                ),
            )
        return payload_ref

    def get_payload(self, organization_id: str, workspace_id: str, payload_ref: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                hosted.sql_query("""
                SELECT payload_json
                FROM effective_context_payloads
                WHERE organization_id=? AND workspace_id=? AND payload_ref=?
                """, """
                SELECT payload_json
                FROM effective_context_payloads
                WHERE organization_id=%s AND workspace_id=%s AND payload_ref=%s
                """),
                (organization_id, workspace_id, payload_ref),
            ).fetchone()
        if row is None:
            raise KeyError(CHECKPOINT_NOT_FOUND)
        return json.loads(row["payload_json"])

    def insert_checkpoint(self, checkpoint: EffectiveContextCheckpoint, verification_reason: str | None = None) -> EffectiveContextCheckpoint:
        checkpoint = checkpoint.with_hash()
        data = checkpoint.model_dump(mode="json")
        with self._connect() as conn:
            hosted.begin_write(conn)
            conn.execute(
                hosted.sql_query("""
                INSERT INTO effective_context_checkpoints
                  (checkpoint_id, organization_id, workspace_id, session_id, boundary_message_id,
                   boundary_event_sequence, knowledge_state_ref, active_contract_ref, authority_pins_ref,
                   transcript_snapshot_ref, trajectory_from_sequence, trajectory_to_sequence,
                   parent_checkpoint_id, lineage_root_checkpoint_id, delta_ref, checkpoint_hash,
                   verification_status, verification_reason, created_at, checkpoint_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO effective_context_checkpoints
                  (checkpoint_id, organization_id, workspace_id, session_id, boundary_message_id,
                   boundary_event_sequence, knowledge_state_ref, active_contract_ref, authority_pins_ref,
                   transcript_snapshot_ref, trajectory_from_sequence, trajectory_to_sequence,
                   parent_checkpoint_id, lineage_root_checkpoint_id, delta_ref, checkpoint_hash,
                   verification_status, verification_reason, created_at, checkpoint_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """),
                (
                    checkpoint.checkpoint_id,
                    checkpoint.organization_id,
                    checkpoint.workspace_id,
                    checkpoint.session_id,
                    checkpoint.boundary_message_id,
                    checkpoint.boundary_event_sequence,
                    checkpoint.knowledge_state_ref,
                    checkpoint.active_contract_ref,
                    checkpoint.authority_pins_ref,
                    checkpoint.transcript_snapshot_ref,
                    checkpoint.trajectory_from_sequence,
                    checkpoint.trajectory_to_sequence,
                    checkpoint.parent_checkpoint_id,
                    checkpoint.lineage_root_checkpoint_id,
                    checkpoint.delta_ref,
                    checkpoint.checkpoint_hash,
                    checkpoint.verification_status,
                    verification_reason,
                    checkpoint.created_at,
                    json.dumps(data, sort_keys=True, separators=(",", ":")),
                ),
            )
            conn.execute(
                hosted.sql_query("""
                UPDATE effective_context_heads
                SET latest_checkpoint_id=?, updated_at=?
                WHERE organization_id=? AND workspace_id=? AND session_id=?
                """, """
                UPDATE effective_context_heads
                SET latest_checkpoint_id=%s, updated_at=%s
                WHERE organization_id=%s AND workspace_id=%s AND session_id=%s
                """),
                (checkpoint.checkpoint_id, utc_now_millis(), checkpoint.organization_id, checkpoint.workspace_id, checkpoint.session_id),
            )
            conn.commit()
        return checkpoint

    def get_checkpoint(self, organization_id: str, workspace_id: str, checkpoint_id: str) -> EffectiveContextCheckpoint:
        with self._connect() as conn:
            row = conn.execute(
                hosted.sql_query("""
                SELECT checkpoint_json
                FROM effective_context_checkpoints
                WHERE organization_id=? AND workspace_id=? AND checkpoint_id=?
                """, """
                SELECT checkpoint_json
                FROM effective_context_checkpoints
                WHERE organization_id=%s AND workspace_id=%s AND checkpoint_id=%s
                """),
                (organization_id, workspace_id, checkpoint_id),
            ).fetchone()
        if row is None:
            raise KeyError(CHECKPOINT_NOT_FOUND)
        return EffectiveContextCheckpoint.model_validate_json(row["checkpoint_json"])

    def list_checkpoints(
        self,
        organization_id: str,
        workspace_id: str,
        session_id: str,
    ) -> list[EffectiveContextCheckpoint]:
        """Return the immutable checkpoint lineage in boundary order."""
        with self._connect() as conn:
            rows = conn.execute(
                hosted.sql_query("""
                SELECT checkpoint_json
                FROM effective_context_checkpoints
                WHERE organization_id=? AND workspace_id=? AND session_id=?
                ORDER BY boundary_event_sequence, created_at, checkpoint_id
                """, """
                SELECT checkpoint_json
                FROM effective_context_checkpoints
                WHERE organization_id=%s AND workspace_id=%s AND session_id=%s
                ORDER BY boundary_event_sequence, created_at, checkpoint_id
                """),
                (organization_id, workspace_id, session_id),
            ).fetchall()
        return [
            EffectiveContextCheckpoint.model_validate_json(row["checkpoint_json"])
            for row in rows
        ]

    def put_eval_result(
        self,
        organization_id: str,
        workspace_id: str,
        session_id: str,
        result: dict[str, Any],
    ) -> str:
        result_id = str(result["result_id"])
        result_json = json.dumps(result, sort_keys=True, separators=(",", ":"))
        result_hash = sha256_hex(result)
        with self._connect() as conn:
            conn.execute(
                hosted.sql_query("""
                INSERT OR IGNORE INTO effective_context_eval_results
                  (result_id, organization_id, workspace_id, session_id, checkpoint_id,
                   result_json, result_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, """
                INSERT INTO effective_context_eval_results
                  (result_id, organization_id, workspace_id, session_id, checkpoint_id,
                   result_json, result_hash, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING"""),
                (
                    result_id,
                    organization_id,
                    workspace_id,
                    session_id,
                    result.get("checkpoint_id"),
                    result_json,
                    result_hash,
                    utc_now_millis(),
                ),
            )
        return result_id

    def get_latest_eval_result(
        self,
        organization_id: str,
        workspace_id: str,
        session_id: str,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                hosted.sql_query("""
                SELECT result_json
                FROM effective_context_eval_results
                WHERE organization_id=? AND workspace_id=? AND session_id=?
                ORDER BY created_at DESC, result_id DESC
                LIMIT 1
                """, """
                SELECT result_json
                FROM effective_context_eval_results
                WHERE organization_id=%s AND workspace_id=%s AND session_id=%s
                ORDER BY created_at DESC, result_id DESC
                LIMIT 1
                """),
                (organization_id, workspace_id, session_id),
            ).fetchone()
        return json.loads(row["result_json"]) if row else None

    def list_eval_results(
        self,
        organization_id: str,
        workspace_id: str,
        session_id: str,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                hosted.sql_query("""
                SELECT result_json
                FROM effective_context_eval_results
                WHERE organization_id=? AND workspace_id=? AND session_id=?
                ORDER BY created_at, result_id
                """, """
                SELECT result_json
                FROM effective_context_eval_results
                WHERE organization_id=%s AND workspace_id=%s AND session_id=%s
                ORDER BY created_at, result_id
                """),
                (organization_id, workspace_id, session_id),
            ).fetchall()
        return [json.loads(row["result_json"]) for row in rows]

    def get_checkpoint_by_boundary(
        self, organization_id: str, workspace_id: str, session_id: str, boundary_message_id: str
    ) -> EffectiveContextCheckpoint | None:
        with self._connect() as conn:
            row = conn.execute(
                hosted.sql_query("""
                SELECT checkpoint_json
                FROM effective_context_checkpoints
                WHERE organization_id=? AND workspace_id=? AND session_id=? AND boundary_message_id=?
                """, """
                SELECT checkpoint_json
                FROM effective_context_checkpoints
                WHERE organization_id=%s AND workspace_id=%s AND session_id=%s AND boundary_message_id=%s
                """),
                (organization_id, workspace_id, session_id, boundary_message_id),
            ).fetchone()
        return EffectiveContextCheckpoint.model_validate_json(row["checkpoint_json"]) if row else None

    def count_checkpoints(self, organization_id: str, workspace_id: str, session_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                hosted.sql_query("""
                SELECT COUNT(*) AS count
                FROM effective_context_checkpoints
                WHERE organization_id=? AND workspace_id=? AND session_id=?
                """, """
                SELECT COUNT(*) AS count
                FROM effective_context_checkpoints
                WHERE organization_id=%s AND workspace_id=%s AND session_id=%s
                """),
                (organization_id, workspace_id, session_id),
            ).fetchone()
        return int(row["count"])
