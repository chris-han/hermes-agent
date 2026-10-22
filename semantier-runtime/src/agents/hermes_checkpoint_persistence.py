"""Observe native Hermes transcript commits without modifying its runtime."""
from __future__ import annotations

from threading import RLock
from typing import Any, Callable

from hermes_state_backend import open_session_db


class CheckpointSessionDB:
    """Emit persisted row envelopes after a native storage commit.

    Native transcripts and ECC are separate stores. A failed sink leaves the transcript
    committed and is retained for the caller to reject a successful outcome.
    """

    def __init__(self, *, sink: Callable[[dict[str, Any]], None], db_path=None):
        self._store = open_session_db(db_path=db_path)
        self._checkpoint_sink = sink
        self._checkpoint_lock = RLock()
        self._checkpoint_failure: Exception | None = None
        self._checkpoint_delivered: set[tuple[str, int]] = set()

    def __getattr__(self, name):
        return getattr(self._store, name)

    def raise_checkpoint_failure(self) -> None:
        if self._checkpoint_failure is not None:
            raise RuntimeError("Effective-context checkpoint persistence failed") from self._checkpoint_failure

    def _emit_committed(self, session_id: str, previous_row_ids: set[int]) -> None:
        self.raise_checkpoint_failure()
        rows = self.get_messages(session_id)
        transcript = []
        for row in rows:
            row_id = row["id"]
            transcript.append({
                "message_id": str(row_id),
                "native_row_id": row_id,
                "role": row["role"],
                "content": row["content"],
                "tool_calls": row.get("tool_calls"),
                "tool_call_id": row.get("tool_call_id"),
                "timestamp": row.get("timestamp"),
            })
            identity = (session_id, row_id)
            if row_id in previous_row_ids or identity in self._checkpoint_delivered or row["role"] not in {"system", "user", "assistant", "tool"}:
                continue
            native_store_identity = getattr(self._store, "storage_identity", None)
            envelope = {
                **transcript[-1], "session_id": session_id,
                "native_store_identity": list(native_store_identity) if native_store_identity else None,
                "transcript_snapshot": {"messages": list(transcript),
                    **({"native_store_identity": list(native_store_identity)} if native_store_identity else {})},
            }
            try:
                self._checkpoint_sink(envelope)
            except Exception as exc:
                self._checkpoint_failure = exc
                raise
            self._checkpoint_delivered.add(identity)

    def append_messages_batch(self, session_id, messages, **kwargs):
        with self._checkpoint_lock:
            self.raise_checkpoint_failure()
            try:
                previous = {row["id"] for row in self.get_messages(session_id)}
                inserted = self._store.append_messages_batch(session_id, messages, **kwargs)
                self._emit_committed(session_id, previous)
                return inserted
            except Exception as exc:
                self._checkpoint_failure = self._checkpoint_failure or exc
                raise

    def append_message(self, session_id, *args, **kwargs):
        with self._checkpoint_lock:
            self.raise_checkpoint_failure()
            try:
                previous = {row["id"] for row in self.get_messages(session_id)}
                row_id = self._store.append_message(session_id, *args, **kwargs)
                self._emit_committed(session_id, previous)
                return row_id
            except Exception as exc:
                self._checkpoint_failure = self._checkpoint_failure or exc
                raise
