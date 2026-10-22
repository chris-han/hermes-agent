"""Append-only suppression and completion state for guidance triggers."""

from __future__ import annotations

from storage import hosted_postgres as hosted

import sqlite3
from pathlib import Path

from contracts.guidance_trigger import GuidanceTriggerV1


class GuidanceTriggerStateStore:
    def __init__(self, path: Path) -> None:
        if hosted.enabled():
            self._connection = hosted.connect_authority('eos', path)
            hosted.require_tables(self._connection, 'guidance_trigger_state_events')
            self._connection.commit()
            return
        self._connection = sqlite3.connect(path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS guidance_trigger_state_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                suppression_key TEXT NOT NULL,
                material_revision TEXT NOT NULL,
                state TEXT NOT NULL CHECK (state IN ('dismissed', 'resolved')),
                actor_or_resolution_ref TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def __enter__(self) -> "GuidanceTriggerStateStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def should_emit(self, trigger: GuidanceTriggerV1) -> bool:
        row = self._connection.execute(
            hosted.sql_query("""
            SELECT 1 FROM guidance_trigger_state_events
            WHERE suppression_key = ? AND material_revision = ?
            LIMIT 1
            """, '\n            SELECT 1 FROM guidance_trigger_state_events\n            WHERE suppression_key = %s AND material_revision = %s\n            LIMIT 1\n            '),
            (trigger.suppression_key, trigger.material_revision),
        ).fetchone()
        return row is None

    def dismiss(self, trigger: GuidanceTriggerV1, *, actor_ref: str) -> None:
        self._append(trigger, state="dismissed", ref=actor_ref)

    def resolve(self, trigger: GuidanceTriggerV1, *, resolution_ref: str) -> None:
        self._append(trigger, state="resolved", ref=resolution_ref)

    def _append(self, trigger: GuidanceTriggerV1, *, state: str, ref: str) -> None:
        if not ref:
            raise ValueError("state event requires an actor or resolution ref")
        self._connection.execute(
            hosted.sql_query("""
            INSERT INTO guidance_trigger_state_events
                (suppression_key, material_revision, state, actor_or_resolution_ref)
            VALUES (?, ?, ?, ?)
            """, '\n            INSERT INTO guidance_trigger_state_events\n                (suppression_key, material_revision, state, actor_or_resolution_ref)\n            VALUES (%s, %s, %s, %s)\n            '),
            (trigger.suppression_key, trigger.material_revision, state, ref),
        )
        self._connection.commit()


__all__ = ["GuidanceTriggerStateStore"]
