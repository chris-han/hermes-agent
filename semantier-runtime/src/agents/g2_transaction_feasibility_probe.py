from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EffectIntent:
    effect_id: str
    workflow_id: str
    effect_type: str
    semantic_round: int
    idempotency_key: str
    request_hash: str
    state: str = "READY"


def derive_effect_identity(*, workflow_id: str, effect_type: str, semantic_round: int, request_payload: bytes) -> EffectIntent:
    request_hash = hashlib.sha256(request_payload).hexdigest()
    stable = f"{workflow_id}|{effect_type}|{semantic_round}|{request_hash}"
    digest = hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24]
    return EffectIntent(
        effect_id=f"effect_{digest}",
        workflow_id=workflow_id,
        effect_type=effect_type,
        semantic_round=semantic_round,
        idempotency_key=f"{workflow_id}:{effect_type}:{semantic_round}:{request_hash}",
        request_hash=request_hash,
    )


class SameTransactionOutboxProbe:
    def __init__(self, path: str | Path) -> None:
        from storage.hosted_postgres import enabled, HostedStorageUnavailable
        if enabled():
            raise HostedStorageUnavailable('the synthetic transaction probe is local-only')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS participants (
                    workflow_id TEXT NOT NULL,
                    attendee_id TEXT NOT NULL,
                    followup_count INTEGER NOT NULL DEFAULT 0,
                    state TEXT NOT NULL DEFAULT 'unresolved',
                    PRIMARY KEY(workflow_id, attendee_id)
                );

                CREATE TABLE IF NOT EXISTS effect_attempts (
                    effect_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    effect_type TEXT NOT NULL,
                    semantic_round INTEGER NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    request_hash TEXT NOT NULL,
                    state TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO participants(workflow_id, attendee_id) VALUES ('n1', 'alice')"
            )

    def commit_reminder_intent(self, *, fail_after_domain_update: bool = False) -> EffectIntent:
        payload = b"reminder:n1:alice:round1"
        effect = derive_effect_identity(
            workflow_id="n1",
            effect_type="REMINDER_SEND",
            semantic_round=1,
            request_payload=payload,
        )
        conn = sqlite3.connect(self.path, isolation_level=None)
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT followup_count, state FROM participants WHERE workflow_id='n1' AND attendee_id='alice'"
            ).fetchone()
            assert row is not None
            followup_count, state = int(row[0]), str(row[1])
            if state != "unresolved" or followup_count >= 3:
                raise RuntimeError("reminder_not_eligible")
            conn.execute(
                "UPDATE participants SET followup_count=followup_count+1 WHERE workflow_id='n1' AND attendee_id='alice'"
            )
            if fail_after_domain_update:
                raise RuntimeError("synthetic_crash_before_effect_intent")
            conn.execute(
                """
                INSERT OR IGNORE INTO effect_attempts(
                    effect_id, workflow_id, effect_type, semantic_round,
                    idempotency_key, request_hash, state
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    effect.effect_id, effect.workflow_id, effect.effect_type,
                    effect.semantic_round, effect.idempotency_key,
                    effect.request_hash, effect.state,
                ),
            )
            conn.commit()
            return effect
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def snapshot(self):
        with sqlite3.connect(self.path) as conn:
            participant = conn.execute(
                "SELECT followup_count, state FROM participants WHERE workflow_id='n1' AND attendee_id='alice'"
            ).fetchone()
            effects = conn.execute(
                "SELECT effect_id, idempotency_key, state FROM effect_attempts ORDER BY effect_id"
            ).fetchall()
            return tuple(participant), [tuple(row) for row in effects]
