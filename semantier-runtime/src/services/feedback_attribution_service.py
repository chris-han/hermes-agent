"""Feedback attribution for graph-schema learning loops."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from eos.artifact_hashing import canonical_json, content_hash
from storage.semantic_feedback_event_store import SemanticFeedbackEventStore
from storage.hosted_postgres import enabled, HostedStorageUnavailable


FEEDBACK_ATTRIBUTION_SCHEMA_VERSION = "feedback_attribution.v1"
AttributionReviewState = Literal["proposed", "human_accepted", "human_edited", "human_rejected"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _loads(value: str) -> Any:
    return json.loads(value) if value else None


@dataclass(frozen=True)
class FeedbackAttribution:
    schema_version: str
    attribution_id: str
    feedback_ref: str
    generated_by: str
    signals: list[dict[str, Any]]
    confidence: float
    review_state: AttributionReviewState
    reviewed_by: str | None
    reviewed_at: str | None
    attribution_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "attribution_id": self.attribution_id,
            "feedback_ref": self.feedback_ref,
            "generated_by": self.generated_by,
            "signals": self.signals,
            "confidence": self.confidence,
            "review_state": self.review_state,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
            "attribution_hash": self.attribution_hash,
        }


class FeedbackAttributionService:
    """Persist proposed feedback attributions and human review state."""

    _process_locks: dict[Path, threading.RLock] = {}
    _process_locks_guard = threading.Lock()

    def __init__(self, workspace_root: str | Path, *, workspace_id: str) -> None:
        if enabled():
            raise HostedStorageUnavailable('Feedback attribution is outside the selected hosted stores; explicit migration is required')
        if not workspace_id:
            raise ValueError("workspace_id is required")
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_id = workspace_id
        self.feedback_store = SemanticFeedbackEventStore(workspace_root, workspace_id=workspace_id)
        self.root = self.workspace_root / ".semantier" / "feedback-attribution"
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "attribution.sqlite3"
        self._initialize()

    @property
    def _lock(self) -> threading.RLock:
        with self._process_locks_guard:
            return self._process_locks.setdefault(self.db_path, threading.RLock())

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_attributions (
                    attribution_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    feedback_ref TEXT NOT NULL,
                    generated_by TEXT NOT NULL CHECK (generated_by IN ('deterministic', 'llm_assisted', 'hybrid')),
                    signals_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    review_state TEXT NOT NULL CHECK (review_state IN ('proposed', 'human_accepted', 'human_edited', 'human_rejected')),
                    reviewed_by TEXT,
                    reviewed_at TEXT,
                    attribution_hash TEXT NOT NULL
                )
                """
            )

    def propose_attribution(
        self,
        *,
        feedback_ref: str,
        signals: list[dict[str, Any]],
        confidence: float,
        generated_by: str = "deterministic",
    ) -> FeedbackAttribution:
        self.feedback_store.get_event(feedback_ref)
        if generated_by not in {"deterministic", "llm_assisted", "hybrid"}:
            raise ValueError("generated_by must be deterministic, llm_assisted, or hybrid")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if not signals:
            raise ValueError("feedback attribution requires at least one signal")
        for signal in signals:
            if signal.get("target_type") not in {
                "lexical_mapping",
                "classification",
                "scope_hierarchy",
                "inheritance_policy",
                "output_suggestion",
                "matcher",
                "semantic_discovery_decision",
            }:
                raise ValueError("unsupported attribution target_type")
            if signal.get("signal") not in {"positive", "negative", "inconclusive"}:
                raise ValueError("unsupported attribution signal")
        payload = {
            "schema_version": FEEDBACK_ATTRIBUTION_SCHEMA_VERSION,
            "attribution_id": f"fattr_{uuid.uuid4().hex}",
            "feedback_ref": feedback_ref,
            "generated_by": generated_by,
            "signals": signals,
            "confidence": confidence,
            "review_state": "proposed",
            "reviewed_by": None,
            "reviewed_at": None,
        }
        payload["attribution_hash"] = f"sha256:{content_hash(payload)}"
        attribution = FeedbackAttribution(**payload)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO feedback_attributions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attribution.attribution_id,
                    self.workspace_id,
                    attribution.schema_version,
                    attribution.feedback_ref,
                    attribution.generated_by,
                    canonical_json(attribution.signals),
                    attribution.confidence,
                    attribution.review_state,
                    attribution.reviewed_by,
                    attribution.reviewed_at,
                    attribution.attribution_hash,
                ),
            )
        return attribution

    def record_human_review(
        self,
        attribution_id: str,
        *,
        review_state: AttributionReviewState,
        reviewed_by: str,
    ) -> FeedbackAttribution:
        if review_state not in {"human_accepted", "human_edited", "human_rejected"}:
            raise ValueError("attribution review must be human_accepted, human_edited, or human_rejected")
        if not reviewed_by:
            raise ValueError("reviewed_by is required")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM feedback_attributions WHERE workspace_id = ? AND attribution_id = ?",
                (self.workspace_id, attribution_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"feedback attribution not found: {attribution_id}")
            connection.execute(
                """
                UPDATE feedback_attributions
                SET review_state = ?, reviewed_by = ?, reviewed_at = ?
                WHERE workspace_id = ? AND attribution_id = ?
                """,
                (review_state, reviewed_by, _utc_now(), self.workspace_id, attribution_id),
            )
        return self.get_attribution(attribution_id)

    def get_attribution(self, attribution_id: str) -> FeedbackAttribution:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM feedback_attributions WHERE workspace_id = ? AND attribution_id = ?",
                (self.workspace_id, attribution_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"feedback attribution not found: {attribution_id}")
        return FeedbackAttribution(
            schema_version=row["schema_version"],
            attribution_id=row["attribution_id"],
            feedback_ref=row["feedback_ref"],
            generated_by=row["generated_by"],
            signals=_loads(row["signals_json"]),
            confidence=row["confidence"],
            review_state=row["review_state"],
            reviewed_by=row["reviewed_by"],
            reviewed_at=row["reviewed_at"],
            attribution_hash=row["attribution_hash"],
        )

