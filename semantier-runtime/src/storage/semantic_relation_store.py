from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from eos.artifact_hashing import canonical_json, content_hash as compute_content_hash
from eos.db import init_db
from runtime_paths import sqlite_db_path
from storage.hosted_postgres import enabled,connect_authority,require_tables,sql_query,begin_write
from storage.transaction_scope import transaction_scope,StoreMutationLock

RELATION_TYPES = {
    "synonym_of",
    "variant_of",
    "projects_to",
    "not_same_as",
    "allowed_context_for",
    "prohibited_context_for",
    "exception_to",
    "conflicts_with",
    "candidate_semantic_neighborhood",
}


def _db_path() -> Path:
    return Path(os.environ.get("SEMANTIER_EOS_DB_PATH", str(sqlite_db_path("eos.db", ".semantier/eos.db"))))


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_json(value: Any) -> str:
    return canonical_json(value if value is not None else {})


def _loads_json(value: str) -> Any:
    return json.loads(value) if value else None


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    if "source_anchor_refs_json" in payload:
        payload["source_anchor_refs"] = _loads_json(payload["source_anchor_refs_json"])
    if "event_json" in payload:
        payload["event"] = _loads_json(payload["event_json"])
    return payload


class SemanticRelationStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        path = Path(db_path) if db_path else _db_path()
        if enabled():
            self._conn=connect_authority("eos",path)
            require_tables(self._conn,'knowledge_builder_curation_events','semantic_relation_candidates')
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

    def create_candidate(
        self,
        *,
        discovery_run_id: str,
        from_node_ref: str,
        to_node_ref: str,
        relation_type: str,
        evidence_summary: str,
        source_anchor_refs: list[str],
        actor: str,
        source_relation_ref: str | None = None,
        reviewer_notes: str | None = None,
        curation_state: str = "CURATED",
    ) -> dict[str, Any]:
        normalized_type = relation_type.strip()
        if normalized_type not in RELATION_TYPES:
            raise ValueError("unsupported relation_type")
        state = curation_state.strip().upper()
        if state not in {"CURATED", "PROPOSED", "REJECTED"}:
            raise ValueError("curation_state must be CURATED, PROPOSED, or REJECTED")
        if not discovery_run_id.strip():
            raise ValueError("discovery_run_id is required")
        if not from_node_ref.strip() or not to_node_ref.strip():
            raise ValueError("from_node_ref and to_node_ref are required")
        candidate_id = f"src_{uuid4().hex}"
        now = _now_iso()
        row = {
            "semantic_relation_candidate_id": candidate_id,
            "source_relation_ref": source_relation_ref,
            "discovery_run_id": discovery_run_id,
            "from_node_ref": from_node_ref,
            "to_node_ref": to_node_ref,
            "relation_type": normalized_type,
            "curation_state": state,
            "evidence_summary": evidence_summary,
            "source_anchor_refs": source_anchor_refs,
            "reviewer_notes": reviewer_notes,
            "created_by": actor,
            "created_at": now,
        }
        row["content_hash"] = compute_content_hash(canonical_json(row))
        with self._lock:
            self._conn.execute(
                sql_query("""
                INSERT INTO semantic_relation_candidates (
                  semantic_relation_candidate_id, source_relation_ref, discovery_run_id,
                  from_node_ref, to_node_ref, relation_type, curation_state,
                  evidence_summary, source_anchor_refs_json, reviewer_notes,
                  created_by, created_at, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO semantic_relation_candidates (\n                  semantic_relation_candidate_id, source_relation_ref, discovery_run_id,\n                  from_node_ref, to_node_ref, relation_type, curation_state,\n                  evidence_summary, source_anchor_refs_json, reviewer_notes,\n                  created_by, created_at, content_hash\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    candidate_id,
                    source_relation_ref,
                    row["discovery_run_id"],
                    row["from_node_ref"],
                    row["to_node_ref"],
                    normalized_type,
                    state,
                    row["evidence_summary"],
                    _as_json(row["source_anchor_refs"]),
                    reviewer_notes,
                    actor,
                    now,
                    row["content_hash"],
                ),
            )
            self._record_event(
                discovery_run_id=row["discovery_run_id"],
                target_kind="relation",
                target_id=candidate_id,
                decision=state.lower(),
                event=row,
                actor=actor,
                created_at=now,
            )
            self._conn.commit()
        return row

    def _record_event(
        self,
        *,
        discovery_run_id: str,
        target_kind: str,
        target_id: str,
        decision: str,
        event: dict[str, Any],
        actor: str,
        created_at: str,
    ) -> dict[str, Any]:
        event_id = f"kbev_{uuid4().hex}"
        row = {
            "curation_event_id": event_id,
            "discovery_run_id": discovery_run_id,
            "target_kind": target_kind,
            "target_id": target_id,
            "decision": decision,
            "event": event,
            "actor": actor,
            "created_at": created_at,
        }
        row["content_hash"] = compute_content_hash(canonical_json(row))
        self._conn.execute(
            sql_query("""
            INSERT INTO knowledge_builder_curation_events (
              curation_event_id, discovery_run_id, target_kind, target_id,
              decision, event_json, actor, created_at, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, '\n            INSERT INTO knowledge_builder_curation_events (\n              curation_event_id, discovery_run_id, target_kind, target_id,\n              decision, event_json, actor, created_at, content_hash\n            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)\n            '),
            (
                event_id,
                discovery_run_id,
                target_kind,
                target_id,
                decision,
                _as_json(event),
                actor,
                created_at,
                row["content_hash"],
            ),
        )
        return row

    def get_relation_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        return _row_to_dict(
            self._conn.execute(
                sql_query("SELECT * FROM semantic_relation_candidates WHERE semantic_relation_candidate_id = ?", 'SELECT * FROM semantic_relation_candidates WHERE semantic_relation_candidate_id = %s'),
                (candidate_id,),
            ).fetchone()
        )

    def list_relation_candidates(self, candidate_ids: list[str]) -> list[dict[str, Any]]:
        if not candidate_ids:
            return []
        rows = []
        for candidate_id in candidate_ids:
            row = self.get_relation_candidate(candidate_id)
            if row is None:
                raise ValueError(f"semantic relation candidate not found: {candidate_id}")
            rows.append(row)
        return rows
