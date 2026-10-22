from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eos.artifact_hashing import canonical_json, content_hash as compute_content_hash
from eos.db import get_conn
from storage.hosted_postgres import sql_query


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class StoreBackedKnowledgeSourceCompilationService:
    def __init__(self, db_path: str | Path | None = None) -> None:
        from storage.knowledge_source_compilation_store import KnowledgeSourceCompilationStore

        self._db_path = Path(db_path) if db_path is not None else None
        self._store = KnowledgeSourceCompilationStore(db_path=db_path)

    def create_run(self, **kwargs: Any) -> dict[str, Any]:
        return self._store.create_run(**kwargs)

    def create_anchor(self, **kwargs: Any) -> dict[str, Any]:
        return self._store.create_anchor(**kwargs)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self._store.get_run(run_id)

    def list_anchors(self, run_id: str) -> list[dict[str, Any]]:
        return self._store.list_anchors(run_id)

    def persist_knowledge_builder_evidence(
        self,
        *,
        organization_id: str,
        workspace_id: str | None,
        knowledge_source_id: str,
        knowledge_source_compilation_run_id: str,
        candidate: dict[str, Any],
        source_anchor_refs: list[str],
        evidence_payload: dict[str, Any],
    ) -> dict[str, Any]:
        if self._db_path is None:
            from runtime_paths import sqlite_db_path

            db_path = sqlite_db_path("eos.db", ".semantier/eos.db")
        else:
            db_path = self._db_path
        evidence_id = "kbre_" + compute_content_hash(
            canonical_json(
                {
                    "knowledge_source_compilation_run_id": knowledge_source_compilation_run_id,
                    "rule_candidate_id": candidate["rule_candidate_id"],
                }
            )
        ).removeprefix("sha256:")[:24]
        now = _now_iso()
        row = {
            "evidence_id": evidence_id,
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "knowledge_source_id": knowledge_source_id,
            "knowledge_source_compilation_run_id": knowledge_source_compilation_run_id,
            "rule_candidate_id": candidate["rule_candidate_id"],
            "candidate_state_snapshot": candidate["candidate_state"],
            "source_anchor_refs": source_anchor_refs,
            "evidence_payload": evidence_payload,
            "created_at": now,
        }
        row["content_hash"] = compute_content_hash(canonical_json(row))
        with get_conn(db_path) as conn:
            conn.execute(
                sql_query("""
                INSERT OR IGNORE INTO knowledge_builder_rule_candidate_evidence (
                  evidence_id, organization_id, workspace_id, knowledge_source_id,
                  knowledge_source_compilation_run_id, rule_candidate_id,
                  candidate_state_snapshot, source_anchor_refs_json,
                  evidence_payload_json, created_at, content_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,'\n                INSERT INTO knowledge_builder_rule_candidate_evidence (\n                  evidence_id, organization_id, workspace_id, knowledge_source_id,\n                  knowledge_source_compilation_run_id, rule_candidate_id,\n                  candidate_state_snapshot, source_anchor_refs_json,\n                  evidence_payload_json, created_at, content_hash\n                )\n                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                \nON CONFLICT DO NOTHING'),
                (
                    row["evidence_id"],
                    row["organization_id"],
                    row["workspace_id"],
                    row["knowledge_source_id"],
                    row["knowledge_source_compilation_run_id"],
                    row["rule_candidate_id"],
                    row["candidate_state_snapshot"],
                    canonical_json(row["source_anchor_refs"]),
                    canonical_json(row["evidence_payload"]),
                    row["created_at"],
                    row["content_hash"],
                ),
            )
        return row
