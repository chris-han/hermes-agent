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
from storage.knowledge_builder_store import KnowledgeBuilderStore


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
    for field in (
        "aliases_json",
        "allowed_contexts_json",
        "prohibited_contexts_json",
        "source_anchor_refs_json",
        "proposed_runtime_effect_json",
    ):
        payload[field.removesuffix("_json")] = _loads_json(payload[field])
    return payload


class CanonicalTermStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        path = Path(db_path) if db_path else _db_path()
        if enabled():
            self._conn=connect_authority("eos",path)
            require_tables(self._conn,'canonical_term_candidates')
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
        self._builder = KnowledgeBuilderStore(db_path=path)

    def create_candidate(
        self,
        *,
        discovery_run_id: str,
        domain: str,
        canonical_label: str,
        definition: str,
        aliases: list[str],
        allowed_contexts: list[str],
        prohibited_contexts: list[str],
        source_anchor_refs: list[str],
        evidence_summary: str,
        proposed_runtime_effect: dict[str, Any],
        created_by: str,
        governance_state: str = "PROPOSED",
    ) -> dict[str, Any]:
        if governance_state not in {"CURATED", "PROPOSED"}:
            raise ValueError("canonical term candidates may start only as CURATED or PROPOSED")
        run = self._builder.get_discovery_run(discovery_run_id)
        if run is None:
            raise ValueError(f"knowledge builder discovery run not found: {discovery_run_id}")
        if not canonical_label.strip():
            raise ValueError("canonical_label is required")
        now = _now_iso()
        term_id = f"ctc_{uuid4().hex}"
        row = {
            "term_candidate_id": term_id,
            "discovery_run_id": discovery_run_id,
            "domain": domain,
            "canonical_label": canonical_label,
            "definition": definition,
            "aliases": aliases,
            "allowed_contexts": allowed_contexts,
            "prohibited_contexts": prohibited_contexts,
            "source_anchor_refs": source_anchor_refs,
            "evidence_summary": evidence_summary,
            "proposed_runtime_effect": proposed_runtime_effect,
            "governance_state": governance_state,
            "created_by": created_by,
            "created_at": now,
        }
        row["content_hash"] = compute_content_hash(canonical_json(row))
        with self._lock:
            self._conn.execute(
                sql_query("""
                INSERT INTO canonical_term_candidates (
                  term_candidate_id, discovery_run_id, domain, canonical_label,
                  definition, aliases_json, allowed_contexts_json, prohibited_contexts_json,
                  source_anchor_refs_json, evidence_summary, proposed_runtime_effect_json,
                  governance_state, created_by, created_at, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO canonical_term_candidates (\n                  term_candidate_id, discovery_run_id, domain, canonical_label,\n                  definition, aliases_json, allowed_contexts_json, prohibited_contexts_json,\n                  source_anchor_refs_json, evidence_summary, proposed_runtime_effect_json,\n                  governance_state, created_by, created_at, content_hash\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    term_id,
                    discovery_run_id,
                    domain,
                    canonical_label,
                    definition,
                    _as_json(aliases),
                    _as_json(allowed_contexts),
                    _as_json(prohibited_contexts),
                    _as_json(source_anchor_refs),
                    evidence_summary,
                    _as_json(proposed_runtime_effect),
                    governance_state,
                    created_by,
                    now,
                    row["content_hash"],
                ),
            )
            self._conn.commit()
        return row

    def get_candidate(self, term_candidate_id: str) -> dict[str, Any] | None:
        return _row_to_dict(
            self._conn.execute(
                sql_query("SELECT * FROM canonical_term_candidates WHERE term_candidate_id = ?", 'SELECT * FROM canonical_term_candidates WHERE term_candidate_id = %s'),
                (term_candidate_id,),
            ).fetchone()
        )

    def transition_candidate(
        self,
        *,
        term_candidate_id: str,
        governance_state: str,
    ) -> dict[str, Any]:
        state = governance_state.strip().upper()
        if state not in {"APPROVED", "ACTIVATED", "REJECTED"}:
            raise ValueError("governance_state must be APPROVED, ACTIVATED, or REJECTED")
        candidate = self.get_candidate(term_candidate_id)
        if candidate is None:
            raise ValueError(f"canonical term candidate not found: {term_candidate_id}")
        current = str(candidate["governance_state"])
        if state == "APPROVED" and current not in {"CURATED", "PROPOSED"}:
            raise ValueError("canonical term candidate must be CURATED or PROPOSED before approval")
        if state == "ACTIVATED" and current != "APPROVED":
            raise ValueError("canonical term candidate must be APPROVED before activation")
        with self._lock:
            self._conn.execute(
                sql_query("""
                UPDATE canonical_term_candidates
                SET governance_state = ?
                WHERE term_candidate_id = ?
                """, '\n                UPDATE canonical_term_candidates\n                SET governance_state = %s\n                WHERE term_candidate_id = %s\n                '),
                (state, term_candidate_id),
            )
            self._conn.commit()
        updated = self.get_candidate(term_candidate_id)
        assert updated is not None
        return updated
