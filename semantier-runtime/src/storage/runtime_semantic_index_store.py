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
from storage.canonical_term_store import CanonicalTermStore
from storage.knowledge_builder_evaluation_store import KnowledgeBuilderEvaluationStore
from storage.knowledge_builder_store import KnowledgeBuilderStore
from storage.semantic_relation_store import SemanticRelationStore

NON_AUTHORITATIVE_READ_MODEL_NOTICE = (
    "derived semantic read models are rebuildable and non-authoritative; runtime authority comes from activated authority versions selected by governed context"
)


def _db_path() -> Path:
    return Path(os.environ.get("SEMANTIER_EOS_DB_PATH", str(sqlite_db_path("eos.db", ".semantier/eos.db"))))


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_json(value: Any) -> str:
    return canonical_json(value if value is not None else {})


def _loads_json(value: str) -> Any:
    return json.loads(value) if value else None


def _hydrate(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    for field in (
        "semantic_relation_candidate_ids_json",
        "semantic_controls_json",
        "classification_controls_json",
        "candidate_lineage_json",
        "relation_ids_json",
        "control_json",
        "lineage_json",
        "evaluation_evidence_json",
    ):
        if field in payload:
            payload[field.removesuffix("_json")] = _loads_json(payload[field])
    if "non_authoritative" in payload:
        payload["non_authoritative"] = bool(payload["non_authoritative"])
    return payload


class RuntimeSemanticIndexStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        path = Path(db_path) if db_path else _db_path()
        if enabled():
            self._conn=connect_authority("eos",path)
            require_tables(self._conn,'audit_path_projection','concept_neighborhood_index','runtime_control_index','runtime_semantic_authority_versions','runtime_semantic_read_model_rebuilds','term_resolution_index')
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
        self._terms = CanonicalTermStore(db_path=path)
        self._relations = SemanticRelationStore(db_path=path)
        self._evaluations = KnowledgeBuilderEvaluationStore(db_path=path)
        self._builder = KnowledgeBuilderStore(db_path=path)

    def promote_and_activate(
        self,
        *,
        organization_id: str,
        workspace_id: str | None,
        term_candidate_id: str,
        semantic_relation_candidate_ids: list[str],
        evaluation_run_id: str,
        approved_by: str,
        activated_by: str,
    ) -> dict[str, Any]:
        term = self._terms.get_candidate(term_candidate_id)
        if term is None:
            raise ValueError(f"canonical term candidate not found: {term_candidate_id}")
        evaluation = self._evaluations.get_evaluation_run(evaluation_run_id)
        if evaluation is None or evaluation["run_status"] != "completed":
            raise ValueError("completed evaluation_run_id is required before activation")
        metrics = evaluation.get("metrics") or {}
        if float(metrics.get("precision") or 0) < 1.0 or float(metrics.get("recall") or 0) < 1.0:
            raise ValueError("evaluation gate requires precision and recall of 1.0 for MVP activation")
        relations = self._relations.list_relation_candidates(semantic_relation_candidate_ids)
        for relation in relations:
            if relation["curation_state"] != "CURATED":
                raise ValueError("semantic relation candidates must be CURATED before activation")
        discovery_run = self._builder.get_discovery_run(term["discovery_run_id"])
        if discovery_run is None:
            raise ValueError("term discovery run is missing")
        now = _now_iso()
        approved_term = self._terms.transition_candidate(
            term_candidate_id=term_candidate_id,
            governance_state="APPROVED",
        )
        activated_term = self._terms.transition_candidate(
            term_candidate_id=term_candidate_id,
            governance_state="ACTIVATED",
        )
        authority_id = f"rsav_{uuid4().hex}"
        semantic_controls = {
            "canonical_label": activated_term["canonical_label"],
            "allowed_contexts": activated_term["allowed_contexts"],
            "prohibited_contexts": activated_term["prohibited_contexts"],
            "relation_types": [relation["relation_type"] for relation in relations],
        }
        classification_controls = {
            "domain": activated_term["domain"],
            "runtime_effect": activated_term["proposed_runtime_effect"],
        }
        lineage = {
            "term_candidate_id": term_candidate_id,
            "discovery_run_id": activated_term["discovery_run_id"],
            "evaluation_run_id": evaluation_run_id,
            "semantic_relation_candidate_ids": semantic_relation_candidate_ids,
            "approved_state": approved_term["governance_state"],
        }
        row = {
            "authority_version_id": authority_id,
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "term_candidate_id": term_candidate_id,
            "canonical_label": activated_term["canonical_label"],
            "domain": activated_term["domain"],
            "semantic_relation_candidate_ids": semantic_relation_candidate_ids,
            "semantic_controls": semantic_controls,
            "classification_controls": classification_controls,
            "evaluation_run_id": evaluation_run_id,
            "candidate_lineage": lineage,
            "authority_state": "ACTIVATED",
            "approved_by": approved_by,
            "approved_at": now,
            "activated_by": activated_by,
            "activated_at": now,
        }
        row["content_hash"] = compute_content_hash(canonical_json(row))
        with self._lock:
            self._conn.execute(
                sql_query("""
                INSERT INTO runtime_semantic_authority_versions (
                  authority_version_id, organization_id, workspace_id, term_candidate_id,
                  canonical_label, domain, semantic_relation_candidate_ids_json,
                  semantic_controls_json, classification_controls_json, evaluation_run_id,
                  candidate_lineage_json, authority_state, approved_by, approved_at,
                  activated_by, activated_at, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO runtime_semantic_authority_versions (\n                  authority_version_id, organization_id, workspace_id, term_candidate_id,\n                  canonical_label, domain, semantic_relation_candidate_ids_json,\n                  semantic_controls_json, classification_controls_json, evaluation_run_id,\n                  candidate_lineage_json, authority_state, approved_by, approved_at,\n                  activated_by, activated_at, content_hash\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    authority_id,
                    organization_id,
                    workspace_id,
                    term_candidate_id,
                    row["canonical_label"],
                    row["domain"],
                    _as_json(semantic_relation_candidate_ids),
                    _as_json(semantic_controls),
                    _as_json(classification_controls),
                    evaluation_run_id,
                    _as_json(lineage),
                    "ACTIVATED",
                    approved_by,
                    now,
                    activated_by,
                    now,
                    row["content_hash"],
                ),
            )
            self._conn.commit()
        return row

    def rebuild_read_models(
        self,
        *,
        authority_version_id: str,
        rebuilt_by: str,
    ) -> dict[str, Any]:
        authority = self.get_authority_version(authority_version_id)
        if authority is None or authority["authority_state"] != "ACTIVATED":
            raise ValueError("activated authority_version_id is required")
        term = self._terms.get_candidate(authority["term_candidate_id"])
        if term is None:
            raise ValueError("activated canonical term candidate is missing")
        now = _now_iso()
        term_rows = [
            self._term_index_row(authority=authority, term_text=item, created_at=now)
            for item in [term["canonical_label"], *term["aliases"]]
        ]
        neighborhood_rows = [
            self._neighborhood_row(authority=authority, created_at=now)
        ]
        control_rows = [
            self._control_row(authority=authority, created_at=now)
        ]
        audit_rows = [
            self._audit_row(authority=authority, term=term, created_at=now)
        ]
        rebuild_id = f"rsrb_{uuid4().hex}"
        rebuild = {
            "rebuild_id": rebuild_id,
            "authority_version_id": authority_version_id,
            "rebuild_status": "completed",
            "term_resolution_index_hash": compute_content_hash(canonical_json(term_rows)),
            "concept_neighborhood_index_hash": compute_content_hash(canonical_json(neighborhood_rows)),
            "runtime_control_index_hash": compute_content_hash(canonical_json(control_rows)),
            "audit_path_projection_hash": compute_content_hash(canonical_json(audit_rows)),
            "non_authoritative_notice": NON_AUTHORITATIVE_READ_MODEL_NOTICE,
            "rebuilt_by": rebuilt_by,
            "rebuilt_at": now,
        }
        rebuild["content_hash"] = compute_content_hash(canonical_json(rebuild))
        with self._lock:
            self._conn.execute(sql_query("DELETE FROM term_resolution_index WHERE authority_version_id = ?", 'DELETE FROM term_resolution_index WHERE authority_version_id = %s'), (authority_version_id,))
            self._conn.execute(sql_query("DELETE FROM concept_neighborhood_index WHERE authority_version_id = ?", 'DELETE FROM concept_neighborhood_index WHERE authority_version_id = %s'), (authority_version_id,))
            self._conn.execute(sql_query("DELETE FROM runtime_control_index WHERE authority_version_id = ?", 'DELETE FROM runtime_control_index WHERE authority_version_id = %s'), (authority_version_id,))
            self._conn.execute(sql_query("DELETE FROM audit_path_projection WHERE authority_version_id = ?", 'DELETE FROM audit_path_projection WHERE authority_version_id = %s'), (authority_version_id,))
            for row in term_rows:
                self._insert_term_row(row)
            for row in neighborhood_rows:
                self._insert_neighborhood_row(row)
            for row in control_rows:
                self._insert_control_row(row)
            for row in audit_rows:
                self._insert_audit_row(row)
            self._conn.execute(
                sql_query("""
                INSERT INTO runtime_semantic_read_model_rebuilds (
                  rebuild_id, authority_version_id, rebuild_status,
                  term_resolution_index_hash, concept_neighborhood_index_hash,
                  runtime_control_index_hash, audit_path_projection_hash,
                  non_authoritative_notice, rebuilt_by, rebuilt_at, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO runtime_semantic_read_model_rebuilds (\n                  rebuild_id, authority_version_id, rebuild_status,\n                  term_resolution_index_hash, concept_neighborhood_index_hash,\n                  runtime_control_index_hash, audit_path_projection_hash,\n                  non_authoritative_notice, rebuilt_by, rebuilt_at, content_hash\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    rebuild_id,
                    authority_version_id,
                    "completed",
                    rebuild["term_resolution_index_hash"],
                    rebuild["concept_neighborhood_index_hash"],
                    rebuild["runtime_control_index_hash"],
                    rebuild["audit_path_projection_hash"],
                    NON_AUTHORITATIVE_READ_MODEL_NOTICE,
                    rebuilt_by,
                    now,
                    rebuild["content_hash"],
                ),
            )
            self._conn.commit()
        return {**rebuild, "indexes": self.get_read_model(authority_version_id)}

    def get_authority_version(self, authority_version_id: str) -> dict[str, Any] | None:
        return _hydrate(
            self._conn.execute(
                sql_query("SELECT * FROM runtime_semantic_authority_versions WHERE authority_version_id = ?", 'SELECT * FROM runtime_semantic_authority_versions WHERE authority_version_id = %s'),
                (authority_version_id,),
            ).fetchone()
        )

    def list_active_authority_versions(
        self,
        *,
        organization_id: str,
        workspace_id: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = [
            _hydrate(row)
            for row in self._conn.execute(
                sql_query("""
                SELECT * FROM runtime_semantic_authority_versions
                WHERE organization_id = ? AND authority_state = 'ACTIVATED'
                ORDER BY activated_at, authority_version_id
                """, "\n                SELECT * FROM runtime_semantic_authority_versions\n                WHERE organization_id = %s AND authority_state = 'ACTIVATED'\n                ORDER BY activated_at, authority_version_id\n                "),
                (organization_id,),
            ).fetchall()
        ]
        if workspace_id is None:
            return rows
        return [row for row in rows if row["workspace_id"] in {None, workspace_id}]

    def get_read_model(self, authority_version_id: str) -> dict[str, Any]:
        return {
            "term_resolution_index": self._list("term_resolution_index", authority_version_id),
            "concept_neighborhood_index": self._list("concept_neighborhood_index", authority_version_id),
            "runtime_control_index": self._list("runtime_control_index", authority_version_id),
            "audit_path_projection": self._list("audit_path_projection", authority_version_id),
        }

    def _list(self, table: str, authority_version_id: str) -> list[dict[str, Any]]:
        return [
            _hydrate(row)
            for row in self._conn.execute(
                sql_query(f"SELECT * FROM {table} WHERE authority_version_id = ? ORDER BY created_at, 1", f"SELECT * FROM {table} WHERE authority_version_id = %s ORDER BY created_at, 1"),
                (authority_version_id,),
            ).fetchall()
        ]

    def _term_index_row(self, *, authority: dict[str, Any], term_text: str, created_at: str) -> dict[str, Any]:
        row = {
            "index_entry_id": f"trix_{uuid4().hex}",
            "authority_version_id": authority["authority_version_id"],
            "term_text": term_text,
            "canonical_label": authority["canonical_label"],
            "domain": authority["domain"],
            "source_authority_hash": authority["content_hash"],
            "non_authoritative": True,
            "created_at": created_at,
        }
        row["content_hash"] = compute_content_hash(canonical_json(row))
        return row

    def _neighborhood_row(self, *, authority: dict[str, Any], created_at: str) -> dict[str, Any]:
        row = {
            "index_entry_id": f"cnix_{uuid4().hex}",
            "authority_version_id": authority["authority_version_id"],
            "canonical_label": authority["canonical_label"],
            "relation_ids": authority["semantic_relation_candidate_ids"],
            "source_authority_hash": authority["content_hash"],
            "non_authoritative": True,
            "created_at": created_at,
        }
        row["content_hash"] = compute_content_hash(canonical_json(row))
        return row

    def _control_row(self, *, authority: dict[str, Any], created_at: str) -> dict[str, Any]:
        row = {
            "index_entry_id": f"rcix_{uuid4().hex}",
            "authority_version_id": authority["authority_version_id"],
            "control_family": str((authority["classification_controls"].get("runtime_effect") or {}).get("control_family") or authority["domain"]),
            "control": authority["semantic_controls"],
            "source_authority_hash": authority["content_hash"],
            "non_authoritative": True,
            "created_at": created_at,
        }
        row["content_hash"] = compute_content_hash(canonical_json(row))
        return row

    def _audit_row(self, *, authority: dict[str, Any], term: dict[str, Any], created_at: str) -> dict[str, Any]:
        evaluation = self._evaluations.get_evaluation_run(authority["evaluation_run_id"]) or {}
        row = {
            "projection_id": f"aprx_{uuid4().hex}",
            "authority_version_id": authority["authority_version_id"],
            "lineage": authority["candidate_lineage"],
            "evaluation_evidence": {
                "evaluation_run_id": authority["evaluation_run_id"],
                "metrics": evaluation.get("metrics") or {},
                "source_anchor_refs": term["source_anchor_refs"],
            },
            "source_authority_hash": authority["content_hash"],
            "non_authoritative": True,
            "created_at": created_at,
        }
        row["content_hash"] = compute_content_hash(canonical_json(row))
        return row

    def _insert_term_row(self, row: dict[str, Any]) -> None:
        self._conn.execute(
            sql_query("""
            INSERT INTO term_resolution_index (
              index_entry_id, authority_version_id, term_text, canonical_label,
              domain, source_authority_hash, non_authoritative, created_at, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, '\n            INSERT INTO term_resolution_index (\n              index_entry_id, authority_version_id, term_text, canonical_label,\n              domain, source_authority_hash, non_authoritative, created_at, content_hash\n            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)\n            '),
            (
                row["index_entry_id"],
                row["authority_version_id"],
                row["term_text"],
                row["canonical_label"],
                row["domain"],
                row["source_authority_hash"],
                1,
                row["created_at"],
                row["content_hash"],
            ),
        )

    def _insert_neighborhood_row(self, row: dict[str, Any]) -> None:
        self._conn.execute(
            sql_query("""
            INSERT INTO concept_neighborhood_index (
              index_entry_id, authority_version_id, canonical_label, relation_ids_json,
              source_authority_hash, non_authoritative, created_at, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, '\n            INSERT INTO concept_neighborhood_index (\n              index_entry_id, authority_version_id, canonical_label, relation_ids_json,\n              source_authority_hash, non_authoritative, created_at, content_hash\n            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)\n            '),
            (
                row["index_entry_id"],
                row["authority_version_id"],
                row["canonical_label"],
                _as_json(row["relation_ids"]),
                row["source_authority_hash"],
                1,
                row["created_at"],
                row["content_hash"],
            ),
        )

    def _insert_control_row(self, row: dict[str, Any]) -> None:
        self._conn.execute(
            sql_query("""
            INSERT INTO runtime_control_index (
              index_entry_id, authority_version_id, control_family, control_json,
              source_authority_hash, non_authoritative, created_at, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, '\n            INSERT INTO runtime_control_index (\n              index_entry_id, authority_version_id, control_family, control_json,\n              source_authority_hash, non_authoritative, created_at, content_hash\n            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)\n            '),
            (
                row["index_entry_id"],
                row["authority_version_id"],
                row["control_family"],
                _as_json(row["control"]),
                row["source_authority_hash"],
                1,
                row["created_at"],
                row["content_hash"],
            ),
        )

    def _insert_audit_row(self, row: dict[str, Any]) -> None:
        self._conn.execute(
            sql_query("""
            INSERT INTO audit_path_projection (
              projection_id, authority_version_id, lineage_json, evaluation_evidence_json,
              source_authority_hash, non_authoritative, created_at, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, '\n            INSERT INTO audit_path_projection (\n              projection_id, authority_version_id, lineage_json, evaluation_evidence_json,\n              source_authority_hash, non_authoritative, created_at, content_hash\n            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)\n            '),
            (
                row["projection_id"],
                row["authority_version_id"],
                _as_json(row["lineage"]),
                _as_json(row["evaluation_evidence"]),
                row["source_authority_hash"],
                1,
                row["created_at"],
                row["content_hash"],
            ),
        )
