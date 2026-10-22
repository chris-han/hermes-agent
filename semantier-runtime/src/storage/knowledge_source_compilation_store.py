from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eos.artifact_hashing import canonical_json, content_hash as compute_content_hash
from eos.db import init_db
from runtime_paths import sqlite_db_path
from storage.hosted_postgres import enabled,connect_authority,require_tables,sql_query,begin_write
from storage.transaction_scope import transaction_scope,StoreMutationLock


def _db_path() -> Path:
    return Path(os.environ.get("SEMANTIER_EOS_DB_PATH", str(sqlite_db_path("eos.db", ".semantier/eos.db"))))


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_json(value: Any) -> str:
    return canonical_json(value if value is not None else [])


def _loads_json(value: str) -> Any:
    return json.loads(value) if value else None


def _assert_ascii(value: str, field: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        raise ValueError(f"{field} is required")
    try:
        normalized.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field} must be an ASCII-stable identifier") from exc
    return normalized


def _assert_optional_ascii(value: str | None, field: str) -> str | None:
    if value is None or not str(value).strip():
        return None
    return _assert_ascii(str(value), field)


def _row_to_run(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    for field in (
        "provider_artifact_refs_json",
        "provider_artifact_hashes_json",
        "mapped_source_anchor_refs_json",
    ):
        payload[field.removesuffix("_json")] = _loads_json(payload[field])
    return payload


def _row_to_anchor(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    payload["anchor_payload"] = _loads_json(payload["anchor_payload_json"])
    return payload


class KnowledgeSourceCompilationStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        path = Path(db_path) if db_path else _db_path()
        if enabled():
            self._conn=connect_authority("eos",path)
            require_tables(self._conn,'knowledge_source_compilation_anchors','knowledge_source_compilation_runs')
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

    def create_run(
        self,
        *,
        knowledge_source_compilation_run_id: str,
        knowledge_source_id: str,
        organization_id: str,
        workspace_id: str | None,
        semantic_purpose: str,
        source_kind: str,
        source_ref: str,
        source_artifact_ref: str,
        source_hash: str,
        normalized_document_artifact_ref: str,
        normalized_document_artifact_hash: str,
        document_schema_version: str,
        parser_method: str,
        parser_version: str,
        provider_artifact_refs: list[str] | None,
        provider_artifact_hashes: list[str] | None,
        compiler_profile_version: str,
        mapped_source_anchor_refs: list[str] | None,
        created_by: str,
    ) -> dict[str, Any]:
        run_id = _assert_ascii(
            knowledge_source_compilation_run_id,
            "knowledge_source_compilation_run_id",
        )
        row = {
            "knowledge_source_compilation_run_id": run_id,
            "knowledge_source_id": _assert_ascii(knowledge_source_id, "knowledge_source_id"),
            "organization_id": _assert_ascii(organization_id, "organization_id"),
            "workspace_id": _assert_optional_ascii(workspace_id, "workspace_id"),
            "semantic_purpose": _assert_ascii(semantic_purpose, "semantic_purpose"),
            "source_kind": _assert_ascii(source_kind, "source_kind"),
            "source_ref": source_ref,
            "source_artifact_ref": source_artifact_ref,
            "source_hash": source_hash,
            "normalized_document_artifact_ref": normalized_document_artifact_ref,
            "normalized_document_artifact_hash": normalized_document_artifact_hash,
            "document_schema_version": document_schema_version,
            "parser_method": parser_method,
            "parser_version": parser_version,
            "provider_artifact_refs": provider_artifact_refs or [],
            "provider_artifact_hashes": provider_artifact_hashes or [],
            "compiler_profile_version": _assert_ascii(
                compiler_profile_version,
                "compiler_profile_version",
            ),
            "mapped_source_anchor_refs": mapped_source_anchor_refs or [],
            "created_by": _assert_ascii(created_by, "created_by"),
            "created_at": _now_iso(),
        }
        row["content_hash"] = compute_content_hash(canonical_json(row))
        with self._lock, transaction_scope(self._conn):
            self._conn.execute(
                sql_query("""
                INSERT INTO knowledge_source_compilation_runs (
                  knowledge_source_compilation_run_id, knowledge_source_id,
                  organization_id, workspace_id, semantic_purpose, source_kind,
                  source_ref, source_artifact_ref, source_hash,
                  normalized_document_artifact_ref, normalized_document_artifact_hash,
                  document_schema_version, parser_method, parser_version,
                  provider_artifact_refs_json, provider_artifact_hashes_json,
                  compiler_profile_version, mapped_source_anchor_refs_json,
                  created_by, created_at, content_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO knowledge_source_compilation_runs (\n                  knowledge_source_compilation_run_id, knowledge_source_id,\n                  organization_id, workspace_id, semantic_purpose, source_kind,\n                  source_ref, source_artifact_ref, source_hash,\n                  normalized_document_artifact_ref, normalized_document_artifact_hash,\n                  document_schema_version, parser_method, parser_version,\n                  provider_artifact_refs_json, provider_artifact_hashes_json,\n                  compiler_profile_version, mapped_source_anchor_refs_json,\n                  created_by, created_at, content_hash\n                )\n                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    row["knowledge_source_compilation_run_id"],
                    row["knowledge_source_id"],
                    row["organization_id"],
                    row["workspace_id"],
                    row["semantic_purpose"],
                    row["source_kind"],
                    row["source_ref"],
                    row["source_artifact_ref"],
                    row["source_hash"],
                    row["normalized_document_artifact_ref"],
                    row["normalized_document_artifact_hash"],
                    row["document_schema_version"],
                    row["parser_method"],
                    row["parser_version"],
                    _as_json(row["provider_artifact_refs"]),
                    _as_json(row["provider_artifact_hashes"]),
                    row["compiler_profile_version"],
                    _as_json(row["mapped_source_anchor_refs"]),
                    row["created_by"],
                    row["created_at"],
                    row["content_hash"],
                ),
            )
        return self.get_run(run_id)  # type: ignore[return-value]

    def create_anchor(
        self,
        *,
        compilation_anchor_ref: str,
        knowledge_source_compilation_run_id: str,
        knowledge_source_id: str,
        canonical_document_anchor_ref: str,
        source_hash: str,
        compiler_profile_version: str,
        anchor_payload: dict[str, Any],
    ) -> dict[str, Any]:
        anchor_ref = _assert_ascii(compilation_anchor_ref, "compilation_anchor_ref")
        row = {
            "compilation_anchor_ref": anchor_ref,
            "knowledge_source_compilation_run_id": _assert_ascii(
                knowledge_source_compilation_run_id,
                "knowledge_source_compilation_run_id",
            ),
            "knowledge_source_id": _assert_ascii(knowledge_source_id, "knowledge_source_id"),
            "canonical_document_anchor_ref": canonical_document_anchor_ref,
            "source_hash": source_hash,
            "compiler_profile_version": _assert_ascii(
                compiler_profile_version,
                "compiler_profile_version",
            ),
            "anchor_payload": anchor_payload,
            "created_at": _now_iso(),
        }
        row["content_hash"] = compute_content_hash(canonical_json(row))
        with self._lock, transaction_scope(self._conn):
            self._conn.execute(
                sql_query("""
                INSERT OR IGNORE INTO knowledge_source_compilation_anchors (
                  compilation_anchor_ref, knowledge_source_compilation_run_id,
                  knowledge_source_id, canonical_document_anchor_ref,
                  source_hash, compiler_profile_version, anchor_payload_json,
                  content_hash, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO knowledge_source_compilation_anchors (\n                  compilation_anchor_ref, knowledge_source_compilation_run_id,\n                  knowledge_source_id, canonical_document_anchor_ref,\n                  source_hash, compiler_profile_version, anchor_payload_json,\n                  content_hash, created_at\n                )\n                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)\n                 ON CONFLICT DO NOTHING'),
                (
                    row["compilation_anchor_ref"],
                    row["knowledge_source_compilation_run_id"],
                    row["knowledge_source_id"],
                    row["canonical_document_anchor_ref"],
                    row["source_hash"],
                    row["compiler_profile_version"],
                    canonical_json(row["anchor_payload"]),
                    row["content_hash"],
                    row["created_at"],
                ),
            )
        return self.get_anchor(anchor_ref)  # type: ignore[return-value]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            sql_query("""
            SELECT * FROM knowledge_source_compilation_runs
            WHERE knowledge_source_compilation_run_id = ?
            """, '\n            SELECT * FROM knowledge_source_compilation_runs\n            WHERE knowledge_source_compilation_run_id = %s\n            '),
            (_assert_ascii(run_id, "knowledge_source_compilation_run_id"),),
        ).fetchone()
        return _row_to_run(row)

    def list_anchors(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            sql_query("""
            SELECT * FROM knowledge_source_compilation_anchors
            WHERE knowledge_source_compilation_run_id = ?
            ORDER BY compilation_anchor_ref
            """, '\n            SELECT * FROM knowledge_source_compilation_anchors\n            WHERE knowledge_source_compilation_run_id = %s\n            ORDER BY compilation_anchor_ref\n            '),
            (_assert_ascii(run_id, "knowledge_source_compilation_run_id"),),
        ).fetchall()
        return [_row_to_anchor(row) for row in rows if row is not None]  # type: ignore[list-item]

    def get_anchor(self, anchor_ref: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            sql_query("""
            SELECT * FROM knowledge_source_compilation_anchors
            WHERE compilation_anchor_ref = ?
            """, '\n            SELECT * FROM knowledge_source_compilation_anchors\n            WHERE compilation_anchor_ref = %s\n            '),
            (_assert_ascii(anchor_ref, "compilation_anchor_ref"),),
        ).fetchone()
        return _row_to_anchor(row)
