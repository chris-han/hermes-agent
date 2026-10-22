from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from contracts.evidence_location import SourceIdentity
from eos.artifact_hashing import canonical_json, content_hash as compute_content_hash
from eos.db import init_db
from runtime_paths import sqlite_db_path
from storage.hosted_postgres import enabled,connect_authority,require_tables,sql_query,begin_write
from storage.transaction_scope import transaction_scope,StoreMutationLock

_STATES = {"DISCOVERED", "CURATED", "PROPOSED", "APPROVED", "ACTIVATED", "REJECTED"}


def _db_path() -> Path:
    return Path(os.environ.get("SEMANTIER_EOS_DB_PATH", str(sqlite_db_path("eos.db", ".semantier/eos.db"))))


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_json(value: Any) -> str:
    return canonical_json(value if value is not None else {})


def _loads_json(value: str) -> Any:
    return json.loads(value) if value else None


def _assert_state(value: str) -> str:
    state = value.strip().upper()
    if state not in _STATES:
        raise ValueError(f"governance_state must be one of {sorted(_STATES)}")
    return state


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    for field in (
        "source_metadata_json",
        "model_provenance_json",
        "procedure_provenance_json",
    ):
        if field in payload:
            payload[field.removesuffix("_json")] = _loads_json(payload[field])
    return payload


_SOURCE_IDENTITY_RECORD_SCHEMA = "semantier.source_identity_registry_record.v1"
_SOURCE_IDENTITY_EVIDENCE_SCHEMA = "semantier.source_identity_registration_evidence.v1"


def _source_identity_payload(source_identity: SourceIdentity) -> dict[str, Any]:
    return source_identity.model_dump(mode="json", by_alias=True, exclude_none=False)


def _source_row_payload(
    row: dict[str, Any], *, include_lineage: bool = True
) -> dict[str, Any]:
    fields = [
        "source_id",
        "organization_id",
        "workspace_id",
        "source_kind",
        "source_ref",
        "source_text",
        "source_hash",
        "source_metadata",
        "governance_state",
        "created_at",
    ]
    if include_lineage:
        fields.insert(1, "lineage_id")
    return {field: row[field] for field in fields}


def _source_row_content_hash(
    row: dict[str, Any], *, include_lineage: bool = True
) -> str:
    return compute_content_hash(
        canonical_json(_source_row_payload(row, include_lineage=include_lineage))
    )


class KnowledgeBuilderStore:
    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        workspace_root: str | Path | None = None,
    ) -> None:
        path = Path(db_path) if db_path else _db_path()
        self._db_path = path
        self.workspace_root = (
            Path(workspace_root).resolve() if workspace_root is not None else None
        )
        if enabled():
            self._conn=connect_authority("eos",path)
            require_tables(self._conn,'knowledge_builder_sources','knowledge_builder_source_identities','knowledge_builder_discovery_runs')
            self._conn.commit()
            self._lock=StoreMutationLock(self._conn)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        prior_db_path = os.environ.get("SEMANTIER_EOS_DB_PATH")
        os.environ["SEMANTIER_EOS_DB_PATH"] = str(path)
        try:
            init_db()
            with sqlite3.connect(str(path)) as schema_probe:
                source_identity_schema_ready = (
                    schema_probe.execute(
                        "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                        "AND name = 'knowledge_builder_source_identities'"
                    ).fetchone()
                    is not None
                )
            if not source_identity_schema_ready:
                init_db(force=True)
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
        source_columns = {
            row["name"]
            for row in self._conn.execute(
                "PRAGMA table_info(knowledge_builder_sources)"
            ).fetchall()
        }
        if "lineage_id" not in source_columns:
            self._conn.execute(
                "ALTER TABLE knowledge_builder_sources ADD COLUMN lineage_id TEXT"
            )
        if "retirement_state" not in source_columns:
            self._conn.execute(
                "ALTER TABLE knowledge_builder_sources ADD COLUMN retirement_state "
                "TEXT NOT NULL DEFAULT 'active'"
            )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_knowledge_builder_sources_lineage_id "
            "ON knowledge_builder_sources(lineage_id)"
        )
        discovery_columns = {
            row["name"]
            for row in self._conn.execute(
                "PRAGMA table_info(knowledge_builder_discovery_runs)"
            ).fetchall()
        }
        if "lineage_id" not in discovery_columns:
            self._conn.execute(
                "ALTER TABLE knowledge_builder_discovery_runs ADD COLUMN lineage_id TEXT"
            )
        if "retirement_state" not in discovery_columns:
            self._conn.execute(
                "ALTER TABLE knowledge_builder_discovery_runs ADD COLUMN retirement_state "
                "TEXT NOT NULL DEFAULT 'active'"
            )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_knowledge_builder_discovery_runs_lineage_id "
            "ON knowledge_builder_discovery_runs(lineage_id)"
        )
        self._conn.commit()

    def close(self) -> None:
        """Release the governed SQLite connection owned by this store."""
        self._conn.close()

    def create_source(
        self,
        *,
        organization_id: str,
        workspace_id: str | None,
        source_kind: str,
        source_ref: str,
        source_text: str,
        source_metadata: dict[str, Any] | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any]:
        kind = source_kind.strip().lower()
        if kind not in {"folder", "file", "text"}:
            raise ValueError("source_kind must be folder, file, or text")
        if not source_ref.strip():
            raise ValueError("source_ref is required")
        resolved_source_text = source_text if source_text.strip() else source_ref
        stable_identity = source_id is not None
        if stable_identity and "canonical_ir_ref" in (source_metadata or {}):
            raise ValueError("SOURCE_IDENTITY_CONFLICT")
        if stable_identity and (not isinstance(source_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", source_id)):
            raise ValueError("SOURCE_IDENTITY_CONFLICT")
        source_id = source_id if stable_identity else f"kbs_{uuid4().hex}"
        now = _now_iso()
        row = {
            "source_id": source_id,
            "lineage_id": source_id,
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "source_kind": kind,
            "source_ref": source_ref,
            "source_text": resolved_source_text,
            "source_hash": compute_content_hash(resolved_source_text),
            "source_metadata": source_metadata or {},
            "governance_state": "DISCOVERED",
            "created_at": now,
        }
        row["content_hash"] = compute_content_hash(canonical_json(row))
        with self._lock, transaction_scope(self._conn):
            if stable_identity:
                begin_write(self._conn)
                try:
                    existing = self.get_source(source_id)
                    if existing is not None:
                        self._verify_source_row(existing)
                        metadata = dict(existing["source_metadata"])
                        # Canonical IR is a later owner-controlled binding, not
                        # part of the source-creation request being replayed.
                        metadata.pop("canonical_ir_ref", None)
                        fields = ("organization_id", "workspace_id", "source_kind", "source_ref", "source_text", "source_hash")
                        if any(existing[field] != row[field] for field in fields) or metadata != row["source_metadata"]:
                            raise ValueError("SOURCE_IDENTITY_CONFLICT")
                        self._conn.commit()
                        return existing
                except BaseException:
                    self._conn.rollback()
                    raise
            self._conn.execute(
                sql_query("""
                INSERT INTO knowledge_builder_sources (
                  source_id, lineage_id, organization_id, workspace_id, source_kind, source_ref,
                  source_text, source_hash, source_metadata_json, governance_state,
                  created_at, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO knowledge_builder_sources (\n                  source_id, lineage_id, organization_id, workspace_id, source_kind, source_ref,\n                  source_text, source_hash, source_metadata_json, governance_state,\n                  created_at, content_hash\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    source_id,
                    source_id,
                    organization_id,
                    workspace_id,
                    kind,
                    source_ref,
                    resolved_source_text,
                    row["source_hash"],
                    _as_json(row["source_metadata"]),
                    row["governance_state"],
                    now,
                    row["content_hash"],
                ),
            )
            self._conn.commit()
        if stable_identity:
            return self.get_source(source_id)
        return row

    def create_discovery_run(
        self,
        *,
        organization_id: str,
        workspace_id: str | None,
        source_id: str | None = None,
        source_kind: str = "text",
        source_ref: str = "inline_tender_sample",
        source_text: str | None = None,
        source_metadata: dict[str, Any] | None = None,
        model_provenance: dict[str, Any] | None = None,
        procedure_provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if source_id:
            source = self.get_source(source_id)
            if (source is None or source["organization_id"] != organization_id or
                    source.get("workspace_id") != workspace_id):
                raise ValueError(f"knowledge builder source not found: {source_id}")
        else:
            source = self.create_source(
                organization_id=organization_id,
                workspace_id=workspace_id,
                source_kind=source_kind,
                source_ref=source_ref,
                source_text=source_text or "",
                source_metadata=source_metadata,
            )
        run_id = f"kbd_{uuid4().hex}"
        now = _now_iso()
        model = model_provenance or {
            "model": "deterministic-lexical-candidate-extractor",
            "version": "knowledge_builder_mvp.v1",
        }
        procedure = procedure_provenance or {
            "procedure": "tender_phrase_anchor_extraction",
            "version": "knowledge_builder_mvp.v1",
        }
        run = {
            "discovery_run_id": run_id,
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "source_id": source["source_id"],
            "lineage_id": source.get("lineage_id") or source["source_id"],
            "run_status": "completed",
            "model_provenance": model,
            "procedure_provenance": procedure,
            "governance_state": "DISCOVERED",
            "started_at": now,
            "completed_at": now,
        }
        run["content_hash"] = compute_content_hash(canonical_json(run))
        with self._lock:
            self._conn.execute(
                sql_query("""
                INSERT INTO knowledge_builder_discovery_runs (
                  discovery_run_id, organization_id, workspace_id, source_id, lineage_id, run_status,
                  model_provenance_json, procedure_provenance_json, governance_state,
                  started_at, completed_at, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, '\n                INSERT INTO knowledge_builder_discovery_runs (\n                  discovery_run_id, organization_id, workspace_id, source_id, lineage_id, run_status,\n                  model_provenance_json, procedure_provenance_json, governance_state,\n                  started_at, completed_at, content_hash\n                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\n                '),
                (
                    run_id,
                    organization_id,
                    workspace_id,
                    source["source_id"],
                    source.get("lineage_id") or source["source_id"],
                    "completed",
                    _as_json(model),
                    _as_json(procedure),
                    "DISCOVERED",
                    now,
                    now,
                    run["content_hash"],
                ),
            )
            self._conn.commit()
        return self.get_discovery_run(run_id)  # type: ignore[return-value]

    def get_source(self, source_id: str) -> dict[str, Any] | None:
        return _row_to_dict(
            self._conn.execute(
                sql_query("SELECT * FROM knowledge_builder_sources WHERE source_id = ?", 'SELECT * FROM knowledge_builder_sources WHERE source_id = %s'),
                (source_id,),
            ).fetchone()
        )

    def bind_canonical_ir_ref(
        self,
        *,
        source_id: str,
        organization_id: str,
        workspace_id: str,
        canonical_ir_ref: str,
        expected_content_hash: str,
    ) -> dict[str, Any]:
        """Compare-and-swap one immutable CanonicalSourceIR parent binding."""

        if not canonical_ir_ref or not canonical_ir_ref.isascii():
            raise ValueError("canonical_ir_ref must be an ASCII-stable reference")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_content_hash):
            raise ValueError("expected source content hash is invalid")
        with self._lock:
            try:
                begin_write(self._conn)
                row = _row_to_dict(
                    self._conn.execute(
                        sql_query("""SELECT * FROM knowledge_builder_sources
                        WHERE source_id = ? AND organization_id = ?
                          AND workspace_id = ?""", 'SELECT * FROM knowledge_builder_sources\n                        WHERE source_id = %s AND organization_id = %s\n                          AND workspace_id = %s'),
                        (source_id, organization_id, workspace_id),
                    ).fetchone()
                )
                if row is None:
                    raise ValueError("existing governed source not found")
                self._verify_source_row(row)
                metadata = dict(row["source_metadata"])
                existing_ref = metadata.get("canonical_ir_ref")
                if existing_ref == canonical_ir_ref:
                    self._conn.commit()
                    return row
                if existing_ref is not None:
                    raise ValueError("governed source CanonicalSourceIR binding conflict")
                if row["content_hash"] != expected_content_hash:
                    raise ValueError("governed source compare-and-swap conflict")
                identity_count = self._conn.execute(
                    sql_query("""SELECT COUNT(*) FROM knowledge_builder_source_identities
                    WHERE source_id = ?""", 'SELECT COUNT(*) FROM knowledge_builder_source_identities\n                    WHERE source_id = %s'),
                    (source_id,),
                ).fetchone()[0]
                if identity_count:
                    raise ValueError(
                        "governed source CanonicalSourceIR binding must precede identity registration"
                    )
                metadata["canonical_ir_ref"] = canonical_ir_ref
                updated = {**row, "source_metadata": metadata}
                updated["content_hash"] = _source_row_content_hash(updated)
                cursor = self._conn.execute(
                    sql_query("""UPDATE knowledge_builder_sources
                    SET source_metadata_json = ?, content_hash = ?
                    WHERE source_id = ? AND organization_id = ?
                      AND workspace_id = ? AND content_hash = ?""", 'UPDATE knowledge_builder_sources\n                    SET source_metadata_json = %s, content_hash = %s\n                    WHERE source_id = %s AND organization_id = %s\n                      AND workspace_id = %s AND content_hash = %s'),
                    (
                        _as_json(metadata),
                        updated["content_hash"],
                        source_id,
                        organization_id,
                        workspace_id,
                        expected_content_hash,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ValueError("governed source compare-and-swap conflict")
                self._conn.commit()
                return updated
            except Exception:
                self._conn.rollback()
                raise

    def register_source_identity(
        self,
        *,
        source_identity: SourceIdentity,
        lineage_id: str,
    ) -> dict[str, Any]:
        """Atomically pin one immutable identity to an existing governed source."""

        if source_identity.created_at.utcoffset() != timezone.utc.utcoffset(None):
            raise ValueError("source identity created_at must be UTC")
        if not lineage_id or not lineage_id.isascii():
            raise ValueError("lineage_id must be an ASCII-stable identifier")
        with self._lock:
            try:
                begin_write(self._conn)
                row = self._source_row_for_lineage(
                    organization_id=source_identity.organization_id,
                    workspace_id=source_identity.workspace_id,
                    lineage_id=lineage_id,
                )
                metadata = row["source_metadata"]
                governed_refs = {
                    str(value)
                    for value in (
                        row.get("source_text"),
                        metadata.get("original_path"),
                    )
                    if value
                }
                if source_identity.artifact_ref not in governed_refs:
                    raise ValueError("governed source artifact ref mismatch")
                stored_hash = self._normalize_source_hash(
                    metadata.get("original_content_hash")
                )
                if stored_hash != source_identity.source_hash:
                    raise ValueError("governed source stored source hash mismatch")
                governed_version = self._governed_source_version(
                    row=row,
                    source_identity=source_identity,
                    lineage_id=lineage_id,
                    stored_hash=stored_hash,
                )
                if (
                    source_identity.source_version is not None
                    and source_identity.source_version != governed_version
                ):
                    raise ValueError("governed source stored source version mismatch")
                source_identity = source_identity.model_copy(
                    update={"source_version": governed_version}
                )
                source_version = governed_version
                self._verify_source_artifact(source_identity, source_metadata=metadata)

                identity_payload = _source_identity_payload(source_identity)
                identity_hash = f"sha256:{compute_content_hash(identity_payload)}"
                existing = self._conn.execute(
                    sql_query("""SELECT * FROM knowledge_builder_source_identities
                    WHERE organization_id = ? AND workspace_id = ?
                      AND source_identity_ref = ?""", 'SELECT * FROM knowledge_builder_source_identities\n                    WHERE organization_id = %s AND workspace_id = %s\n                      AND source_identity_ref = %s'),
                    (
                        source_identity.organization_id,
                        source_identity.workspace_id,
                        source_identity.source_identity_ref,
                    ),
                ).fetchone()
                if existing is not None:
                    verified = self._verified_source_identity_row(existing)
                    if (
                        _source_identity_payload(verified) != identity_payload
                        or existing["source_id"] != row["source_id"]
                        or existing["source_version"] != source_version
                    ):
                        raise ValueError("source identity registry conflict")
                    self._conn.commit()
                    return dict(existing)

                version_conflict = self._conn.execute(
                    sql_query("""SELECT source_identity_ref
                    FROM knowledge_builder_source_identities
                    WHERE source_id = ? AND source_version = ?""", 'SELECT source_identity_ref\n                    FROM knowledge_builder_source_identities\n                    WHERE source_id = %s AND source_version = %s'),
                    (row["source_id"], source_version),
                ).fetchone()
                if version_conflict is not None:
                    raise ValueError("source version conflict")

                prior_count = self._conn.execute(
                    sql_query("""SELECT COUNT(*) FROM knowledge_builder_source_identities
                    WHERE source_id = ?""", 'SELECT COUNT(*) FROM knowledge_builder_source_identities\n                    WHERE source_id = %s'),
                    (row["source_id"],),
                ).fetchone()[0]
                supersedes = source_identity.supersedes_source_identity_ref
                if supersedes == source_identity.source_identity_ref:
                    raise ValueError("SourceIdentity cannot supersede itself")
                if supersedes is not None:
                    predecessor = self._conn.execute(
                        sql_query("""SELECT * FROM knowledge_builder_source_identities
                        WHERE organization_id = ? AND workspace_id = ?
                          AND source_identity_ref = ?""", 'SELECT * FROM knowledge_builder_source_identities\n                        WHERE organization_id = %s AND workspace_id = %s\n                          AND source_identity_ref = %s'),
                        (
                            source_identity.organization_id,
                            source_identity.workspace_id,
                            supersedes,
                        ),
                    ).fetchone()
                    expected_scope = (
                        row["source_id"],
                        source_identity.organization_id,
                        source_identity.workspace_id,
                        lineage_id,
                    )
                    if predecessor is None:
                        raise ValueError(
                            "SourceIdentity supersedes target must exist in the same source lineage"
                        )
                    self._verified_source_identity_row(predecessor)
                    predecessor_scope = tuple(
                        predecessor[field]
                        for field in (
                            "source_id",
                            "organization_id",
                            "workspace_id",
                            "lineage_id",
                        )
                    )
                    if predecessor_scope != expected_scope:
                        raise ValueError(
                            "SourceIdentity supersedes target must exist in the same source lineage"
                        )
                elif prior_count:
                    raise ValueError(
                        "additional SourceIdentity must name a supersedes target"
                    )

                evidence = {
                    "schema_version": _SOURCE_IDENTITY_EVIDENCE_SCHEMA,
                    "source_id": row["source_id"],
                    "source_ref": row["source_ref"],
                    "source_text": row["source_text"],
                    "source_record_content_hash": row["content_hash"],
                    "stored_original_content_hash": stored_hash,
                    "source_version": source_version,
                    "artifact_ref": source_identity.artifact_ref,
                    "artifact_hash": source_identity.source_hash,
                }
                evidence_hash = f"sha256:{compute_content_hash(evidence)}"
                record = {
                    "schema_version": _SOURCE_IDENTITY_RECORD_SCHEMA,
                    "source_identity_ref": source_identity.source_identity_ref,
                    "source_id": row["source_id"],
                    "organization_id": source_identity.organization_id,
                    "workspace_id": source_identity.workspace_id,
                    "lineage_id": lineage_id,
                    "source_hash": source_identity.source_hash,
                    "source_version": source_version,
                    "supersedes_source_identity_ref": supersedes,
                    "artifact_ref": source_identity.artifact_ref,
                    "artifact_hash": source_identity.source_hash,
                    "media_type": source_identity.media_type,
                    "original_name": source_identity.original_name,
                    "created_at": source_identity.created_at.isoformat(),
                    "source_identity": identity_payload,
                    "source_identity_hash": identity_hash,
                    "registration_evidence": evidence,
                    "registration_evidence_hash": evidence_hash,
                }
                record_hash = f"sha256:{compute_content_hash(record)}"
                self._conn.execute(
                    sql_query("""INSERT INTO knowledge_builder_source_identities (
                      source_identity_ref, source_id, organization_id, workspace_id,
                      lineage_id, source_hash, source_version,
                      supersedes_source_identity_ref, artifact_ref, artifact_hash,
                      media_type, original_name, created_at, schema_version,
                      source_identity_json, source_identity_hash,
                      registration_evidence_json, registration_evidence_hash,
                      content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", 'INSERT INTO knowledge_builder_source_identities (\n                      source_identity_ref, source_id, organization_id, workspace_id,\n                      lineage_id, source_hash, source_version,\n                      supersedes_source_identity_ref, artifact_ref, artifact_hash,\n                      media_type, original_name, created_at, schema_version,\n                      source_identity_json, source_identity_hash,\n                      registration_evidence_json, registration_evidence_hash,\n                      content_hash\n                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'),
                    (
                        record["source_identity_ref"], record["source_id"],
                        record["organization_id"], record["workspace_id"],
                        record["lineage_id"], record["source_hash"],
                        record["source_version"],
                        record["supersedes_source_identity_ref"],
                        record["artifact_ref"], record["artifact_hash"],
                        record["media_type"], record["original_name"],
                        record["created_at"], record["schema_version"],
                        canonical_json(identity_payload), identity_hash,
                        canonical_json(evidence), evidence_hash, record_hash,
                    ),
                )
                self._conn.commit()
                return {**record, "content_hash": record_hash}
            except sqlite3.IntegrityError as exc:
                self._conn.rollback()
                if "source_version" in str(exc):
                    raise ValueError("source version conflict") from exc
                raise ValueError("source identity registry conflict") from exc
            except Exception:
                self._conn.rollback()
                raise

    def resolve_source_identity_lineage(
        self, *, source_identity_ref: str, organization_id: str, workspace_id: str,
    ) -> tuple[str, SourceIdentity]:
        """Resolve one exact governed identity without inferring lineage from its ref."""
        rows = self._conn.execute(
            sql_query("SELECT * FROM knowledge_builder_source_identities "
            "WHERE organization_id=? AND workspace_id=? AND source_identity_ref=?", 'SELECT * FROM knowledge_builder_source_identities WHERE organization_id=%s AND workspace_id=%s AND source_identity_ref=%s'),
            (organization_id, workspace_id, source_identity_ref),
        ).fetchall()
        if len(rows) != 1:
            raise ValueError("SOURCE_IDENTITY_LINEAGE_NOT_FOUND_OR_AMBIGUOUS")
        identity = self._verified_source_identity_row(rows[0])
        return rows[0]["lineage_id"], identity

    def get_source_identity(
        self,
        *,
        source_identity_ref: str,
        organization_id: str,
        workspace_id: str,
        lineage_id: str,
    ) -> SourceIdentity | None:
        identities = self.list_source_identities(
            organization_id=organization_id,
            workspace_id=workspace_id,
            lineage_id=lineage_id,
        )
        return next(
            (
                identity
                for identity in identities
                if identity.source_identity_ref == source_identity_ref
            ),
            None,
        )

    def permits_historical_null_source_version(
        self,
        *,
        supplied_identity: SourceIdentity,
        registered_identity: SourceIdentity,
        lineage_id: str,
    ) -> bool:
        """Prove the sole legacy null-version alias from the governed source row."""

        if (
            supplied_identity.source_version is not None
            or registered_identity.source_version is None
        ):
            return False
        row = self._source_row_for_lineage(
            organization_id=supplied_identity.organization_id,
            workspace_id=supplied_identity.workspace_id,
            lineage_id=lineage_id,
        )
        if row["source_metadata"].get("source_version") is not None:
            return False
        stored_hash = KnowledgeBuilderStore._normalize_source_hash(
            row["source_metadata"].get("original_content_hash")
        )
        if stored_hash != supplied_identity.source_hash:
            return False
        expected = KnowledgeBuilderStore._governed_source_version(
            row=row,
            source_identity=supplied_identity,
            lineage_id=lineage_id,
            stored_hash=stored_hash,
        )
        if registered_identity.source_version != expected:
            return False
        supplied = supplied_identity.model_dump(
            mode="json", by_alias=True, exclude_none=False
        )
        registered = registered_identity.model_dump(
            mode="json", by_alias=True, exclude_none=False
        )
        registered["sourceVersion"] = None
        return supplied == registered

    def list_source_identities(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        lineage_id: str,
    ) -> list[SourceIdentity]:
        rows = self._conn.execute(
            sql_query("""SELECT * FROM knowledge_builder_source_identities
            WHERE organization_id = ? AND workspace_id = ? AND lineage_id = ?
            ORDER BY created_at, source_identity_ref""", 'SELECT * FROM knowledge_builder_source_identities\n            WHERE organization_id = %s AND workspace_id = %s AND lineage_id = %s\n            ORDER BY created_at, source_identity_ref'),
            (organization_id, workspace_id, lineage_id),
        ).fetchall()
        try:
            return [self._verified_source_identity_row(row) for row in rows]
        except ValueError as exc:
            if "source identity registry integrity" in str(exc):
                raise
            raise ValueError("source identity registry integrity failure") from exc

    def _source_row_for_lineage(
        self,
        *,
        organization_id: str,
        workspace_id: str,
        lineage_id: str,
        required: bool = True,
    ) -> dict[str, Any] | None:
        rows = self._conn.execute(
            sql_query("""SELECT * FROM knowledge_builder_sources
            WHERE organization_id = ? AND workspace_id = ?
              AND COALESCE(lineage_id, source_id) = ?
            ORDER BY created_at, source_id""", 'SELECT * FROM knowledge_builder_sources\n            WHERE organization_id = %s AND workspace_id = %s\n              AND COALESCE(lineage_id, source_id) = %s\n            ORDER BY created_at, source_id'),
            (organization_id, workspace_id, lineage_id),
        ).fetchall()
        if not rows:
            if required:
                raise ValueError("existing governed source not found")
            return None
        if len(rows) != 1:
            raise ValueError("source identity registry integrity duplicate lineage")
        row = _row_to_dict(rows[0])
        if row is None:
            raise ValueError("source identity registry integrity invalid source row")
        self._verify_source_row(row)
        return row

    @staticmethod
    def _verify_source_row(row: dict[str, Any]) -> None:
        if not isinstance(row.get("source_metadata"), dict):
            raise ValueError("source identity registry integrity invalid metadata")
        hashes = {_source_row_content_hash(row)}
        if row.get("lineage_id") is None:
            hashes.add(_source_row_content_hash(row, include_lineage=False))
        if row.get("content_hash") not in hashes:
            raise ValueError("source identity registry integrity record hash mismatch")

    @staticmethod
    def _normalize_source_hash(value: Any) -> str:
        raw = str(value or "")
        if re.fullmatch(r"[0-9a-f]{64}", raw):
            return f"sha256:{raw}"
        if re.fullmatch(r"sha256:[0-9a-f]{64}", raw):
            return raw
        raise ValueError("governed source stored source hash mismatch")

    @staticmethod
    def _governed_source_version(
        *,
        row: dict[str, Any],
        source_identity: SourceIdentity,
        lineage_id: str,
        stored_hash: str,
    ) -> str:
        stored_version = row["source_metadata"].get("source_version")
        if stored_version is not None:
            value = str(stored_version)
            if not value or not value.isascii():
                raise ValueError(
                    "governed source stored source version must be ASCII-stable"
                )
            return value
        version_binding = {
            "schema_version": "semantier.governed_source_record_version.v1",
            "source_id": row["source_id"],
            "organization_id": row["organization_id"],
            "workspace_id": row["workspace_id"],
            "lineage_id": lineage_id,
            "source_record_content_hash": row["content_hash"],
            "source_identity_ref": source_identity.source_identity_ref,
            "source_hash": stored_hash,
            "artifact_ref": source_identity.artifact_ref,
            "supersedes_source_identity_ref": (
                source_identity.supersedes_source_identity_ref
            ),
        }
        return f"source-record:sha256:{compute_content_hash(version_binding)}"

    def _verified_source_identity_row(self, row: sqlite3.Row) -> SourceIdentity:
        try:
            payload = json.loads(row["source_identity_json"])
            evidence = json.loads(row["registration_evidence_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "source identity registry integrity invalid JSON"
            ) from exc
        if not isinstance(payload, dict) or not isinstance(evidence, dict):
            raise ValueError("source identity registry integrity invalid payload")
        identity_hash = f"sha256:{compute_content_hash(payload)}"
        evidence_hash = f"sha256:{compute_content_hash(evidence)}"
        if row["schema_version"] != _SOURCE_IDENTITY_RECORD_SCHEMA:
            raise ValueError("source identity registry integrity record schema")
        if row["source_identity_hash"] != identity_hash:
            raise ValueError("source identity registry integrity identity hash")
        if evidence.get("schema_version") != _SOURCE_IDENTITY_EVIDENCE_SCHEMA:
            raise ValueError("source identity registry integrity evidence schema")
        if row["registration_evidence_hash"] != evidence_hash:
            raise ValueError("source identity registry integrity evidence hash")
        identity = SourceIdentity.model_validate(payload)
        expected_columns = {
            "source_identity_ref": identity.source_identity_ref,
            "organization_id": identity.organization_id,
            "workspace_id": identity.workspace_id,
            "source_hash": identity.source_hash,
            "source_version": identity.source_version,
            "supersedes_source_identity_ref": identity.supersedes_source_identity_ref,
            "artifact_ref": identity.artifact_ref,
            "artifact_hash": identity.source_hash,
            "media_type": identity.media_type,
            "original_name": identity.original_name,
            "created_at": identity.created_at.isoformat(),
        }
        if any(row[field] != value for field, value in expected_columns.items()):
            raise ValueError("source identity registry integrity column pin")
        evidence_columns = {
            "source_id": row["source_id"],
            "source_version": row["source_version"],
            "artifact_ref": row["artifact_ref"],
            "artifact_hash": row["artifact_hash"],
        }
        if any(evidence.get(field) != value for field, value in evidence_columns.items()):
            raise ValueError("source identity registry integrity evidence pin")
        record = {
            "schema_version": row["schema_version"],
            "source_identity_ref": row["source_identity_ref"],
            "source_id": row["source_id"],
            "organization_id": row["organization_id"],
            "workspace_id": row["workspace_id"],
            "lineage_id": row["lineage_id"],
            "source_hash": row["source_hash"],
            "source_version": row["source_version"],
            "supersedes_source_identity_ref": row["supersedes_source_identity_ref"],
            "artifact_ref": row["artifact_ref"],
            "artifact_hash": row["artifact_hash"],
            "media_type": row["media_type"],
            "original_name": row["original_name"],
            "created_at": row["created_at"],
            "source_identity": payload,
            "source_identity_hash": row["source_identity_hash"],
            "registration_evidence": evidence,
            "registration_evidence_hash": row["registration_evidence_hash"],
        }
        expected_record_hash = f"sha256:{compute_content_hash(record)}"
        if row["content_hash"] != expected_record_hash:
            raise ValueError("source identity registry integrity record hash")
        return identity

    def _verify_source_artifact(self, identity: SourceIdentity, *, source_metadata: dict[str, Any] | None = None) -> Path:
        if self.workspace_root is None:
            raise ValueError(
                "governed workspace root is required for source identity authority"
            )
        artifact_ref = Path(identity.artifact_ref)
        if artifact_ref.is_absolute():
            raise ValueError("governed source artifact must be workspace-relative")
        workspace_path = (
            self.workspace_root
            if self.workspace_root.name == identity.workspace_id
            and self.workspace_root.parent.name == "workspaces"
            else self.workspace_root / "workspaces" / identity.workspace_id
        ).resolve()
        artifact = (workspace_path / artifact_ref).resolve()
        if workspace_path not in artifact.parents:
            raise ValueError("governed source artifact escapes workspace")
        if not artifact.is_file():
            raise ValueError("governed source artifact not found")
        actual_hash = f"sha256:{sha256(artifact.read_bytes()).hexdigest()}"
        if actual_hash != identity.source_hash:
            raise ValueError("governed source artifact hash mismatch")
        digest = identity.source_hash.removeprefix("sha256:")
        canonical_original = f"artifacts/workspace-originals/sha256/{digest[:2]}/{digest}"
        expected_name = artifact.name
        if identity.artifact_ref == canonical_original:
            if source_metadata is None:
                row = self._conn.execute(
                    sql_query("SELECT * FROM knowledge_builder_source_identities WHERE organization_id=? AND workspace_id=? AND source_identity_ref=?", 'SELECT * FROM knowledge_builder_source_identities WHERE organization_id=%s AND workspace_id=%s AND source_identity_ref=%s'),
                    (identity.organization_id, identity.workspace_id, identity.source_identity_ref),
                ).fetchone()
                if row is None or self._verified_source_identity_row(row) != identity:
                    raise ValueError("governed source original name identity mismatch")
                source = self._source_row_for_lineage(
                    organization_id=identity.organization_id,
                    workspace_id=identity.workspace_id,
                    lineage_id=row["lineage_id"],
                )
                source_metadata = source["source_metadata"]
            expected_name = source_metadata.get("original_name")
            if not isinstance(expected_name, str) or not expected_name or any(character in expected_name for character in ("/", "\\", "\x00", ":")):
                raise ValueError("governed source original name metadata required")
        if expected_name != identity.original_name:
            raise ValueError("governed source original name mismatch")
        return artifact

    def find_source_by_governed_id(self, knowledge_source_id: str) -> dict[str, Any] | None:
        for row in self._conn.execute("SELECT * FROM knowledge_builder_sources").fetchall():
            item = _row_to_dict(row)
            if item and (item.get("source_metadata") or {}).get("knowledge_source_id") == knowledge_source_id:
                return item
        return None

    def get_discovery_run(self, run_id: str) -> dict[str, Any] | None:
        return _row_to_dict(
            self._conn.execute(
                sql_query("SELECT * FROM knowledge_builder_discovery_runs WHERE discovery_run_id = ?", 'SELECT * FROM knowledge_builder_discovery_runs WHERE discovery_run_id = %s'),
                (run_id,),
            ).fetchone()
        )

    def _list(self, table: str, column: str, value: str, order_by: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            sql_query(f"SELECT * FROM {table} WHERE {column} = ? ORDER BY {order_by}", f"SELECT * FROM {table} WHERE {column} = %s ORDER BY {order_by}"),
            (value,),
        ).fetchall()
        return [_row_to_dict(row) for row in rows if row is not None]  # type: ignore[list-item]
